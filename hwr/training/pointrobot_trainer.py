import os
import sys
import time
import logging
import argparse
import joblib
import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.spaces import Box

from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer

# sys.path.append(os.path.join(os.getcwd(), "lib"))
from hwr.cae.cae import CAE
from hwr.relabeling.pointrobot_relabeling import PointrobotRelabeler
from hwr.utils import visualize_trajectory


if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


class PointrobotTrainer:
    def __init__(
            self,
            policy,
            env,
            params,
            test_env=None):
        """Initializing the training instance."""

        self._params = params
        self._set_from_params()
        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env
        args = self._get_args_from_params()

        # Convolutional Autoencoder:
        self._CAE = CAE(pooling=self._params["cae"]["pooling"],
                        latent_dim=self._params["cae"]["latent_dim"],
                        input_shape=self._env.workspace.shape,
                        conv_filters=self._params["cae"]["conv_filters"])
        self._CAE.build(input_shape=(1, self._env.workspace.shape[0], self._env.workspace.shape[1], 1))
        self._CAE.load_weights(filepath=self._params["cae"]["weights_path"])
        for layer, _ in self._CAE._get_trainable_state().items():
            layer.trainable = False

        #Initialize array for trajectory storage
        self.trajectory=[]

        # Initialize workspace relabeler:
        self._relabeler = PointrobotRelabeler(
            ws_shape=(self._env.grid_size, self._env.grid_size),
            mode=params["trainer"]["relabeling_mode"]
            )

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            suffix="{}_{}".format(self._policy.policy_name, params["trainer"]["dir_suffix"]))
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(params["trainer"]["logging_level"]),
            output_dir=self._output_dir)
        if self._save_test_path_sep:
            sep_logdirs = ['successful_trajs', 'unsuccessful_trajs', 'unfinished_trajs']
            for logdir in sep_logdirs:
                if not os.path.exists(os.path.join(self._logdir, logdir)):
                    os.makedirs(os.path.join(self._logdir, logdir))

        if params["trainer"]["mode"] == "evaluate":
            assert glob.glob(os.path.join(params["trainer"]["model_dir"], '*'))
        self._set_check_point(params["trainer"]["model_dir"])

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()

        # relabeling visualization:
        self._relabel_fig = plt.figure(2)


    def _set_check_point(self, model_dir):
        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=model_dir, max_to_keep=5)

        if model_dir is not None:
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))


    def train(self):
        """method for training an agent with Hindsight Workspace Relabeling"""

        # training mode:
        self._policy.eval_mode = False

        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0
        success_traj_train = 0.

        relabeling_times, training_times = [], []

        #Initialize replay buffer
        self._replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)

        # resetting:
        self.trajectory = []
        workspace, goal, obs = self._env.reset()
        
        #Concatenate position observation with start, goal, and reduced workspace
        reduced_workspace = self._CAE.evaluate(workspace)
        obs_full = np.concatenate((obs, goal, reduced_workspace))
        

        while total_steps < self._max_steps:
        
            #Visualize environment if "show_progess"
            if self._show_progress and \
                ((n_episode % self._show_progress_interval) == 0) and \
                total_steps > self._policy.n_warmup:
                self._env.render()

            #Get action randomly for warmup /from Actor-NN otherwise
            if total_steps < self._policy.n_warmup:
                action = self._env.action_space.sample()
            else:
                action = self._policy.get_action(obs_full)

            #Take action and get next_obs, reward and done_flag from environment
            next_obs, reward, done, _ = self._env.step(action)
            next_obs_full = np.concatenate((next_obs, goal, reduced_workspace))

            # add the new point to replay buffer
            self._replay_buffer.add(obs=obs_full, act=action,
                next_obs=next_obs_full, rew=reward, done=done)

            #Add obersvation to the trajectory storage
            self.trajectory.append({'workspace': workspace,'position': obs,
                'next_position': next_obs,'goal': goal, 'action': action, 'reward': reward, 'done': done})

            obs = next_obs
            obs_full = next_obs_full        
            
            episode_steps += 1
            episode_return += reward
            total_steps += 1
            tf.summary.experimental.set_step(total_steps)

            if done or episode_steps == self._episode_max_steps:
                
                if (reward != self._env.goal_reward):
                    """Workspace relabeling"""

                    relabeling_begin = time.time()
                    # Create new workspace for the trajectory:
                    relabeled_trajectory = self._relabeler.relabel(trajectory=self.trajectory, env=self._env)

                    if len(relabeled_trajectory) != 0:
                        relabeled_ws = relabeled_trajectory[0]['workspace']
                        relabeled_reduced_ws = self._CAE.evaluate(relabeled_ws)
                        
                        # adding the points of the relabeled trajectory to the replay buffer:
                        for point in relabeled_trajectory:
                            relabeled_obs_full = np.concatenate((point['position'],
                                point['goal'], relabeled_reduced_ws))
                            relabeled_next_obs_full = np.concatenate((point['next_position'],
                                point['goal'], relabeled_reduced_ws))
                            self._replay_buffer.add(obs=relabeled_obs_full, act=point['action'],
                                next_obs=relabeled_next_obs_full, rew=point['reward'], done=point['done'])

                    relabeling_times.append(time.time() - relabeling_begin)

                    # plotting the relabeled trajectory:
                    self._relabel_fig = visualize_trajectory(
                        trajectory=relabeled_trajectory,
                        fig=self._relabel_fig,
                        env=self._env
                        )
                    plt.pause(1)
                else:
                    success_traj_train += 1

                # resetting:
                workspace, goal, obs = self._env.reset()
                reduced_workspace = self._CAE.evaluate(workspace)
                obs_full = np.concatenate((obs, goal, reduced_workspace))
                self.trajectory = []

                #Print out train accuracy
                n_episode += 1
                if n_episode % self._test_episodes == 0:
                    train_sucess_rate = success_traj_train / self._test_episodes

                    fps = episode_steps / (time.perf_counter() - episode_start_time)
                    self.logger.info("Total Epi: {0: 5} Train sucess rate: {1: 5.4f} Total Steps: {2: 7} Episode Steps: {3: 5} Return: {4: 5.4f} Last reward: {5: 5.4f} FPS: {6: 5.2f}".format(
                        n_episode, train_sucess_rate, total_steps, episode_steps, episode_return, reward, fps))
                    tf.summary.scalar(
                        name="Common/training_return", data=episode_return)
                    tf.summary.scalar(
                        name="Common/training_success_rate", data=train_sucess_rate)
                    success_traj_train = 0

                    if len(relabeling_times) != 0:
                        print('average relabeling time: {}'.format(sum(relabeling_times) / len(relabeling_times)))
                        relabeling_times = []
                    if len(training_times) != 0:
                        print('average training time: {}'.format(sum(training_times) / len(training_times)))
                        training_times = []

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

            #While warmup, we only produce experiences without training 
            if total_steps <= self._policy.n_warmup:
                continue
            
            # After every Update_interval we want to train/update the Actor-NN, Critic-NN, 
            # and the Target-Actor-NN & Target-Critic-NN
            if total_steps % self._policy.update_interval == 0:
                training_begin = time.time()
                #Sample a new batch of experiences from the replay buffer for training
                samples = self._replay_buffer.sample(self._policy.batch_size)
                
                with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                    # Here we update the Actor-NN, Critic-NN, and the Target-Actor-NN & Target-Critic-NN 
                    # after computing the Critic-loss and the Actor-loss
                    self._policy.train(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32),
                        None if not self._use_prioritized_rb else samples["weights"])
            
                if self._use_prioritized_rb:
                    #Here we compute the Td-Critic-Loss/error
                    td_error = self._policy.compute_td_error(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32))
                    self._replay_buffer.update_priorities(
                        samples["indexes"], np.abs(td_error) + 1e-6)

                training_times.append(time.time() - training_begin)

            # Every test_interval we want to test our agent 
            if total_steps % self._test_interval == 0:
                
                # setting evaluation mode for deterministic actions:
                self._policy.eval_mode = True

                avg_test_return, success_rate = self.evaluate_policy(total_steps)
                self.logger.info("Evaluation: Total Steps: {0: 7} Average Reward {1: 5.4f} and Sucess rate: {2: 5.4f} for {3: 2} episodes".format(
                    total_steps, avg_test_return, success_rate, self._test_episodes))
                tf.summary.scalar(
                    name="Common/average_test_return", data=avg_test_return)
                tf.summary.scalar(
                    name="Common/test_success_rate", data=success_rate)
                tf.summary.scalar(name="Common/fps", data=fps)
                self.writer.flush()

                # setting evaluation mode back to false:
                self._policy.eval_mode = False

            # Every save_model_interval we save the model
            if total_steps % self._save_model_interval == 0:
                self.checkpoint_manager.save()

        tf.summary.flush()

    
    def evaluate(self):
        """method for evaluating a pretrained agent for some episodes."""
        self._policy.eval_mode = True

        avg_test_return, success_rate = self.evaluate_policy(total_steps=0)
        print("----- Evaluation -----")
        print("average test return: {}".format(avg_test_return))
        print("average test success rate: {}".format(success_rate))


    def evaluate_policy_continuously(self):
        """
        Periodically search the latest checkpoint, and keep evaluating with the latest model until user kills process.
        """
        if self._model_dir is None:
            self.logger.error("Please specify model directory by passing command line argument `--model-dir`")
            exit(-1)

        self.evaluate_policy(total_steps=0)
        while True:
            latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)
            if self._latest_path_ckpt != latest_path_ckpt:
                self._latest_path_ckpt = latest_path_ckpt
                self._checkpoint.restore(self._latest_path_ckpt)
                self.logger.info("Restored {}".format(self._latest_path_ckpt))
            self.evaluate_policy(total_steps=0)

    def evaluate_policy(self, total_steps):
        """evaluating the policy."""

        tf.summary.experimental.set_step(total_steps)
        
        total_test_return = 0.
        success_traj = 0
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)

        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []
            workspace, goal, obs = self._test_env.reset()
            reduced_workspace = self._CAE.evaluate(workspace)
            #Concatenate position observation with start, goal, and reduced workspace!!
            obs_full = np.concatenate((obs, goal, reduced_workspace))

            for _ in range(self._episode_max_steps):
                action = self._policy.get_action(obs_full)
                next_obs, reward, done, _ = self._test_env.step(action)
                #Concatenate position observation with start, goal, and reduced workspace!!
                next_obs_full = np.concatenate((obs, goal, reduced_workspace))

                # Add obersvation to the trajectory storage
                self.trajectory.append({'workspace': workspace,'position': obs,
                    'next_position': next_obs,'goal': goal, 'action': action, 'reward': reward, 'done': done})
                
                if self._save_test_path:
                    replay_buffer.add(obs=obs_full, act=action,
                                next_obs=next_obs_full, rew=reward, done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='plot'))

                elif self._show_test_progress:
                    self._test_env.render()

                episode_return += reward
                obs = next_obs
                obs_full = next_obs_full
                
                if done:
                    break

            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
                total_steps, i, episode_return)

            if self._save_test_path:
                save_path(replay_buffer._encode_sample(np.arange(self._episode_max_steps)),
                          os.path.join(self._output_dir, prefix + ".pkl"))
                replay_buffer.clear()

            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)

            if self._save_test_path_sep:
                self._save_traj_separately(prefix)
                   
            total_test_return += episode_return

            if reward == self._test_env.goal_reward:        
                success_traj += 1

            # empty trajectory:
            self.trajectory = []

        if self._show_test_images:
            images = tf.cast(
                tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=2),
                tf.uint8)
            tf.summary.image('train/input_img', images,)

        avg_test_return = total_test_return / self._test_episodes
        success_rate = success_traj / self._test_episodes

        return avg_test_return, success_rate


    def _save_traj_separately(self, prefix):
        """Saves the test trajectories into separate folders under the logdir
        based on the ending of the trajectory.
        """
        last_reward = self.trajectory[-1]['reward']

        if last_reward == self._env.goal_reward:
            log_dir = os.path.join(self._logdir, 'successful_trajs')
        elif last_reward == self._env.collision_reward:
            log_dir = os.path.join(self._logdir, 'unsuccessful_trajs')
        else:
            log_dir = os.path.join(self._logdir, 'unfinished_trajs')

        file_name = os.path.join(log_dir, prefix + '.pkl')
        joblib.dump(self.trajectory, file_name)


    def _set_from_params(self):
        # experiment settings
        self._max_steps = self._params["trainer"]["max_steps"]
        self._episode_max_steps = self._params["trainer"]["episode_max_steps"] \
            if self._params["trainer"]["episode_max_steps"] is not None \
            else self._params["trainer"]["max_steps"]
        self._n_experiments = self._params["trainer"]["n_experiments"]
        self._show_progress = self._params["trainer"]["show_progress"]
        self._show_progress_interval = self._params["trainer"]["show_progress_interval"]
        self._save_model_interval = self._params["trainer"]["save_model_interval"]
        self._save_summary_interval = self._params["trainer"]["save_summary_interval"]
        self._normalize_obs = self._params["trainer"]["normalize_obs"]
        self._logdir = self._params["trainer"]["logdir"]
        self._model_dir = self._params["trainer"]["model_dir"]
        # replay buffer
        self._use_prioritized_rb = self._params["trainer"]["use_prioritized_rb"]
        self._use_nstep_rb = self._params["trainer"]["use_nstep_rb"]
        self._n_step = self._params["trainer"]["n_step"]
        # test settings
        self._test_interval = self._params["trainer"]["test_interval"]
        self._show_test_progress = self._params["trainer"]["show_test_progress"]
        self._test_episodes = self._params["trainer"]["test_episodes"]
        self._save_test_path = self._params["trainer"]["save_test_path"]
        self._save_test_path_sep = self._params["trainer"]["save_test_path_sep"]
        self._save_test_movie = self._params["trainer"]["save_test_movie"]
        self._show_test_images = self._params["trainer"]["show_test_images"]


    def _get_args_from_params(self):
        """creates an argparse Namespace object from params for the tf2rl based classes."""

        args = {}
        for key in self._params["trainer"]:
            args[key] = self._params["trainer"][key]

        return argparse.Namespace(**args)

