import os
import sys
import time
import logging
import argparse

import numpy as np
import tensorflow as tf
from gym.spaces import Box

from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer

# sys.path.append(os.path.join(os.getcwd(), "lib"))
from lib.cae.cae import CAE
from lib.relabeling.pointrobot_relabeling import PointrobotRelabeler


if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


class Trainer:
    def __init__(
            self,
            policy,
            env,
            args,
            test_env=None):
        self._set_from_args(args)
        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env

        # Convolutional Autoencoder:
        self._CAE = CAE(pooling=self._cae_pooling,
                        latent_dim=self._cae_latent_dim,
                        input_shape=self._env.workspace.shape,
                        conv_filters=self._cae_conv_filters)
        self._CAE.build(input_shape=(1, self._env.workspace.shape[0], self._env.workspace.shape[1], 1))
        self._CAE.load_weights(filepath=self._cae_weights_path)
        for layer, _ in self._CAE._get_trainable_state().items():
            layer.trainable = False

        #Initialize array for trajectory storage
        self.trajectory=[]

        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(
                shape=env.observation_space.shape)

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix))
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir)

        if args.evaluate:
            assert args.model_dir is not None
        self._set_check_point(args.model_dir)

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()

    def _set_check_point(self, model_dir):
        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=self._output_dir, max_to_keep=5)

        if model_dir is not None:
            assert os.path.isdir(model_dir)
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))

    def __call__(self):
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0

        #Initialize replay buffer
        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)

        # Empty trajectory list:
        self.trajectory = []

        obs = self._env.reset()
        
        #Concatenate position observation with start, goal, and reduced workspace!!
        reduced_workspace = self._CAE.evaluate(self._env.workspace)
        obs_full = [obs, self._env.start, self._env.goal, reduced_workspace]

        while total_steps < self._max_steps:
            #Get action randomly for warmup /from Actor-NN otherwise
            
            if total_steps < self._policy.n_warmup:
                action = self._env.action_space.sample()
            else:
                action = self._policy.get_action(obs_full)

            #Take action and get next_obs, reward and done_flag from environment
            next_obs, reward, done, _ = self._env.step(action)
            
            #Concatenate position observation with start, goal, and reduced workspace!!
            next_obs_full=[next_obs, self._env.start, self._env.goal, self._env.reduced_workspace]

            #Add obersvation to the trajectory storage
            self.trajectory.append({'workspace':self._env.workspace,'position':obs,
                'next_position':next_obs,'start':self._env.start,'goal':self._env.goal, 'action':action, 'reward':reward, 'done':done})

            #Visualize environment if "show_progess"
            if self._show_progress:
                self._env.render()
            
            episode_steps += 1
            episode_return += reward
            total_steps += 1
            tf.summary.experimental.set_step(total_steps)

            done_flag = done

            if hasattr(self._env, "_max_episode_steps") and \
                    episode_steps == self._env._max_episode_steps:
                done_flag = False

            # Add last obersvation to replay buffer. 
            # This is where the workspace relabeling comes into play!:
            #If the robot crashed (reward=-1), then the whole trajectory must be relabeld with a new workspace, a new goal_position, and a reward of 1 for the last element.

            if reward == -1:
                # Add fail obersvation to replay buffer
                replay_buffer.add(obs=obs_full, act=action,
                              next_obs=next_obs_full, rew=reward, done=done_flag)

                # Create new workspace ################ Insert better workspace relbeling function later ############
                buffer_index= np.random.randint(low=0, high=self._env.buffer_size-1, size=None)
                relabeled_workspace=self._env.workspace_buffer[buffer_index]
                
                #Shrink workspace to latent space
                CAE_model = CAE(pooling='max', latent_dim=16, input_shape=(self._env.grid_size, self._env.grid_size), conv_filters=[4, 8, 16])
                CAE_model.load_weights(os.path.join(os.getcwd(), "models/cae/model_num_5_size_8.h5"))
                relabeled_reduced_workspace = CAE_model.evaluate(relabeled_workspace)
                
                #Relabel all observations in the trajectory so far with the new workspace and insert to replay buffer
                relabeld_goal = self.trajectory[-1]["next_position"]
                for tra_i in self.trajectory[:-1]:
                    obs_relabeld = [tra_i["position"], self._env.start, relabeld_goal, relabeled_reduced_workspace]
                    next_obs_relabeld = [tra_i["next_position"], self._env.start, relabeld_goal, relabeled_reduced_workspace] 

                    replay_buffer.add(obs=obs_relabeld, act=tra_i["action"],
                                next_obs=next_obs_relabeld, rew=tra_i["reward"], done=tra_i["done"])

                # The last element in the trajectory must be relabeld with a new reward=1 (goal reached)
                last_traj_element=self.trajectory[-1]
                obs_relabeld = [last_traj_element["position"], self._env.start, relabeld_goal, relabeled_reduced_workspace]
                next_obs_relabeld = [last_traj_element["next_position"], self._env.start, relabeld_goal, relabeled_reduced_workspace] 

                replay_buffer.add(obs=obs_relabeld, act=last_traj_element["action"],
                            next_obs=next_obs_relabeld, rew=1, done=last_traj_element["done"])

            #Normal observation adding to replay buffer 
            else:
                # Add success obersvation to replay buffer
                replay_buffer.add(obs=obs_full, act=action,
                              next_obs=next_obs_full, rew=reward, done=done_flag)

            obs = next_obs
            obs_full = next_obs_full


            if done or episode_steps == self._episode_max_steps:
                obs = self._env.reset()

                #Concatenate position observation with start, goal, and reduced workspace!!
                obs_full = [obs, self._env.start, self._env.goal, self._env.reduced_workspace]

                n_episode += 1
                fps = episode_steps / (time.perf_counter() - episode_start_time)
                self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                    n_episode, total_steps, episode_steps, episode_return, fps))
                tf.summary.scalar(
                    name="Common/training_return", data=episode_return)

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

            #While warmup, we only produce experiences without training 
            if total_steps < self._policy.n_warmup:
                continue
            
            # After every Update_interval we want to train/update the Actor-NN, Critic-NN, and the Target-Actor-NN & Target-Critic-NN
            if total_steps % self._policy.update_interval == 0:

                #Sample a new batch of experiences from the replay buffer for training
                samples = replay_buffer.sample(self._policy.batch_size)
                
                with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                    # Here we update the Actor-NN, Critic-NN, and the Target-Actor-NN & Target-Critic-NN after computing the Critic-loss and the Actor-loss
                    self._policy.train(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32),
                        None if not self._use_prioritized_rb else samples["weights"])
               
                if self._use_prioritized_rb:
                    #Here we compute the Td-Critic-Loss/error
                    td_error = self._policy.compute_td_error(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32))
                    replay_buffer.update_priorities(
                        samples["indexes"], np.abs(td_error) + 1e-6)

            # Every test_interval we want to test our agent 
            if total_steps % self._test_interval == 0:
                #Here we evaluate the policy
                avg_test_return = self.evaluate_policy(total_steps)
                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes))
                tf.summary.scalar(
                    name="Common/average_test_return", data=avg_test_return)
                tf.summary.scalar(name="Common/fps", data=fps)
                self.writer.flush()

            # Every save_model_interval we save the model of our agent with the Actor-NN, Critic-NN, and the Target-Actor-NN & Target-Critic-NN so far.

            if total_steps % self._save_model_interval == 0:
                self.checkpoint_manager.save()

        tf.summary.flush()

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
        tf.summary.experimental.set_step(total_steps)
        if self._normalize_obs:
            self._test_env.normalizer.set_params(
                *self._env.normalizer.get_params())
        
        avg_test_return = 0.
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)

        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []
            obs = self._test_env.reset()
            #Concatenate position observation with start, goal, and reduced workspace!!
            obs_full = [obs, self._env.start, self._env.goal, self._env.reduced_workspace]

            for _ in range(self._episode_max_steps):
                action = self._policy.get_action(obs_full, test=True)

                next_obs, reward, done, _ = self._test_env.step(action)
                #Concatenate position observation with start, goal, and reduced workspace!!
                next_obs_full = [obs, self._env.start, self._env.goal, self._env.reduced_workspace]

                # Add obersvation to the trajectory storage
                self.trajectory.append({'workspace':self._env.workspace,'position':obs,
                    'next_position':next_obs,'start':self._env.start,'goal':self._env.goal, 'action':action, 'reward':reward, 'done':done})
                
                if self._save_test_path:
                    if reward == -1:
                        # Add fail obersvation to replay buffer
                        replay_buffer.add(obs=obs_full, act=action,
                                    next_obs=next_obs_full, rew=reward, done=done)

                        # Create new workspace ################ Insert better workspace relbeling function later ############
                        buffer_index= np.random.randint(low=0, high=self._env.buffer_size-1, size=None)
                        relabeled_workspace=self._env.workspace_buffer[buffer_index]
                        
                        #Shrink workspace to latent space
                        CAE_model = CAE(pooling='max', latent_dim=16, input_shape=(self._env.grid_size, self._env.grid_size), conv_filters=[4, 8, 16])
                        CAE_model.load_weights(os.path.join(os.getcwd(), "models/cae/model_num_5_size_8.h5"))
                        relabeled_reduced_workspace = CAE_model.evaluate(relabeled_workspace)
                        
                        #Relabel all observations in the trajectory so far with the new workspace and insert to replay buffer
                        relabeld_goal = self.trajectory[-1]["next_position"]
                        for tra_i in self.trajectory[:-1]:
                            obs_relabeld = [tra_i["position"], self._env.start, relabeld_goal, relabeled_reduced_workspace]
                            next_obs_relabeld = [tra_i["next_position"], self._env.start, relabeld_goal, relabeled_reduced_workspace] 

                            replay_buffer.add(obs=obs_relabeld, act=tra_i["action"],
                                        next_obs=next_obs_relabeld, rew=tra_i["reward"], done=tra_i["done"])

                        # The last element in the trajectory must be relabeld with a new reward=1 (goal reached)
                        last_traj_element=self.trajectory[-1]
                        obs_relabeld = [last_traj_element["position"], self._env.start, relabeld_goal, relabeled_reduced_workspace]
                        next_obs_relabeld = [last_traj_element["next_position"], self._env.start, relabeld_goal, relabeled_reduced_workspace] 

                        replay_buffer.add(obs=obs_relabeld, act=last_traj_element["action"],
                                    next_obs=next_obs_relabeld, rew=1, done=last_traj_element["done"])

                    else:
                        # Add success obersvation to replay buffer
                        replay_buffer.add(obs=obs_full, act=action,
                                    next_obs=next_obs_full, rew=reward, done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))

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
            
            avg_test_return += episode_return

        if self._show_test_images:
            images = tf.cast(
                tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
                tf.uint8)
            tf.summary.image('train/input_img', images,)
        return avg_test_return / self._test_episodes

    def _set_from_args(self, args):
        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = args.episode_max_steps \
            if args.episode_max_steps is not None \
            else args.max_steps
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images
        # autoencoder settings
        self._cae_pooling = args.cae_pooling
        self._cae_latent_dim = args.cae_latent_dim
        self._cae_conv_filters = args.cae_conv_filters
        self._cae_weights_path = args.cae_weights_path

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env.')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode')
        parser.add_argument('--n-experiments', type=int, default=1,
                            help='Number of experiments')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process')
        parser.add_argument('--save-model-interval', type=int, default=int(1e4),
                            help='Interval to save model')
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help='Interval to save summary')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results')
        parser.add_argument('--normalize-obs', action='store_true',
                            help='Normalize observation')
        parser.add_argument('--logdir', type=str, default='results',
                            help='Output directory')
        # test settings
        parser.add_argument('--evaluate', action='store_true',
                            help='Evaluate trained model')
        parser.add_argument('--test-interval', type=int, default=int(1e4),
                            help='Interval to evaluate trained model')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process')
        parser.add_argument('--test-episodes', type=int, default=5,
                            help='Number of episodes to evaluate at once')
        parser.add_argument('--save-test-path', action='store_true',
                            help='Save trajectories of evaluation')
        parser.add_argument('--show-test-images', action='store_true',
                            help='Show input images to neural networks when an episode finishes')
        parser.add_argument('--save-test-movie', action='store_true',
                            help='Save rendering results')
        # replay buffer
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level')
        # autoencoder related
        parser.add_argument('--cae_pooling', type=str, default='max',
                            help='pooling type of the CAE. default: max')
        parser.add_argument('--cae_latent_dim', type=int, default=16,
                            help='latent dimension of the CAE. default: 16')
        parser.add_argument('--cae_conv_filters', type=int, nargs='+', default=[4, 8, 16],
                            help='number of filters in the conv layers. default: [4, 8, 16]')
        parser.add_argument('--cae_weights_path', type=str, default='models/cae/model_num_5_size_8.h5',
                            help='path to saved CAE weights. default: models/cae/model_num_5_size_8.h5')

        return parser