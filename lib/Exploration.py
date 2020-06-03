import os
import sys
import time
import logging
import argparse

import numpy as np
import tensorflow as tf

from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger

sys.path.append(os.path.join(os.getcwd(), "lib"))
from cae import CAE 


if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


class Exploration:
    def __init__(
            self,
            policy,
            env,
            reduced_workspace,
            workspace_buffer
            args,
            test_env=None):
        self._set_from_args(args)
        self._policy = policy
        self._env = env
        self._reduced_workspace = reduced_workspace
        self._test_env = self._env if test_env is None else test_env

       # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix))
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir)
    
    
    def __call__(self, n_episode, total_steps):
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
     
        #Initialize array for trajectory storage
        trajectory=[]
        #Initialize relay buffer
        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)

        obs = self._env.reset()
        
        # Add obersvation to the trajectory storage
            trajectory.add({'position':obs,'start':self._env.start,'goal':self._env.goal})
        
        #Concatenate position observation with start, goal, and reduced workspace!!
        obs_full = [obs, self._env.start, self._env.goal, self._reduced_workspace]
        
        while done==0:
            #Get action randomly for warmup /from Actor-NN otherwise
            if total_steps < self._policy.n_warmup:
                action = self._env.action_space.sample()
            else:
                action = self._policy.get_action(obs_full)

            #Take action and get next_obs, reward and done_flag from environment
            next_obs, reward, done, _ = self._env.step(action)
            
            # Add obersvation to the trajectory storage
            trajectory.add({'position':obs,'start':self._env.start,'goal':self._env.goal})
            
            #Concatenate position observation with start, goal, and reduced workspace!!
            next_obs_full=[next_obs, self._env.start, self._env.goal, self._reduced_workspace]
            
            #Visualize environment if "show_progess"
            if self._show_progress:
                self._env.render()
            
            episode_steps += 1
            episode_return += reward
            total_steps += 1
            #tf.summary.experimental.set_step(total_steps)


            done_flag = done
            
            # Add obersvation to replay buffer
            if reward == -1
                replay_buffer.add(obs=obs_full, act=action,
                              next_obs=next_obs_full, rew=reward, done=done_flag)

                buffer_index= np.random.randint(low=0, high=buffer_size-1, size=None)
                relabeld_workspace=workspace_buffer[buffer_index]
                
                CAE_model = CAE(pooling='max', latent_dim=16, input_shape=(ws_height, ws_width), conv_filters=[4, 8, 16])
                CAE_model.load_weights(os.path.join(os.getcwd(), "models/cae/model_num_5_size_8.h5"))
                relabeld_reduced_workspace = CAE_model.evaluate(relabeld_wokspace)
                
                #Relabeld observations for replay buffer
                obs_relabeld = [obs, self._env.start, self._env.goal, relabeld_reduced_workspace]
                next_obs_relabeld = [obs, self._env.start, self._env.goal, self.relabeld_reduced_workspace] 

                replay_buffer.add(obs=obs_relabeld, act=action,
                              next_obs=next_obs_relabeld, rew=reward, done=done_flag)

            else:
                replay_buffer.add(obs=obs_full, act=action,
                              next_obs=next_obs_full, rew=reward, done=done_flag)
            obs = next_obs
            obs_full = next_obs_full
        
        obs = self._env.reset()
        
        fps = episode_steps / (time.perf_counter() - episode_start_time)
        self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
            n_episode, total_steps, episode_steps, episode_return, fps))
        tf.summary.scalar(
            name="Common/training_return", data=episode_return)

        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        
        
        
        return  replay_buffer, episode_return, total_steps

    def _set_from_args(self, args):
        # experiment settings
        self._episode_max_steps = args.episode_max_steps \
            if args.episode_max_steps is not None \
            else args.max_steps
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step


    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
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
    
        return parser