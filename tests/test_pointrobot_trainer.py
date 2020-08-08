import unittest
import os
import sys
import numpy as np
import tensorflow as tf

import gym
import gym_pointrobo

from hwr.agents.pointrobo_ddpg import DDPG
from hwr.cae.cae import CAE
from hwr.training.pointrobot_trainer import PointrobotTrainer
from hwr.utils import load_params


class PointrobotTrainerTests(unittest.TestCase):
    """For testing the Pointrobot trainer."""

    def setUp(self):
        """setup"""
        self.params = load_params('params/test_params.json')
        self.parser = PointrobotTrainer.get_argument()
        self.parser = DDPG.get_argument(self.parser)
        self.parser.add_argument('--env-name', type=str, default="pointrobo-v0")
        self.args = self.parser.parse_args()
        self.args.save_test_path_sep = False

        self.env = gym.make(
            self.args.env_name,
            params=self.params
            )
        self.test_env = gym.make(
            self.args.env_name,
            params=self.params
            )

        self.policy = DDPG(
            state_shape=self.env.observation_space.shape,
            action_dim=self.env.action_space.high.size,
            gpu=self.args.gpu,
            memory_capacity=self.args.memory_capacity,
            #max_action=env.action_space.high[0],
            batch_size=self.args.batch_size,
            n_warmup=self.args.n_warmup)

        self.cae = CAE(pooling='max',
                    latent_dim=16,
                    input_shape=(32, 32),
                    conv_filters=[4, 8, 16])
        self.cae.build(input_shape=(1, 32, 32, 1))
        self.cae.load_weights(filepath='../models/cae/model_num_5_size_8.h5')


    def test_pointrobot_trainer_init(self):
        """tests the __init__() function of the pointrobot trainer"""
        trainer = PointrobotTrainer(
            self.policy,
            self.env,
            self.args,
            test_env=self.test_env)
    

    def test_evaluation(self):
        """tests the evaluation method of the pointrobot trainer"""
        
        total_steps = 10

        self.args.batch_size = 100
        self.args.n_warmup = 10
        self.args.max_steps = 100
        self.args.save_test_path_sep = False

        trainer = PointrobotTrainer(
            self.policy,
            self.env,
            self.args,
            test_env=self.test_env)

        avg_return = trainer.evaluate_policy(total_steps=total_steps)
    

    def test_training(self):
        """sanity check of the training method."""

        self.args.batch_size = 100
        self.args.n_warmup = 10
        self.args.max_steps = 1000
        self.args.save_test_path_sep = False
        self.args.update_interval = 2
        self.args.show_progress = False

        trainer = PointrobotTrainer(
            self.policy,
            self.env,
            self.args,
            test_env=self.test_env)
        trainer()


if __name__ == '__main__':
    unittest.main()
    

    
