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

        self.env = gym.make(
            self.params["env"]["name"],
            params=self.params
            )
        self.test_env = gym.make(
            self.params["env"]["name"],
            params=self.params
            )

        self.policy = DDPG(
            env=self.env,
            params=self.params
            )

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
            self.params,
            test_env=self.test_env
            )
    

    def test_evaluation(self):
        """tests the evaluation method of the pointrobot trainer"""
        
        total_steps = 10

        trainer = PointrobotTrainer(
            self.policy,
            self.env,
            self.params,
            test_env=self.test_env)

        trainer.evaluate_policy(total_steps=total_steps)
    

    def test_training(self):
        """sanity check of the training method."""

        self.params["trainer"]["max_steps"] = 1e4

        trainer = PointrobotTrainer(
            self.policy,
            self.env,
            self.params,
            test_env=self.test_env)
        trainer()


if __name__ == '__main__':
    unittest.main()
    

    
