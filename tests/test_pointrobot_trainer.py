import os
import sys
import numpy as np
import tensorflow as tf

import gym
import gym_pointrobo

from tf2rl.algos.ddpg import DDPG

from hwr.cae.cae import CAE
from hwr.training.pointrobot_trainer import PointrobotTrainer


def test_pointrobot_trainer_init():

    parser = PointrobotTrainer.get_argument()
    parser = DDPG.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="pointrobo-v0")
    args = parser.parse_args()

    #Initialize the environment
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
    trainer = PointrobotTrainer(policy, env, args, test_env=test_env)


if __name__ == '__main__':

    #test_pointrobot_trainer_init()
    #print('All tests have run successfully!')

    model = CAE(pooling='max',
                    latent_dim=16,
                    input_shape=(32, 32),
                    conv_filters=[4, 8, 16])
    model.build(input_shape=(1, 32, 32, 1))
    model.load_weights(filepath='../models/cae/model_num_5_size_8.h5')

    for k, _ in model._get_trainable_state().items():
        k.trainable = False

    env = gym.make("pointrobo-v0")
    workspace, goal, obs = env.reset()
    print('ws type: {}, shape: {}'.format(type(workspace), workspace.shape))
    print('goal type: {}, shape: {}'.format(type(goal), goal.shape))
    print('obs type: {}, shape: {}'.format(type(obs), obs.shape))

    print(np.concatenate((goal, obs)).shape)
