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
    print('-' * 5 + 'test_pointrobot_trainer_init' + '-' * 5)

    parser = PointrobotTrainer.get_argument()
    parser = DDPG.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="pointrobo-v0")
    args = parser.parse_args()

    args.save_test_path_sep = True

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


def test_state_concatenation():
    print('-' * 5 + 'test_state_concatenation' + '-' * 5)

    model = CAE(pooling='max',
                latent_dim=16,
                input_shape=(32, 32),
                conv_filters=[4, 8, 16])
    model.build(input_shape=(1, 32, 32, 1))
    model.load_weights(filepath='models/cae/model_num_5_size_8.h5')

    for layer, _ in model._get_trainable_state().items():
        layer.trainable = False

    env = gym.make("pointrobo-v0")
    workspace, goal, obs = env.reset()
    print('ws type: {}, dtype: {}, shape: {}'.format(type(workspace), workspace.dtype, workspace.shape))
    print('goal type: {}, dtype: {}, shape: {}'.format(type(goal), goal.dtype, goal.shape))
    print('obs type: {}, dtype: {}, shape: {}'.format(type(obs), obs.dtype, obs.shape))

    reduced_ws = model.evaluate(workspace)
    complete_state = np.concatenate((obs, goal, reduced_ws))
    print('complete_state type: {}, dtype: {}, shape: {}'.format(type(complete_state), complete_state.dtype, complete_state.shape))


def test_evaluation():
    """Tests only the evaluation method of the PointrobotTrainer class"""
    print('-' * 5 + 'test_evaluation' + '-' * 5)
    
    total_steps = 10

    parser = PointrobotTrainer.get_argument()
    parser = DDPG.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="pointrobo-v0")
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10)
    args = parser.parse_args()

    args.max_steps = 100
    args.save_test_path_sep = True

    #######
    # possibly set some args attributes to small numbers, so that testing does not last that long.
    # like for example args.n_warmup = 10, args.max_steps = 100...
    #######

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

    avg_return = trainer.evaluate_policy(total_steps=total_steps)


def test_training():

    print('-' * 5 + 'test_training' + '-' * 5)

    parser = PointrobotTrainer.get_argument()
    parser = DDPG.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="pointrobo-v0")
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10)
    parser.set_defaults(update_interval=2)
    args = parser.parse_args()

    args.max_steps = 1000
    args.show_progress = False

    #######
    # possibly set some args attributes to small numbers, so that testing does not last that long.
    # like for example args.n_warmup = 10, args.max_steps = 100...
    #######

    #Initialize the environment
    env = gym.make(
        args.env_name,
        goal_reward=5,
        collision_reward=-1,
        step_reward=-0.01,
        buffer_size=100,
        grid_size=32,
        num_obj_max=args.num_obj_max,
        obj_size_avg=args.obj_size_avg,
        )
    test_env = gym.make(
        args.env_name,
        goal_reward=5,
        collision_reward=-1,
        step_reward=-0.01,
        buffer_size=100,
        grid_size=32,
        num_obj_max=5,
        obj_size_avg=8,
        )

    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        update_interval=args.update_interval,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
    trainer = PointrobotTrainer(policy, env, args, test_env=test_env)
    trainer()


if __name__ == '__main__':

    test_pointrobot_trainer_init()
    test_state_concatenation()
    test_evaluation()
    test_training()
    print('All tests have run successfully!')
    

    
