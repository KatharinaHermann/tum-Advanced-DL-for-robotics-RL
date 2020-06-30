import os
import sys
import numpy as np
import tensorflow as tf

import gym
import gym_pointrobo

from tf2rl.algos.ddpg import DDPG

from hwr.cae.cae import CAE
from hwr.training.pointrobot_trainer import PointrobotTrainer


parser = PointrobotTrainer.get_argument()
parser = DDPG.get_argument(parser)
parser.add_argument('--env-name', type=str, default="pointrobo-v0")
parser.set_defaults(batch_size=1024)
parser.set_defaults(n_warmup=10000)
parser.set_defaults(update_interval=1)

args = parser.parse_args()

hyperparameter_search = False

args.max_steps = 1e6
args.test_interval = 10000
args.episode_max_steps = 50
args.test_episodes = 100
args.save_test_path_sep = True
#args.save_test_movie = True
args.show_progress = True
args.num_obj_max = 5
#args.n_warmup = 0

lr_actor = 1e-4
lr_critic = 1e-4

#Initialize the environment
env = gym.make(
    args.env_name,
    goal_reward=10,
    collision_reward=-1,
    step_reward=-0.05,
    buffer_size=1000,
    grid_size=32,
    num_obj_max=args.num_obj_max,
    obj_size_avg=args.obj_size_avg,
    )
test_env = gym.make(
    args.env_name,
    goal_reward=10,
    collision_reward=-1,
    step_reward=-0.05,
    buffer_size=1000,
    grid_size=32,
    num_obj_max=args.num_obj_max,
    obj_size_avg=args.obj_size_avg,
    )


if hyperparameter_search:
    # Hyperparameter grid search
    for lr in [2e-6, 8e-6, 1e-5]:
        for sig in [ 0.01, 0.1, 0.5]:
            for tau in [0.05, 0.005, 0.0005]:
                print("Learning rate: {0: 5.6f} Sigma_action: {1: 5.6f} Tau_Target_update: {2: 5.6f} ".format(
                            lr, sig, tau))

                # initialize the agent:
                policy = DDPG(
                    state_shape=env.observation_space.shape,
                    action_dim=env.action_space.high.size,
                    gpu=args.gpu,
                    memory_capacity=args.memory_capacity,
                    update_interval=args.update_interval,
                    max_action=env.action_space.high[0], #max action =1
                    lr_actor=lr, #0.001 hyperparamter learning rate actor network
                    lr_critic=lr, #hyperparamter learning rate critic network
                    actor_units=[400, 300],
                    critic_units=[400, 300],
                    batch_size=args.batch_size,
                    sigma=sig,#0.1 hyperparamter: standard deviation for nrmal distributed for randomization of action with my action = 1
                    tau = tau, #0.005, #weight used to gate the update. The permitted range is 0 < tau <= 1, with small tau representing an incremental update, and tau == 1 representing a full update (that is, a straight copy).
                    n_warmup=args.n_warmup)

                trainer = PointrobotTrainer(policy, env, args, test_env=test_env)

                print('-' * 5 + "Let's start training" + '-' * 5)

                trainer()

                print('-' * 5 + "We succeeeeeded!!!!!!!!!!!!!" + '-' * 5)
else:
    # initialize the agent:
    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        update_interval=args.update_interval,
        max_action=env.action_space.high[0], #max action =1
        lr_actor=lr_actor, #0.001 hyperparamter learning rate actor network
        lr_critic=lr_critic, #hyperparamter learning rate critic network
        actor_units=[400, 300],
        critic_units=[400, 300],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)

    trainer = PointrobotTrainer(policy, env, args, test_env=test_env)

    print('-' * 5 + "Let's start training" + '-' * 5)

    trainer()

    print('-' * 5 + "We succeeeeeded!!!!!!!!!!!!!" + '-' * 5)