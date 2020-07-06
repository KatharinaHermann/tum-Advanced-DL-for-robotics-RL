import os
import sys
import numpy as np
import tensorflow as tf

import gym
import gym_pointrobo

from hwr.agents.pointrobo_ddpg import DDPG

from hwr.cae.cae import CAE
from hwr.training.pointrobot_trainer import PointrobotTrainer


parser = PointrobotTrainer.get_argument()
parser = DDPG.get_argument(parser)
parser.add_argument('--env-name', type=str, default="pointrobo-v0")
parser.set_defaults(batch_size=1024)
parser.set_defaults(n_warmup=10000)
parser.set_defaults(update_interval=10)

args = parser.parse_args()

args.max_steps = 1e6
args.test_interval = 10000
args.episode_max_steps = 50
args.test_episodes = 100
args.save_test_path_sep = False
args.save_test_movie = False
args.show_progress = True
args.num_obj_max = 5

lr_actor = 1e-7
lr_critic = 1e-7

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

# initialize the agent:
policy = DDPG(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=args.gpu,
    memory_capacity=args.memory_capacity,
    update_interval=args.update_interval,
    #max_action=env.action_space.high[0], #max action =1
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