import os
import sys
import numpy as np
import tensorflow as tf

import gym
import gym_pointrobo

from tf2rl.algos.ddpg import DDPG

#sys.path.append(os.path.join(os.getcwd(), "lib"))
from lib.cae.cae import CAE
from lib.training.pointrobot_trainer import Trainer


parser = Trainer.get_argument()
parser = DDPG.get_argument(parser)
parser.add_argument('--env-name', type=str, default="pointrobo-v0")
parser.set_defaults(batch_size=100)
parser.set_defaults(n_warmup=10000)
args = parser.parse_args()

#Initialize the environment
env = gym.make(args.env_name)
test_env = gym.make(args.env_name)
#env = PointroboEnv()
#test_env = PointroboEnv()

policy = DDPG(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=args.gpu,
    memory_capacity=args.memory_capacity,
    max_action=env.action_space.high[0],
    batch_size=args.batch_size,
    n_warmup=args.n_warmup)
trainer = Trainer(policy, env, args, test_env=test_env)
trainer()