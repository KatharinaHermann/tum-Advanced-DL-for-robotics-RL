import os
import sys
import numpy as np
import tensorflow as tf
import glob

import gym
import gym_pointrobo

from hwr.agents.pointrobo_ddpg import DDPG
from hwr.cae.cae import CAE
from hwr.training.pointrobot_trainer import PointrobotTrainer
from hwr.utils import load_params


# loading params:
params = load_params('src/params/training_without_obstacles.json')

if params["trainer"]["train_from_scratch"]:
    # deleting the previous checkpoints:
    ckp_files = glob.glob(os.path.join(params["trainer"]["model_dir"], '*'))
    for f in ckp_files:
        os.remove(f)
    print('-' * 5 + 'TRAINING FROM SCRATCH!! --> DELETED CHECKPOINTS!' + '-' * 5)

#Initialize the environment
env = gym.make(
    params["env"]["name"],
    params=params,
    )
test_env = gym.make(
    params["env"]["name"],
    params=params
    )

# initialize the agent:
policy = DDPG(
    env=env,
    params=params
    )

# initialize the trainer:
trainer = PointrobotTrainer(
    policy,
    env,
    params,
    test_env=test_env)

print('-' * 5 + "Let's start training" + '-' * 5)

trainer()

print('-' * 5 + "We succeeeeeded!!!!!!!!!!!!!" + '-' * 5)