import os
import sys
import glob
import shutil
import numpy as np
import tensorflow as tf

import gym
import gym_pointrobo

from hwr.agents.pointrobo_ddpg import DDPG
from hwr.cae.cae import CAE
from hwr.training.pointrobot_trainer import PointrobotTrainer
from hwr.utils import load_params, get_random_params, export_params


# loading params:
params = load_params('params/hyperparam_tuning_params.json')

# deleting the previous runs logs:
logdir_files = glob.glob(os.path.join('results', 'hyperparam_tuning'))
for f in logdir_files:
    if os.path.isdir(f):
        shutil.rmtree(f)
    else:
        os.remove(f)

for run in range(params["hyper_tuning"]["num_of_runs"]):

    # getting random hyperparams according to the ranges and placing them into params.
    params = get_random_params(params)

    # setting up logdir for the current hyperparams:
    logdir = os.path.join('results', 'hyperparam_tuning', str(run))
    os.makedirs(logdir)
    params["trainer"]["logdir"] = logdir

    # write the actual hyperparams into a file:
    export_params(params)

    #Initialize the environment
    env = gym.make(
        params["env"]["name"],
        params=params,
        )
    test_env = gym.make(
        params["env"]["name"],
        params=params
        )

    # deleting the previous checkpoints:
    ckp_files = glob.glob(params["trainer"]["model_dir"])
    for f in ckp_files:
        os.remove(f)

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
        test_env=test_env
        )

    trainer.train()