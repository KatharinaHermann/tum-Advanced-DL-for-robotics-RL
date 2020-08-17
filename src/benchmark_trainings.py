import os
import sys
import numpy as np
import tensorflow as tf
import glob
import json
import shutil

import gym
import gym_pointrobo

from hwr.agents.pointrobo_ddpg import DDPG
from hwr.cae.cae import CAE
from hwr.training.pointrobot_trainer import PointrobotTrainer
from hwr.utils import load_params, set_up_benchmark_params


# loading the params:
params = load_params('params/benchmark_trainings.json')
benchmark_keys = params["benchmark"].keys()

# deleting the previous runs logs:
logdir_files = glob.glob(os.path.join(params["trainer"]["logdir"], "*"))
for f in logdir_files:
    if os.path.isdir(f):
        shutil.rmtree(f)
    else:
        os.remove(f)

for key in benchmark_keys:
    # loading original params:
    params = load_params('params/benchmark_params.json')

    # deleting the previous checkpoints:
    if os.path.isdir(params["trainer"]["model_dir"]):
        ckp_files = glob.glob(os.path.join(params["trainer"]["model_dir"], '*'))
        for f in ckp_files:
            os.remove(f)

    # setting up training run:
    params = set_up_benchmark_params(params, key)
    params["trainer"]["logdir"] = os.path.join(params["trainer"]["logdir"], key)
    param_log_path = os.path.join(params["trainer"]["logdir"], "params.json")        

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

    # saving the params:
    param_log_path = os.path.join(params["trainer"]["logdir"], "params.json")
    with open(param_log_path, 'w') as f:
        json.dump(params["benchmark"][key], f)

    trainer.train()