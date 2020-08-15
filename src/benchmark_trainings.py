import os
import sys
import numpy as np
import tensorflow as tf
import glob
import joblib

import gym
import gym_pointrobo

from hwr.agents.pointrobo_ddpg import DDPG
from hwr.cae.cae import CAE
from hwr.training.pointrobot_trainer import PointrobotTrainer
from hwr.utils import load_params, set_up_benchmark_params


for key in params["benchmark"]:
    # loading original params:
    params = load_params('params/benchmark_params.json')

    # deleting the previous checkpoints:
    ckp_files = glob.glob(os.path.join(params["trainer"]["model_dir"], '*'))
    for f in ckp_files:
        os.remove(f)

    # setting up training run:
    params = set_up_benchmark_params(params, key)
    params["training"]["logdir"] = os.path.join(params["training"]["logdir"], key)
    param_log_path = os.path.join(params["training"]["logdir"], "params.json")
    joblib.dump(params["benchmark"][key], param_log_path)

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

    if params["trainer"]["mode"] == "train":
        trainer.train()
    elif params["trainer"]["mode"] == "evaluate":
        trainer.evaluate()
    else:
        print("Unknown training mode. Expected \"train\" or \"evaluate\", received: {}".format(params["trainer"]["mode"]))