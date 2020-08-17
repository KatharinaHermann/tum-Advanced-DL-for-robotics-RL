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
params = load_params('params/benchmark_evaluations.json')
benchmark_keys = params["benchmark"].keys()

for key in benchmark_keys:
    # loading original params:
    params = load_params('params/benchmark_evaluations.json')

    # setting up training run:
    params = set_up_benchmark_params(params, key)
    params["trainer"]["logdir"] = os.path.join(params["trainer"]["logdir"], key)

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

    avg_test_return, success_rate, ratio_straight_lines, success_rate_straight_line, success_rate_no_straight_line = trainer.evaluate()

    # writing the results into a log file.
    results_path = os.path.join(params["trainer"]["logdir"], "results.txt")
    with open(results_path, 'a') as f:
        f.write("model_dir: {}".format(params["trainer"]["model_dir"]) + "\n")
        f.write("WS_level: {}".format(params["env"]["WS_level"]) + "\n")
        f.write("num_obj_max: {}".format(params["env"]["num_obj_max"]) + "\n")
        f.write("avg_test_return: {}".format(avg_test_return) + "\n")
        f.write("success_rate: {}".format(success_rate) + "\n")
        f.write("ratio_straight_lines: {}".format(ratio_straight_lines) + "\n")
        f.write("success_rate_straight_line: {}".format(success_rate_straight_line) + "\n")
        f.write("success_rate_no_straight_line: {}".format(success_rate_no_straight_line) + "\n")
