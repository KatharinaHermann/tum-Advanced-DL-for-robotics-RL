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
from hwr.utils import load_params


# loading params:
params = load_params('params/hyperparam_tuning_params.json')

#Initialize the environment
env = gym.make(
    params["env"]["name"],
    params=params,
    )
test_env = gym.make(
    params["env"]["name"],
    params=params
    )

# deleting the previous runs logs:
logdir_files = glob.glob(params["trainer"]["logdir"])
for f in logdir_files:
    if os.path.isdir(f):
        shutil.rmtree(f)
    else:
        os.remove(f)


# Hyperparameter grid search
for lr_i, lr in enumerate([5e-4, 1e-4, 5e-5]):
    for max_grad_i, max_grad in enumerate([1, 0.5, 0.1]):
        for tau_i, tau in enumerate([0.005, 0.001, 0.0005]):
            for memory_capacity_i, memory_capacity in enumerate([1e6]):
                print("Learning rate: {0: 1.8f} max_grad: {1: 3.2f} Tau_Target_update: {2: 1.3f}  memory_capacity: {3: 4}".format(
                            lr, max_grad, tau, memory_capacity))

                # the actual parameters:
                params["agent"]["lr_actor"] = lr
                params["agent"]["lr_critic"] = lr
                params["agent"]["max_grad"] = max_grad
                params["agent"]["tau"] = tau
                params["agent"]["memory_capacity"] = memory_capacity
                
                # setting up logdir for the current hyperparams:
                logdir = os.path.join('results/hyperparam_tuning',
                    str(lr_i)+str(max_grad_i)+str(tau_i)+str(memory_capacity_i))
                if not os.path.exists(logdir):
                    os.makedirs(logdir)
                logdir_files = glob.glob(logdir + '/*')
                for f in logdir_files:
                    if os.path.isdir(f):
                        shutil.rmtree(f)
                    else:
                        os.remove(f)

                # writing the hyperparameters into a file:
                info_file = os.path.join(logdir, 'params.txt')
                with open(info_file, 'a') as f:
                    f.write('learning rate: {0: 1.8f}'.format(lr) + '\n')
                    f.write('max_grad: {0: 3.2f}'.format(max_grad) + '\n')
                    f.write('tau: {0: 1.3f}'.format(tau) + '\n')
                    f.write('batch size: {0: 4}'.format(memory_capacity) + '\n')

                # deleting the previous checkpoints:
                ckp_files = glob.glob('../models/agents/*')
                for f in ckp_files:
                    os.remove(f)

                params["trainer"]["logdir"] = logdir

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

                print('-' * 5 + "Let's start training" + '-' * 5)

                trainer()