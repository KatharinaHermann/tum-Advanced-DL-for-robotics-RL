import os
import sys
import glob
import shutil
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
parser.set_defaults(batch_size=10)
parser.set_defaults(n_warmup=100)
parser.set_defaults(update_interval=1)

args = parser.parse_args()

args.max_steps = 2e2
args.test_interval = 10000
args.episode_max_steps = 50
args.test_episodes = 100
args.num_obj_max = 5


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



# Hyperparameter grid search
for lr_i, lr in enumerate([2e-7, 5e-7, 8e-7, 1e-6]):
    for max_grad_i, max_grad in enumerate([0.1, 1, 10]):
        for tau_i, tau in enumerate([0.1, 0.01, 0.001]):
            for batch_size_i, batch_size in enumerate([1024, 5000]):
                print("Learning rate: {0: 1.8f} max_grad: {1: 3.2f} Tau_Target_update: {2: 1.3f}  Batch size: {3: 4}".format(
                            lr, max_grad, tau, batch_size))
                
                # setting up logdir for the current hyperparams:
                logdir = os.path.join('src/results/hyperparam_tuning',
                    str(lr_i)+str(max_grad_i)+str(tau_i)+str(batch_size_i))
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
                    f.write('batch size: {0: 4}'.format(batch_size) + '\n')

                # deleting the previous checkpoints:
                ckp_files = glob.glob('models/agents/*')
                for f in ckp_files:
                    os.remove(f)

                args.batch_size = batch_size
                args.logdir = logdir
                # initialize the agent:
                policy = DDPG(
                    state_shape=env.observation_space.shape,
                    action_dim=env.action_space.high.size,
                    gpu=args.gpu,
                    memory_capacity=args.memory_capacity,
                    update_interval=args.update_interval,
                    max_action=env.action_space.high[0],
                    lr_actor=lr, 
                    lr_critic=lr,
                    max_grad=max_grad,
                    actor_units=[400, 300],
                    critic_units=[400, 300],
                    batch_size=args.batch_size,
                    tau = tau,
                    n_warmup=args.n_warmup)

                trainer = PointrobotTrainer(policy, env, args, test_env=test_env)

                print('-' * 5 + "Let's start training" + '-' * 5)

                trainer()