import os
import sys
import numpy as np
import tensorflow as tf

import gym
import gym_pointrobo

from tf2rl.algos.ddpg import DDPG

sys.path.append(os.path.join(os.getcwd(), "lib"))
from cae import CAE 
from Trainer import Trainer
from Exploration import Exploration



if __name__ == '__main__':
    parser = Explorer.get_argument()
    parser = DDPG.get_argument(parser)
    args_E = parser.parse_args()


    parser = Trainer.get_argument()
    parser = DDPG.get_argument(parser)
    args_T = parser.parse_args()

    #Initialize workspace buffer
    workspace_buffer = create_workspace_buffer()

    #Initialize workspace, start &, goal, and reduced representation
    workspace, start, goal, reduced_workspace = setup_rndm_workspace_from_buffer(workspace_buffer)
    
    env = PointroboEnv(start_pos=start, goal_pos=goal, workspace=workspace)
    test_env = PointroboEnv(start_pos=start, goal_pos=goal, workspace=workspace)
    
    #env = gym.make(args.env_name)
    #test_env = gym.make(args.env_name)
    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=100,
        n_warmup=10000)
    
    total_steps= 0
    n_episode = 1
    max_steps = int(1e6)
    
    while total_steps < max_steps
    
        # Make an episode of explorations in the sampled workspace
        explorer = Exploration(policy, env, reduced_workspace, workspace_buffer, args, test_env=test_env)
        replay_buffer, episode_return, total_steps = explorer(n_episode, total_steps)

        n_episode += 1

        # Sample a new workspace and environment for the next run
        workspace, start, goal, reduced_workspace = setup_rndm_workspace_from_buffer(workspace_buffer)
        
        env = PointroboEnv(start_pos=start, goal_pos=goal, workspace=workspace)

        # Train from the replay _buffer
        trainer = Trainer(policy, env, reduced_workspace, args, test_env=test_env)
        trainer(replay_buffer, episode_return, total_steps)



def create_workspace_buffer():
    #Create workspace buffer of size "buffer_size"
    workspace_buffer= []
    buffer_size=100
    grid_size=32
    num_obj_max=10
    obj_size_avg=5

    for i in range (buffer_size)
        random_workspace=random_workspace(grid_size, num_obj_max, obj_size_avg)
        workspace_buffer.append(random_workspace)
    return workspace_buffer

def setup_rndm_workspace_from_buffer(workspace_buffer):
    #Choose random workspace from buffer
    buffer_index= np.random.randint(low=0, high=buffer_size-1, size=None)
    workspace=workspace_buffer[buffer_index]
    start, goal = get_start_goal_for_workspace(workspace)

    CAE_model = CAE(pooling='max', latent_dim=16, input_shape=(ws_height, ws_width), conv_filters=[4, 8, 16])
    CAE_model.load_weights(os.path.join(os.getcwd(), "models/cae/model_num_5_size_8.h5"))
    reduced_workspace = CAE_model.evaluate(wokspace)
    
    return workspace, start, goal, reduced_workspace