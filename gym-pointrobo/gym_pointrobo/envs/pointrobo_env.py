import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "lib"))
from random_workspace import * 
from cae.cae import CAE


"""
Install the environment with: pip install -e .
Use it then with:
    import gym
    import gym_pointrobo
    env = gym.make('pointrobo-v0')
"""

class PointroboEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, MAX_EPISODE_STEPS=int(1e3)):
        super(PointroboEnv, self).__init__()

        # Initialize the agent
        self.workspace_buffer, self.grid_size, self.buffer_size  = create_workspace_buffer()
        self.workspace, self.start_pos, self.goal_pos = setup_rndm_workspace_from_buffer(self.workspace_buffer, self.grid_size, self.buffer_size)
        self.agent_pos = np.asarray(self.start_pos).astype(np.float64)
        self.MAX_EPISODE_STEPS=MAX_EPISODE_STEPS
        self.current_step=1

        # Define action and observation space
        # They must be gym.spaces objects
        
        # Continuous action space with velocities up to 10m/s in x- & y- direction
        self.action_space = spaces.Box(low=0, high=1, shape=
                        (2, 1), dtype=np.float32)
        
        #The observation will be the coordinate of the agent 
        #this can be described by Box space
        self.observation_space = spaces.Box(low=0.0, high=self.grid_size,
                                            shape=(2,1), dtype=np.float32)


    def step(self, action):
        """Implements the step function for taking an action and evaluating it"""
        take_action(action, self.agent_pos) 

        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.grid_size)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.grid_size)

        self.current_step += 1
        
        done = 0
        #Episode stop
        #Have we reached the goal?
        if (self.agent_pos[0] == self.goal_pos[0]) and (self.agent_pos[1] == self.goal_pos[1]):
            done = 1
        #Have we reached the maximum episode steps?
        elif self.current_step==self.MAX_EPISODE_STEPS:
            done = 1
        #Have we hit an obstacle?
        elif collision_check(self.workspace, self.agent_pos)==1:
            done = 1

        #Goal reached: Reward=1; Obstacle Hit: Reward=-1; Step made=-0.01
        if (self.agent_pos[0] == self.goal_pos[0]) and (self.agent_pos[1] == self.goal_pos[1]):
            reward = 1
        #Have we hit an obstacle?
        elif collision_check(self.workspace, self.agent_pos)==1:
            reward = -1
        #We made another step
        else: 
            reward = -0.01       

        return np.asarray(self.agent_pos).astype(np.float32), reward, done, {}

    def reset(self):
        """Resets the robot state to the initial state"""
        # here we convert start to float32 to make it more general (we want to use continuous actions)
        
        self.workspace, self.start_pos, self.goal_pos = setup_rndm_workspace_from_buffer(self.workspace_buffer, self.grid_size, self.buffer_size)
        self.agent_pos = np.asarray(self.start_pos).astype(np.float64)

        
        return self.agent_pos

    def render(self, mode='console', close=False):
        """"The x-axis of the environment is pointing from left to right. 
            The y-axis is pointing downwards. 
            The origin of the KOSY is in the top left corner."""
        # Render the environment to the screen
        if mode != 'console':
            raise NotImplementedError()
    
        # represend environment
        self.workspace[self.start_pos[0],self.start_pos[1]]=2
        self.workspace[self.goal_pos[0],self.goal_pos[1]]=4

        workspace_fig = visualize_workspace(self.workspace)
        robot = visualize_robot(self.agent_pos)
        distance_fig= visualize_distance_field(self.workspace)
        robot = visualize_robot(self.agent_pos)
        plt.show()


def take_action(action, agent_pos):
    "The action is encoded like a real velocity vector with the first element pointing in x-direction and the second element pointing in y-direction"
    action_x = action[0]
    action_y = action[1]
    t=1.0

    agent_pos[1] += action_x*t
    agent_pos[0] += action_y*t


def collision_check(workspace, agent_pos):
    """Checks whether the robot' collides with an object or the workspace boundary. If it collides collision ist set to 1"""
    
    collision = 0

    #Treat boundaries as obstacles
    padding_workspace=np.ones((34,34))
    padding_workspace[1:33,1:33]=workspace

    #Create distance map
    dist_img = ndimage.distance_transform_edt(-padding_workspace + 1)  # Expects blocks as 0 and free space as 1
    pixel_size = 1 #10/32
    
    #Compute distance to the nearest obstacle at the center
    dist_fun = image_interpolation(img=dist_img, pixel_size=pixel_size)
    x = agent_pos
    nearest_dist = dist_fun(x=x)

    print("Distance to the nearest obstacle at the center: ",
        nearest_dist)
        
    if nearest_dist <= 1.5:
        collision = 1
        print("Collision is: ",
        collision)
    
    return collision


def image_interpolation(*, img, pixel_size=1, order=1, mode='nearest'):

    factor = 1 / pixel_size

    def interp_fun(x):
        x2 = x.copy()

        if x2.ndim == 1:
            x2 = x2[np.newaxis, :]
        # Transform physical coordinates to image coordinates 
        x2 *= factor
        x2 += 0.5

        return ndimage.map_coordinates(input=img, coordinates=x2.T, order=order, mode=mode).T

    return interp_fun

def create_workspace_buffer():
    #Create workspace buffer of size "buffer_size"
    workspace_buffer = []
    buffer_size = 100
    grid_size = 32
    num_obj_max = 10
    obj_size_avg = 5

    for i in range (buffer_size):
        random_ws = random_workspace(grid_size, num_obj_max, obj_size_avg)
        workspace_buffer.append(random_ws)
    return workspace_buffer, grid_size, buffer_size

def setup_rndm_workspace_from_buffer(workspace_buffer, grid_size, buffer_size):
    #Choose random workspace from buffer
    buffer_index = np.random.randint(low=0, high=buffer_size-1, size=None)
    workspace = workspace_buffer[buffer_index]
    start, goal = get_start_goal_for_workspace(workspace)

    #Shrink workspace to latent space
    #CAE_model = CAE(pooling='max', latent_dim=16, input_shape=(grid_size, grid_size), conv_filters=[4, 8, 16])
    #CAE_model.load_weights(os.path.join(os.getcwd(), "models/cae/model_num_5_size_8.h5"))
    #reduced_workspace = CAE_model.evaluate(workspace)
    
    return workspace, start, goal, #reduced_workspace



#***************TEST THE ENVIRONMENT******************************
if __name__ == '__main__':

    env = PointroboEnv()
    obs = env.reset()
    env.render()

    # Hardcoded agent: always go diagonal
    action=[0.5,0.5]

    n_steps = 20
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(action)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        
        if done:
            print("reward=", reward)
            if reward==1:
                print ("Goal reached!")
            elif reward==-1:
                print ("OOOOpssss you crashed!!")
            break
        env.render()

