import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

import os
import sys

#sys.path.append(os.path.join(os.getcwd(), "lib"))
from hwr.random_workspace import * 
from hwr.cae.cae import CAE


"""
Install the environment with: pip install -e .
Use it then with:
    import gym
    import gym_pointrobo
    env = gym.make('pointrobo-v0')
"""

class PointroboEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self): 
        super(PointroboEnv, self).__init__()

        # workspace related inits:
        self.buffer_size = 100
        self.grid_size = 32
        self.num_obj_max = 10
        self.obj_size_avg = 5

        # Define action and observation space
        # They must be gym.spaces objects
        
        # Continuous action space with velocities up to 10m/s in x- & y- direction
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        
        #The observation will be the coordinate of the agent 
        #this can be described by Box space
        self.observation_space = spaces.Box(low=0.0, high=self.grid_size,
                                            shape=(2,), dtype=np.float32)

        # Initialize the agent
        self.current_step = 0
        self.create_workspace_buffer()

        # reset. With this the first workspace, agent position, goal position is created.
        _, _, _, = self.reset()
        

    def step(self, action):
        """Implements the step function for taking an action and evaluating it"""

        self.take_action(action)
        self.current_step += 1        

        #Goal reached: Reward=1; Obstacle Hit: Reward=-1; Step made=-0.01
        if (self.agent_pos == self.goal_pos).all():
            reward = 1
            done = True
        #Have we hit an obstacle?
        elif self.collision_check():
            reward = -1
            done = True
        #We made another step
        else: 
            reward = -0.01
            done = False 

        return self.agent_pos, reward, done, {}


    def reset(self):
        """Resets the robot state to the initial state"""        
        self.setup_rndm_workspace_from_buffer()
        self.agent_pos = self.start_pos
   
        return self.workspace.astype(np.float32), self.goal_pos.astype(np.float32), self.agent_pos.astype(np.float32)


    def render(self, mode='console', close=False):
        """"The x-axis of the environment is pointing from left to right. 
            The y-axis is pointing downwards. 
            The origin of the KOSY is in the top left corner."""
        # Render the environment to the screen
        if mode != 'console':
            raise NotImplementedError()
    
        # represend environment
        self.workspace[int(self.start_pos[0]), int(self.start_pos[1])] = 2
        self.workspace[int(self.goal_pos[0]), int(self.goal_pos[1])] = 4

        workspace_fig = visualize_workspace(self.workspace)
        robot = visualize_robot(self.agent_pos)
        distance_fig= visualize_distance_field(self.workspace)
        robot = visualize_robot(self.agent_pos)
        plt.show()


    def take_action(self, action):
        """The action is encoded like a real velocity vector with the first element 
        pointing in x-direction and the second element pointing in y-direction
        """
        t = 1.0
        self.agent_pos += action * t
        self.agent_pos = np.clip(self.agent_pos, [0.0, 0.0], [float(self.grid_size), float(self.grid_size)])


    def create_workspace_buffer(self):
        """Create workspace buffer of size buffer_size"""
        self.workspace_buffer = [random_workspace(self.grid_size, self.num_obj_max, self.obj_size_avg)\
                                    for _ in range(self.buffer_size)]


    def setup_rndm_workspace_from_buffer(self):
        """Choose random workspace from buffer"""
        buffer_index = np.random.randint(low=0, high=self.buffer_size - 1)
        self.workspace = self.workspace_buffer[buffer_index]
        self.start_pos, self.goal_pos = get_start_goal_for_workspace(self.workspace)


    def collision_check(self):
        """Checks whether the robot' collides with an object or the workspace boundary. If it collides collision ist set to 1"""
        
        collision = False

        #Treat boundaries as obstacles
        padding_workspace = np.ones((self.grid_size+2, self.grid_size+2))
        padding_workspace[1: -1, 1: -1] = self.workspace

        #Create distance map
        dist_img = ndimage.distance_transform_edt(-padding_workspace + 1)  # Expects blocks as 0 and free space as 1
        pixel_size = 1 #10/32
        
        #Compute distance to the nearest obstacle at the center
        dist_fun = image_interpolation(img=dist_img, pixel_size=pixel_size)
        x = self.agent_pos
        nearest_dist = dist_fun(x=x)

        #print("Distance to the nearest obstacle at the center: ", nearest_dist)
            
        if nearest_dist <= 1.5:
            collision = True
            #print("Collision is: ", collision)
        
        return collision