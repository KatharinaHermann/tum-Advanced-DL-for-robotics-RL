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

    def __init__(self, 
                 goal_reward=5,
                 collision_reward=-1,
                 step_reward=-0.01,
                 buffer_size=100,
                 grid_size=32,
                 num_obj_max=5,
                 obj_size_avg=8, 
                 robot_radius=1): 

        super(PointroboEnv, self).__init__()

        # rewards
        self.goal_reward = goal_reward
        self.collision_reward = collision_reward
        self.step_reward = step_reward
        self.robot_radius = robot_radius

        # workspace related inits:
        self.buffer_size = buffer_size
        self.grid_size = grid_size
        self.num_obj_max = num_obj_max
        self.obj_size_avg =obj_size_avg

        # Define action and observation space
        # They must be gym.spaces objects
        
        # Continuous action space with velocities up to 10m/s in x- & y- direction
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        
        #The observation will be the coordinate of the agent 
        #this can be described by Box space
        self.observation_space = spaces.Box(low=0.0, high=self.grid_size,
                                            shape=(20,), dtype=np.float32)

        # Initialize the agent
        self.current_step = 0
        self.create_workspace_buffer()

        # reset. With this the first workspace, agent position, goal position is created.
        _, _, _, = self.reset()

        # Visulaization initializations:
        self._fig = None
        self._ax = None
        self._robo_artist = None
        self._goal_artist = None

        

    def step(self, action):
        """Implements the step function for taking an action and evaluating it"""

        self.take_action(action)
        self.current_step += 1        

        #Goal reached: Reward=1; Obstacle Hit: Reward=-1; Step made: Reward=-0.01
        # Tolerance of distance 3, that the robot reached the goal!
        if (np.linalg.norm(self.agent_pos-self.goal_pos) < self.robot_radius): 
            reward = self.goal_reward
            done = True
        #Have we hit an obstacle?
        elif self.collision_check():
            reward = self.collision_reward
            done = True
        #We made another step
        else: 
            reward = self.step_reward
            done = False 

        return self.agent_pos, reward, done, {}


    def reset(self):
        """Resets the robot state to the initial state"""        
        self.setup_rndm_workspace_from_buffer()
        self.agent_pos = self.start_pos.astype(np.float32)
   
        return self.workspace.astype(np.float32), self.goal_pos.astype(np.float32), self.agent_pos.astype(np.float32)


    def render(self, mode='plot', close=False):
        """"The x-axis of the environment is pointing from left to right. 
            The y-axis is pointing downwards. 
            The origin of the KOSY is in the top left corner."""
        # Render the environment to the screen
        if mode != 'plot':
            raise NotImplementedError()
            
        if self._fig is None:
            self._fig = plt.figure(1)
            self._ax = plt.gca()
            self._robo_artist = plt.Circle((self.agent_pos[0], self.agent_pos[1]), self.robot_radius, color='m') 
            self._ax.add_artist(self._robo_artist)
            self._goal_artist = plt.Circle((self.goal_pos[0], self.goal_pos[1]), self.robot_radius, color='b')
            self._ax.add_artist(self._goal_artist)
        
        self._ax.matshow(self.workspace)
        self._robo_artist.set_center((self.agent_pos[0], self.agent_pos[1]))
        self._goal_artist.set_center((self.goal_pos[0], self.goal_pos[1]))

        plt.pause(0.1)


    def take_action(self, action):
        """The action is encoded like a real velocity vector with the first element 
        pointing in x-direction and the second element pointing in y-direction
        """
        t = 1.0
        self.agent_pos += action * t
        self.agent_pos = np.clip(self.agent_pos, [0.0, 0.0], [float(self.grid_size-1), float(self.grid_size-1)])


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
        dist_fun = image_interpolation(img=dist_img.T, pixel_size=pixel_size)
        x = self.agent_pos
        nearest_dist = dist_fun(x=x)

        #print("Distance to the nearest obstacle at the center: ", nearest_dist)
        # With -0.5 we count for the obstacle expansion
        if nearest_dist - self.robot_radius - 0.5 < 0 :
            collision = True
            #print("Collision is: ", collision)
        
        return collision