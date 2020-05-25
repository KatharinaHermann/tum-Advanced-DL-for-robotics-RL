import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np

class PointroboEnv(gym.Env):
  #metadata = {'render.modes': ['human']}

  def __init__(self, grid_size, start_pos, goal_pos, workspace,MAX_EPISODE_STEPS):
     super(PointroboEnv, self).__init__()

     # Size of the 1D-grid
     self.grid_size = grid_size
     # Initialize the agent at the right of the grid
     self.agent_pos = start_pos
     self.start_pos = start_pos
     self.goal_pos = goal_pos
     self.workspace= workspace
     self.MAX_EPISODE_STEPS=MAX_EPISODE_STEPS

     # Define action and observation space
     # They must be gym.spaces objects
    
     # Continuous action space with velocities in x- & y- direction
     self.action_space = spaces.Box(low=0, high=255, shape=
                    (2, 1), dtype=np.uint8)
    
     # The observation will be the coordinate of the agent
    # this can be described by Box space
     self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                        shape=(1,), dtype=np.float32)


  def step(self, action):
      self._take_action(action) 

      self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.grid_size)
      self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.grid_size)

      self.current_step += 1
    
      #Episode stop
      #Have we reached the goal?
      if self.agent_pos == self.goal_pos
        done = 1
        #Have we reached the maximum episode steps?
        elif self.current_step==self.MAX_EPISODE_STEPS:
            done = 1
            #Have we hit an obstacle?
            elif self.collision_check(self.agent_pos)==1:
                done = 1

      #Goal reached: Reward=1; Obstacle Hit: Reward=-1; Step made=-0.01
      if self.agent_pos == self.goal_pos:
        reward = 1
        #Have we hit an obstacle?
        elif self.collision_check(self.agent_pos)==1:
            reward=-1
            #We made another step
            else: 
             reward=-0.01       

      return np.array([self.agent_pos]).astype(np.float32), reward, done, {}
    ...
  def reset(self):
    self.agent_pos = self.start_pos
    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    return np.array([self.agent_pos]).astype(np.float32)

  def render(self, mode='console', close=False):
    # Render the environment to the screen
    if mode != 'console':
      raise NotImplementedError()
    # agent is represented as a cross, rest as a dot

    self.workspace[self.start_pos[0,0],self.start_pos[0,1]]=2
    self.workspace[self.goal_pos[0,0],self.goal_pos[0,1]]=4
    self.workspace[self.agent_pos[0],self.agent_pos[0]]]=3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(self.workspace_sample)
    fig.colorbar(cax)

  def _take_action(self, action):
    action_x = action[0]
    action_y = action[1]
    t=1

    self.agent_pos[1] += action_x*t
    self.agent_pos[0] += action_y*t

  def collision_check(self, self.agent_pos):
      collision = 0
      if self.workspace[self.agent_pos[0],self.agent_pos[1]]==1:
        collision=1
      return collision
