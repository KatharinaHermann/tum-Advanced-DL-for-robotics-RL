import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import sys
sys.path.insert(1, 'C:/Users/Katharina Hermann/Documents/UniMaster/2.Semester/ADLR/Project/GITHUB/project/lib')
#sys.path.append("C:/Users/Katharina Hermann/Documents/UniMaster/2.Semester/ADLR/Project/GITHUB/project/lib")
from random_workspace import random_workspace 

class PointroboEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, grid_size, start_pos, goal_pos, workspace, MAX_EPISODE_STEPS):
     super(PointroboEnv, self).__init__()

     # Size of the 1D-grid
     self.grid_size = grid_size
     # Initialize the agent at the right of the grid
     self.agent_pos = start_pos
     self.start_pos = start_pos
     self.goal_pos = goal_pos
     self.workspace= workspace
     self.MAX_EPISODE_STEPS=MAX_EPISODE_STEPS
     self.current_step=1

     # Define action and observation space
     # They must be gym.spaces objects
    
     # Continuous action space with velocities up to 10m/s in x- & y- direction
     self.action_space = spaces.Box(low=0, high=10, shape=
                    (2, 1), dtype=np.uint8)
    
     # The observation will be the coordinate of the agent *****SHOULD WE OBSERVE MORE?**********
     #this can be described by Box space
     self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                        shape=(1,), dtype=np.float32)


  def step(self, action):
      
      self._take_action(action) 

      self.agent_pos[0,0] = np.clip(self.agent_pos[0,0], 0, self.grid_size)
      self.agent_pos[0,1] = np.clip(self.agent_pos[0,1], 0, self.grid_size)

      self.current_step += 1
      done = 0
      #Episode stop
      #Have we reached the goal?
      if (self.agent_pos[0,0] == self.goal_pos[0,0]) and (self.agent_pos[0,1] == self.goal_pos[0,1]):
        done = 1
        #Have we reached the maximum episode steps?
      elif self.current_step==self.MAX_EPISODE_STEPS:
        done = 1
        #Have we hit an obstacle?
      elif self.collision_check()==1:
        done = 1

      #Goal reached: Reward=1; Obstacle Hit: Reward=-1; Step made=-0.01
      if (self.agent_pos[0,0] == self.goal_pos[0,0]) and (self.agent_pos[0,1] == self.goal_pos[0,1]):
        reward = 1
        #Have we hit an obstacle?
      elif self.collision_check()==1:
        reward=-1
        #We made another step
      else: 
        reward=-0.01       

      return np.array([self.agent_pos]).astype(np.float32), reward, done, {}

  def reset(self):
    self.agent_pos = self.start_pos
    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    return np.array([self.agent_pos]).astype(np.float32)

  def render(self, mode='console', close=False):
    # Render the environment to the screen
    if mode != 'console':
      raise NotImplementedError()
    
    # represend environment
    self.workspace[self.start_pos[0,0],self.start_pos[0,1]]=2
    self.workspace[self.goal_pos[0,0],self.goal_pos[0,1]]=4
    self.workspace[self.agent_pos[0,0],self.agent_pos[0,1]]=3

    fig = plt.figure()
    ax = fig.add_subplot(221)
    cax = ax.matshow(self.workspace)
    fig.colorbar(cax)

    dist_img = ndimage.distance_transform_edt(-self.workspace + 1)  # Excpects blocks as 0 and free space as 1

    dm = fig.add_subplot(212)
    cdm=dm.imshow(dist_img, cmap='Reds')
    cdm=dm.imshow(self.workspace, cmap='Blues', alpha=0.3)
    plt.show()

  def _take_action(self, action):
    action_x = action[0]
    action_y = action[1]
    t=1

    self.agent_pos[0,1] += action_x*t
    self.agent_pos[0,0] += action_y*t

  def collision_check(self):
    collision = 0

    #Treat boundaries as obstacles
    padding_workspace=np.ones((34,34))
    padding_workspace[1:33,1:33]=self.workspace

    #Create distance map
    dist_img = ndimage.distance_transform_edt(-padding_workspace + 1)  # Excpects blocks as 0 and free space as 1
    pixel_size=1 #10/32
    #Compute distance to the nearest obstacle at the center
    dist_fun = image_interpolation(img=dist_img, pixel_size=pixel_size)
    x=np.array([self.agent_pos[0,0],self.agent_pos[0,1]])
    nearest_dist = dist_fun(x=x)
    print("Distance to the nearest obstacle at the center: ",
        nearest_dist)
      
    if nearest_dist <= 2:
        collision=1
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
      #x2 *= factor
      #x2 -= 0.5

    return ndimage.map_coordinates(input=img, coordinates=x2.T, order=order, mode=mode).T

  return interp_fun



#***************TEST THE ENVIRONMENT******************************
grid_size=32
start,goal,workspace=random_workspace(grid_size, 10, 5)
env = PointroboEnv(grid_size=grid_size, start_pos=start, goal_pos=goal, workspace=workspace, MAX_EPISODE_STEPS=30)
obs = env.reset()
env.render()

action=[1,1]
# Hardcoded best agent: always go left!
n_steps = 20
for step in range(n_steps):
  print("Step {}".format(step + 1))
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render()
  if done:
    print("reward=", reward)
    if reward==1:
      print ("Goal reached!")
    elif reward==-1:
      print ("OOOOpssss you crashed!!")
    break
