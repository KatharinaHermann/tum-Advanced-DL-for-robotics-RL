import numpy as np
import gym
import gym_pointrobo
import matplotlib.pyplot as plt

from hwr.utils import load_params


def test_pointrobot_gym_goal():
    params = load_params("params/test_params.json")
    params["env"]["grid_size"] = 20
    params["env"]["goal_reward"] = -0.01
    params["env"]["collision_reward"] = -1
    params["env"]["step_reward"] = -0.01
    params["env"]["max_goal_dist"] = 5

    env = gym.make(
        params["env"]["name"], 
        params=params
        )

    workspace, goal, obs = env.reset()
    env.render()        
    env.workspace = np.zeros((20, 20))
    env.agent_pos = np.array([6.5, 6.5])
    env.goal_pos = np.array([10.0, 10.0])
    env.render()

    # Hardcoded agent: always go diagonal
    action=np.array([0.5, 0.5])

    n_steps = 40
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(action)
        print('obs={}, reward={}, done={}'.format(obs, reward, done))
        env.render()
        dist = np.linalg.norm(env.agent_pos-env.goal_pos)
        print(dist)

        if dist <= 2* env.robot_radius:
            assert (done==True) and (reward==env.goal_reward), "Done should be one and reward should be goal reward as the robot has reached it's goal!"

        
        if done:
            print("reward={}".format(reward))
            if (reward == env.goal_reward) and (dist <= 2* env.robot_radius):
                print ("Correct! Goal reached - correct goal reward assigned!")
            if (reward == env.goal_reward) and (dist > 2* env.robot_radius):
                print ("Wrong! Goal not reached - Reward should not be goal reward!")
            break
        env.render()
    
    plt.show()


def test_pointrobot_gym_obstacle():
    params = load_params("params/test_params.json")
    params["env"]["grid_size"] = 20
    params["env"]["goal_reward"] = -0.01
    params["env"]["collision_reward"] = -1
    params["env"]["step_reward"] = -0.01
    params["env"]["max_goal_dist"] = 5

    env = gym.make(
        params["env"]["name"], 
        params=params
        )
    workspace, goal, obs = env.reset()
    env.render()        
    env.workspace = np.zeros((20, 20))
    env.render()
    env.workspace[5:10, 5:10] = 1
    env.render()
    env.agent_pos = np.array([3.0, 3.0])
    env.goal_pos = np.array([17.0, 17.0])
    env.render()

    # Hardcoded agent: always go diagonal
    action=np.array([0.5, 0.5])

    n_steps = 40
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(action)
        print('obs={}, reward={}, done={}'.format(obs, reward, done))
        env.render()
        dist = np.linalg.norm(env.agent_pos-np.array([4.5, 4.5]))
        print(dist)

        if dist < env.robot_radius:
            assert (done==True) and (reward==env.collision_reward), "Done should be one and reward should be collision reward as the robot has crashed!"
        
        if done:
            print("reward={}".format(reward))
            if (reward == env.collision_reward) and (dist <= env.robot_radius):
                print ("Correct! Robot has crashed!")
            elif (reward == env.collision_reward) and (dist > env.robot_radius):
                print ("Wrong!Robot has not crashed!")
          
            break
        
        #env.workspace = np.zeros((32, 32))
        #env.workspace[0:4, 0:4] = 1
        #env.agent_pos = np.array([29.0, 29.0])
        env.render()
    
    plt.show()

def test_pointrobot_gym_boundaries():
    params = load_params("params/test_params.json")
    params["env"]["grid_size"] = 20
    params["env"]["goal_reward"] = -0.01
    params["env"]["collision_reward"] = -1
    params["env"]["step_reward"] = -0.01
    params["env"]["max_goal_dist"] = 5
    
    env = gym.make(
        params["env"]["name"], 
        params=params
        )
    workspace, goal, obs = env.reset()
    print(goal, obs)
    env.render()        
    env.workspace = np.zeros((20, 20))
    env.render()
    env.agent_pos = np.array([17.0, 17.0])
    env.goal_pos = np.array([10.0, 10.0])
    env.render()

    # Hardcoded agent: always go diagonal
    action=np.array([0, 0.5])

    n_steps = 40
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(action)
        print('obs={}, reward={}, done={}'.format(obs, reward, done))
        env.render()
        dist = np.linalg.norm(env.agent_pos-np.array([17.0, 19.5]))
        print(dist)

        if dist < env.robot_radius:
            assert (done==True) and (reward==env.collision_reward), "Done should be one and reward should be collision reward as the robot has crashed!"
        
        if done:
            print("reward={}".format(reward))
            if (reward == env.collision_reward) and (dist <= env.robot_radius):
                print ("Correct! Robot has crashed!")
            elif (reward == env.collision_reward) and (dist > env.robot_radius):
                print ("Wrong!Robot has not crashed!")
          
            break
        
        #env.workspace = np.zeros((32, 32))
        #env.workspace[0:4, 0:4] = 1
        #env.agent_pos = np.array([29.0, 29.0])
        env.render()
    
    plt.show()
        


if __name__ == '__main__':

    test_pointrobot_gym_goal()
    test_pointrobot_gym_obstacle()
    test_pointrobot_gym_boundaries()
    print('All tests have run successfully!')
