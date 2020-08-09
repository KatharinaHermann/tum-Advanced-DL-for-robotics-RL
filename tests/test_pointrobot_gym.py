import unittest
import numpy as np
import gym
import gym_pointrobo
import matplotlib.pyplot as plt

from hwr.utils import load_params, normalize, rescale


def test_pointrobot_gym_goal(params):

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


def test_pointrobot_gym_obstacle(params):

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

def test_pointrobot_gym_boundaries(params):

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

        env.render()
    
    plt.show()

class PointrobotGymTests(unittest.TestCase):
    """for testing the Pointrobot environment."""

    def setUp(self):
        """setUp"""
        self.params = load_params("params/test_params.json")

        # environment without normalization:
        self.params["env"]["normalize"] = False
        self.env = gym.make(
            self.params["env"]["name"], 
            params=self.params
            )
        # environment with normaluzation:
        self.params["env"]["normalize"] = True
        self.env_norm = gym.make(
            self.params["env"]["name"], 
            params=self.params
            )
    

    def test_step(self):
        """testing the step function of the environment."""

        num_of_cases = 100
        # random actions:
        actions = [np.random.uniform(self.env.action_space.low, 
            self.env.action_space.high) for _ in range(num_of_cases)]
        actions_norm = [normalize(action, self.env.action_space) for action in actions]
        
        # running the corresponding actions through the environments:
        for i in range(num_of_cases):
            workspace, goal_pos, agent_pos = self.env.reset()
            self.env_norm.workspace = workspace
            self.env_norm.goal_pos = goal_pos
            self.env_norm.agent_pos = agent_pos

            # stepping both of the environments:
            next_pos, reward, done, _ = self.env.step(actions[i])
            next_pos_norm, reward_norm, done_norm, _ = self.env_norm.step(actions_norm[i])

            # assertions:
            self.assertTrue(np.isclose(normalize(next_pos, self.env.pos_bounds), next_pos_norm, atol=1e-6).all())
            self.assertTrue(np.isclose(next_pos, self.env_norm.agent_pos, atol=1e-6).all())
            self.assertEqual(reward, reward_norm)
            self.assertEqual(done, done_norm)


    def test_reset(self):
        """tests the reset function of the environment."""

        num_of_cases = 100
        for _ in range(num_of_cases):
            _, goal_pos_norm, agent_pos_norm = self.env_norm.reset()

            # assertions:
            self.assertTrue(np.isclose(rescale(goal_pos_norm, self.env_norm.pos_bounds),
                self.env_norm.goal_pos, atol=1e-6).all())
            self.assertTrue(np.isclose(rescale(agent_pos_norm, self.env_norm.pos_bounds),
                self.env_norm.agent_pos, atol=1e-6).all())



if __name__ == '__main__':

    """params = load_params("params/test_params.json")
    params["env"]["grid_size"] = 20
    params["env"]["goal_reward"] = -0.01
    params["env"]["collision_reward"] = -1
    params["env"]["step_reward"] = -0.01
    params["env"]["max_goal_dist"] = 5

    test_pointrobot_gym_goal(params)
    test_pointrobot_gym_obstacle(params)
    test_pointrobot_gym_boundaries(params)
    print('All tests have run successfully!')"""
    
    unittest.main()
