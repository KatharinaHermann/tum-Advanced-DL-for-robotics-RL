import numpy as np
import gym
import gym_pointrobo
import matplotlib.pyplot as plt

from hwr.utils import load_params


def test_pointrobot_gym():
    params = load_params("params/test_params.json")
    env = gym.make('pointrobo-v0', 
                   params=params)
    workspace, goal, obs = env.reset()

    # Hardcoded agent: always go diagonal
    action=np.array([0.5, 0.5])

    n_steps = 40
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(action)
        print('obs={}, reward={}, done={}'.format(obs, reward, done))
        
        if done:
            print("reward={}".format(reward))
            if reward == env.goal_reward:
                print ("Goal reached!")
            elif reward == env.collision_reward:
                print ("OOOOpssss you crashed!!")
            break
        env.render()
    
    plt.show()
        


if __name__ == '__main__':

    test_pointrobot_gym()
    print('All tests have run successfully!')
