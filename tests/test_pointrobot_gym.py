import numpy as np
import gym
import gym_pointrobo
import matplotlib.pyplot as plt



def test_pointrobot_gym():
    env = gym.make('pointrobo-v0', 
                   goal_reward=5, 
                   collision_reward=-1,
                   step_reward=-0.01)
    workspace, goal, obs = env.reset()
    #env.render()

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
        #env.workspace = np.zeros((32, 32))
        #env.workspace[0:4, 0:4] = 1
        #env.agent_pos = np.array([29.0, 29.0])
        env.render()
    
    plt.show()
        


if __name__ == '__main__':

    test_pointrobot_gym()
    print('All tests have run successfully!')
