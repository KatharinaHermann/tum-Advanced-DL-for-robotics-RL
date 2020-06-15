import numpy as np
import gym
import gym_pointrobo



def test_pointrobot_gym():
    env = gym.make('pointrobo-v0', 
                   goal_reward=5, 
                   collision_reward=-1,
                   step_reward=-0.01)
    workspace, goal, obs = env.reset()
    env.render()

    # Hardcoded agent: always go diagonal
    action=np.array([0.5, 0.5])

    n_steps = 20
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(action)
        print('obs={}, reward={}, done={}'.format(obs, reward, done))
        
        if done:
            print("reward={}".format(reward))
            if reward == 1:
                print ("Goal reached!")
            elif reward == -1:
                print ("OOOOpssss you crashed!!")
            break
        env.render()


if __name__ == '__main__':

    test_pointrobot_gym()
    print('All tests have run successfully!')