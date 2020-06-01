

import gym
import gym_pointrobo

from tf2rl.algos.ddpg import DDPG
#from tf2rl.experiments.trainer import Trainer


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = DDPG.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="pointrobo-v0")
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000)
    args = parser.parse_args()

    #Create workspace buffer of size "buffer_size"
    workspace_buffer= []
    buffer_size=100
    grid_size=32
    num_obj_max=10
    obj_size_avg=5

    for i in range (buffer_size)
        random_workspace=random_workspace(grid_size, num_obj_max, obj_size_avg)
        workspace_buffer.append(random_workspace)

    #Choose random workspace from buffer
    index_buffer= np.random.randint(low=0, high=buffer_size-1, size=None)
    workspace=workspace_buffer[index_buffer]
    start, goal = get_start_goal_for_workspace(workspace)
    reduced_workspace=##############AE(wokspace)

    env = PointroboEnv(grid_size=grid_size, start_pos=start, goal_pos=goal, workspace=workspace, MAX_EPISODE_STEPS=30)

    #env = gym.make(args.env_name)
    #test_env = gym.make(args.env_name)
    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
