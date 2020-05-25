from gym.envs.registration import register

register(
    id='pointrobo-v0',
    entry_point='gym_pointrobo.envs:PointroboEnv',
)