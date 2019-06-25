from gym.envs.registration import register

register(
    id='Overcooked-v1',
    entry_point='overcooked_gridworld.mdp.overcooked_env:Overcooked',
)
