import numpy as np
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import save_as_json

# The point of this file is to generate serialized trajectories for MDP dynamics consistency testing
# NOTE: If intentionally updating MDP dynamics, this function should be used

def generate_serialized_trajectory(mdp, save_path):
    # Saving trajectory for dynamics consistency test
    seed = 0
    sparse_reward = 0
    while sparse_reward <= 0:
        np.random.seed(seed)
        ae = AgentEvaluator.from_mdp(mdp, env_params={"horizon": 1500})
        test_trajs = ae.evaluate_random_pair(all_actions=True, num_games=1)
        sparse_reward = np.mean(test_trajs["ep_returns"])
        seed += 1

    AgentEvaluator.save_traj_as_json(test_trajs, save_path)
