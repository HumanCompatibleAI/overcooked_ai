import numpy as np
from overcooked_ai_py import COMMON_TESTS_DIR
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import save_as_json

# NOTE: This code was used to create the common test trajectories. The purpose of these
# tests is to sanity check the consistency of dynamics and encodings across the 
# overcooked python and javascript implementations.
# If changing the overcooked environment in ways that affect trajectories, one 
# should also run this file again, and make sure `npm run test` will pass in
# overcooked_ai_js

# Saving trajectory for dynamics consistency test
np.random.seed(0)
ae = AgentEvaluator(mdp_params={"layout_name": "cramped_room"}, env_params={"horizon": 1500})
test_trajs = ae.evaluate_random_pair(all_actions=True, num_games=1)
assert test_trajs["ep_returns"][0] > 0, "Choose a different seed, we should have a test trajectory that gets some reward"

test_trajs_path = COMMON_TESTS_DIR + "trajectory_tests/trajs.json"
AgentEvaluator.save_traj_as_json(test_trajs, test_trajs_path)

# Saving encondings for encoding tests
load_traj = AgentEvaluator.load_traj_from_json(test_trajs_path)
mdp_params = load_traj["mdp_params"][0]
env_params = load_traj["env_params"][0]
mdp = AgentEvaluator(mdp_params, env_params).mdp_fn()
for i in range(2):
    lossless_path = COMMON_TESTS_DIR + "encoding_tests/lossless_py{}.json".format(i)
    encoded_states = [mdp.lossless_state_encoding(s)[i].tolist() for s in np.concatenate(load_traj["ep_states"])]
    save_as_json(encoded_states, lossless_path)

    featurization_path = COMMON_TESTS_DIR + "encoding_tests/featurized_py{}.json".format(i)
    encoded_states = [mdp.featurize_state(s, ae.mlp)[i].tolist() for s in np.concatenate(load_traj["ep_states"])]
    save_as_json(encoded_states, featurization_path)
