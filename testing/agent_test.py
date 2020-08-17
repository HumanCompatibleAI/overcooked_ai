import pickle
import time
import unittest
import numpy as np

from overcooked_ai_py.agents.agent import AgentPair, FixedPlanAgent, GreedyHumanModel, CoupledPlanningPair, RandomAgent
from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

np.random.seed(42)

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

force_compute_large = False
force_compute = True
DISPLAY = False

simple_mdp = OvercookedGridworld.from_layout_name('cramped_room')
large_mdp = OvercookedGridworld.from_layout_name('corridor')


class TestAgentEvaluator(unittest.TestCase):

    def setUp(self):
        self.agent_eval = AgentEvaluator({"layout_name": "cramped_room"}, {"horizon": 100})
        
    def test_human_model_pair(self):
        trajs = self.agent_eval.evaluate_human_model_pair()
        try:
            AgentEvaluator.check_trajectories(trajs)
        except AssertionError as e:
            self.fail("Trajectories were not returned in standard format:\n{}".format(e))

    def test_rollouts(self):
        ap = AgentPair(RandomAgent(), RandomAgent())
        trajs = self.agent_eval.evaluate_agent_pair(ap, num_games=5)
        try:
            AgentEvaluator.check_trajectories(trajs)
        except AssertionError as e:
            self.fail("Trajectories were not returned in standard format:\n{}".format(e))
        
    def test_mlp_computation(self):
        try:
            self.agent_eval.env.mlp
        except Exception as e:
            self.fail("Failed to compute MediumLevelPlanner:\n{}".format(e))


class TestBasicAgents(unittest.TestCase):

    def setUp(self):
        self.mlp_large = MediumLevelPlanner.from_pickle_or_compute(large_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute_large)

    def test_fixed_plan_agents(self):
        a0 = FixedPlanAgent([s, e, n, w])
        a1 = FixedPlanAgent([s, w, n, e])
        agent_pair = AgentPair(a0, a1)
        env = OvercookedEnv.from_mdp(large_mdp, horizon=10)
        trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        end_state = trajectory[-1][0]
        self.assertEqual(time_taken, 10)
        self.assertEqual(env.mdp.get_standard_start_state().player_positions, end_state.player_positions)

    def test_two_greedy_human_open_map(self):
        scenario_2_mdp = OvercookedGridworld.from_layout_name('scenario2')
        mlp = MediumLevelPlanner.from_pickle_or_compute(scenario_2_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute)        
        a0 = GreedyHumanModel(mlp)
        a1 = GreedyHumanModel(mlp)
        agent_pair = AgentPair(a0, a1)
        start_state = OvercookedState(
            [P((8, 1), s),
             P((1, 1), s)],
            {})
        env = OvercookedEnv.from_mdp(scenario_2_mdp, start_state_fn=lambda: start_state, horizon=100)
        trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)


if __name__ == '__main__':
    unittest.main()
