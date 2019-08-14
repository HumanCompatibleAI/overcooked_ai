import pickle
import time
import unittest
import numpy as np

from overcooked_ai_py.agents.agent import Agent, AgentPair, FixedPlanAgent, CoupledPlanningAgent, GreedyHumanModel, CoupledPlanningPair, EmbeddedPlanningAgent, RandomAgent
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
DISPLAY = True
simple_mdp = OvercookedGridworld.from_layout_name('simple', start_order_list=['any'], cook_time=5)
large_mdp = OvercookedGridworld.from_layout_name('corridor', start_order_list=['any'], cook_time=5)


class TestAgentEvaluator(unittest.TestCase):

    def setUp(self):
        self.agent_eval = AgentEvaluator({"layout_name": "simple"}, {"horizon": 100})
        
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
            self.agent_eval.mlp
        except Exception as e:
            self.fail("Failed to compute MediumLevelPlanner:\n{}".format(e))


class TestAgents(unittest.TestCase):

    def setUp(self):
        self.mlp_large = MediumLevelPlanner.from_pickle_or_compute(large_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute_large)

    def test_fixed_plan_agents(self):
        a0 = FixedPlanAgent([s, e, n, w])
        a1 = FixedPlanAgent([s, w, n, e])
        agent_pair = AgentPair(a0, a1)
        env = OvercookedEnv(large_mdp, horizon=10)
        trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        end_state = trajectory[-1][0]
        self.assertEqual(time_taken, 10)
        self.assertEqual(env.mdp.get_standard_start_state().player_positions, end_state.player_positions)

    def test_two_coupled_agents(self):
        a0 = CoupledPlanningAgent(self.mlp_large)
        a1 = CoupledPlanningAgent(self.mlp_large)
        agent_pair = AgentPair(a0, a1)
        start_state = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {}, order_list=['any'])
        env = OvercookedEnv(large_mdp, start_state_fn=lambda: start_state)
        trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        end_state = trajectory[-1][0]
        self.assertEqual(end_state.order_list, [])

    def test_two_coupled_agents_coupled_pair(self):
        mlp_simple = MediumLevelPlanner.from_pickle_or_compute(simple_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute)
        cp_agent = CoupledPlanningAgent(mlp_simple)
        agent_pair = CoupledPlanningPair(cp_agent)
        start_state = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {}, order_list=['any'])
        env = OvercookedEnv(simple_mdp, start_state_fn=lambda: start_state)
        trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        end_state = trajectory[-1][0]
        self.assertEqual(end_state.order_list, [])
    
    def test_one_coupled_one_fixed(self):
        a0 = CoupledPlanningAgent(self.mlp_large)
        a1 = FixedPlanAgent([s, e, n, w])
        agent_pair = AgentPair(a0, a1)
        env = OvercookedEnv(large_mdp, horizon=10)
        trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        self.assertEqual(time_taken, 10)

    def test_one_coupled_one_greedy_human(self):
        # Even though in the first ~10 timesteps it seems like agent 1 is wasting time
        # it turns out that this is actually not suboptimal as the true bottleneck is 
        # going to be agent 0 later on (when it goes to get the 3rd onion)
        a0 = GreedyHumanModel(self.mlp_large)
        a1 = CoupledPlanningAgent(self.mlp_large)
        agent_pair = AgentPair(a0, a1)
        start_state = OvercookedState(
            [P((2, 1), s),
             P((1, 1), s)],
            {}, order_list=['onion'])
        env = OvercookedEnv(large_mdp, start_state_fn=lambda: start_state)
        trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        end_state = trajectory[-1][0]
        self.assertEqual(end_state.order_list, [])

    def test_two_greedy_human_open_map(self):
        scenario_2_mdp = OvercookedGridworld.from_layout_name('scenario2', start_order_list=['any'], cook_time=5)
        mlp = MediumLevelPlanner.from_pickle_or_compute(scenario_2_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute)        
        a0 = GreedyHumanModel(mlp)
        a1 = GreedyHumanModel(mlp)
        agent_pair = AgentPair(a0, a1)
        start_state = OvercookedState(
            [P((8, 1), s),
             P((1, 1), s)],
            {}, order_list=['onion'])
        env = OvercookedEnv(scenario_2_mdp, start_state_fn=lambda: start_state, horizon=100)
        trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        end_state = trajectory[-1][0]
        self.assertEqual(len(end_state.order_list), 0)

    def test_embedded_planning_agent(self):
        agent_evaluator = AgentEvaluator({"layout_name": "simple"}, {"horizon": 100})
        other_agent = GreedyHumanModel(agent_evaluator.mlp)
        epa = EmbeddedPlanningAgent(other_agent, agent_evaluator.mlp, agent_evaluator.env, delivery_horizon=1)
        ap = AgentPair(epa, other_agent)
        agent_evaluator.evaluate_agent_pair(ap, num_games=1, display=True)


class TestScenarios(unittest.TestCase):

    """
    Corridor: assuming optimality / planning horizon – scenario1_s
    Assuming optimality – scenario2
    Unidentifiable plan – unident_s
    Schelling – schelling_s
    """

    def compare_times(self, evaluator, h_idx=0):
        trajectory_hr = evaluator.evaluate_one_optimal_one_greedy_human(h_idx=h_idx, display=DISPLAY)
        time_taken_hr = trajectory_hr["ep_lengths"][0]

        print("\n"*5, "\n" , "-"*50)
        trajectory_rr = evaluator.evaluate_optimal_pair(display=DISPLAY)
        time_taken_rr = trajectory_rr["ep_lengths"][0]

        print("H+R time taken: ", time_taken_hr)
        print("R+R time taken: ", time_taken_rr)

        self.assertGreater(time_taken_hr, time_taken_rr)

    def test_scenario_1(self):
        # Myopic corridor collision
        #
        # X X X X X O X D X X X X X
        # X   ↓Ho     X           X
        # X     X X X X X X X ↓R  X
        # X                       X
        # X S X X X X X X X X P P X
        #
        # H on left with onion, further away to the tunnel entrance than R.
        # Optimal planner tells R to go first and that H will wait
        # for R to pass. H however, starts going through the tunnel
        # and they get stuck. The H plan is a bit extreme (it would probably
        # realize that it should retrace it's steps at some point)
        scenario_1_mdp = OvercookedGridworld.from_layout_name('small_corridor', start_order_list=['any'], cook_time=5)
        mlp = MediumLevelPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute)
        a0 = GreedyHumanModel(mlp)
        a1 = CoupledPlanningAgent(mlp)
        agent_pair = AgentPair(a0, a1)
        start_state = OvercookedState(
            [P((2, 1), s, Obj('onion', (2, 1))),
             P((10, 2), s)],
            {}, order_list=['onion'])
        env = OvercookedEnv(scenario_1_mdp, start_state_fn=lambda: start_state)
        env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)

    def test_scenario_1_s(self):
        # Smaller version of the corridor collisions scenario above
        # to facilitate DRL training
        scenario_1_mdp = OvercookedGridworld.from_layout_name('scenario1_s', start_order_list=['any'], cook_time=5)
        mlp = MediumLevelPlanner.from_pickle_or_compute(scenario_1_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute)
        a0 = GreedyHumanModel(mlp)
        a1 = CoupledPlanningAgent(mlp)
        agent_pair = AgentPair(a0, a1)
        start_state = OvercookedState(
            [P((2, 1), s, Obj('onion', (2, 1))),
             P((4, 2), s)],
            {}, order_list=['onion'])
        env = OvercookedEnv(scenario_1_mdp, start_state_fn=lambda: start_state)
        trajectory, time_taken_hr, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        env.reset()

        print("\n"*5)
        print("-"*50)

        a0 = CoupledPlanningAgent(mlp)
        a1 = CoupledPlanningAgent(mlp)
        agent_pair = AgentPair(a0, a1)
        trajectory, time_taken_rr, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)

        print("H+R time taken: ", time_taken_hr)
        print("R+R time taken: ", time_taken_rr)
        self.assertGreater(time_taken_hr, time_taken_rr)

    def test_scenario_2(self):
        # Simple asymmetric advantages scenario
        #
        # X X X X X O X X X X
        # S                 O
        # D         ↑H  ↑R  X
        # X X X X X X P=X X X
        #
        # Worse version of scenario 3 (probably to be deleted) 
        #
        # The optimal thing to do for the human is to go and get a dish
        # so that by the time it gets back to the pot, the soup will be ready.
        # However, H goes to get the onion, and so does R initially, as it
        # assumes H will go and get the dish. Once H has picked up the onion,
        # R realizes that it should go and get the dish itself. This leads to
        # a couple of timesteps lost (the difference could be made bigger with a
        # better thought through map)

        start_state = OvercookedState(
            [P((5, 2), n),
             P((7, 2), n)],
            {(6, 3): Obj('soup', (6, 3), ('onion', 2, 0))},
            order_list=['onion'])

        mdp_params = {"layout_name": "scenario2", "cook_time": 5}
        env_params = {"start_state_fn": lambda: start_state}
        eva = AgentEvaluator(mdp_params, env_params)
        self.compare_times(eva)

    def test_scenario_3(self):
        # Another asymmetric advantage scenario
        #
        # X X X X X O X X X X
        # S           X X P=X
        # X         ↑H      X
        # D   X X X X!X X   X
        # X           →R    O
        # X X X X X X X X X X
        # 
        # Human plan is suboptimal, and R relies on H switching
        #
        # The optimal thing to do for the human is to go and get a dish
        # so that by the time it gets back to the pot, the soup will be ready.
        # However, H goes to get the onion, and so does R initially, as it
        # assumes H will go and get the dish. Once H has picked up the onion,
        # R realizes that it should go and get the dish itself. This leads to
        # more time wasted compared to the R-R case

        mdp_params = {"layout_name": "scenario3", "cook_time": 5}
        mdp = OvercookedGridworld.from_layout_name(**mdp_params)
        start_state = mdp.get_standard_start_state()
        start_state.objects = {(8, 1): Obj('soup', (8, 1), ('onion', 2, 0))}
        start_state.order_list = ['onion']

        valid_counters = [(5, 3)]
        one_counter_params = {
            'start_orientations': False,
            'wait_allowed': False,
            'counter_goals': valid_counters,
            'counter_drop': valid_counters,
            'counter_pickup': [],
            'same_motion_goals': True
        }

        env_params = {"start_state_fn": lambda: start_state, "horizon": 1000}
        eva = AgentEvaluator(mdp_params, env_params, mlp_params=one_counter_params, force_compute=force_compute)

        self.compare_times(eva)

    def test_scenario_4(self):
        # Yet another asymmetric advantage scenario
        #
        # X X X X X O X X X X
        # S             X P=X
        # D         ↑H      X
        # X X X X X X X X   X
        # X X X X X X →R    O
        # X X X X X X X X X X
        # 
        # Similar to scenario 3, just keeping for reference for now. 
        # In this case we only have human suboptimality, and R
        # assuming H optimality does not end up to be a problem
        mdp_params = {"layout_name": "scenario4", "cook_time": 5}
        mdp = OvercookedGridworld.from_layout_name(**mdp_params)
        
        start_state = mdp.get_standard_start_state()
        start_state.objects = {(8, 1): Obj('soup', (8, 1), ('onion', 2, 5))}
        start_state.order_list = ['onion']

        env_params = {"start_state_fn": lambda: start_state, "horizon": 1000}
        eva = AgentEvaluator(mdp_params, env_params, force_compute=force_compute)
        self.compare_times(eva)

    def test_schelling_s(self):
        # Schelling failure scenario
        #
        # X X S P-D X X
        # X     ↓R    X
        # X     X     X
        # O           O
        # X     X     X
        # X     ↓H    X
        # X X D P-S X X
        #
        # The layout is completely symmetric. Both pots need 2 more onions,
        # and only one delivery is left. The best thing to do would be to split up
        # towards the different pots, but the agents must somehow coordinate on the
        # first step. In the H+R case, this doesn't work, but in the R+R it does.
        #
        eva = AgentEvaluator({"layout_name": "schelling_s", "start_order_list": ["any", "any"], "cook_time": 5}, force_compute=force_compute)
        start_state = eva.env.mdp.get_standard_start_state()
        start_state.objects = {(2, 0): Obj('soup', (2, 0), ('onion', 2, 5)),
                               (2, 4): Obj('soup', (2, 4), ('onion', 2, 5))}
        eva.start_state = start_state
        self.compare_times(eva, h_idx=1)
    
    def test_unidentifiable(self):
        # Scenario with unidentifiable human plan (and asymmetric advantages)
        #
        # X O X X X
        # X     ↓RX
        # X X     X
        # X X     X
        # X S     D
        # X X P=P5X
        # X O     D
        # X X     X
        # X X ↑H  X
        # X       X
        # X S X X X
        # 
        # The human goes up towards either the onion or a dish
        # The robot can't really deduce which one the human is going for,
        # but if the human was optimal, it would go for the onion. Therefore,
        # R assumes H will take care of the last onion necessary, and heads 
        # to the dish dispenser. However, by the time R gets there it is clear
        # that H has decided to get a dish, so the optimal action now becomes
        # going for the onion, wasting quite a lot of time.

        eva = AgentEvaluator({"layout_name": "unident", "start_order_list": ["any", "any"], "cook_time": 5}, force_compute=force_compute)
        start_state = eva.env.mdp.get_standard_start_state()
        start_state.objects = {(5, 2): Obj('soup', (5, 2), ('onion', 2, 0)),
                               (5, 3): Obj('soup', (5, 3), ('onion', 3, 5))}
        eva.start_state = start_state
        self.compare_times(eva, h_idx=0)

    def test_unidentifiable_s(self):
        # Same as above, but smaller layout to facilitate DRL training

        eva = AgentEvaluator({"layout_name": "unident_s", "start_order_list": ["any", "any"], "cook_time": 5}, force_compute=force_compute)
        start_state = eva.env.mdp.get_standard_start_state()
        start_state.objects = {(4, 2): Obj('soup', (4, 2), ('onion', 2, 0)),
                               (4, 3): Obj('soup', (4, 3), ('onion', 3, 5))}
        eva.start_state = start_state
        self.compare_times(eva, h_idx=0)

if __name__ == '__main__':
    unittest.main()
