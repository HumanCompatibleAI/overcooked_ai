import unittest
import numpy as np

from overcooked_ai_py.agents.agent import AgentPair, FixedPlanAgent, GreedyHumanModel, \
    SimpleGreedyHumanModel, RandomAgent, SampleAgent, SlowedDownAgent
from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
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
        self.agent_eval = AgentEvaluator.from_layout_name({"layout_name": "cramped_room"}, {"horizon": 100})
        
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
        
    def test_mlam_computation(self):
        try:
            self.agent_eval.env.mlam
        except Exception as e:
            self.fail("Failed to compute MediumLevelActionManager:\n{}".format(e))


class TestBasicAgents(unittest.TestCase):

    def setUp(self):
        self.mlam_large = MediumLevelActionManager.from_pickle_or_compute(large_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute_large)

    def test_fixed_plan_agents(self):
        a0 = FixedPlanAgent([s, e, n, w])
        a1 = FixedPlanAgent([s, w, n, e])
        agent_pair = AgentPair(a0, a1)
        env = OvercookedEnv.from_mdp(large_mdp, horizon=10)
        trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
        end_state = trajectory[-1][0]
        self.assertEqual(time_taken, 10)
        self.assertEqual(env.mdp.get_standard_start_state().player_positions, end_state.player_positions)

    def test_agents_on_open_map(self):
        scenario_2_mdp = OvercookedGridworld.from_layout_name('scenario2')
        mlam = MediumLevelActionManager.from_pickle_or_compute(scenario_2_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute)
        agent_pairs = [
            AgentPair(GreedyHumanModel(mlam), GreedyHumanModel(mlam)),
            AgentPair(SimpleGreedyHumanModel(mlam), SimpleGreedyHumanModel(mlam)),
            AgentPair(RandomAgent(all_actions=True), RandomAgent(all_actions=True)),
            AgentPair(RandomAgent(all_actions=False), RandomAgent(all_actions=False))
        ]

        start_state = OvercookedState(
            [P((8, 1), s),
             P((1, 1), s)],
            {},
            all_orders=scenario_2_mdp.start_all_orders
        )
        for agent_pair in agent_pairs:
            env = OvercookedEnv.from_mdp(scenario_2_mdp, start_state_fn=lambda: start_state, horizon=100)
            trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)

    def test_sample_agent(self):
        agent = SampleAgent([RandomAgent(all_actions=False), RandomAgent(all_actions=True)])
        probs = agent.action(None)[1]["action_probs"]
        expected_probs = np.array([0.18333333, 0.18333333, 0.18333333, 0.18333333, 0.18333333, 0.08333333])
        self.assertTrue(np.allclose(probs, expected_probs))

    def test_slowed_down_agent(self):
        def should_stop(step_num, stop_every_n_steps):
            # currently SlowDownAgent always stops at 2nd step 
            return not bool((i-1) % stop_every_n_steps)

        horizon = 100
        #NOTE: if stop_every_n_steps is 3 this would not work because of rounding error
        #   (ok for practical purposes, will just skip turn later but would fail test below)
        for stop_every_n_steps in [2, 4]:
            slowdown_rate = 1 - 1/stop_every_n_steps
            
            agent_pair = AgentPair(
                SlowedDownAgent(RandomAgent(), slowdown_rate), 
                SlowedDownAgent(RandomAgent(), slowdown_rate)
                )
            skip_action_probs = SlowedDownAgent(RandomAgent()).skip_action[1]["action_probs"].tolist()
            env = OvercookedEnv.from_mdp(large_mdp, horizon=horizon)
            trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)

            for i, traj_step in enumerate(trajectory):
                (s_t, a_t, r_t, done, info) = traj_step 
                if not done:
                    agent_0_probs = info["agent_infos"][0]["action_probs"]
                    agent_1_probs = info["agent_infos"][1]["action_probs"]
                    if should_stop(i, stop_every_n_steps):
                        self.assertEqual(agent_0_probs.tolist(), skip_action_probs)
                        self.assertEqual(agent_1_probs.tolist(), skip_action_probs)
                    else:
                        self.assertNotEqual(agent_0_probs.tolist(), skip_action_probs)
                        self.assertNotEqual(agent_1_probs.tolist(), skip_action_probs)
    
    def test_pedagogical_ingredients_picking(self):
        mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages_tomato")
        mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=force_compute)
        agent = GreedyHumanModel(mlam, choose_ingredients_pedagogically=True)

        self.assertEqual(agent.next_ingredients([], ['tomato', 'tomato', 'tomato'] ), {"tomato"})
        self.assertEqual(agent.next_ingredients([], ['onion', 'onion', 'tomato']), {'tomato', 'onion'})
        self.assertEqual(agent.next_ingredients([], ['onion', 'onion', 'onion']), {"onion"})

        self.assertEqual(agent.next_ingredients(["onion"], ["onion", "onion", "tomato"]), {"tomato"})
        self.assertEqual(agent.next_ingredients(["onion"], ["onion", "onion", "onion"]), {"onion"})

        self.assertEqual(agent.next_ingredients(["tomato"], ["onion", "onion", "tomato"]), {"onion"})
        self.assertEqual(agent.next_ingredients(["tomato"], ["tomato", "tomato", "tomato"]), {"tomato"})


        

class TestAgentEvaluatorStatic(unittest.TestCase):

    layout_name_lst = ["asymmetric_advantages", "asymmetric_advantages_tomato", "bonus_order_test", "bottleneck",
                       "centre_objects", "centre_pots", "corridor", "forced_coordination_tomato", "unident",
                       "marshmallow_experiment", "marshmallow_experiment_coordination", "you_shall_not_pass"]

    def test_from_mdp(self):
        for layout_name in self.layout_name_lst:
            original_mdp = OvercookedGridworld.from_layout_name(layout_name)
            ae = AgentEvaluator.from_mdp(mdp=original_mdp, env_params={"horizon": 400})
            ae_mdp = ae.env.mdp
            self.assertTrue(original_mdp.ids_independent_equal(ae_mdp), "mdp with name " + layout_name + " experienced an inconsistency")

    def test_from_mdp_params_layout(self):
        for layout_name in self.layout_name_lst:
            original_mdp = OvercookedGridworld.from_layout_name(layout_name)
            ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name": layout_name}, env_params={"horizon": 400})
            ae_mdp = ae.env.mdp
            self.assertTrue(original_mdp.ids_independent_equal(ae_mdp), "mdp with name " + layout_name + " experienced an inconsistency")

    mdp_gen_params_1 = {
        "inner_shape": (10, 7),
        "prop_empty": 0.95,
        "prop_feats": 0.1,
        "start_all_orders": [
            {"ingredients": ["onion", "onion", "onion"]}
        ],
        "display": False,
    }

    mdp_gen_params_2 = {
        "inner_shape": (10, 7),
        "prop_empty": 0.7,
        "prop_feats": 0.5,
        "start_all_orders": [
            {"ingredients": ["onion", "onion", "onion"]}
        ],
        "display": False,
    }

    mdp_gen_params_3 = {
        "inner_shape": (10, 7),
        "prop_empty": 0.5,
        "prop_feats": 0.4,
        "start_all_orders": [
            {"ingredients": ["onion", "onion", "onion"]}
        ],
        "display": False,
    }

    mdp_gen_params_lst = [mdp_gen_params_1, mdp_gen_params_2, mdp_gen_params_3]

    outer_shape = (10, 7)

    def test_from_mdp_params_variable_across(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae0 = AgentEvaluator.from_mdp_params_infinite(mdp_params=mdp_gen_params,
                                                          env_params={"horizon": 400, "num_mdp": np.inf},
                                                          outer_shape=self.outer_shape)
            ae1 = AgentEvaluator.from_mdp_params_infinite(mdp_params=mdp_gen_params,
                                                          env_params={"horizon": 400, "num_mdp": np.inf},
                                                          outer_shape=self.outer_shape)
            self.assertFalse(ae0.env.mdp == ae1.env.mdp,
                             "2 randomly generated layouts across 2 evaluators are the same, which is wrong")

    def test_from_mdp_params_variable_infinite(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_infinite(mdp_params=mdp_gen_params,
                                                         env_params={"horizon": 400, "num_mdp": np.inf},
                                                         outer_shape=self.outer_shape)
            mdp_0 = ae.env.mdp.copy()
            for _ in range(5):
                ae.env.reset(regen_mdp=True)
                mdp_1 = ae.env.mdp
                self.assertFalse(mdp_0 == mdp_1,
                                 "with infinite layout generator and regen_mdp=True, the 2 layouts should not be the same")

    def test_from_mdp_params_variable_infinite_no_regen(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_infinite(mdp_params=mdp_gen_params,
                                                         env_params={"horizon": 400, "num_mdp": np.inf},
                                                         outer_shape=self.outer_shape)
            mdp_0 = ae.env.mdp.copy()
            for _ in range(5):
                ae.env.reset(regen_mdp=False)
                mdp_1 = ae.env.mdp
                self.assertTrue(mdp_0 == mdp_1,
                                 "with infinite layout generator and regen_mdp=False, the 2 layouts should be the same")

    def test_from_mdp_params_variable_infinite_specified(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_infinite(mdp_params=mdp_gen_params,
                                                         env_params={"horizon": 400, "num_mdp": np.inf},
                                                         outer_shape=self.outer_shape)
            mdp_0 = ae.env.mdp.copy()
            for _ in range(5):
                ae.env.reset(regen_mdp=True)
                mdp_1 = ae.env.mdp
                self.assertFalse(mdp_0 == mdp_1,
                                 "with infinite layout generator and regen_mdp=True, the 2 layouts should not be the same")

    def test_from_mdp_params_variable_finite(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_finite(mdp_params=mdp_gen_params,
                                                       env_params={"horizon": 400, "num_mdp": 2},
                                                       outer_shape=self.outer_shape)
            mdp_0 = ae.env.mdp.copy()
            seen = [mdp_0]
            for _ in range(20):
                ae.env.reset(regen_mdp=True)
                mdp_i = ae.env.mdp
                if len(seen) == 1:
                    if mdp_i != seen[0]:
                        seen.append(mdp_i.copy())
                elif len(seen) == 2:
                    mdp_0, mdp_1 = seen
                    self.assertTrue((mdp_i == mdp_0 or mdp_i == mdp_1),
                                        "more than 2 mdp was created, the function failed to perform")
                else:
                    self.assertTrue(False, "theoretically unreachable statement")

    layout_name_short_lst = ["cramped_room", "cramped_room_tomato", "simple_o", "simple_tomato", "simple_o_t"]
    biased = [0.1, 0.15, 0.2, 0.25, 0.3]
    num_reset = 200000

    def test_from_mdp_lst_default(self):
        mdp_lst = [OvercookedGridworld.from_layout_name(name) for name in self.layout_name_short_lst]
        ae = AgentEvaluator.from_mdp_lst(mdp_lst=mdp_lst, env_params={"horizon": 400})
        counts = {}

        for _ in range(self.num_reset):
            ae.env.reset(regen_mdp=True)
            if ae.env.mdp.layout_name in counts:
                counts[ae.env.mdp.layout_name] += 1
            else:
                counts[ae.env.mdp.layout_name] = 1

        for k, v in counts.items():
            self.assertAlmostEqual(0.2, v/self.num_reset, 2, "more than 2 places off for " + k)

    def test_from_mdp_lst_uniform(self):
        mdp_lst = [OvercookedGridworld.from_layout_name(name) for name in self.layout_name_short_lst]
        ae = AgentEvaluator.from_mdp_lst(mdp_lst=mdp_lst, env_params={"horizon": 400}, sampling_freq=[0.2, 0.2, 0.2, 0.2, 0.2])
        counts = {}

        for _ in range(self.num_reset):
            ae.env.reset(regen_mdp=True)
            if ae.env.mdp.layout_name in counts:
                counts[ae.env.mdp.layout_name] += 1
            else:
                counts[ae.env.mdp.layout_name] = 1

        for k, v in counts.items():
            self.assertAlmostEqual(0.2, v/self.num_reset, 2, "more than 2 places off for " + k)

    def test_from_mdp_lst_biased(self):
        mdp_lst = [OvercookedGridworld.from_layout_name(name) for name in self.layout_name_short_lst]
        ae = AgentEvaluator.from_mdp_lst(mdp_lst=mdp_lst, env_params={"horizon": 400}, sampling_freq=self.biased)
        counts = {}

        for _ in range(self.num_reset):
            ae.env.reset(regen_mdp=True)
            if ae.env.mdp.layout_name in counts:
                counts[ae.env.mdp.layout_name] += 1
            else:
                counts[ae.env.mdp.layout_name] = 1

        # construct the ground truth
        gt = {self.layout_name_short_lst[i]: self.biased[i] for i in range(len(self.layout_name_short_lst))}

        for k, v in counts.items():
            self.assertAlmostEqual(gt[k], v/self.num_reset, 2, "more than 2 places off for " + k)


if __name__ == '__main__':
    unittest.main()
