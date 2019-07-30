import unittest, time, pickle
from overcooked_ai_py.planning.planners import MediumLevelPlanner, Heuristic, HighLevelActionManager, HighLevelPlanner
from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

force_compute = True
force_compute_large = False

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState


# Simple MDP Setup
simple_mdp = OvercookedGridworld.from_layout_name('simple_tomato', start_order_list=['any'], cook_time=5)

base_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': simple_mdp.terrain_pos_dict['X'],
    'counter_drop': simple_mdp.terrain_pos_dict['X'][1:2],
    'counter_pickup': [],
    'same_motion_goals': True
}
action_manger_filename = "simple_1_am.pkl"
ml_planner_simple = MediumLevelPlanner.from_pickle_or_compute(
    simple_mdp, mlp_params=base_params, custom_filename=action_manger_filename, force_compute=force_compute)
ml_planner_simple.env = OvercookedEnv(simple_mdp)

base_params_start_or = {
    'start_orientations': True,
    'wait_allowed': False,
    'counter_goals': simple_mdp.terrain_pos_dict['X'],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': False
}
action_manger_filename = "simple_2_am.pkl"
or_ml_planner_simple = MediumLevelPlanner.from_pickle_or_compute(
    simple_mdp, mlp_params=base_params_start_or, custom_filename=action_manger_filename, force_compute=force_compute)


# Large MDP Setup
large_mdp = OvercookedGridworld.from_layout_name('corridor', start_order_list=['any'], cook_time=5)

no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': False
}
action_manger_filename = "corridor_no_shared_motion_goals_am.pkl"
ml_planner_large_no_shared = MediumLevelPlanner.from_pickle_or_compute(
    large_mdp, no_counters_params, custom_filename=action_manger_filename, force_compute=force_compute_large)

same_goals_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}
action_manger_filename = "corridor_am.pkl"
ml_planner_large = MediumLevelPlanner.from_pickle_or_compute(
    large_mdp, same_goals_params, custom_filename=action_manger_filename, force_compute=force_compute_large)

hlam = HighLevelActionManager(ml_planner_large)
hlp = HighLevelPlanner(hlam)

class TestMotionPlanner(unittest.TestCase):

    def test_gridworld_distance(self):
        planner = ml_planner_simple.ml_action_manager.joint_motion_planner.motion_planner
        start = ((2, 1), e)
        end = ((1, 1), w)
        dist = planner.get_gridworld_distance(start, end)
        self.assertEqual(dist, 1)

        start = ((2, 1), e)
        end = ((1, 1), n)
        dist = planner.get_gridworld_distance(start, end)
        self.assertEqual(dist, 2)

        start = (2, 1)
        end = (1, 1)
        dist = planner.get_gridworld_pos_distance(start, end)
        self.assertEqual(dist, 1)

        start = (1, 1)
        end = (3, 2)
        dist = planner.get_gridworld_pos_distance(start, end)
        self.assertEqual(dist, 3)

    def test_simple_mdp(self):
        planner = ml_planner_simple.ml_action_manager.joint_motion_planner.motion_planner
        self.simple_mdp_already_at_goal(planner)
        self.simple_mdp_orientation_change(planner)
        self.simple_mdp_basic_plan(planner)
        self.simple_mdp_orientation_optimization_dependent_plans(planner)
        
    def simple_mdp_already_at_goal(self, planner):
        start_status = goal_status = ((1, 1), n)
        self.check_single_motion_plan(planner, start_status, goal_status, expected_length=1)

    def simple_mdp_orientation_change(self, planner):
        start_status = ((1, 1), n)
        goal_status = ((1, 1), w)
        self.check_single_motion_plan(planner, start_status, goal_status, expected_length=2)

    def simple_mdp_basic_plan(self, planner):
        start_status = ((1, 1), n)
        goal_status = ((3, 1), n)
        self.check_single_motion_plan(planner, start_status, goal_status, expected_length=4)

    def simple_mdp_orientation_optimization_dependent_plans(self, planner):
        start_status = ((2, 1), n)
        goal_status = ((1, 2), w)
        self.check_single_motion_plan(planner, start_status, goal_status, expected_length=3)

        goal_status = ((1, 2), s)
        self.check_single_motion_plan(planner, start_status, goal_status, expected_length=3)

    def test_larger_mdp(self):
        planner = ml_planner_large.ml_action_manager.joint_motion_planner.motion_planner
        self.large_mdp_basic_plan(planner)

    def large_mdp_basic_plan(self, planner):
        start_status = ((1, 2), n)
        goal_status = ((8, 1), n)
        self.check_single_motion_plan(planner, start_status, goal_status)

    def check_single_motion_plan(self, motion_planner, start_pos_and_or, goal_pos_and_or, expected_length=None):
        dummy_agent = P((3, 2), n)
        start_state = OvercookedState([P(*start_pos_and_or), dummy_agent], {}, order_list=['any', 'any'])
        action_plan, pos_and_or_plan, plan_cost = motion_planner.get_plan(start_pos_and_or, goal_pos_and_or)
        
        # Checking that last state obtained matches goal position
        self.assertEqual(pos_and_or_plan[-1], goal_pos_and_or)

        # In single motion plans the graph cost should be equal to
        # the plan cost (= plan length) as agents should never STAY
        graph_plan_cost = sum([motion_planner._graph_action_cost(a) for a in action_plan])
        self.assertEqual(plan_cost, graph_plan_cost)

        joint_action_plan = [(a, stay) for a in action_plan]
        env = OvercookedEnv(motion_planner.mdp, horizon=1000)
        resulting_state, _ = env.execute_plan(start_state, joint_action_plan)
        self.assertEqual(resulting_state.players_pos_and_or[0], goal_pos_and_or)

        if expected_length is not None: 
            self.assertEqual(len(action_plan), expected_length)


class TestJointMotionPlanner(unittest.TestCase):

    def test_same_start_and_end_pos_with_no_start_orientations(self):
        jm_planner = ml_planner_simple.ml_action_manager.joint_motion_planner
        start = (((1, 1), w), ((1, 2), s))
        goal = (((1, 1), n), ((2, 1), n))

        joint_action_plan, end_jm_state, finshing_times = jm_planner.get_low_level_action_plan(start, goal)
        optimal_plan = [(n, e), (interact, n)]
        self.assertEqual(joint_action_plan, optimal_plan)
        
        optimal_end_jm_state = (((1, 1), n), ((2, 1), n))
        self.assertEqual(end_jm_state, optimal_end_jm_state)

        optimal_finshing_times = (2, 3)
        self.assertEqual(finshing_times, optimal_finshing_times)

    def test_with_start_orientations_simple_mdp(self):
        jm_planner = or_ml_planner_simple.ml_action_manager.joint_motion_planner
        self.simple_mdp_suite(jm_planner)
    
    def test_without_start_orientations_simple_mdp(self):
        jm_planner = ml_planner_simple.ml_action_manager.joint_motion_planner
        self.simple_mdp_suite(jm_planner)

    def simple_mdp_suite(self, jm_planner):
        self.simple_mdp_already_at_goal(jm_planner)
        self.simple_mdp_only_orientations_switch(jm_planner)
        self.simple_mdp_one_at_goal(jm_planner)
        self.simple_mdp_position_swap(jm_planner)
        self.simple_mdp_one_at_goal_other_conflicting_path(jm_planner)
        self.simple_mdp_test_final_orientation_optimization(jm_planner)

    def simple_mdp_already_at_goal(self, planner):
        a1_start = a1_goal = ((1, 1), n)
        a2_start = a2_goal = ((2, 1), n)
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal, times=(1, 1), min_t=1)

        a1_start = a1_goal = ((1, 1), w)
        a2_start = a2_goal = ((1, 2), s)
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal, times=(1, 1), min_t=1)

    def simple_mdp_only_orientations_switch(self, planner):
        a1_start = ((1, 1), s)
        a1_goal = ((1, 1), w)
        a2_start = ((1, 2), s)
        a2_goal = ((1, 2), w)
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal, times=(2, 2), min_t=2)

    def simple_mdp_one_at_goal(self, planner):
        a1_start = ((3, 2), s)
        a1_goal = ((3, 2), s)
        a2_start = ((2, 1), w)
        a2_goal = ((1, 1), w)
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal, times=(1, 2))

    def simple_mdp_position_swap(self, planner):
        a1_start = ((1, 1), w)
        a2_start = ((3, 2), s)
        a1_goal = a2_start
        a2_goal = a1_start
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal)

    def simple_mdp_one_at_goal_other_conflicting_path(self, planner):
        a1_start = ((1, 1), w)
        a1_goal = ((3, 1), e)
        a2_start = a2_goal = ((2, 1), n)
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal, min_t=1)

    def simple_mdp_test_final_orientation_optimization(self, planner):
        a1_start = ((2, 1), n)
        a1_goal = ((1, 2), w)
        a2_start = a2_goal = ((3, 2), s)
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        # NOTE: Not considering all plans with same cost yet, this won't work
        # check_joint_plan(planner, mdp, start, goal, times=(3, 1))

        a1_goal = ((1, 2), s)
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal, times=(3, 1))

    def test_large_mdp_suite_shared_motion_goals(self):
        jmp = ml_planner_large.ml_action_manager.joint_motion_planner
        self.large_mdp_test_basic_plan(jmp)
        self.large_mdp_test_shared_motion_goal(jmp)
        self.large_mdp_test_shared_motion_goal_with_conflict(jmp)
        self.large_mdp_test_shared_motion_goal_with_conflict_other(jmp)

    def large_mdp_test_basic_plan(self, planner):
        a1_start = ((5, 1), n)
        a2_start = ((8, 1), n)
        a1_goal = a2_start
        a2_goal = a1_start
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal)

    def large_mdp_test_shared_motion_goal(self, planner):
        a1_start = ((4, 1), n)
        a2_start = ((1, 1), n)
        a1_goal = ((5, 1), n)
        a2_goal = ((5, 1), n)
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal, min_t=3)

    def large_mdp_test_shared_motion_goal_with_conflict(self, planner):
        assert planner.same_motion_goals
        # When paths conflict for same goal, will resolve by making
        # one agent wait (the one that results in the shortest plan)
        a1_start = ((5, 2), n)
        a2_start = ((4, 1), n)
        a1_goal = ((5, 1), n)
        a2_goal = ((5, 1), n)
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal, min_t=2)

    def large_mdp_test_shared_motion_goal_with_conflict_other(self, planner):
        assert planner.same_motion_goals
        a1_start = ((4, 2), e)
        a2_start = ((4, 1), e)
        a1_goal = ((5, 1), n)
        a2_goal = ((5, 1), n)
        start = (a1_start, a2_start)
        goal = (a1_goal, a2_goal)
        self.check_joint_plan(planner, start, goal, min_t=3)

    def check_joint_plan(self, joint_motion_planner, start, goal, times=None, min_t=None, display=False):
        """Runs the plan in the environment and checks that the intended goals are achieved."""
        debug = False
        action_plan, end_pos_and_orients, plan_lengths = joint_motion_planner.get_low_level_action_plan(start, goal)
        if debug: print("Start state: {}, Goal state: {}, Action plan: {}".format(start, goal, action_plan))

        start_state = OvercookedState([P(*start[0]), P(*start[1])], {}, order_list=['any', 'any'])
        env = OvercookedEnv(joint_motion_planner.mdp, horizon=1000)
        resulting_state, _ = env.execute_plan(start_state, action_plan, display=display)

        self.assertTrue(any([agent_goal in resulting_state.players_pos_and_or for agent_goal in goal]))
        self.assertEqual(resulting_state.players_pos_and_or, end_pos_and_orients)
        self.assertEqual(len(action_plan), min(plan_lengths))

        if min_t is not None: self.assertEqual(len(action_plan), min_t)
        if times is not None: self.assertEqual(plan_lengths, times)


class TestMediumLevelPlanner(unittest.TestCase):

    def test_simple_mdp_without_start_orientations(self):
        print("Simple - no start orientations (& shared motion goals)")
        mlp = ml_planner_simple
        self.simple_mpd_already_done(mlp)
        self.simple_mdp_get_and_serve_soup(mlp)
        self.simple_mdp_get_onion_then_serve(mlp)
        self.simple_mdp_one_delivery_from_start(mlp)
        self.simple_mdp_two_deliveries(mlp)

    def test_simple_mdp_with_start_orientations(self):
        print("Simple - with start orientations (no shared motion goals)")
        mlp = or_ml_planner_simple
        self.simple_mpd_already_done(mlp)
        self.simple_mdp_get_and_serve_soup(mlp)
        self.simple_mdp_get_onion_then_serve(mlp)
        self.simple_mdp_one_delivery_from_start(mlp)
        self.simple_mdp_two_deliveries(mlp)

    def test_large_mdp(self):
        print("Corridor - shared motion goals")
        mlp = ml_planner_large
        self.large_mdp_get_and_serve_soup(mlp)
        self.large_mdp_get_onion_then_serve(mlp)
        self.large_mdp_one_delivery_from_start(mlp)
        self.large_mdp_two_deliveries_from_start(mlp)

    def test_large_mdp_no_shared(self):
        print("Corridor - no shared motion goals")
        mlp = ml_planner_large_no_shared
        self.large_mdp_get_and_serve_soup(mlp)
        self.large_mdp_get_onion_then_serve(mlp)
        self.large_mdp_one_delivery_from_start(mlp)
        self.large_mdp_two_deliveries_from_start(mlp)

    def simple_mpd_already_done(self, planner):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {}, order_list=[])
        self.check_full_plan(s, planner)

    def simple_mdp_get_and_serve_soup(self, planner):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 3, 5))},
            order_list=['onion'])
        self.check_full_plan(s, planner, debug=False)

    def simple_mdp_get_onion_then_serve(self, planner):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 2, 5))},
            order_list=['onion'])
        self.check_full_plan(s, planner)

    def simple_mdp_one_delivery_from_start(self, planner):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {}, order_list=['onion'])
        self.check_full_plan(s, planner)

    def simple_mdp_two_deliveries(self, planner):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {}, order_list=['onion', 'any'])
        self.check_full_plan(s, planner, debug=False)

    def large_mdp_get_and_serve_soup(self, planner):
        s = OvercookedState(
            [P((8, 1), n),
             P((11, 4), n, Obj('dish', (11, 4)))],
            {(8, 8): Obj('soup', (8, 8), ('tomato', 3, 1))},
            order_list=['any'])
        self.check_full_plan(s, planner, debug=False)

    def large_mdp_get_onion_then_serve(self, planner):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {(8, 8): Obj('soup', (8, 8), ('onion', 2, 5))},
            order_list=['onion'])
        self.check_full_plan(s, planner, debug=False)
    
    def large_mdp_one_delivery_from_start(self, planner):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {}, order_list=['any'])
        self.check_full_plan(s, planner, debug=False)

    def large_mdp_two_deliveries_from_start(self, planner):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {}, order_list=['onion', 'any'])
        self.check_full_plan(s, planner, debug=False)
        
    def check_full_plan(self, start_state, planner, debug=False):
        heuristic = Heuristic(planner.mp)
        joint_action_plan = planner.get_low_level_action_plan(start_state, heuristic.simple_heuristic, debug=debug, goal_info=debug)
        env = OvercookedEnv(planner.mdp, horizon=1000)
        resulting_state, _ = env.execute_plan(start_state, joint_action_plan, display=False)
        self.assertEqual(len(resulting_state.order_list), 0)
        

class TestHighLevelPlanner(unittest.TestCase):
    """The HighLevelPlanner class has been mostly discontinued"""
    
    def test_basic_hl_planning(self):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {}, order_list=[])
        h = Heuristic(hlp.mp)
        hlp.get_hl_plan(s, h.simple_heuristic)

        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {}, order_list=['any', 'any', 'any'])
        
        hlp.get_low_level_action_plan(s, h.simple_heuristic)
        # hlp.get_low_level_action_plan(s, h.hard_heuristic)

        # heuristic = Heuristic(ml_planner_large.mp)
        # ml_planner_large.get_low_level_action_plan(s, heuristic.simple_heuristic)
        # ml_planner_large.get_low_level_action_plan(s, heuristic.hard_heuristic)

if __name__ == '__main__':
    unittest.main()
