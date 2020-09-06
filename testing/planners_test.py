import unittest
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, SoupState, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel

large_mdp_tests = False
force_compute = True
force_compute_large = False

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

# Simple MDP Setup
simple_mdp = OvercookedGridworld.from_layout_name('simple_o')


base_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': simple_mdp.terrain_pos_dict['X'],
    'counter_drop': simple_mdp.terrain_pos_dict['X'][1:2],
    'counter_pickup': simple_mdp.terrain_pos_dict['X'],
    'same_motion_goals': True
}
action_manger_filename = "simple_1_am.pkl"
ml_action_manager_simple = MediumLevelActionManager.from_pickle_or_compute(
    simple_mdp, mlam_params=base_params, custom_filename=action_manger_filename, force_compute=force_compute)
ml_action_manager_simple.env = OvercookedEnv.from_mdp(simple_mdp)

base_params_start_or = {
    'start_orientations': True,
    'wait_allowed': False,
    'counter_goals': simple_mdp.terrain_pos_dict['X'],
    'counter_drop': [],
    'counter_pickup': simple_mdp.terrain_pos_dict['X'],
    'same_motion_goals': False
}
action_manger_filename = "simple_2_am.pkl"
or_ml_action_manager_simple = MediumLevelActionManager.from_pickle_or_compute(
    simple_mdp, mlam_params=base_params_start_or, custom_filename=action_manger_filename, force_compute=force_compute)

if large_mdp_tests:
    # Not testing by default

    # Large MDP Setup
    large_mdp = OvercookedGridworld.from_layout_name('corridor', cook_time=5)

    no_counters_params = {
        'start_orientations': False,
        'wait_allowed': False,
        'counter_goals': [],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }
    action_manger_filename = "corridor_no_shared_motion_goals_am.pkl"
    ml_planner_large_no_shared = MediumLevelActionManager.from_pickle_or_compute(
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
    ml_planner_large = MediumLevelActionManager.from_pickle_or_compute(
        large_mdp, same_goals_params, custom_filename=action_manger_filename, force_compute=force_compute_large)
    # Deprecated.
    # hlam = HighLevelActionManager(ml_planner_large)
    # hlp = HighLevelPlanner(hlam)


def done_soup_obj(soup_loc, num_onion_inside=3):
    return soup_obj(soup_loc, num_onion_inside, 20)


def idle_soup_obj(soup_loc, num_onion_inside):
    return soup_obj(soup_loc, num_onion_inside, -1)


def cooking_soup_obj(soup_loc, num_onion_inside=3, cooking_tick=0):
    assert cooking_tick >= 0
    assert num_onion_inside >= 0
    return soup_obj(soup_loc, num_onion_inside, cooking_tick)


def soup_obj(soup_loc, num_onion_inside, cooking_tick):
    ingredient_obj_lst = [Obj('onion', soup_loc)] * num_onion_inside
    return SoupState(soup_loc, ingredient_obj_lst, cooking_tick)


class TestMotionPlanner(unittest.TestCase):

    def test_gridworld_distance(self):
        planner = ml_action_manager_simple.joint_motion_planner.motion_planner
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
        planner = ml_action_manager_simple.joint_motion_planner.motion_planner
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
        if large_mdp_tests:
            planner = ml_planner_large.ml_action_manager.joint_motion_planner.motion_planner
            self.large_mdp_basic_plan(planner)

    def large_mdp_basic_plan(self, planner):
        start_status = ((1, 2), n)
        goal_status = ((8, 1), n)
        self.check_single_motion_plan(planner, start_status, goal_status)

    def check_single_motion_plan(self, motion_planner, start_pos_and_or, goal_pos_and_or, expected_length=None):
        dummy_agent = P((3, 2), n)
        start_state = OvercookedState([P(*start_pos_and_or), dummy_agent], {}, all_orders=simple_mdp.start_all_orders)
        action_plan, pos_and_or_plan, plan_cost = motion_planner.get_plan(start_pos_and_or, goal_pos_and_or)
        
        # Checking that last state obtained matches goal position
        self.assertEqual(pos_and_or_plan[-1], goal_pos_and_or)

        # In single motion plans the graph cost should be equal to
        # the plan cost (= plan length) as agents should never STAY
        graph_plan_cost = sum([motion_planner._graph_action_cost(a) for a in action_plan])
        self.assertEqual(plan_cost, graph_plan_cost)

        joint_action_plan = [(a, stay) for a in action_plan]
        env = OvercookedEnv.from_mdp(motion_planner.mdp, horizon=1000)
        resulting_state, _ = env.execute_plan(start_state, joint_action_plan)
        self.assertEqual(resulting_state.players_pos_and_or[0], goal_pos_and_or)

        if expected_length is not None: 
            self.assertEqual(len(action_plan), expected_length)


class TestJointMotionPlanner(unittest.TestCase):

    def test_same_start_and_end_pos_with_no_start_orientations(self):
        jm_planner = ml_action_manager_simple.joint_motion_planner
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
        jm_planner = or_ml_action_manager_simple.joint_motion_planner
        self.simple_mdp_suite(jm_planner)
    
    def test_without_start_orientations_simple_mdp(self):
        jm_planner = ml_action_manager_simple.joint_motion_planner
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
        if large_mdp_tests:
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

        start_state = OvercookedState([P(*start[0]), P(*start[1])], {}, all_orders=simple_mdp.start_all_orders)
        env = OvercookedEnv.from_mdp(joint_motion_planner.mdp, horizon=1000)
        resulting_state, _ = env.execute_plan(start_state, action_plan, display=display)

        self.assertTrue(any([agent_goal in resulting_state.players_pos_and_or for agent_goal in goal]))
        self.assertEqual(resulting_state.players_pos_and_or, end_pos_and_orients)
        self.assertEqual(len(action_plan), min(plan_lengths))

        if min_t is not None: self.assertEqual(len(action_plan), min_t)
        if times is not None: self.assertEqual(plan_lengths, times)


# Rewritten because the previous test depended on Heuristic, and Heuristic has been deprecated
class TestMediumLevelActionManagerSimple(unittest.TestCase):
    def test_simple_mdp_without_start_orientations(self):
        print("Simple - no start orientations (& shared motion goals)")
        mlam = ml_action_manager_simple
        self.simple_mpd_empty_hands(mlam)
        self.simple_mdp_deliver_soup(mlam)
        self.simple_mdp_pickup_counter_soup(mlam)
        self.simple_mdp_pickup_counter_dish(mlam)
        self.simple_mdp_pickup_counter_onion(mlam)
        self.simple_mdp_drop_useless_dish_with_soup_idle(mlam)
        self.simple_mdp_pickup_soup(mlam)
        self.simple_mdp_pickup_dish(mlam)
        self.simple_mdp_start_good_soup_cooking(mlam)
        self.simple_mdp_start_bad_soup_cooking(mlam)
        self.simple_mdp_start_1_onion_soup_cooking(mlam)
        self.simple_mdp_drop_useless_onion_good_soup(mlam)
        self.simple_mdp_drop_useless_onion_bad_soup(mlam)
        self.simple_mdp_add_3rd_onion(mlam)
        self.simple_mdp_add_2nd_onion(mlam)
        self.simple_mdp_drop_useless_dish(mlam)

    def test_simple_mdp_with_start_orientations(self):
        print("Simple - with start orientations (no shared motion goals)")
        mlam = or_ml_action_manager_simple
        self.simple_mpd_empty_hands(mlam, counter_drop_forbidden=True)
        self.simple_mdp_deliver_soup(mlam, counter_drop_forbidden=True)
        self.simple_mdp_pickup_counter_soup(mlam, counter_drop_forbidden=True)
        self.simple_mdp_pickup_counter_dish(mlam, counter_drop_forbidden=True)
        self.simple_mdp_pickup_counter_onion(mlam, counter_drop_forbidden=True)
        self.simple_mdp_drop_useless_dish_with_soup_idle(mlam, counter_drop_forbidden=True)
        self.simple_mdp_pickup_soup(mlam, counter_drop_forbidden=True)
        self.simple_mdp_pickup_dish(mlam, counter_drop_forbidden=True)
        self.simple_mdp_start_good_soup_cooking(mlam, counter_drop_forbidden=True)
        self.simple_mdp_start_bad_soup_cooking(mlam, counter_drop_forbidden=True)
        self.simple_mdp_start_1_onion_soup_cooking(mlam, counter_drop_forbidden=True)
        self.simple_mdp_drop_useless_onion_good_soup(mlam, counter_drop_forbidden=True)
        self.simple_mdp_drop_useless_onion_bad_soup(mlam, counter_drop_forbidden=True)
        self.simple_mdp_add_3rd_onion(mlam, counter_drop_forbidden=True)
        self.simple_mdp_add_2nd_onion(mlam, counter_drop_forbidden=True)
        self.simple_mdp_drop_useless_dish(mlam, counter_drop_forbidden=True)

    ONION_PICKUP = ((3, 2), (1, 0))
    DISH_PICKUP = ((2, 2), (0, 1))
    COUNTER_DROP = ((1, 1), (0, -1))
    COUNTER_PICKUP = ((1, 2), (-1, 0))
    POT_INTERACT = ((2, 1), (00, -1))
    SOUP_DELIVER = ((3, 2), (0, 1))

    def simple_mpd_empty_hands(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {},
            all_orders=simple_mdp.start_all_orders)
        self.check_ml_action_manager(s, planner,
                                     [self.ONION_PICKUP, self.DISH_PICKUP],
                                     [self.ONION_PICKUP, self.DISH_PICKUP]
                                     )

    def simple_mdp_deliver_soup(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n, done_soup_obj((2, 1)))],
            {},
            all_orders=simple_mdp.start_all_orders)

        if counter_drop_forbidden:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP],
                                         [self.SOUP_DELIVER]
                                         )
        else:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP],
                                         [self.COUNTER_DROP, self.SOUP_DELIVER]
                                         )

    def simple_mdp_pickup_counter_soup(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {(0, 2): done_soup_obj((0, 2))},
            all_orders=simple_mdp.start_all_orders)
        self.check_ml_action_manager(s, planner,
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP],
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP]
                                     )

    def simple_mdp_pickup_counter_dish(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {(0, 2): Obj('dish', (0, 2))},
            all_orders=simple_mdp.start_all_orders)
        self.check_ml_action_manager(s, planner,
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP],
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP]
                                     )

    def simple_mdp_pickup_counter_onion(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {(0, 2): Obj('onion', (0, 2))},
            all_orders=simple_mdp.start_all_orders)
        self.check_ml_action_manager(s, planner,
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP],
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP]
                                     )

    def simple_mdp_drop_useless_dish_with_soup_idle(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n, Obj('dish', (2, 1)))],
            {(2, 0): idle_soup_obj((2, 0), 3)},
            all_orders=simple_mdp.start_all_orders)
        if counter_drop_forbidden:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                         []
                                         )
        else:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                         [self.COUNTER_DROP]
                                         )

    def simple_mdp_pickup_soup(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n, Obj('dish', (2, 1)))],
            {(2, 0): done_soup_obj((2, 0))},
            all_orders=simple_mdp.start_all_orders)
        if counter_drop_forbidden:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP],
                                         [self.POT_INTERACT]
                                         )
        else:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP],
                                         [self.COUNTER_DROP, self.POT_INTERACT]
                                         )

    def simple_mdp_pickup_dish(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {(2, 0): done_soup_obj((2, 0))},
            all_orders=simple_mdp.start_all_orders)
        self.check_ml_action_manager(s, planner,
                                     [self.ONION_PICKUP, self.DISH_PICKUP],
                                     [self.ONION_PICKUP, self.DISH_PICKUP]
                                     )

    def simple_mdp_start_good_soup_cooking(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {(2, 0): idle_soup_obj((2, 0), 3)},
            all_orders=simple_mdp.start_all_orders)
        self.check_ml_action_manager(s, planner,
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT]
                                     )

    def simple_mdp_start_bad_soup_cooking(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {(2, 0): idle_soup_obj((2, 0), 2)},
            all_orders=simple_mdp.start_all_orders)
        self.check_ml_action_manager(s, planner,
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT]
                                     )

    def simple_mdp_start_1_onion_soup_cooking(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n)],
            {(2, 0): idle_soup_obj((2, 0), 1)},
            all_orders=simple_mdp.start_all_orders)
        self.check_ml_action_manager(s, planner,
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                     [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT]
                                     )

    def simple_mdp_drop_useless_onion_good_soup(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n, Obj('onion', (2, 1)))],
            {(2, 0): done_soup_obj((2, 0))},
            all_orders=simple_mdp.start_all_orders)
        if counter_drop_forbidden:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP],
                                         []
                                         )
        else:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP],
                                         [self.COUNTER_DROP]
                                         )

    def simple_mdp_drop_useless_onion_bad_soup(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n, Obj('onion', (2, 1)))],
            {(2, 0): done_soup_obj((2, 0), 2)},
            all_orders=simple_mdp.start_all_orders)
        if counter_drop_forbidden:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP],
                                         []
                                         )
        else:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP],
                                         [self.COUNTER_DROP]
                                         )

    def simple_mdp_add_3rd_onion(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n, Obj('onion', (2, 1)))],
            {(2, 0): idle_soup_obj((2, 0), 2)},
            all_orders=simple_mdp.start_all_orders)
        if counter_drop_forbidden:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                         [self.POT_INTERACT]
                                         )
        else:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                         [self.COUNTER_DROP, self.POT_INTERACT]
                                         )

    def simple_mdp_add_2nd_onion(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n, Obj('onion', (2, 1)))],
            {(2, 0): idle_soup_obj((2, 0), 1)},
            all_orders=simple_mdp.start_all_orders)
        if counter_drop_forbidden:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                         [self.POT_INTERACT]
                                         )
        else:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                         [self.COUNTER_DROP, self.POT_INTERACT]
                                         )

    def simple_mdp_drop_useless_dish(self, planner, counter_drop_forbidden=False):
        s = OvercookedState(
            [P((2, 2), n),
             P((2, 1), n, Obj('dish', (2, 1)))],
            {(2, 0): idle_soup_obj((2, 0), 1)},
            all_orders=simple_mdp.start_all_orders)
        if counter_drop_forbidden:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                         [self.POT_INTERACT]
                                         )
        else:
            self.check_ml_action_manager(s, planner,
                                         [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
                                         [self.COUNTER_DROP, self.POT_INTERACT]
                                         )

    def check_ml_action_manager(self, state, am, expected_mla_0, expected_mla_1, debug=False):
        """
        args:
            state (OvercookedState): an overcooked state
            am (MediumLevelActionManager): the planer whose action manager will be tested

        This function checks if all the mid-level actions make sense for each player state inside STATE
        """
        player_0, player_1 = state.players

        mla_0 = am.get_medium_level_actions(state, player_0)
        mla_1 = am.get_medium_level_actions(state, player_1)


        if debug:
            print("Player 0 mla", mla_0)
            print("Player 1 mla", mla_1)
            print(am.mdp.state_string(state))

        self.assertEqual(set(mla_0), set(expected_mla_0),
                         "player 0's ml_action should be " + str(expected_mla_0) +
                         " but get " + str(mla_0))
        self.assertEqual(set(mla_1), set(expected_mla_1),
                         "player 0's ml_action should be " + str(expected_mla_1) +
                         " but get " + str(mla_1))

class TestScenarios(unittest.TestCase):
    def repetative_runs(self, evaluator, num_games=10):
        trajectory_0 = evaluator.evaluate_human_model_pair(num_games=num_games, native_eval=True)
        trajectory_1 = evaluator.evaluate_human_model_pair(num_games=num_games, native_eval=True)

        h0 = GreedyHumanModel(evaluator.env.mlam)
        h1 = GreedyHumanModel(evaluator.env.mlam)
        ap_hh_2 = AgentPair(h0, h1)
        trajectory_2 = evaluator.evaluate_agent_pair(agent_pair=ap_hh_2, num_games=num_games, native_eval=True)

        h3 = GreedyHumanModel(evaluator.env.mlam)
        h4 = GreedyHumanModel(evaluator.env.mlam)
        ap_hh_3 = AgentPair(h3, h4)
        trajectory_3 = evaluator.evaluate_agent_pair(agent_pair=ap_hh_3, num_games=num_games, native_eval=True)


    def test_scenario_3_no_counter(self):
        # Asymmetric advantage scenario
        #
        # X X X X X O X X X X
        # S           X X P X
        # X         ↑H      X
        # D   X X X X!X X   X
        # X           →R    O
        # X X X X X X X X X X
        #
        # This test does not allow counter by using the default NO_COUNTER_PARAMS when calling from_layout_name

        mdp_params = {"layout_name": "scenario3"}
        mdp = OvercookedGridworld.from_layout_name(**mdp_params)
        start_state = mdp.get_standard_start_state()

        env_params = {"start_state_fn": lambda: start_state, "horizon": 1000}
        eva = AgentEvaluator.from_layout_name(mdp_params, env_params, force_compute=force_compute)

        self.repetative_runs(eva)


    def test_scenario_3_yes_counter(self):
        # Asymmetric advantage scenario
        #
        # X X X X X O X X X X
        # S           X X P X
        # X         ↑H      X
        # D   X X X X!X X   X
        # X           →R    O
        # X X X X X X X X X X
        #
        # This test does not allow only (5. 3) as the only counter

        mdp_params = {"layout_name": "scenario3"}
        mdp = OvercookedGridworld.from_layout_name(**mdp_params)
        start_state = mdp.get_standard_start_state()

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
        eva = AgentEvaluator.from_layout_name(mdp_params, env_params, mlam_params=one_counter_params, force_compute=force_compute)

        self.repetative_runs(eva)
# # Deprecated. because of Heuristic
# class TestHighLevelPlanner(unittest.TestCase):
#     """The HighLevelPlanner class has been mostly discontinued"""
#
#     def test_basic_hl_planning(self):
#         if large_mdp_tests:
#             s = OvercookedState(
#                 [P((2, 2), n),
#                 P((2, 1), n)],
#                 {}, order_list=[])
#             h = Heuristic(hlp.mp)
#             hlp.get_hl_plan(s, h.simple_heuristic)
#
#             s = OvercookedState(
#                 [P((2, 2), n),
#                 P((2, 1), n)],
#                 {}, order_list=['any', 'any', 'any'])
#
#             hlp.get_low_level_action_plan(s, h.simple_heuristic)
#         # hlp.get_low_level_action_plan(s, h.hard_heuristic)
#
#         # heuristic = Heuristic(ml_planner_large.mp)
#         # ml_planner_large.get_low_level_action_plan(s, heuristic.simple_heuristic)
#         # ml_planner_large.get_low_level_action_plan(s, heuristic.hard_heuristic)

if __name__ == '__main__':
    unittest.main()
