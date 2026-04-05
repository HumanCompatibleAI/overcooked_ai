from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
    SoupState,
)
from overcooked_ai_py.planning.planners import MediumLevelActionManager

large_mdp_tests = False
force_compute = True
force_compute_large = False

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

# Simple MDP Setup
simple_mdp = OvercookedGridworld.from_layout_name("simple_o")


base_params = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": simple_mdp.terrain_pos_dict["X"],
    "counter_drop": simple_mdp.terrain_pos_dict["X"][1:2],
    "counter_pickup": simple_mdp.terrain_pos_dict["X"],
    "same_motion_goals": True,
}
action_manger_filename = "simple_1_am.pkl"
ml_action_manager_simple = MediumLevelActionManager.from_pickle_or_compute(
    simple_mdp,
    mlam_params=base_params,
    custom_filename=action_manger_filename,
    force_compute=force_compute,
)
ml_action_manager_simple.env = OvercookedEnv.from_mdp(simple_mdp)

base_params_start_or = {
    "start_orientations": True,
    "wait_allowed": False,
    "counter_goals": simple_mdp.terrain_pos_dict["X"],
    "counter_drop": [],
    "counter_pickup": simple_mdp.terrain_pos_dict["X"],
    "same_motion_goals": False,
}
action_manger_filename = "simple_2_am.pkl"
or_ml_action_manager_simple = MediumLevelActionManager.from_pickle_or_compute(
    simple_mdp,
    mlam_params=base_params_start_or,
    custom_filename=action_manger_filename,
    force_compute=force_compute,
)

if large_mdp_tests:
    large_mdp = OvercookedGridworld.from_layout_name("corridor", cook_time=5)

    no_counters_params = {
        "start_orientations": False,
        "wait_allowed": False,
        "counter_goals": [],
        "counter_drop": [],
        "counter_pickup": [],
        "same_motion_goals": False,
    }
    action_manger_filename = "corridor_no_shared_motion_goals_am.pkl"
    ml_planner_large_no_shared = (
        MediumLevelActionManager.from_pickle_or_compute(
            large_mdp,
            no_counters_params,
            custom_filename=action_manger_filename,
            force_compute=force_compute_large,
        )
    )

    same_goals_params = {
        "start_orientations": False,
        "wait_allowed": False,
        "counter_goals": [],
        "counter_drop": [],
        "counter_pickup": [],
        "same_motion_goals": True,
    }
    action_manger_filename = "corridor_am.pkl"
    ml_planner_large = MediumLevelActionManager.from_pickle_or_compute(
        large_mdp,
        same_goals_params,
        custom_filename=action_manger_filename,
        force_compute=force_compute_large,
    )


def done_soup_obj(soup_loc, num_onion_inside=3):
    return soup_obj(soup_loc, num_onion_inside, 20)


def idle_soup_obj(soup_loc, num_onion_inside):
    return soup_obj(soup_loc, num_onion_inside, -1)


def cooking_soup_obj(soup_loc, num_onion_inside=3, cooking_tick=0):
    assert cooking_tick >= 0
    assert num_onion_inside >= 0
    return soup_obj(soup_loc, num_onion_inside, cooking_tick)


def soup_obj(soup_loc, num_onion_inside, cooking_tick):
    ingredient_obj_lst = [Obj("onion", soup_loc)] * num_onion_inside
    return SoupState(soup_loc, ingredient_obj_lst, cooking_tick)


def _check_single_motion_plan(
    motion_planner,
    start_pos_and_or,
    goal_pos_and_or,
    expected_length=None,
):
    start_state = OvercookedState(
        [P(*start_pos_and_or), P((3, 2), n)],
        {},
        all_orders=simple_mdp.start_all_orders,
    )
    action_plan, pos_and_or_plan, plan_cost = motion_planner.get_plan(
        start_pos_and_or, goal_pos_and_or
    )

    assert pos_and_or_plan[-1] == goal_pos_and_or

    graph_plan_cost = sum(
        [motion_planner._graph_action_cost(a) for a in action_plan]
    )
    assert plan_cost == graph_plan_cost

    joint_action_plan = [(a, stay) for a in action_plan]
    env = OvercookedEnv.from_mdp(motion_planner.mdp, horizon=1000)
    resulting_state, _ = env.execute_plan(start_state, joint_action_plan)
    assert resulting_state.players_pos_and_or[0] == goal_pos_and_or

    if expected_length is not None:
        assert len(action_plan) == expected_length


def _check_joint_plan(
    joint_motion_planner,
    start,
    goal,
    times=None,
    min_t=None,
    display=False,
):
    (
        action_plan,
        end_pos_and_orients,
        plan_lengths,
    ) = joint_motion_planner.get_low_level_action_plan(start, goal)

    start_state = OvercookedState(
        [P(*start[0]), P(*start[1])],
        {},
        all_orders=simple_mdp.start_all_orders,
    )
    env = OvercookedEnv.from_mdp(joint_motion_planner.mdp, horizon=1000)
    resulting_state, _ = env.execute_plan(
        start_state, action_plan, display=display
    )

    assert any(
        agent_goal in resulting_state.players_pos_and_or
        for agent_goal in goal
    )
    assert resulting_state.players_pos_and_or == end_pos_and_orients
    assert len(action_plan) == min(plan_lengths)

    if min_t is not None:
        assert len(action_plan) == min_t
    if times is not None:
        assert plan_lengths == times


def _check_ml_action_manager(state, am, expected_mla_0, expected_mla_1):
    player_0, player_1 = state.players
    mla_0 = am.get_medium_level_actions(state, player_0)
    mla_1 = am.get_medium_level_actions(state, player_1)

    assert set(mla_0) == set(expected_mla_0), (
        f"player 0's ml_action should be {expected_mla_0} but got {mla_0}"
    )
    assert set(mla_1) == set(expected_mla_1), (
        f"player 1's ml_action should be {expected_mla_1} but got {mla_1}"
    )


class TestMotionPlanner:
    def test_gridworld_distance(self):
        planner = ml_action_manager_simple.joint_motion_planner.motion_planner
        assert planner.get_gridworld_distance(((2, 1), e), ((1, 1), w)) == 1
        assert planner.get_gridworld_distance(((2, 1), e), ((1, 1), n)) == 2
        assert planner.get_gridworld_pos_distance((2, 1), (1, 1)) == 1
        assert planner.get_gridworld_pos_distance((1, 1), (3, 2)) == 3

    def test_simple_mdp(self):
        planner = ml_action_manager_simple.joint_motion_planner.motion_planner
        # already at goal
        _check_single_motion_plan(planner, ((1, 1), n), ((1, 1), n), expected_length=1)
        # orientation change
        _check_single_motion_plan(planner, ((1, 1), n), ((1, 1), w), expected_length=2)
        # basic plan
        _check_single_motion_plan(planner, ((1, 1), n), ((3, 1), n), expected_length=4)
        # orientation optimization dependent plans
        _check_single_motion_plan(planner, ((2, 1), n), ((1, 2), w), expected_length=3)
        _check_single_motion_plan(planner, ((2, 1), n), ((1, 2), s), expected_length=3)

    def test_larger_mdp(self):
        if large_mdp_tests:
            planner = (
                ml_planner_large.ml_action_manager.joint_motion_planner.motion_planner
            )
            _check_single_motion_plan(planner, ((1, 2), n), ((8, 1), n))


class TestJointMotionPlanner:
    def test_same_start_and_end_pos_with_no_start_orientations(self):
        jm_planner = ml_action_manager_simple.joint_motion_planner
        start = (((1, 1), w), ((1, 2), s))
        goal = (((1, 1), n), ((2, 1), n))

        (
            joint_action_plan,
            end_jm_state,
            finshing_times,
        ) = jm_planner.get_low_level_action_plan(start, goal)
        assert joint_action_plan == [(n, e), (interact, n)]
        assert end_jm_state == (((1, 1), n), ((2, 1), n))
        assert finshing_times == (2, 3)

    def test_with_start_orientations_simple_mdp(self):
        jm_planner = or_ml_action_manager_simple.joint_motion_planner
        self._simple_mdp_suite(jm_planner)

    def test_without_start_orientations_simple_mdp(self):
        jm_planner = ml_action_manager_simple.joint_motion_planner
        self._simple_mdp_suite(jm_planner)

    def _simple_mdp_suite(self, jm_planner):
        # already at goal
        _check_joint_plan(jm_planner, (((1, 1), n), ((2, 1), n)), (((1, 1), n), ((2, 1), n)), times=(1, 1), min_t=1)
        _check_joint_plan(jm_planner, (((1, 1), w), ((1, 2), s)), (((1, 1), w), ((1, 2), s)), times=(1, 1), min_t=1)
        # only orientations switch
        _check_joint_plan(jm_planner, (((1, 1), s), ((1, 2), s)), (((1, 1), w), ((1, 2), w)), times=(2, 2), min_t=2)
        # one at goal
        _check_joint_plan(jm_planner, (((3, 2), s), ((2, 1), w)), (((3, 2), s), ((1, 1), w)), times=(1, 2))
        # position swap
        _check_joint_plan(jm_planner, (((1, 1), w), ((3, 2), s)), (((3, 2), s), ((1, 1), w)))
        # one at goal other conflicting path
        _check_joint_plan(jm_planner, (((1, 1), w), ((2, 1), n)), (((3, 1), e), ((2, 1), n)), min_t=1)
        # final orientation optimization
        a1_start = ((2, 1), n)
        a2_start = a2_goal = ((3, 2), s)
        a1_goal = ((1, 2), s)
        _check_joint_plan(jm_planner, (a1_start, a2_start), (a1_goal, a2_goal), times=(3, 1))

    def test_large_mdp_suite_shared_motion_goals(self):
        if large_mdp_tests:
            jmp = ml_planner_large.ml_action_manager.joint_motion_planner
            # basic plan
            _check_joint_plan(jmp, (((5, 1), n), ((8, 1), n)), (((8, 1), n), ((5, 1), n)))
            # shared motion goal
            _check_joint_plan(jmp, (((4, 1), n), ((1, 1), n)), (((5, 1), n), ((5, 1), n)), min_t=3)
            # shared motion goal with conflict
            assert jmp.same_motion_goals
            _check_joint_plan(jmp, (((5, 2), n), ((4, 1), n)), (((5, 1), n), ((5, 1), n)), min_t=2)
            # shared motion goal with conflict other
            _check_joint_plan(jmp, (((4, 2), e), ((4, 1), e)), (((5, 1), n), ((5, 1), n)), min_t=3)


class TestMediumLevelActionManagerSimple:
    ONION_PICKUP = ((3, 2), (1, 0))
    DISH_PICKUP = ((2, 2), (0, 1))
    COUNTER_DROP = ((1, 1), (0, -1))
    COUNTER_PICKUP = ((1, 2), (-1, 0))
    POT_INTERACT = ((2, 1), (00, -1))
    SOUP_DELIVER = ((3, 2), (0, 1))

    def test_simple_mdp_without_start_orientations(self):
        mlam = ml_action_manager_simple
        self._run_all_scenarios(mlam, counter_drop_forbidden=False)

    def test_simple_mdp_with_start_orientations(self):
        mlam = or_ml_action_manager_simple
        self._run_all_scenarios(mlam, counter_drop_forbidden=True)

    def _run_all_scenarios(self, planner, counter_drop_forbidden):
        self._empty_hands(planner, counter_drop_forbidden)
        self._deliver_soup(planner, counter_drop_forbidden)
        self._pickup_counter_soup(planner, counter_drop_forbidden)
        self._pickup_counter_dish(planner, counter_drop_forbidden)
        self._pickup_counter_onion(planner, counter_drop_forbidden)
        self._drop_useless_dish_with_soup_idle(planner, counter_drop_forbidden)
        self._pickup_soup(planner, counter_drop_forbidden)
        self._pickup_dish(planner, counter_drop_forbidden)
        self._start_good_soup_cooking(planner, counter_drop_forbidden)
        self._start_bad_soup_cooking(planner, counter_drop_forbidden)
        self._start_1_onion_soup_cooking(planner, counter_drop_forbidden)
        self._drop_useless_onion_good_soup(planner, counter_drop_forbidden)
        self._drop_useless_onion_bad_soup(planner, counter_drop_forbidden)
        self._add_3rd_onion(planner, counter_drop_forbidden)
        self._add_2nd_onion(planner, counter_drop_forbidden)
        self._drop_useless_dish(planner, counter_drop_forbidden)

    def _empty_hands(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n)], {},
            all_orders=simple_mdp.start_all_orders,
        )
        _check_ml_action_manager(
            s, planner,
            [self.ONION_PICKUP, self.DISH_PICKUP],
            [self.ONION_PICKUP, self.DISH_PICKUP],
        )

    def _deliver_soup(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n, done_soup_obj((2, 1)))], {},
            all_orders=simple_mdp.start_all_orders,
        )
        if counter_drop_forbidden:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP], [self.SOUP_DELIVER])
        else:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP], [self.COUNTER_DROP, self.SOUP_DELIVER])

    def _pickup_counter_soup(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n)],
            {(0, 2): done_soup_obj((0, 2))},
            all_orders=simple_mdp.start_all_orders,
        )
        _check_ml_action_manager(s, planner,
            [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP],
            [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP])

    def _pickup_counter_dish(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n)],
            {(0, 2): Obj("dish", (0, 2))},
            all_orders=simple_mdp.start_all_orders,
        )
        _check_ml_action_manager(s, planner,
            [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP],
            [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP])

    def _pickup_counter_onion(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n)],
            {(0, 2): Obj("onion", (0, 2))},
            all_orders=simple_mdp.start_all_orders,
        )
        _check_ml_action_manager(s, planner,
            [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP],
            [self.ONION_PICKUP, self.DISH_PICKUP, self.COUNTER_PICKUP])

    def _drop_useless_dish_with_soup_idle(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n, Obj("dish", (2, 1)))],
            {(2, 0): idle_soup_obj((2, 0), 3)},
            all_orders=simple_mdp.start_all_orders,
        )
        if counter_drop_forbidden:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT], [])
        else:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT], [self.COUNTER_DROP])

    def _pickup_soup(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n, Obj("dish", (2, 1)))],
            {(2, 0): done_soup_obj((2, 0))},
            all_orders=simple_mdp.start_all_orders,
        )
        if counter_drop_forbidden:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP], [self.POT_INTERACT])
        else:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP], [self.COUNTER_DROP, self.POT_INTERACT])

    def _pickup_dish(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n)],
            {(2, 0): done_soup_obj((2, 0))},
            all_orders=simple_mdp.start_all_orders,
        )
        _check_ml_action_manager(s, planner,
            [self.ONION_PICKUP, self.DISH_PICKUP],
            [self.ONION_PICKUP, self.DISH_PICKUP])

    def _start_good_soup_cooking(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n)],
            {(2, 0): idle_soup_obj((2, 0), 3)},
            all_orders=simple_mdp.start_all_orders,
        )
        _check_ml_action_manager(s, planner,
            [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
            [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT])

    def _start_bad_soup_cooking(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n)],
            {(2, 0): idle_soup_obj((2, 0), 2)},
            all_orders=simple_mdp.start_all_orders,
        )
        _check_ml_action_manager(s, planner,
            [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
            [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT])

    def _start_1_onion_soup_cooking(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n)],
            {(2, 0): idle_soup_obj((2, 0), 1)},
            all_orders=simple_mdp.start_all_orders,
        )
        _check_ml_action_manager(s, planner,
            [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT],
            [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT])

    def _drop_useless_onion_good_soup(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n, Obj("onion", (2, 1)))],
            {(2, 0): done_soup_obj((2, 0))},
            all_orders=simple_mdp.start_all_orders,
        )
        if counter_drop_forbidden:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP], [])
        else:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP], [self.COUNTER_DROP])

    def _drop_useless_onion_bad_soup(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n, Obj("onion", (2, 1)))],
            {(2, 0): done_soup_obj((2, 0), 2)},
            all_orders=simple_mdp.start_all_orders,
        )
        if counter_drop_forbidden:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP], [])
        else:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP], [self.COUNTER_DROP])

    def _add_3rd_onion(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n, Obj("onion", (2, 1)))],
            {(2, 0): idle_soup_obj((2, 0), 2)},
            all_orders=simple_mdp.start_all_orders,
        )
        if counter_drop_forbidden:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT], [self.POT_INTERACT])
        else:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT], [self.COUNTER_DROP, self.POT_INTERACT])

    def _add_2nd_onion(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n, Obj("onion", (2, 1)))],
            {(2, 0): idle_soup_obj((2, 0), 1)},
            all_orders=simple_mdp.start_all_orders,
        )
        if counter_drop_forbidden:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT], [self.POT_INTERACT])
        else:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT], [self.COUNTER_DROP, self.POT_INTERACT])

    def _drop_useless_dish(self, planner, counter_drop_forbidden):
        s = OvercookedState(
            [P((2, 2), n), P((2, 1), n, Obj("dish", (2, 1)))],
            {(2, 0): idle_soup_obj((2, 0), 1)},
            all_orders=simple_mdp.start_all_orders,
        )
        if counter_drop_forbidden:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT], [self.POT_INTERACT])
        else:
            _check_ml_action_manager(s, planner,
                [self.ONION_PICKUP, self.DISH_PICKUP, self.POT_INTERACT], [self.COUNTER_DROP, self.POT_INTERACT])


class TestScenarios:
    def _repetitive_runs(self, evaluator, num_games=10):
        trajectory_0 = evaluator.evaluate_human_model_pair(
            num_games=num_games, native_eval=True
        )
        trajectory_1 = evaluator.evaluate_human_model_pair(
            num_games=num_games, native_eval=True
        )

        h0 = GreedyHumanModel(evaluator.env.mlam)
        h1 = GreedyHumanModel(evaluator.env.mlam)
        ap_hh_2 = AgentPair(h0, h1)
        trajectory_2 = evaluator.evaluate_agent_pair(
            agent_pair=ap_hh_2, num_games=num_games, native_eval=True
        )

        h3 = GreedyHumanModel(evaluator.env.mlam)
        h4 = GreedyHumanModel(evaluator.env.mlam)
        ap_hh_3 = AgentPair(h3, h4)
        trajectory_3 = evaluator.evaluate_agent_pair(
            agent_pair=ap_hh_3, num_games=num_games, native_eval=True
        )

    def test_scenario_3_no_counter(self):
        mdp_params = {"layout_name": "scenario3"}
        mdp = OvercookedGridworld.from_layout_name(**mdp_params)
        start_state = mdp.get_standard_start_state()

        env_params = {"start_state_fn": lambda: start_state, "horizon": 1000}
        eva = AgentEvaluator.from_layout_name(
            mdp_params, env_params, force_compute=force_compute
        )
        self._repetitive_runs(eva)

    def test_scenario_3_yes_counter(self):
        mdp_params = {"layout_name": "scenario3"}
        mdp = OvercookedGridworld.from_layout_name(**mdp_params)
        start_state = mdp.get_standard_start_state()

        valid_counters = [(5, 3)]
        one_counter_params = {
            "start_orientations": False,
            "wait_allowed": False,
            "counter_goals": valid_counters,
            "counter_drop": valid_counters,
            "counter_pickup": [],
            "same_motion_goals": True,
        }

        env_params = {"start_state_fn": lambda: start_state, "horizon": 1000}
        eva = AgentEvaluator.from_layout_name(
            mdp_params,
            env_params,
            mlam_params=one_counter_params,
            force_compute=force_compute,
        )
        self._repetitive_runs(eva)
