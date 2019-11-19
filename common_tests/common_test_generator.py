from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, ObjectState, PlayerState
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
S, P, Obj = OvercookedState, PlayerState, ObjectState

delivery_reward = 20

s_a_r_pairs = [
    (S([P((1, 2), n), P((3, 1), n)], {}, order_list=None), [n, e], 0),
    (S([P((1, 1), n), P((3, 1), e)], {}, order_list=None), [w, interact], 0),
    (S([P((1, 1), w), P((3, 1), e, Obj('onion', (3, 1)))], {}, order_list=None), [interact, w], 0),
    (S([P((1, 1), w, Obj('onion', (1, 1))),P((2, 1), w, Obj('onion', (2, 1)))],{}, order_list=None), [e, n], 0),
    (S([P((1, 1), e, Obj('onion', (1, 1))),P((2, 1), n, Obj('onion', (2, 1)))],{}, order_list=None), [stay, interact], 0),
    (S([P((1, 1), e, Obj('onion', (1, 1))),P((2, 1), n)],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, order_list=None), [e, e], 0),
    (S([P((2, 1), e, Obj('onion', (2, 1))),P((3, 1), e)],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, order_list=None), [n, interact], 0),
    (S([P((2, 1), n, Obj('onion', (2, 1))),P((3, 1), e, Obj('onion', (3, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, order_list=None), [interact, w], 0),
    (S([P((2, 1), n),P((3, 1), w, Obj('onion', (3, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 2, 0))}, order_list=None), [w, w], 0),
    (S([P((1, 1), w),P((2, 1), w, Obj('onion', (2, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 2, 0))}, order_list=None), [s, n], 0),
    (S([P((1, 2), s),P((2, 1), n, Obj('onion', (2, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 2, 0))}, order_list=None), [interact, interact], 0),
    (S([P((1, 2), s, Obj('dish', (1, 2))),P((2, 1), n)],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 1))}, order_list=None), [e, s], 0),
    (S([P((1, 2), e, Obj('dish', (1, 2))),P((2, 1), s)],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 2))}, order_list=None), [e, interact], 0),
    (S([P((2, 2), e, Obj('dish', (2, 2))),P((2, 1), s)],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 3))}, order_list=None), [n, e], 0),
    (S([P((2, 1), n, Obj('dish', (2, 1))),P((3, 1), e)],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 4))}, order_list=None), [interact, interact], 0),
    (S([P((2, 1), n, Obj('dish', (2, 1))),P((3, 1), e, Obj('onion', (3, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 5))}, order_list=None), [stay, stay], 0),
    (S([P((2, 1), n, Obj('dish', (2, 1))),P((3, 1), e, Obj('onion', (3, 1)))],{(2, 0): Obj('soup', (2, 0), ('onion', 3, 5))}, order_list=None), [interact, interact], 0),
    (S([P((2, 1), n, Obj('soup', (2, 1), ('onion', 3, 5))),P((3, 1), e, Obj('onion', (3, 1)))],{}, order_list=None), [e, w], 0),
    (S([P((2, 1), e, Obj('soup', (2, 1), ('onion', 3, 5))),P((3, 1), w, Obj('onion', (3, 1)))],{}, order_list=None), [e, s], 0),
    (S([P((3, 1), e, Obj('soup', (3, 1), ('onion', 3, 5))),P((3, 2), s, Obj('onion', (3, 2)))],{}, order_list=None), [s, interact], 0),
    (S([P((3, 1), s, Obj('soup', (3, 1), ('onion', 3, 5))),P((3, 2), s, Obj('onion', (3, 2)))],{}, order_list=None), [s, w], 0),
    (S([P((3, 2), s, Obj('soup', (3, 2), ('onion', 3, 5))),P((2, 2), w, Obj('onion', (2, 2)))],{}, order_list=None), [interact, n], delivery_reward),
    (S([P((3, 2), s),P((2, 1), n, Obj('onion', (2, 1)))],{}, order_list=None), [e, interact], 0),
    (S([P((3, 2), e),P((2, 1), n)],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, order_list=None), [interact, s], 0),
    (S([P((3, 2), e, Obj('tomato', (3, 2))),P((2, 2), s)],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, order_list=None), [w, w], 0),
    (S([P((2, 2), w, Obj('tomato', (2, 2))),P((1, 2), w)],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, order_list=None), [n, interact], 0),
    (S([P((2, 1), n, Obj('tomato', (2, 1))),P((1, 2), w, Obj('tomato', (1, 2)))],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, order_list=None), [interact, interact], 0),
    (S([P((2, 1), n, Obj('tomato', (2, 1))),P((1, 2), w, Obj('tomato', (1, 2)))],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, order_list=None), [s, interact], 0),
    (S([P((2, 2), s, Obj('tomato', (2, 2))),P((1, 2), w, Obj('tomato', (1, 2)))],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, order_list=None), [interact, interact], 0),
    (S([P((2, 2), s),P((1, 2), w, Obj('tomato', (1, 2)))],{(2, 0): Obj('soup', (2, 0), ('onion', 1, 0)),(2, 3): Obj('soup', (2, 3), ('tomato', 1, 0))}, order_list=None), [interact, interact], 0)
]

curr_ep_rewards = [s_a_r[2] for s_a_r in s_a_r_pairs]

traj = {
    "ep_observations": [[s_a_r[0] for s_a_r in s_a_r_pairs]],
    "ep_actions": [[tuple(s_a_r[1]) for s_a_r in s_a_r_pairs]],
    "ep_rewards": [curr_ep_rewards],
    "ep_dones": [False] * len(s_a_r_pairs),

    "ep_returns": [sum(curr_ep_rewards)],
    "ep_returns_sparse": [sum(curr_ep_rewards)],
    "ep_lengths": [len(curr_ep_rewards)],
    "mdp_params": [{
        "layout_name": "mdp_test",
        "cook_time": 5,
        "start_order_list": None,
        "num_items_for_soup": 3,
        "rew_shaping_params": None
    }],
    "env_params": [{
        "horizon": 100,
        "start_state_fn": None
    }]
}

traj["ep_actions_probs"] = [[tuple(list(Agent.a_probs_from_action(a)) for a in j_a) for j_a in traj["ep_actions"][0]]]

AgentEvaluator.save_traj_as_json(traj, "trajectory_tests/test_full_traj")
