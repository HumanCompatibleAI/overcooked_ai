"""Rule-based experiment agents built on the MediumLevelActionManager.

OnionLoaderAgent and ServerAgent form a complementary pair: the loader only
stocks pots with onions, the server does everything downstream (start
cooking, fetch dish, collect soup, deliver). Together they implement an
exact division of labor; paired with other agent types they exhibit role
overlap/conflict, which makes them useful as distinct strategic profiles.

Run/visualize them with:  python -m overcooked_ai_py.agents.visualize_agents
"""
import itertools
from collections import defaultdict

import numpy as np

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action


def _first_step_toward_cheapest_goal(am, player, goals):
    """Filter unreachable goals, then take the first action of the cheapest
    motion plan. Returns Action.STAY when no goal is reachable."""
    goals = [
        g
        for g in goals
        if am.motion_planner.is_valid_motion_start_goal_pair(
            player.pos_and_or, g
        )
    ]
    if not goals:
        return Action.STAY
    best_goal = min(
        goals,
        key=lambda g: am.motion_planner.get_plan(player.pos_and_or, g)[2],
    )
    return am.motion_planner.get_plan(player.pos_and_or, best_goal)[0][0]


class OnionLoaderAgent(Agent):
    """Pot-stocking specialist. Rules, in priority order:

    - holding nothing -> pick up an onion (dispenser or counter)
    - holding an onion -> put it in a non-full pot
    - holding anything else -> drop it on a counter

    Memoryless: replans from scratch every tick, no anti-deadlock logic.
    """

    def __init__(self, mlam):
        self.mlam = mlam
        super().__init__()

    def action(self, state):
        player = state.players[self.agent_index]
        am = self.mlam

        if not player.has_object():
            counter_objects = am.mdp.get_counter_objects_dict(state)
            goals = am.pickup_onion_actions(counter_objects)
        elif player.get_object().name == "onion":
            pot_states = am.mdp.get_pot_states(state)
            goals = am.put_onion_in_pot_actions(pot_states)
        else:
            goals = am.place_obj_on_counter_actions(state)

        chosen = _first_step_toward_cheapest_goal(am, player, goals)
        return chosen, {"action_probs": Agent.a_probs_from_action(chosen)}


class ServerAgent(Agent):
    """Serving specialist, the complement of OnionLoaderAgent. Rules:

    - empty-handed:
        * a pot has 3 onions and isn't cooking -> go start it cooking
        * a soup is cooking or ready           -> go fetch a dish
        * otherwise                            -> wait
    - holding a dish -> collect the (nearly-)ready soup
    - holding a soup -> deliver it
    - holding anything else -> drop it on a counter

    Carries one step of history for GreedyHumanModel-style auto-unstuck.
    NOTE: if no soup is in progress while holding a dish it just waits, so
    pairing it with a partner that also fetches dishes can livelock (both
    stuck holding dishes over an empty pot).
    """

    def __init__(self, mlam):
        self.mlam = mlam
        super().__init__()

    def reset(self):
        super().reset()
        self.prev_state = None

    def action(self, state):
        player = state.players[self.agent_index]
        am = self.mlam
        pot_states = am.mdp.get_pot_states(state)

        if not player.has_object():
            full_pots = pot_states["3_items"]
            soup_in_progress = pot_states["ready"] or pot_states["cooking"]
            if full_pots:
                # start_cooking_actions would also start partially-full pots;
                # restrict to full ones so we never cook a sub-3-onion soup
                only_full_pots = defaultdict(list)
                only_full_pots["3_items"] = full_pots
                goals = am.start_cooking_actions(only_full_pots)
            elif soup_in_progress:
                counter_objects = am.mdp.get_counter_objects_dict(state)
                goals = am.pickup_dish_actions(counter_objects)
            else:
                goals = []
        else:
            obj_name = player.get_object().name
            if obj_name == "dish":
                goals = am.pickup_soup_with_dish_actions(
                    pot_states, only_nearly_ready=True
                )
            elif obj_name == "soup":
                goals = am.deliver_soup_actions()
            else:
                goals = am.place_obj_on_counter_actions(state)

        chosen = _first_step_toward_cheapest_goal(am, player, goals)

        # Anti-deadlock (same trick as GreedyHumanModel.auto_unstuck): if no
        # one moved since last tick, pick a random action that would change
        # our position assuming the partner stays put.
        if (
            self.prev_state is not None
            and state.players_pos_and_or == self.prev_state.players_pos_and_or
        ):
            if self.agent_index == 0:
                joint_actions = list(
                    itertools.product(Action.ALL_ACTIONS, [Action.STAY])
                )
            else:
                joint_actions = list(
                    itertools.product([Action.STAY], Action.ALL_ACTIONS)
                )
            unblocking = [
                j_a
                for j_a in joint_actions
                if am.mdp.get_state_transition(state, j_a)[0].player_positions
                != self.prev_state.player_positions
            ]
            if not unblocking:
                unblocking = [(Action.STAY, Action.STAY)]
            chosen = unblocking[np.random.choice(len(unblocking))][
                self.agent_index
            ]

        self.prev_state = state
        return chosen, {"action_probs": Agent.a_probs_from_action(chosen)}
