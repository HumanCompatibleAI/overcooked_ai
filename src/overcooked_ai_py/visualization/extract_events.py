import numpy as np
import itertools, copy
from collections import defaultdict
from overcooked_ai_py.utils import numpy_to_native
def extract_events(trajectories, traj_idx=0, cumulative_events_description=[]):
    """
    extracts events from trajectories for plots
    """

    def find_last_event(events, event_type, player_num, last_timestep=None):
        for event in reversed(events):
            if event.get("player") == player_num and event.get("action") == event_type \
                    and (last_timestep is None or event["timestep"] < last_timestep):
                return event
            
    def holding_event(events, player_num, timestep, obj):
        last_pickup_t = (find_last_event(events, "pickup", player_num, timestep) or {}).get("timestep")

        start_t = last_pickup_t or 0
        event = {
            "player": player_num,
            "start_timestep":start_t,
            "end_timestep":timestep,
            "action": "holding",
            "object": obj
        }
        return event

    def get_holding_events(events, game_states):
        holding_events = [holding_event(events, event.get("player"), event.get("timestep"), event.get("object")) 
            for event in events if event.get("action") in ["drop", "potting", "delivery"]]
        # check if object is held at end of the episode
        for player_num, player in enumerate(game_states[-1].players):
            if player.held_object:
                holding_events.append(holding_event(events, player_num, len(game_states)-1, player.held_object.to_dict()))
        return holding_events


    def get_cumulative_events(events, last_timestep, cumulative_events_desription):
        """
        Receives events for scatter plot and returns events for cumulative plot
        """
        def add_cumulative_event(events, events_sum, timestep, actions=None, adjectives=None, player_num=None, obj=None):
            event = {"sum": events_sum,
                    "timestep": timestep}

            if actions is not None: event["actions"] = actions
            if adjectives is not None: event["adjectives"] = adjectives
            if player_num is not None: event["player"] = player_num
            if obj is not None: event["object"] = obj
            events.append(event)
    
        def is_matching_adjectives(event, adjectives):
            event_adjectives = event.get("adjectives", [])
            if not adjectives: # assuming that there was no supplied adjectives because all are allowed
                return True
            no_adjectives_allowed = None in adjectives or "" in adjectives
            if no_adjectives_allowed and not event_adjectives:
                return True
            return bool(set(event_adjectives).intersection(adjectives))

        def is_matching_actions(event, actions):
            if not actions: # assuming that there was no supplied actions because all are allowed
                return True
            return event.get("action") in actions
    
        cumulative_events = []

        for description in cumulative_events_desription:
            actions = description.get("actions")
            adjectives = description.get("adjectives")
            all_events_sum = 0
            player_sums = defaultdict(int)
            # add zero timestep events
            first_timestep = 0
            add_cumulative_event(cumulative_events, all_events_sum, first_timestep, actions, adjectives)
            for player_num in set(map(lambda e: e["player"], events)):
                add_cumulative_event(cumulative_events, all_events_sum, first_timestep, actions, adjectives, player_num)

            for event in filter(lambda e: is_matching_actions(e, actions) and is_matching_adjectives(e, adjectives), events):
                player_num = event["player"]
                timestep = event["timestep"]
                all_events_sum +=1
                player_sums[player_num] +=1
                add_cumulative_event(cumulative_events, all_events_sum, timestep, actions, adjectives)
                add_cumulative_event(cumulative_events, player_sums[player_num], timestep, actions, adjectives, player_num)

            # add cumulative events at last timestep for graph ending with the last timestep
            add_cumulative_event(cumulative_events, all_events_sum, last_timestep, actions, adjectives)
            for player_num in set(map(lambda x: x["player"], events)):
                add_cumulative_event(cumulative_events, player_sums[player_num], last_timestep, actions, adjectives, player_num)
        return cumulative_events
        
    ep_states = trajectories["ep_states"][traj_idx]
    events = copy.deepcopy(trajectories["ep_infos"][0][-1]["episode"]["events_list"])
    
    events += get_holding_events(events, ep_states)

    
    if cumulative_events_description:
        events += get_cumulative_events(events, len(ep_states), cumulative_events_description)

    # clearing possible numpy data types from data to allow json.dumps of the data
    events = [{k:numpy_to_native(v) for k,v in event.items()} for event in events]
    
    return events
