import numpy as np
import itertools, copy
from collections import defaultdict
from overcooked_ai_py.utils import numpy_to_native
def extract_events(trajectories, traj_idx=0, add_cumulative_events=True):
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


    def get_cumulative_events(events, last_timestep):
        """
        Receives events for scatter plot and returns events for cumulative plot
        """
        def add_cumulative_event(events, events_sum, timestep, action, player_num=None, obj=None):
            event = {"action": action,
                    "sum": events_sum,
                    "timestep": timestep,}
            if player_num is not None: event["player"] = player_num
            if obj is not None: event["object"] = obj
            events.append(event)

        cumulative_events = []

        all_events_sum = 0
        player_sums = defaultdict(int)
        action_name = "pickups_and_drops"
        
        # add zero timestep events
        first_timestep = 0
        add_cumulative_event(cumulative_events, all_events_sum, first_timestep, action_name)
        for player_num in set(map(lambda x: x["player"], events)):
            add_cumulative_event(cumulative_events, all_events_sum, first_timestep, action_name, player_num)

        for event in filter(lambda x:x["action"] in ["pickup", "drop", "delivery", "potting"], events):
            player_num = event["player"]
            timestep = event["timestep"]
            all_events_sum +=1
            player_sums[player_num] +=1
            add_cumulative_event(cumulative_events, all_events_sum, timestep, action_name)
            add_cumulative_event(cumulative_events, player_sums[player_num], timestep, action_name, player_num)

        # add cumulative events at last timestep for nice graph
        add_cumulative_event(cumulative_events, all_events_sum, last_timestep, action_name)
        for player_num in set(map(lambda x: x["player"], events)):
            add_cumulative_event(cumulative_events, player_sums[player_num], last_timestep, action_name, player_num)
        return cumulative_events
        
    ep_states = trajectories["ep_states"][traj_idx]
    events = copy.deepcopy(trajectories["ep_infos"][0][-1]["episode"]["events_list"])
    
    events += get_holding_events(events, ep_states)
    
    if add_cumulative_events:
        events += get_cumulative_events(events, len(ep_states))
    
    # clearing possible numpy data types from data to allow json.dumps of the data
    events = [{k:numpy_to_native(v) for k,v in event.items()} for event in events]
    
    return events
