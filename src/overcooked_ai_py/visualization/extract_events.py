import numpy as np
import itertools, copy
from collections import defaultdict
from overcooked_ai_py.utils import numpy_to_native
def extract_events(trajectories, traj_idx=0, add_cumulative_events=True):
    """
    extracts events from trajectories for plots
    """

    def find_last_event(events, event_type, p, last_timestep=None):
        for event in reversed(events):
            if event.get("player") == p and event.get("action") == event_type \
                    and (last_timestep is None or event["timestep"] < last_timestep):
                return event
            
    def holding_event(events, p, t, obj):
        last_pickup_t = (find_last_event(events, "pickup", p, t) or {}).get("timestep")

        start_t = last_pickup_t or 0
        event = {
            "player": p,
            "start_timestep":start_t,
            "end_timestep":t,
            "action": "holding",
            "object": obj
        }
        return event

    def get_holding_events(events, game_states):
        holding_events = [holding_event(events, e.get("player"), e.get("timestep"), e.get("object")) 
            for e in events if e.get("action") in ["drop", "potting", "delivery"]]
        # check if object is held at end of the episode
        for p, player in enumerate(game_states[-1].players):
            if player.held_object:
                holding_events.append(holding_event(events, p, len(game_states)-1, player.held_object.to_dict()))
        return holding_events


    def get_cumulative_events(events, last_timestep):
        """
        Receives events for scatter plot and returns events for cumulative plot
        """
        def add_cumulative_event(events, s, timestep, action, player=None, obj=None):
            event = {"action": action,
                    "sum": s,
                    "timestep": timestep,}
            if player is not None: event["player"] = player
            if obj is not None: event["object"] = obj
            events.append(event)

        cumulative_events = []

        s = 0
        player_sums = defaultdict(int)
        a = "pickups_and_drops"
        
        # add zero timestep events
        first_timestep = 0
        add_cumulative_event(cumulative_events, s, first_timestep, a)
        for p in set(map(lambda x: x["player"], events)):
            add_cumulative_event(cumulative_events, s, first_timestep, a, p)

        for e in filter(lambda x:x["action"] in ["pickup", "drop"], events):
            p = e["player"]
            t = e["timestep"]
            s+=1
            player_sums[p]+=1
            add_cumulative_event(cumulative_events, s, t, a)
            add_cumulative_event(cumulative_events, player_sums[p], t, a, p)

        # add cumulative events at last timestep for nice graph
        add_cumulative_event(cumulative_events, s, last_timestep, a)
        for p in set(map(lambda x: x["player"], events)):
            add_cumulative_event(cumulative_events, player_sums[p], last_timestep, a, p)
        return cumulative_events
        
    ep_states = trajectories["ep_states"][traj_idx]
    events = copy.deepcopy(trajectories["ep_infos"][0][-1]["episode"]["events_list"])
    
    events += get_holding_events(events, ep_states)
    
    if add_cumulative_events:
        events += get_cumulative_events(events, len(ep_states))
    
    # clearing possible numpy data types from data to allow json.dumps of the data
    events = [{k:numpy_to_native(v) for k,v in event.items()} for event in events]
    
    return events


