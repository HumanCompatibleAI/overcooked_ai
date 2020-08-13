import numpy as np
import itertools
from collections import defaultdict

def extract_events(trajectories, traj_idx=0, add_cumulative_events=True):
    """
    extracts events from trajectories for plots
    """
    terrain = np.array(trajectories["mdp_params"][traj_idx]["terrain"]).T    
    
    def find_last_pickup(events, p):
        for event in reversed(events):
            if event["player"] == p and event["action"] == "pickup":
                return event
            
    def front_of_player_pos(player):
        return [player["position"][i] + player["orientation"][i]
                                         for i in range(len(player["position"]))]

    def add_holding_event(events, p, t, held_object):
        last_pickup_t = (find_last_pickup(events, p) or {}).get("timestep")
        if last_pickup_t is None:
            start_t = 0
        else:
            start_t = last_pickup_t
        event = {
            "player": p,
            "start_timestep":start_t,
            "end_timestep":t,
            "action": "holding",
            "item_name": held_object["name"],
            "item_id": held_object["object_id"]
        }
        events.append(event)

    def add_drop_item_event(events, player, p, t, dropped_obj, terrain):
        player_pos = list(player["position"])
        front_pos = front_of_player_pos(player)
        front_terrain = terrain[tuple(front_pos)]
        event = {
            "player": p,
            "timestep":t,
            "action": "drop",
            "item_name": dropped_obj["name"],
            "front_terrain": front_terrain,
            "front_pos": front_pos,
            "player_pos": player_pos,
            "item_id": dropped_obj["object_id"]
            }
        events.append(event)
    
    def add_pickup_item_event(events, player, p, t, picked_obj, terrain):
        player_pos = list(player["position"])
        front_pos = front_of_player_pos(player)
        front_terrain = terrain[tuple(front_pos)]
        event = {
            "player": p,
            "timestep":t,
            "action": "pickup",
            "item_name": picked_obj["name"],
            "front_terrain": front_terrain,
            "front_terrain_pos": front_pos,
            "player_pos": player_pos,
            "item_id": picked_obj["object_id"]

        }
        events.append(event)
    
    def get_cumulative_events(events, last_timestep):
        """
        Receives events for scatter plot and returns events for cumulative plot
        """
        def add_cumulative_event(events, s, timestep, action, player=None, item=None):
            event = {"action": action,
                    "sum": s,
                    "timestep": timestep}
            if player is not None: event["player"] = player
            if item is not None: event["item"] = item
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

    game_states = [state.to_dict() for state in trajectories["ep_states"][traj_idx]]
    events = []
    
    for t, state in enumerate(game_states):
        for p, player in enumerate(state["players"]):
            held_object = player["held_object"] or {}
            if (t+1 < len(game_states)):
                held_object_next_t = game_states[t+1]["players"][p]["held_object"] or {}
                if held_object:
                    if held_object.get("name") != held_object_next_t.get("name"):
                        add_drop_item_event(events, player, p, t, held_object, terrain)
                        add_holding_event(events, p, t, held_object)
                
                if held_object_next_t and held_object.get("name") != held_object_next_t.get("name"):
                    add_pickup_item_event(events, player, p, t, held_object_next_t, terrain)
                    
            else: #last timestep
                if held_object:
                    add_holding_event(events, p, t, held_object)
    
    if add_cumulative_events:
        events += get_cumulative_events(events, len(game_states))
    
    # clearing possible numpy data types from data to allow json.dumps of the data
    def numpy_to_native(x):
        if isinstance(x, np.generic):
            return x.item()
        else:
            return x
    events = [{k:numpy_to_native(v) for k,v in event.items()} for event in events]
    
    return events


