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
        
    def add_object_ids_to_game_states(game_states, id_key="object_id"):
        """
        Receives list of game states as dicts.
        Adds id to every pickable object as long as they are not moving without player action.
        This function is most likely temporary until object_id will be implemented in mdp code.
        """
        id_generator = itertools.count()
        
        def update_object_ids_of_state_objects(state, id_generator):
            for item in state["objects"]:
                if item.get(id_key) is None:
                    if t == 0:
                        item[id_key] = next(id_generator)
                    else:
                        object_id = (find_object_in_game_state(game_states[t-1],
                                                item.get("name"), item.get("position"))
                                                or {}).get(id_key)
                        if object_id is None:
                            object_id = next(id_generator)
                        item[id_key] = object_id
                        
        def find_object_in_game_state(state, name, pos):
            pos = tuple(pos)
            for item in state["objects"]:
                if item["name"] == name and tuple(item["position"]) == pos:
                    return item
        
        def update_object_id_in_state(state, name, pos, obj_id):
            found_obj = find_object_in_game_state(state, name, pos)
            if found_obj:
                found_obj[id_key] = obj_id
                
        def front_of_player_pos(player):
            return [player["position"][i] + player["orientation"][i]
                                            for i in range(len(player["position"]))]
        
        for t, state in enumerate(game_states):
            update_object_ids_of_state_objects(state, id_generator)
            for p, player in enumerate(state["players"]):
                held_object = player["held_object"] or {}
                if (t+1 < len(game_states)):
                    front_pos = front_of_player_pos(player)
                    held_object_next_t = game_states[t+1]["players"][p]["held_object"] or {}
                    if held_object:
                        if held_object.get(id_key) is None:
                            held_object[id_key] = next(id_generator)
                        if held_object.get("name") == held_object_next_t.get("name"):
                            held_object_next_t[id_key] = held_object[id_key]
                        else:
                            # item drop event
                            update_object_id_in_state(game_states[t+1], held_object["name"], front_pos, held_object["object_id"])
                    
                    # item pickup event
                    if held_object_next_t and held_object.get("name") != held_object_next_t.get("name"):
                        held_object_next_t[id_key] = (find_object_in_game_state(state, held_object_next_t["name"], front_pos) or {}).get("object_id")
                        if held_object_next_t[id_key] is None:
                            held_object_next_t[id_key] = next(id_generator)
        return game_states
    
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
    game_states = add_object_ids_to_game_states(game_states)
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


