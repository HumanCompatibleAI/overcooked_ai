import random
import copy
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner
INTERACT_TRANSITION_COST = 2

INFINITY = np.inf

# the location for secondary agent as we are moving the primary agent
UNDEFIND_LOCATION = "UND_L"

# the undefined action
UNDEFIND_ACTION = "UND_A"

ENTROPY_RHO = 5


import heapq


def midpoint(point_0, point_1, terrain_matrix):
    if point_0[0] == point_1[0] or point_0[1] == point_1[1]:
        return tuple([(point_0[0] + point_1[0])//2, (point_0[1] + point_1[1])//2])
    else:
        # handling the diagonal handover
        if terrain_matrix[point_0[0]][point_1[1]] == "X":
            return tuple([point_0[0], point_1[1]])
        elif terrain_matrix[point_1[0]][point_0[1]] == "X":
            return tuple([point_1[0], point_0[1]])
        else:
            raise NotImplementedError("neither of the diagonal entries has a counter")

def add_action_from_location(curr_loc, next_loc, actions):
    """
    Arguments:
        curr_loc (tuple): a tuple for the current location of the agent (x, y)
        next_loc (tuple): a tuple for the next location of the agent (x, y)
            UNDEFIND_LOCATION if the agent is currently at the end of its path
    This function modifies the action list to include the relevant action given the two locations
    """

    # case where both curr_loc and next_loc are valid locations
    if curr_loc != UNDEFIND_LOCATION and next_loc != UNDEFIND_LOCATION:
        # if previous action was undefined, then we are picking up from a counter
        # replace the undefined action with the interact action
        if len(actions) > 0 and actions[len(actions) - 1] == UNDEFIND_ACTION:
            actions[len(actions) - 1] = 'interact'
        else:
            actions.append((next_loc[1] - curr_loc[1], next_loc[0] - curr_loc[0]))
    # case where agent is at the end of its path
    elif curr_loc != UNDEFIND_LOCATION:
        actions.append('interact')
    # case where current location is undefined (location before picking up from counter)
    elif next_loc != UNDEFIND_LOCATION:
        actions.append(UNDEFIND_ACTION)
    # both current and next locations are undefined
    else:
        return

def remove_extra_action(actions):
    """
    Arguments:
        actions (list): a list of tuples containing valid movement actions including interact
    return:
        updated list of actions with extra actions removed
    """
    # case to check if should remove second to last because no turn in place exists
    if len(actions) > 2:
        last_movement_action = actions[len(actions) - 2]
        second_last_movement_action = actions[len(actions) - 3]
        if last_movement_action == second_last_movement_action:
            return actions[:len(actions) - 2] + actions[len(actions) - 1:]
    return actions

def path_to_actions(path_0, path_1, terrain_mtx):
    """
    Arguments:
        path_0 (list): a list of tuble for locations of agent 0
        path_1 (list): a list of tuble for locations of agent 1
        terrain_mtx (list of list): the terrain matrix. You might need this to identify counters / pots, etc.
    return:
        the list of actions that will be used to fulfill the path
    """
    assert len(path_0) == len(path_1), "input path should have the same length. Please check the SearchNode for this"
    actions_0 = []
    actions_1 = []

    # fill in both action lists with actions of movement and interacting
    for index in range(len(path_0)):

        # Set current and next locations to determine action
        curr_loc_0 = path_0[index]
        curr_loc_1 = path_1[index]
        next_loc_0 = UNDEFIND_LOCATION
        next_loc_1 = UNDEFIND_LOCATION

        if index != (len(path_0) - 1):
            next_loc_0 = path_0[index + 1]
            next_loc_1 = path_1[index + 1]

        add_action_from_location(curr_loc_0, next_loc_0, actions_0)
        add_action_from_location(curr_loc_1, next_loc_1, actions_1)

    # remove unnecessary action at second to last spot if no in place turning required
    actions_0 = remove_extra_action(actions_0)
    actions_1 = remove_extra_action(actions_1)

    # padding the two agent's action lists

    # case where agent 0 did not do anything
    if len(actions_0) == 0:
        actions_0 = [UNDEFIND_ACTION] * len(actions_1)
    # case where agent 1 did not do anything
    elif len(actions_1) == 0:
        actions_1 = [UNDEFIND_ACTION] * len(actions_0)
    # case where counter was used by both agents
    else:
        # case where agent 0 is picking up from the counter
        if actions_0[0] == 'interact':
            temp_len = len(actions_0)
            actions_0 = [UNDEFIND_ACTION] * len(actions_1) + actions_0
            actions_1 = actions_1 + [UNDEFIND_ACTION] * temp_len
        # case where agent 1 is picking up from the counter
        elif actions_1[0] == 'interact':
            temp_len = len(actions_1)
            actions_1 = [UNDEFIND_ACTION] * len(actions_0) + actions_1
            actions_0 = actions_0 + [UNDEFIND_ACTION] * temp_len

    assert len(actions_0) == len(actions_1), " resulting actions should have the same length. Please pad if otherwise"
    return actions_0, actions_1

def path_to_actions_with_padding(path_0, path_1, terrain_mtx, prev_loc_0, prev_loc_1):
    """
    Arguments:
        path_0 (list): a list of tuble for locations of agent 0
        path_1 (list): a list of tuble for locations of agent 1
        terrain_mtx (list of list): the terrain matrix. You might need this to identify counters / pots, etc.
    return:
        the list of actions that will be used to fulfill the path
    """
    assert len(path_0) == len(path_1), "input path should have the same length. Please check the SearchNode for this"
    actions_0 = []
    actions_1 = []

    counter_pickup_position_0 = None
    counter_pickup_0 = False

    counter_pickup_position_1 = None
    counter_pickup_1 = False

    # fill in both action lists with actions of movement and interacting
    for index in range(len(path_0)):

        # Set current and next locations to determine action
        curr_loc_0 = path_0[index]
        curr_loc_1 = path_1[index]
        next_loc_0 = UNDEFIND_LOCATION
        next_loc_1 = UNDEFIND_LOCATION

        if index != (len(path_0) - 1):
            next_loc_0 = path_0[index + 1]
            next_loc_1 = path_1[index + 1]

        add_action_from_location(curr_loc_0, next_loc_0, actions_0)
        add_action_from_location(curr_loc_1, next_loc_1, actions_1)

        # logic to keep track of the counter pickup location
        if counter_pickup_0:
            orientation = (curr_loc_0[0]-next_loc_0[0], curr_loc_0[1]-next_loc_0[1])
            counter_pickup_position_0 = curr_loc_0
        if counter_pickup_1:
            orientation = (curr_loc_1[0]-next_loc_1[0], curr_loc_1[1]-next_loc_1[1])
            counter_pickup_position_1 = curr_loc_1
        if len(actions_0) > 0:
            counter_pickup_0 = actions_0[len(actions_0)-1] == UNDEFIND_ACTION
        if len(actions_1) > 0:
            counter_pickup_1 = actions_1[len(actions_1)-1] == UNDEFIND_ACTION


    # remove unnecessary action at second to last spot if no in place turning required
    actions_0 = remove_extra_action(actions_0)
    actions_1 = remove_extra_action(actions_1)

    # padding the two agent's action lists

    # case where agent 0 did not do anything
    if len(actions_0) == 0:
        actions_0 = [(0, 0)] * len(actions_1)
    # case where agent 1 did not do anything
    elif len(actions_1) == 0:
        actions_1 = [(0, 0)] * len(actions_0)
    # case where counter was used by both agents
    else:
        walk_graph = walk_graph_from_terrain(terrain_mtx)
        # case where agent 0 is picking up from the counter
        if actions_0[0] == 'interact':
            temp_len = len(actions_0)

            # calculate the action path to get the agent from where it ended previously to where it needs to pickup
            action_pad = shortest_walk_path(walk_graph, prev_loc_0, counter_pickup_position_0, terrain_mtx)
            pad_len = len(action_pad)

            # remove last repetitive action if turning is not necessary
            if pad_len > 1 and action_pad[pad_len-1] == action_pad[pad_len-2]:
                action_pad = action_pad[:pad_len-1]

            if len(action_pad) > len(actions_1):
                actions_0 = action_pad + actions_0
                actions_1 = actions_1 + [(0, 0)] * (temp_len + len(action_pad) - len(actions_1))
            else:
                actions_0 = [(0, 0)] * (len(actions_1) - len(action_pad)) + action_pad + actions_0
                actions_1 = actions_1 + [(0, 0)] * temp_len
        # case where agent 1 is picking up from the counter
        elif actions_1[0] == 'interact':
            temp_len = len(actions_1)

            # calculate the action path to get the agent from where it ended previously to where it needs to pickup
            action_pad = shortest_walk_path(walk_graph, prev_loc_1, counter_pickup_position_1, terrain_mtx)
            pad_len = len(action_pad)

            # remove last repetitive action if turning is not necessary
            if pad_len > 1 and action_pad[pad_len - 1] == action_pad[pad_len - 2]:
                action_pad = action_pad[:pad_len - 1]

            if len(action_pad) > len(actions_0):
                actions_1 = action_pad + actions_1
                actions_0 = actions_0 + [(0, 0)] * (temp_len + len(action_pad) - len(actions_0))
            else:
                actions_1 = [(0, 0)] * (len(actions_0) - len(action_pad)) + action_pad + actions_1
                actions_0 = actions_0 + [(0, 0)] * temp_len

    assert len(actions_0) == len(actions_1), " resulting actions should have the same length. Please pad if otherwise"
    return actions_0, actions_1

def connect_action_path(action_paths):
    """
        Arguments:
            action_paths (list of lists): a list of action paths to connect together
        return:
            connected action paths
            currently just concatenates paths regardless of undefined actions
        TODO: fill in undefined actions
    """
    total_path = []
    for path in action_paths:
        total_path += path
    return total_path

def calculate_entropy_of_path(path, rho):
    """
        Arguments:
            path (list): a list of tuble for locations of agent 0
            rho: parameter used in calculation of entropy
        return:
            the calculated entropy of the path
            entropy of a sequence of one action 'a' repeated 'n' times is -ln(n) + ln(rho)
            total entropy is the sum of all the sequences in the path
    """
    total_entropy = 0
    const_entropy = np.log(rho)
    curr_length = 1
    curr_action = path[0]

    # calculate and add entropy for each sequence of actions

    for action in path[1:]:
        if action == curr_action:
            curr_length += 1
        else:
            # if the sequence of actions that is ending is undefined, then we don't add the entropy
            if curr_action == UNDEFIND_ACTION or curr_action == (0, 0):
                curr_length = 1
                curr_action = action
                continue
            entropy = -np.log(curr_length) + const_entropy
            total_entropy += entropy
            curr_length = 1
            curr_action = action

    # add entropy for last sequence of actions if the last sequence is undefined
    if curr_action != UNDEFIND_ACTION and curr_action != (0, 0):
        entropy = -np.log(curr_length) + const_entropy
        total_entropy += entropy

    return total_entropy




class OvercookedSearchNode:
    def __init__(self, primary_idx, agent_0_loc, agent_1_loc, agent_0_path, agent_1_path, num_counter_ops):
        """
        Search node used in UCS
        Arguments:
            primary_idx (int): the index of the agent that is moving in the search process
            agent_0_loc (tuple of length 2): the location of agent 0
            agent_1_loc (tuple of length 2): the location of agent 1
            agent_0_path (list of variable length): the cummulative list of locations traversed by agent 0
            agent_1_path (list of variable length): the cummulative list of locations traversed by agent 1
        """
        self.primary_idx = primary_idx
        self.agent_0_loc = agent_0_loc
        self.agent_1_loc = agent_1_loc
        self.agent_0_path = agent_0_path.copy()
        self.agent_1_path = agent_1_path.copy()
        self.num_counter_ops = num_counter_ops

    def primary_agent_loc(self):
        if self.primary_idx == 0:
            return self.agent_0_loc
        else:
            return self.agent_1_loc

    def secondary_agent_loc(self):
        if self.primary_idx == 0:
            return self.agent_1_loc
        else:
            return self.agent_0_loc

    def primary_path(self):
        if self.primary_idx == 0:
            return self.agent_0_path
        else:
            return self.agent_1_path

    def secondary_path(self):
        if self.primary_idx == 0:
            return self.agent_1_path
        else:
            return self.agent_0_path

    def correct_primary_agent_loc(self):
        # correct that primary agent location is located at the feature instead of in an empty squre
        if self.primary_idx == 0:
            self.agent_0_loc = self.primary_path()[-2]
        else:
            self.agent_1_loc = self.primary_path()[-2]

    def update_primary_agent_loc(self, new_primary_agent_loc, new_secondary_agent_loc=UNDEFIND_LOCATION, secondary_path_UDF=False):
        """
        This function is mostly used when generating successors of a search node
        Arguments:
            new_primary_agent_loc (tuple of length 2): the new location of the primary agent
            new_secondary_agent_loc (tuple of length 2): the new location of th secondary agent
        """
        if self.primary_idx == 0:
            self.agent_0_loc = new_primary_agent_loc
            if new_secondary_agent_loc != UNDEFIND_LOCATION:
                self.agent_1_loc = new_secondary_agent_loc
            self.agent_0_path += [new_primary_agent_loc]
            # if we want to override the path update to UDF, we can still do this
            if secondary_path_UDF:
                self.agent_1_path += [UNDEFIND_LOCATION]
            else:
                self.agent_1_path += [new_secondary_agent_loc]
        else:
            self.agent_1_loc = new_primary_agent_loc
            if new_secondary_agent_loc != UNDEFIND_LOCATION:
                self.agent_0_loc = new_secondary_agent_loc
            self.agent_1_path += [new_primary_agent_loc]
            # if we want to override the path update to UDF, we can still do this
            if secondary_path_UDF:
                self.agent_0_path += [UNDEFIND_LOCATION]
            else:
                self.agent_0_path += [new_secondary_agent_loc]

    def to_tuple(self):
        # convert the entire node nodes to a hashable tuple
        return tuple([self.primary_idx, self.agent_0_loc, self.agent_1_loc, self.agent_0_path, self.agent_1_path, self.num_counter_ops])

    def hash_key(self, at_goal=False):
        # produce a hash key to be used when returned by the UCS
        # if the primary_agent_loc is at goal, use the location before that for physical location
        if at_goal:
            self.correct_primary_agent_loc()
        return tuple([self.primary_idx, self.primary_agent_loc(), self.secondary_agent_loc()])

    def path_length(self):
        return len(self.primary_path())

    def copy(self):
        return OvercookedSearchNode(
            self.primary_idx, self.agent_0_loc, self.agent_1_loc, self.agent_0_path,
            self.agent_1_path, self.num_counter_ops
        )

    def successor(self, primary_agent_new_loc, counter_opposite_loc=None, terrain_mtx=None):
        successor_node = self.copy()
        if counter_opposite_loc:
            counter_loc = midpoint(primary_agent_new_loc, counter_opposite_loc, terrain_mtx)
            # interact to get the iter onto the counter
            successor_node.update_primary_agent_loc(counter_loc, counter_loc)
            # change control
            successor_node.primary_idx = 1 - self.primary_idx
            # interact to get the item from the counter
            successor_node.update_primary_agent_loc(primary_agent_new_loc, counter_opposite_loc, True)
            # increment the counter_ops (1 drop off, 1 pickup)
            successor_node.num_counter_ops += 2
            return successor_node

        else:
            # only change the location of primary agent
            successor_node.update_primary_agent_loc(primary_agent_new_loc)

        return successor_node

    def __str__(self):
        # string representation of search node in UCS
        output = ""
        output += "primary agent: " + str(self.primary_idx) + "\n"
        output += "locations: " + str(self.agent_0_loc) + " " + str(self.agent_1_loc) + "\n"
        output += "path 0: " + str(self.agent_0_path) + "\n"
        output += "path 1: " + str(self.agent_1_path) + "\n"
        output += "num counter operations: " + str(self.num_counter_ops) + "\n"
        return output


class OvercookedMLASearchNode:
    def __init__(self, primary_idx, agent_0_loc, agent_1_loc, pot_loc, agent_0_act_dict, agent_1_act_dict,
                 num_counter_ops, terrain_mtx):
        """
        Search Node used for a composition of medium level tasks (picking up an onion, dishing a soup, etc)
        Arguments:
            primary_idx (int): the index of the agent that is moving in the search process
            agent_0_loc (tuple of length 2): the location of agent 0
            agent_1_loc (tuple of length 2): the location of agent 1
            agent_0_act_dict (dictionary of list of variable length):
                keys: medium levthe cummulative list of locations traversed by agent 0
            agent_1_act_dict (dictionary of list of variable length): the cummulative list of locations traversed by agent 1

        """
        self.primary_idx = primary_idx
        self.agent_0_loc = agent_0_loc
        self.agent_1_loc = agent_1_loc
        self.pot_loc = pot_loc
        self.agent_0_act_dict = agent_0_act_dict.copy()
        self.agent_1_act_dict = agent_1_act_dict.copy()
        self.num_counter_ops = num_counter_ops
        self.terrain_mtx = terrain_mtx

    def append_path_for_task(self, task_name, agent_0_path_task, agent_1_path_task):
        act_paths = path_to_actions_with_padding(agent_0_path_task, agent_1_path_task, self.terrain_mtx,
                                                 self.agent_0_loc, self.agent_1_loc)
        self.agent_0_act_dict[task_name] = act_paths[0]
        self.agent_1_act_dict[task_name] = act_paths[1]

    def total_path_length(self):
        # the total path length of this
        return sum([len(self.agent_0_act_dict[task]) for task in self.agent_0_act_dict.keys()])


    def update_pot_loc(self, pot_loc):
        # update the target pot when starting to make a plan to cook soups
        if self.pot_loc and self.pot_loc != pot_loc:
            raise NotImplementedError("cannot switch pot halfway through")
        self.pot_loc = pot_loc

    def update_from_search_node(self, task_name, search_node: OvercookedSearchNode, pot_loc=None):
        """
        This function is mostly used when generating successors of a mla search node.
        It will parse information from the lower level search node returned by UCS
        Arguments:
            task_name (str): the name of the medium level task carried out by the search_node
            search_node (OvercookedSearchNode): the resulting search node returned by UCS
        """
        updated_mla_search_node = self.copy()
        updated_mla_search_node.primary_idx = search_node.primary_idx
        updated_mla_search_node.append_path_for_task(task_name, search_node.agent_0_path, search_node.agent_1_path)
        updated_mla_search_node.agent_0_loc = search_node.agent_0_loc
        updated_mla_search_node.agent_1_loc = search_node.agent_1_loc
        updated_mla_search_node.num_counter_ops += search_node.num_counter_ops
        if pot_loc:
            updated_mla_search_node.update_pot_loc(pot_loc)
        return updated_mla_search_node

    def hash_key(self):
        # the hash code for the mla search node
        return tuple([self.agent_0_loc, self.agent_1_loc, self.pot_loc, self.primary_idx])

    def agent_positions(self):
        return tuple([self.agent_0_loc, self.agent_1_loc])

    def copy(self):
        return OvercookedMLASearchNode(
            self.primary_idx,
            self.agent_0_loc,
            self.agent_1_loc,
            self.pot_loc,
            self.agent_0_act_dict,
            self.agent_1_act_dict,
            self.num_counter_ops,
            self.terrain_mtx
        )

    def __str__(self):
        output = ""
        output += "primary agent: " + str(self.primary_idx) + "\n"
        output += "locations: " + str(self.agent_0_loc) + " " + str(self.agent_1_loc) + "\n"
        output += "pot: " + str(self.pot_loc) + "\n"
        output += "action dict 0: " + str(self.agent_0_act_dict) + "\n"
        output += "action dict 1: " + str(self.agent_1_act_dict) + "\n"
        output += "num counter operations: " + str(self.num_counter_ops) + "\n"
        return output

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


def shortest_walk_dist(walk_graph, start_loc, goal_loc, terrain_mtx, debug=False):
    walk_graph_copy = copy.deepcopy(walk_graph)

    l, w = len(terrain_mtx), len(terrain_mtx[0])

    # add incoming edge for goal_loc
    goal_loc_i, goal_loc_j = goal_loc
    if goal_loc_i > 0 and terrain_mtx[goal_loc_i - 1][goal_loc_j] == ' ':
        walk_graph_copy[(goal_loc_i - 1, goal_loc_j)].append(goal_loc)
    if goal_loc_i < l - 1 and terrain_mtx[goal_loc_i + 1][goal_loc_j] == ' ':
        walk_graph_copy[(goal_loc_i + 1, goal_loc_j)].append(goal_loc)
    if goal_loc_j > 0 and terrain_mtx[goal_loc_i][goal_loc_j - 1] == ' ':
        walk_graph_copy[(goal_loc_i, goal_loc_j - 1)].append(goal_loc)
    if goal_loc_j < w - 1 and terrain_mtx[goal_loc_i][goal_loc_j + 1] == ' ':
        walk_graph_copy[(goal_loc_i, goal_loc_j + 1)].append(goal_loc)

    closed = set([])
    fringe = PriorityQueue()
    # Initialize empty start node.
    # All search nodes have the form (LOCATION, TOTAL_WALK_COST)
    startInfo = (start_loc, 0)
    fringe.push(startInfo, 0)
    # Continue till fringe empty
    while (not fringe.isEmpty()):
        (loc, total_walk_cost_so_far) = fringe.pop()
        # Goal state check
        if loc == goal_loc:
            return total_walk_cost_so_far
        # If necessary, add successors
        if (loc not in closed):
            closed.add(loc)
            for sucLoc in walk_graph_copy[loc]:
                fringe.push((sucLoc, total_walk_cost_so_far + 1), total_walk_cost_so_far + 1)
    if debug:
        print("!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!")
        print("cannot find a walk path. Please use uniform_cost_search to utilize handover_graph")
        print("!!!!!!!!!!!!!END WARNING!!!!!!!!!!!!!!!!!!")
    return INFINITY

def shortest_walk_path(walk_graph, start_loc, goal_loc, terrain_mtx, debug=False):
    walk_graph_copy = copy.deepcopy(walk_graph)

    l, w = len(terrain_mtx), len(terrain_mtx[0])

    # add incoming edge for goal_loc
    goal_loc_i, goal_loc_j = goal_loc
    if goal_loc_i > 0 and terrain_mtx[goal_loc_i - 1][goal_loc_j] == ' ':
        walk_graph_copy[(goal_loc_i - 1, goal_loc_j)].append(goal_loc)
    if goal_loc_i < l - 1 and terrain_mtx[goal_loc_i + 1][goal_loc_j] == ' ':
        walk_graph_copy[(goal_loc_i + 1, goal_loc_j)].append(goal_loc)
    if goal_loc_j > 0 and terrain_mtx[goal_loc_i][goal_loc_j - 1] == ' ':
        walk_graph_copy[(goal_loc_i, goal_loc_j - 1)].append(goal_loc)
    if goal_loc_j < w - 1 and terrain_mtx[goal_loc_i][goal_loc_j + 1] == ' ':
        walk_graph_copy[(goal_loc_i, goal_loc_j + 1)].append(goal_loc)

    closed = set([])
    fringe = PriorityQueue()
    # Initialize empty start node.
    # All search nodes have the form (LOCATION, TOTAL_WALK_COST, CURRENT_ACTION_PATH)
    startInfo = (start_loc, 0, [])
    fringe.push(startInfo, 0)
    # Continue till fringe empty
    while (not fringe.isEmpty()):
        (loc, total_walk_cost_so_far, action_path) = fringe.pop()
        # Goal state check
        if loc == goal_loc:
            return action_path
        # If necessary, add successors
        if (loc not in closed):
            closed.add(loc)
            for sucLoc in walk_graph_copy[loc]:
                fringe.push((sucLoc, total_walk_cost_so_far + 1,
                             action_path + [(sucLoc[1]-loc[1], sucLoc[0]-loc[0])]),
                            total_walk_cost_so_far + 1)
    if debug:
        print("!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!")
        print("cannot find a walk path. Please use uniform_cost_search to utilize handover_graph")
        print("!!!!!!!!!!!!!END WARNING!!!!!!!!!!!!!!!!!!")
    return INFINITY



def uniform_cost_search(walk_graph, handover_graph, terrain_mtx, agent_locations, start_agent_idx, goal_loc):
    """
    Arguments:
        walk_graph: the dictionary representation of the walking graph (reachable by walking)
        handover_graph: the dictionary representation of the handover graph (reachable by placing on counter)
        terrain_mtx: matrix representation of the grid world. Does not include information about agents
        agent_locations: tuple of tuple, locations of agents
        start_agent_idx: the agent starting to act first
        goal_loc: the location we would like to reach
    :return: dictionary of
        {number of counter_operations: [length of the path, the ending physical location, the path itself]}
        note: the number of counter_operations should always be even because it takes 1 to drop something down,
        and 1 to pick something up

    """
    walk_graph_copy = copy.deepcopy(walk_graph)
    handover_graph_copy = copy.deepcopy(handover_graph)

    l, w = len(terrain_mtx), len(terrain_mtx[0])

    for agent_loc in agent_locations:
        agent_loc_i, agent_loc_j = agent_loc
        assert terrain_mtx[agent_loc_i][agent_loc_j] == ' ', "starting location %s is not an empty square" % str(agent_loc)

    # add incoming edge for goal_loc
    goal_loc_i, goal_loc_j = goal_loc
    if goal_loc_i > 0 and terrain_mtx[goal_loc_i - 1][goal_loc_j] == ' ':
        walk_graph_copy[(goal_loc_i - 1, goal_loc_j)].append(goal_loc)
    if goal_loc_i < l - 1 and terrain_mtx[goal_loc_i + 1][goal_loc_j] == ' ':
        walk_graph_copy[(goal_loc_i + 1, goal_loc_j)].append(goal_loc)
    if goal_loc_j > 0 and terrain_mtx[goal_loc_i][goal_loc_j - 1] == ' ':
        walk_graph_copy[(goal_loc_i, goal_loc_j - 1)].append(goal_loc)
    if goal_loc_j < w - 1 and terrain_mtx[goal_loc_i][goal_loc_j + 1] == ' ':
        walk_graph_copy[(goal_loc_i, goal_loc_j + 1)].append(goal_loc)

    res = {}
    # now the closed set keep track of how many counter_ops we need.
    # because counter operation should only be giving shortcupts if we have visited a state with 0 counter
    closed = {0: set([])}
    def truly_closed(loc, counter_op_so_far):
        for k in closed:
            if k <= counter_op_so_far and loc in closed[k]:
                return True
        return False
    fringe = PriorityQueue()

    # we would like to start the secondary agent's path on UND
    if start_agent_idx == 0:
        path_0 = [agent_locations[0]]
        path_1 = [UNDEFIND_LOCATION]
    else:
        path_0 = [UNDEFIND_LOCATION]
        path_1 = [agent_locations[1]]

    # All search nodes have the form (LOCATION, OTHER_AGENT_LOCATION, PATH, TOTAL_WALK_COST, COUNTER_OPERATION)
    startNode = OvercookedSearchNode(
        start_agent_idx,
        agent_locations[0],
        agent_locations[1],
        path_0,
        path_1,
        0
    )
    fringe.push(startNode, 0)
    # Continue till fringe empty
    while (not fringe.isEmpty()):
        curNode = fringe.pop()
        loc = curNode.primary_agent_loc()
        other_agent_loc = curNode.secondary_agent_loc()
        path_so_far = curNode.primary_path()
        counter_op_so_far = curNode.num_counter_ops
        # Goal state check
        if loc == goal_loc:
            ending_hash = curNode.hash_key(True)

            # create the empty list if key does not exist
            if ending_hash not in res.keys():
                res[ending_hash] = []
            res[ending_hash].append(curNode)
            continue
        # If necessary, add successors
        if counter_op_so_far not in closed.keys():
            closed[counter_op_so_far] = set([])
        if not truly_closed(loc, counter_op_so_far):
            closed[counter_op_so_far].add(loc)
            # first type of successor: same agent moving
            for suc_loc in walk_graph_copy[loc]:
                # keep original agent acting
                newNode = curNode.successor(suc_loc)
                fringe.push(newNode, newNode.path_length())
            # second type of succesor: handover
            if loc in handover_graph_copy:
                for suc_loc in handover_graph_copy[loc]:
                    # first we need to check if the other agent can actually make the walk:
                    other_agent_walk_dist = shortest_walk_dist(walk_graph, other_agent_loc, suc_loc, terrain_mtx)
                    if other_agent_walk_dist != INFINITY:
                        # give control over to the other agent, and become the other agent by "taking its place on the map"\
                        newNode = curNode.successor(suc_loc, loc, terrain_mtx)
                        fringe.push(newNode, newNode.path_length())
    return res


def empty_space(terrain_mtx, loc):
    i, j = loc
    l, w = len(terrain_mtx), len(terrain_mtx[0])
    empty_spaces = []
    if i > 0 and terrain_mtx[i-1][j] == ' ':
        empty_spaces.append((i-1, j))
    if i < l-1 and terrain_mtx[i+1][j] == ' ':
        empty_spaces.append((i+1, j))
    if j > 0 and terrain_mtx[i][j-1] == ' ':
        empty_spaces.append((i, j-1))
    if j < w-1 and terrain_mtx[i][j+1] == ' ':
        empty_spaces.append((i, j+1))
    return empty_spaces


def walk_graph_from_terrain(terrain_mtx):
    l, w = len(terrain_mtx), len(terrain_mtx[0])
    walk_graph = {}
    for i in range(l):
        for j in range(w):
            cur_feature = terrain_mtx[i][j]
            # EMPTY's successors can be EMPTY or COUNTER (for the hop)
            if cur_feature == ' ' or cur_feature == 'O' or cur_feature == 'T' or cur_feature == 'P' or cur_feature == 'D':
                walk_graph[(i, j)] = []
                if i - 1 >= 0 and terrain_mtx[i - 1][j] == ' ':
                    walk_graph[(i, j)].append((i - 1, j))
                if i + 1 <= l - 1 and terrain_mtx[i + 1][j] == ' ':
                    walk_graph[(i, j)].append((i + 1, j))
                if j - 1 >= 0 and terrain_mtx[i][j - 1] == ' ':
                    walk_graph[(i, j)].append((i, j - 1))
                if j + 1 <= w - 1 and terrain_mtx[i][j + 1] == ' ':
                    walk_graph[(i, j)].append((i, j + 1))

    return walk_graph


def graph_from_terrain(terrain_mtx):
    """
    Arguments:
        terrain_mtx: the terrain matrix
    Return:
         a walk_graph, in format of
        {current_loc : [[next_loc_0, 1, counter_op_0], [next_loc_1, 1, counter_op_1]...], ...}
        handover_graph, in format of
        {current_loc: [next_loc_0, next_loc_1, ...], ...}
    """
    l, w = len(terrain_mtx), len(terrain_mtx[0])
    walk_graph = {}
    handover_graph = {}
    for i in range(l):
        for j in range(w):
            cur_feature = terrain_mtx[i][j]
            # EMPTY's successors can be EMPTY or COUNTER (for the hop)
            if cur_feature == ' ':
                walk_graph[(i, j)] = []
                if i - 1 >= 0 and terrain_mtx[i-1][j] == ' ':
                    walk_graph[(i, j)].append((i-1, j))
                if i + 1 <= l-1 and terrain_mtx[i+1][j] == ' ':
                    walk_graph[(i, j)].append((i+1, j))
                if j - 1 >= 0 and terrain_mtx[i][j-1] == ' ':
                    walk_graph[(i, j)].append((i, j-1))
                if j + 1 <= w-1 and terrain_mtx[i][j+1] == ' ':
                    walk_graph[(i, j)].append((i, j+1))

                # if len(walk_graph[(i, j)]) == 0:
                #     walk_graph.pop((i, j), None)

                handover_graph[(i, j)] = []
                # a handover
                if i - 1 >= 0 and terrain_mtx[i-1][j] == 'X':
                    if i - 2 >= 0 and terrain_mtx[i - 2][j] == ' ':
                        handover_graph[(i, j)].append((i - 2, j))
                    if j - 1 >= 0 and terrain_mtx[i - 1][j - 1] == ' ':
                        handover_graph[(i, j)].append((i - 1, j - 1))
                    if j + 1 <= w-1 and terrain_mtx[i - 1][j + 1] == ' ':
                        handover_graph[(i, j)].append((i - 1, j + 1))
                if i + 1 <= l - 1 and terrain_mtx[i + 1][j] == 'X':
                    if i + 2 <= l - 1 and terrain_mtx[i + 2][j] == ' ':
                        handover_graph[(i, j)].append((i + 2, j))
                    if j - 1 >= 0 and terrain_mtx[i + 1][j - 1] == ' ':
                        handover_graph[(i, j)].append((i + 1, j - 1))
                    if j + 1 <= w-1 and terrain_mtx[i + 1][j + 1] == ' ':
                        handover_graph[(i, j)].append((i + 1, j + 1))
                if j - 1 >= 0 and terrain_mtx[i][j-1] == 'X':
                    if j - 2 >= 0 and terrain_mtx[i][j-2] == ' ':
                        handover_graph[(i, j)].append((i, j-2))
                    if i - 1 >= 0 and terrain_mtx[i - 1][j - 1] == ' ':
                        handover_graph[(i, j)].append((i - 1, j - 1))
                    if i + 1 <= l-1 and terrain_mtx[i + 1][j - 1] == ' ':
                        handover_graph[(i, j)].append((i + 1, j - 1))
                if j + 1 <= w - 1 and terrain_mtx[i][j + 1] == 'X':
                    if j + 2 <= w - 1 and terrain_mtx[i][j + 2] == ' ':
                        handover_graph[(i, j)].append((i, j + 2))
                    if i - 1 >= 0 and terrain_mtx[i - 1][j + 1] == ' ':
                        handover_graph[(i, j)].append((i - 1, j + 1))
                    if i + 1 <= l-1 and terrain_mtx[i + 1][j + 1] == ' ':
                        handover_graph[(i, j)].append((i + 1, j + 1))

                if len(handover_graph[(i, j)]) == 0:
                    handover_graph.pop((i, j), None)

    return walk_graph, handover_graph

def get_feature_locations(terrain_mtx, feature):
    # Return all (i, j) locations of feature
    l, w = len(terrain_mtx), len(terrain_mtx[0])
    res = []
    for i in range(l):
        for j in range(w):
            if terrain_mtx[i][j] == feature:
                res.append((i, j))
    return res

def perform_action(prev_loc, action, terrain_mtx):
    curr_loc = prev_loc
    if action != 'interact':
        new_loc = (curr_loc[0] + action[1], curr_loc[1] + action[0])
        if terrain_mtx[new_loc[0]][new_loc[1]] == ' ':
            curr_loc = new_loc
    return curr_loc

def get_open_neighbor(loc, other_agent_loc, terrain_mtx):
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    locs = [(loc[0] + dir[1], loc[1] + dir[0]) for dir in dirs]

    for i in range(len(locs)):
        new_loc = locs[i]
        if new_loc != other_agent_loc and terrain_mtx[new_loc[0]][new_loc[1]] == ' ':
            return new_loc, dirs[i]

    return None, None



def perform_mla(feature_locations, prev_mla_dict, walk_graph, handover_graph, terrain_mtx, task_name,
                        both_idx=False, is_initial_potting=False, is_returning_to_pot=False, best_only=False):
    """
    This function perform one mla action from the previous mla node, recorded in prev_mla_dict
    Arguments:
       feature_locations (list of tuples): the locations of destination features for this mla action
       prev_mla_dict (dict): a dictionary of mla nodes up to this point
       walk_graph (dict): walk graph found by graph_from_terrain
       handover_graph (dict): handover graph found by graph_from_terrain
       terrain_mtx (list of list): matrix for the terrain
       task_name (str): name of the mla action
       both_idx (bool): whether this action can be performed by agents at both agent, and not just the primary agent
       is_initial_potting (bool): whether we need to update the pot location when create the new mla node
       is_returning_to_pot (bool): whether we are using the pot_loc as the goal for the motion plan
       best_only (bool): only keeping the best path at each of the forward_mla_hash
    """
    new_dict = {}
    assert is_returning_to_pot or feature_locations != [None]

    # construct empty terrain mtx in which start locations are emptied out for use in constructing a location path from an action path
    empty_terrain_mtx = copy.deepcopy(terrain_mtx)
    agent_0_start = get_feature_locations(empty_terrain_mtx, "1")
    agent_1_start = get_feature_locations(empty_terrain_mtx, "2")
    if len(agent_0_start) > 0:
        empty_terrain_mtx[agent_0_start[0][0]][agent_0_start[0][1]] = ' '
    if len(agent_1_start) > 0:
        empty_terrain_mtx[agent_1_start[0][0]][agent_1_start[0][1]] = ' '

    for f_location in feature_locations:
        for backward_mla_hash in prev_mla_dict.keys():
            for backward_mla_node in prev_mla_dict[backward_mla_hash]:
                # For this part, it doesn't matter who is picking up the dish
                agent_positions = backward_mla_node.agent_positions()
                # give the option to let both agents participat in a task if it does not rely on primary agent having something
                if both_idx:
                    start_idx_lst = [0, 1]
                else:
                    start_idx_lst = [backward_mla_node.primary_idx]
                # if we are dishing a cooked soup, the distination has to be the current pot location
                if is_returning_to_pot:
                    f_location = backward_mla_node.pot_loc

                for start_idx in start_idx_lst:
                    # Also, there is nothing to be handed over, so passing in empty dictionary
                    options = uniform_cost_search(walk_graph, handover_graph, terrain_mtx, agent_positions, start_idx, f_location)
                    for ending_hash_i in options.keys():
                        for forward_node in options[ending_hash_i]:
                            # we only update the pot location in the mla node if we are potting onion
                            if is_initial_potting:
                                forward_mla_node = backward_mla_node.update_from_search_node(task_name, forward_node, f_location)
                            else:
                                forward_mla_node = backward_mla_node.update_from_search_node(task_name, forward_node)

                            # check the action/location paths for collision and update the mla node if necessary

                            # previous locations and current unmodified action_paths
                            prev_locs = [backward_mla_node.agent_0_loc, backward_mla_node.agent_1_loc]
                            mla_key = list(forward_mla_node.agent_0_act_dict.keys())[-1]
                            act_paths = [forward_mla_node.agent_0_act_dict[mla_key], forward_mla_node.agent_1_act_dict[mla_key]]

                            # construct empty action paths to be filled out as we traverse to check for collisions
                            new_act_paths = [[], []]
                            # keep track of our current location
                            curr_locs = prev_locs
                            # keep track of number of whether there is an unavoidable collision
                            collision = 0

                            # traverse through each joint action
                            for i in range(len(act_paths[0])):

                                # the would be new locations if these actions were taken
                                locs = [perform_action(curr_locs[0], act_paths[0][i], empty_terrain_mtx),
                                        perform_action(curr_locs[1], act_paths[1][i], empty_terrain_mtx)]

                                # extracting primary and non primary indexes
                                primary_idx = forward_mla_node.primary_idx
                                non_primary_idx = 1 - primary_idx

                                # check if there is a collision, if there is: move the non primary agent to a
                                # empty neighbor and modify its action and new location
                                non_primary_loc = locs[non_primary_idx]
                                non_primary_act = act_paths[non_primary_idx][i]

                                if locs[0] == locs[1]:
                                    non_primary_idx = 1 - primary_idx
                                    new_loc, new_act = get_open_neighbor(locs[non_primary_idx], prev_locs[primary_idx], empty_terrain_mtx)

                                    # if there is no empty neighbor to move to then notify the outer loop and break
                                    if new_loc == None:
                                        collision = 1
                                        break

                                    non_primary_loc = new_loc
                                    non_primary_act = new_act

                                # update and append to our new list of action paths and the curr location
                                curr_locs[primary_idx] = locs[primary_idx]
                                new_act_paths[primary_idx].append(act_paths[primary_idx][i])

                                curr_locs[non_primary_idx] = non_primary_loc
                                new_act_paths[non_primary_idx].append(non_primary_act)

                            # if there is an unavoidable collision, do not hash and index this search node
                            if collision:
                                continue

                            # update the action paths and previous location in the mla search node
                            forward_mla_node.agent_0_act_dict[mla_key] = new_act_paths[0]
                            forward_mla_node.agent_1_act_dict[mla_key] = new_act_paths[1]
                            forward_mla_node.agent_0_loc = curr_locs[0]
                            forward_mla_node.agent_1_loc = curr_locs[1]
                            
                            forward_mla_hash = forward_mla_node.hash_key()

                            if forward_mla_hash not in new_dict.keys():
                                new_dict[forward_mla_hash] = [forward_mla_node]
                            else:
                                if not best_only:
                                    new_dict[forward_mla_hash].append(forward_mla_node)
                                else:
                                    # retrieve the previous entry
                                    forward_mla_node_prev = new_dict[forward_mla_hash][0]
                                    # if the current node's path length is lower, we will replace it
                                    if forward_mla_node.total_path_length() < forward_mla_node_prev.total_path_length():
                                        new_dict[forward_mla_hash] = [forward_mla_node]
    return new_dict


def add_cooking_waiting(act_paths, difference):
    """

    Args:
        act_path (tuple of two list of tuples): the act_paths that needs to be amended
        difference (int) the number of (0, 0) needs to be inserted

    Returns:
        the amended act_paths

    """
    insertion = [(0, 0)] * difference
    amended_act_paths = []
    amended_act_paths.append(act_paths[0][:-1] + insertion + [act_paths[0][-1]])
    amended_act_paths.append(act_paths[1][:-1] + insertion + [act_paths[1][-1]])
    amended_act_paths = tuple(amended_act_paths)
    return amended_act_paths


def terrain_analysis(terrain_mtx, silent=True, best_only=True):
    """
    Arguments:
        terrain_mtx (list of list): 2 dimensional terrain matrix which represent the grid
            details for conventions can be found at overcooked_ai_py.mdp.layout_generator
        silent (bool): whether to print the details
        best_only (bool): whether to only consider the best mla_node at each mla_hash
    Returns:
        a dictionary containing the stage score, various individual paths for both agents
        to make a 3 onion soup, and these paths sorted by length
    """
    start_player_positions = [None, None]

    start_player_1_positions = get_feature_locations(terrain_mtx, "1")
    if len(start_player_1_positions) > 0:
        start_player_1_position = random.choice(start_player_1_positions)
        for i, j in start_player_1_positions:
            terrain_mtx[i][j] = ' '
        start_player_positions[0] = start_player_1_position[::-1]

    start_player_2_positions = get_feature_locations(terrain_mtx, "2")
    if len(start_player_2_positions) > 0:
        start_player_2_position = random.choice(start_player_2_positions)
        for i, j in start_player_2_positions:
            terrain_mtx[i][j] = ' '
        start_player_positions[1] = start_player_2_position[::-1]

    walk_graph, handover_graph = graph_from_terrain(terrain_mtx)

    empty_locations = get_feature_locations(terrain_mtx, ' ')

    if start_player_positions[0] == None:
        p0_starting = random.choice(empty_locations)
    else:
        # the start_player_position is flipped
        p0_starting_pre = start_player_positions[0]
        p0_starting = (p0_starting_pre[1], p0_starting_pre[0])


    if start_player_positions[1] == None:
        p1_starting = random.choice(empty_locations)
        while p1_starting == p0_starting:
            p1_starting = random.choice(empty_locations)
    else:
        # the start_player_position is flipped
        p1_starting_pre = start_player_positions[1]
        p1_starting = (p1_starting_pre[1], p1_starting_pre[0])

    stage_score = []

    p0_i, p0_j = p0_starting
    p1_i, p1_j = p1_starting
    terrain_mtx_rep = copy.deepcopy(terrain_mtx)
    terrain_mtx_rep[p0_i][p0_j] = '1'
    terrain_mtx_rep[p1_i][p1_j] = '2'
    if not silent:
        print("P0 starting at ", p0_starting)
        print("P1 starting at ", p1_starting)
        for line in terrain_mtx_rep:
            print(line)

    modified_terrain_mtx = copy.deepcopy(terrain_mtx)
    modified_terrain_mtx[start_player_1_position[0]][start_player_1_position[1]] = ' '
    modified_terrain_mtx[start_player_2_position[0]][start_player_2_position[1]] = ' '

    # keep track of the (position of the agent) and lowest walking cost for each counter operation cost so far
    """
    Format:
    key: (p0_loc, p1_loc, pot_loc, num_counter_operation)

    """
    starting_mla_search_node = OvercookedMLASearchNode(-1, p0_starting, p1_starting, None, {}, {}, 0, modified_terrain_mtx)
    possible_starting_agents_positions = {
        # format: p0 location, p1 location,
        starting_mla_search_node.hash_key(): [starting_mla_search_node]
    }

    if not silent:
        print("possible agents positions before starting", possible_starting_agents_positions)

    # first we need the first onion
    onion_dispenser_locations = get_feature_locations(terrain_mtx, 'O')
    possible_onion_1_agent_positions = perform_mla(
        onion_dispenser_locations,
        possible_starting_agents_positions,
        walk_graph,
        {},
        terrain_mtx,
        "0_onion_pickup_1",
        both_idx=True,
        best_only=best_only
    )

    if len(possible_onion_1_agent_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)

    if not silent:
        print("possible positions after collecting onion 1")
        for k in possible_onion_1_agent_positions:
            print(k)
            for rep in possible_onion_1_agent_positions[k]:
                print(str(rep))
            print("----")
        print("*************************************")

    # then we need to put the onion to the pot
    pot_locations = get_feature_locations(terrain_mtx, 'P')
    possible_agents_and_cooking_pot_1_positions = perform_mla(
        pot_locations,
        possible_onion_1_agent_positions,
        walk_graph,
        handover_graph,
        terrain_mtx,
        "1_onion_drop_1",
        is_initial_potting=True,
        best_only=best_only
    )

    if len(possible_agents_and_cooking_pot_1_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)

    if not silent:
        print("possible positions of agents and cooking pot 1")
        for k in possible_agents_and_cooking_pot_1_positions:
            print(k)
            for rep in possible_agents_and_cooking_pot_1_positions[k]:
                print(str(rep))
            print("----")
        print("*************************************")

    # second we need the second onion
    possible_onion_2_agent_positions = perform_mla(
        onion_dispenser_locations,
        possible_agents_and_cooking_pot_1_positions,
        walk_graph,
        {},
        terrain_mtx,
        "2_onion_pickup_2",
        both_idx=True,
        best_only=best_only
    )

    if len(possible_onion_2_agent_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)

    if not silent:
        print("possible positions after collecting onion 2")
        for k in possible_onion_2_agent_positions:
            print(k)
            for rep in possible_onion_2_agent_positions[k]:
                print(str(rep))
            print("----")
        print("*************************************")

    # then we need to put the onion to the pot
    possible_agents_and_cooking_pot_2_positions = perform_mla(
        [None],
        possible_onion_2_agent_positions,
        walk_graph,
        handover_graph,
        terrain_mtx,
        "3_onion_drop_2",
        is_returning_to_pot=True,
        best_only=best_only
    )

    if len(possible_agents_and_cooking_pot_2_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)

    if not silent:
        print("possible positions of agents and cooking pot 2")
        for k in possible_agents_and_cooking_pot_2_positions:
            print(k)
            for rep in possible_agents_and_cooking_pot_2_positions[k]:
                print(str(rep))
            print("----")
        print("*************************************")


    # third we need the third onion
    possible_onion_3_agent_positions = perform_mla(
        onion_dispenser_locations,
        possible_agents_and_cooking_pot_2_positions,
        walk_graph,
        {},
        terrain_mtx,
        "4_onion_pickup_3",
        both_idx=True,
        best_only=best_only
    )

    if len(possible_onion_3_agent_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)

    if not silent:
        print("possible positions after collecting onion 3")
        for k in possible_onion_3_agent_positions:
            print(k)
            for rep in possible_onion_3_agent_positions[k]:
                print(str(rep))
            print("----")
        print("*************************************")

    # then we need to put the onion to the pot
    possible_agents_and_full_pot_3_positions = perform_mla(
        [None],
        possible_onion_3_agent_positions,
        walk_graph,
        handover_graph,
        terrain_mtx,
        "5_onion_drop_3",
        is_returning_to_pot=True,
        best_only=best_only
    )

    if len(possible_agents_and_full_pot_3_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)

    if not silent:
        print("possible positions of agents and cooking pot 3")
        for k in possible_agents_and_full_pot_3_positions:
            print(k)
            for rep in possible_agents_and_full_pot_3_positions[k]:
                print(str(rep))
            print("----")
        print("*************************************")

    # then we need to start cooking
    possible_agents_and_cooking_pot_3_positions = perform_mla(
        [None],
        possible_agents_and_full_pot_3_positions,
        walk_graph,
        {},
        terrain_mtx,
        "6_start_cooking",
        is_returning_to_pot=True,
        best_only=best_only
    )

    if len(possible_agents_and_cooking_pot_3_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)

    if not silent:
        print("possible positions of agents and cooking pot 3")
        for k in possible_agents_and_cooking_pot_3_positions:
            print(k)
            for rep in possible_agents_and_cooking_pot_3_positions[k]:
                print(str(rep))
            print("----")
        print("*************************************")

    # then we need to pick up the dish from a dispenser
    dish_dispenser_locations = get_feature_locations(terrain_mtx, 'D')
    possible_dish_agent_and_cooking_pot_positions = perform_mla(
        dish_dispenser_locations,
        possible_agents_and_cooking_pot_3_positions,
        walk_graph,
        {},
        terrain_mtx,
        "7_dish_pickup",
        both_idx=True,
        best_only=best_only
    )

    if len(possible_dish_agent_and_cooking_pot_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)
    if not silent:
        print("possible positions after collecting dish and cooking pot")
        for k in possible_dish_agent_and_cooking_pot_positions:
            print(k)
            for rep in possible_dish_agent_and_cooking_pot_positions[k]:
                print(str(rep))
            print("----")
        print("*************************************")

    # then we need to return the dish to the cooked pot
    possible_agent_and_cooked_dished_pot_positions = perform_mla(
        [None], # this should be ignored anyway
        possible_dish_agent_and_cooking_pot_positions,
        walk_graph,
        handover_graph,
        terrain_mtx,
        "8_dishing_soup",
        is_returning_to_pot=True,
        best_only=best_only
    )

    if len(possible_agent_and_cooked_dished_pot_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)
    if not silent:
        print("possible positions of agents and cooked dished pot")
        for k in possible_agent_and_cooked_dished_pot_positions:
            print(k)
            for rep in possible_agent_and_cooked_dished_pot_positions[k]:
                print(str(rep))
            print("----")
        print("*************************************")

    # In the end, we need to deliver the meal to the serving point
    serving_locations = get_feature_locations(terrain_mtx, 'S')
    possible_agents_served_positions = perform_mla(
        serving_locations,
        possible_agent_and_cooked_dished_pot_positions,
        walk_graph,
        handover_graph,
        terrain_mtx,
        "9_serving",
        best_only=best_only
    )
    if len(possible_agents_served_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)
    if not silent:
        print("possible positions of agents after serving")
        for k in possible_agents_served_positions:
            print(k)
            for rep in possible_agents_served_positions[k]:
                print(str(rep))
            print("----")
        print("*************************************")

    # list to store all possible full action paths for each player
    player_1_action_paths = []
    player_2_action_paths = []
    pairs_of_action_paths_by_total_length = {}

    for key in possible_agents_served_positions:
        for rep in possible_agents_served_positions[key]:

            # for each possible end mla node we find all the location paths from all the mlas
            act_path_dict_0 = rep.agent_0_act_dict
            act_path_dict_1 = rep.agent_1_act_dict
            assert set(act_path_dict_0.keys()) == set(act_path_dict_1.keys()), "different subtasks were recorded"

            act_path_0 = []
            act_path_1 = []

            # soup cooking wait time, so that the entire sequence will result in the correct timing to dish the soup
            soup_cooking_tick = 0
            # currently only support soup with cook time 20
            soup_cooking_time = 20

            # for each mla location path we convert it to actions and add that to our list of action_lists for this node
            for mla in sorted(list(act_path_dict_0.keys())):
                act_paths = [act_path_dict_0[mla], act_path_dict_1[mla]]
                if mla == "7_dish_pickup":
                    soup_cooking_tick += len(act_paths[0]) + 1 # the plus one because soup ticking starts at 1
                elif mla == "8_dishing_soup":
                    soup_cooking_tick += len(act_paths[0]) - 1
                    if soup_cooking_tick < soup_cooking_time:
                        act_paths = add_cooking_waiting(act_paths, soup_cooking_time - soup_cooking_tick)
                act_path_0.append(act_paths[0])
                act_path_1.append(act_paths[1])

            # we add the full action_list for this end node to our list of all possible action paths for each player
            connected_action_path_0 = connect_action_path(act_path_0)
            connected_action_path_1 = connect_action_path(act_path_1)
            player_1_action_paths.append(connected_action_path_0)
            player_2_action_paths.append(connected_action_path_1)

            # categorize the paths by its total length
            pair_of_action_paths = [connected_action_path_0, connected_action_path_1]
            assert len(connected_action_path_0) == len(connected_action_path_1)
            total_length = len(connected_action_path_0)
            if total_length not in pairs_of_action_paths_by_total_length:
                pairs_of_action_paths_by_total_length[total_length] = [pair_of_action_paths]
            else:
                pairs_of_action_paths_by_total_length[total_length].append(pair_of_action_paths)

    return_dict = {}
    return_dict['stage score'] = stage_score
    return_dict['player 1 action paths'] = player_1_action_paths
    return_dict['player 2 action paths'] = player_2_action_paths
    return_dict['pairs of action paths by total length'] = pairs_of_action_paths_by_total_length

    return return_dict


def stats_from_analaysis(terrain_mtx, best_only=True):
    dic = terrain_analysis(terrain_mtx, best_only=best_only)
    pairs_of_action_paths_by_total_length = dic['pairs of action paths by total length']
    stats_by_length = {}
    for total_length in sorted(pairs_of_action_paths_by_total_length.keys()):
        stats_by_length[total_length] = pairs_of_action_paths_by_total_length[total_length]
    return stats_by_length

