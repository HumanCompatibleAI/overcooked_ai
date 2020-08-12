import random
import copy
import numpy as np
INTERACT_TRANSITION_COST = 2

INFINITY = np.inf

import heapq

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


def uniform_cost_search(walk_graph, handover_graph, terrain_mtx, agent_locations, start_agent_idx, goal_loc):
    """
    :param walk_graph: the dictionary representation of the walking graph (reachable by walking)
    :param handover_graph: the dictionary representation of the handover graph (reachable by placing on counter)
    :param terrain_mtx: matrix representation of the grid world. Does not include information about agents
    :param agent_locations: tuple of tuple, locations of agents
    :param start_agent_idx: the agent starting to act first
    :param goal_loc: the location we would like to reach
    :return: dictionary of
        {number of counter_operations: [length of the path, the ending physical location, the path itself]}
        note: the number of counter_operations should always be even because it takes 1 to drop something down,
        and 1 to pick something up

    """
    walk_graph_copy = copy.deepcopy(walk_graph)
    handover_graph_copy = copy.deepcopy(handover_graph)

    # print("start agent location", agent_locations, "agent who act first is ", start_agent_idx)
    # print("goal", goal_loc, "of type", terrain_mtx[goal_loc[0]][goal_loc[1]])

    l, w = len(terrain_mtx), len(terrain_mtx[0])

    for agent_loc in agent_locations:
        agent_loc_i, agent_loc_j = agent_loc
        assert terrain_mtx[agent_loc_i][agent_loc_j] == ' ', "starting location is not an empty square"

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
    start_loc = agent_locations[start_agent_idx]
    other_agent_start_loc = agent_locations[1-start_agent_idx]
    fringe = PriorityQueue()
    # All search nodes have the form (LOCATION, OTHER_AGENT_LOCATION, PATH, TOTAL_WALK_COST, COUNTER_OPERATION)
    startInfo = (start_loc, other_agent_start_loc, [start_loc], 0, 0)
    fringe.push(startInfo, 0)
    # Continue till fringe empty
    while (not fringe.isEmpty()):
        (loc, other_agent_loc, path_so_far, total_walk_cost_so_far, counter_op_so_far) = fringe.pop()
        # Goal state check
        if loc == goal_loc:
            if len(path_so_far) < 2:
                agents_ending_positions = (start_loc, other_agent_loc)
                print("tru path_so_far", path_so_far)
                path_so_far = [start_loc, goal_loc]
                print("edited path_so_far", path_so_far)
            else:
                agents_ending_positions = (path_so_far[-2], other_agent_loc)
            # create the empty list if key does not exist
            if agents_ending_positions not in res.keys():
                res[agents_ending_positions] = []
            # the content of the res[agents_ending_loc] is
            # [counter_op_cost, walk_cost, path taken]
            # this means the starting position was right there
            res[agents_ending_positions].append([counter_op_so_far, total_walk_cost_so_far, path_so_far])

            continue
        # If necessary, add successors
        if counter_op_so_far not in closed.keys():
            closed[counter_op_so_far] = set([])
        if not truly_closed(loc, counter_op_so_far):
            closed[counter_op_so_far].add(loc)
            # first type of successor: same agent moving
            for suc_loc in walk_graph_copy[loc]:
                # keep original agent acting
                fringe.push((suc_loc, other_agent_loc, path_so_far + [suc_loc], total_walk_cost_so_far + 1,
                             counter_op_so_far), total_walk_cost_so_far + 1 + counter_op_so_far)
            # second type of succesor: handover
            if loc in handover_graph_copy:
                for suc_loc in handover_graph_copy[loc]:
                    # first we need to check if the other agent can actually make the walk:
                    other_agent_walk_dist = shortest_walk_dist(walk_graph, other_agent_loc, suc_loc)
                    if other_agent_walk_dist != INFINITY:
                        # give control over to the other agent, and become the other agent by "taking its place on the map"
                        fringe.push((suc_loc, loc, path_so_far + ["COUNTER", suc_loc], total_walk_cost_so_far + 2,
                                     counter_op_so_far + 2), total_walk_cost_so_far + 2 + counter_op_so_far + 2)
                        # print("switched, agent taking control at", suc_loc)
                    # else:
                        # print("the other agent cannot move from", other_agent_loc, "to", suc_loc)
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
    :param terrain_mtx: the terrain matrix
    :return: a walk_graph, in format of
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


def terrain_analysis(terrain_mtx, start_player_positions = [None, None], silent = True):
    """
    :param terrain_mtx: 2 dimensional terrain matrix which represent the grid
        details for conventions can be found at overcooked_ai_py.mdp.layout_generator
    :return:
    """

    def get_feature_locations(terrain_mtx, feature):
        # Return all (i, j) locations of feature
        l, w = len(terrain_mtx), len(terrain_mtx[0])
        res = []
        for i in range(l):
            for j in range(w):
                if terrain_mtx[i][j] == feature:
                    res.append((i, j))
        return res


    empty_locations = get_feature_locations(terrain_mtx, ' ')

    walk_graph, handover_graph = graph_from_terrain(terrain_mtx)

    """
    print("WALK GRAPH")
    print(walk_graph)
    print("HANDOVER GRAPH")
    print(handover_graph)  
    """

    if start_player_positions[0] == None:
        p1_starting = random.choice(empty_locations)
    else:
        # the start_player_position is flipped
        p1_starting_pre = start_player_positions[0]
        p1_starting = (p1_starting_pre[1], p1_starting_pre[0])


    if start_player_positions[1] == None:
        p2_starting = random.choice(empty_locations)
        while p2_starting == p1_starting:
            p2_starting = random.choice(empty_locations)
    else:
        # the start_player_position is flipped
        p2_starting_pre = start_player_positions[1]
        p2_starting = (p2_starting_pre[1], p2_starting_pre[0])



    stage_score = []

    p1_i, p1_j = p1_starting
    p2_i, p2_j = p2_starting
    terrain_mtx_rep = copy.deepcopy(terrain_mtx)
    terrain_mtx_rep[p1_i][p1_j] = '1'
    terrain_mtx_rep[p2_i][p2_j] = '2'
    if not silent:
        print("P1 starting at ", p1_starting)
        print("P2 starting at ", p2_starting)
        for line in terrain_mtx_rep:
            print(line)

    # keep track of the (position of the agent) and lowest walking cost for each counter operation cost so far
    possible_agents_positions = {(p1_starting, p2_starting, (-1, -1)): [[0, 0, []]]}

    possible_onion_agent_positions = {}

    if not silent:
        print("possible agents positions before starting", possible_agents_positions)




    # first we need the onions
    onion_dispenser_locations = get_feature_locations(terrain_mtx, 'O')
    for o_location in onion_dispenser_locations:
        for agent_pot_positions in possible_agents_positions.keys():
            for backwards_counter_op, backwards_walking_cost, backwards_path in possible_agents_positions[agent_pot_positions]:
                agent_positions = agent_pot_positions[:2]
                # For this part, it doesn't matter who is picking up the onion
                for start_idx in [0, 1]:
                    # Also, there is nothing to be handed over, so passing in empty dictionary
                    options = uniform_cost_search(walk_graph, {}, terrain_mtx, agent_positions, start_idx, o_location)
                    for agents_ending_positions_i in options.keys():
                        for forward_counter_op, forward_walking_cost, forward_path in options[agents_ending_positions_i]:
                            if agents_ending_positions_i not in possible_onion_agent_positions.keys():
                                possible_onion_agent_positions[agents_ending_positions_i] = []
                            possible_onion_agent_positions[agents_ending_positions_i].append([
                                backwards_counter_op + forward_counter_op,
                                backwards_walking_cost + forward_walking_cost + 1,
                                backwards_path + forward_path]
                            )
    if len(possible_onion_agent_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)

    if not silent:
        print("possible positions after collecting onion")
        for k in possible_onion_agent_positions:
            print(k, "     ", possible_onion_agent_positions[k])
            print("----")
        print("*************************************")


    # PLEASE NOTE THAT NOW AGENT AT INDEX 0 of onion_agent_position has the onion!!
    # then we need to put the onion to the pot
    pot_locations = get_feature_locations(terrain_mtx, 'P')
    possible_agents_and_cooking_pot_positions = {}
    for p_location in pot_locations:
        for onion_agent_positions in possible_onion_agent_positions.keys():
            for backwards_counter_op, backwards_walking_cost, backwards_path in possible_onion_agent_positions[
                    onion_agent_positions]:
                # Only agent at index 0 has the onion
                options = uniform_cost_search(walk_graph, handover_graph, terrain_mtx, onion_agent_positions, 0,
                                              p_location)
                for agents_ending_positions_i in options.keys():
                    for forward_counter_op, forward_walking_cost, forward_path in options[agents_ending_positions_i]:
                        agents_ending_positions_and_cooking_pot_i = (agents_ending_positions_i[0], agents_ending_positions_i[1], p_location)
                        if agents_ending_positions_and_cooking_pot_i not in possible_agents_and_cooking_pot_positions.keys():
                            possible_agents_and_cooking_pot_positions[agents_ending_positions_and_cooking_pot_i] = []
                        possible_agents_and_cooking_pot_positions[agents_ending_positions_and_cooking_pot_i].append([
                            backwards_counter_op + forward_counter_op,
                            backwards_walking_cost + forward_walking_cost + 1,
                            backwards_path + forward_path]
                        )

    if len(possible_agents_and_cooking_pot_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)

    if not silent:
        print("possible positions of agents and cooking pot")
        for k in possible_agents_and_cooking_pot_positions:
            print(k, "     ", possible_agents_and_cooking_pot_positions[k])
            print("----")
        print("*************************************")



    # then we need to pick up the dish
    dish_dispenser_locations = get_feature_locations(terrain_mtx, 'D')
    possible_dish_agent_and_cooking_pot_positions = {}
    for d_location in dish_dispenser_locations:
        for agent_and_cooking_pot_position in possible_agents_and_cooking_pot_positions.keys():
            for backwards_counter_op, backwards_walking_cost, backwards_path in possible_agents_and_cooking_pot_positions[
                    agent_and_cooking_pot_position]:
                # For this part, it doesn't matter who is picking up the dish
                agent_positions = agent_and_cooking_pot_position[:2]
                cooking_pot_position = agent_and_cooking_pot_position[2]
                for start_idx in [0, 1]:
                    # Also, there is nothing to be handed over, so passing in empty dictionary
                    options = uniform_cost_search(walk_graph, {}, terrain_mtx, agent_positions, start_idx, d_location)
                    for agents_ending_positions_i in options.keys():
                        for forward_counter_op, forward_walking_cost, forward_path in options[agents_ending_positions_i]:
                            agents_ending_positions_and_cooking_pot_i = (agents_ending_positions_i[0], agents_ending_positions_i[1], cooking_pot_position)
                            if agents_ending_positions_and_cooking_pot_i not in possible_dish_agent_and_cooking_pot_positions.keys():
                                possible_dish_agent_and_cooking_pot_positions[agents_ending_positions_and_cooking_pot_i] = []
                            possible_dish_agent_and_cooking_pot_positions[agents_ending_positions_and_cooking_pot_i].append([
                                backwards_counter_op + forward_counter_op,
                                backwards_walking_cost + forward_walking_cost + 1,
                                backwards_path + forward_path]
                            )

    if len(possible_dish_agent_and_cooking_pot_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)
    if not silent:
        print("possible positions after collecting dish and cooking pot")
        for k in possible_dish_agent_and_cooking_pot_positions:
            print(k, "     ", possible_dish_agent_and_cooking_pot_positions[k])
            print("----")
        print("*************************************")

    # PLEASE NOTE THAT NOW AGENT AT INDEX 0 of dish_agent_and_cooking_pot_position has the dish!!
    # then we need to return the dish to the cooked pot

    possible_agent_and_cooked_dished_pot_positions = {}

    for dish_agent_and_cooking_pot_position in possible_dish_agent_and_cooking_pot_positions.keys():
        for backwards_counter_op, backwards_walking_cost, backwards_path in possible_dish_agent_and_cooking_pot_positions[
                dish_agent_and_cooking_pot_position]:
            dish_agent_positions = dish_agent_and_cooking_pot_position[:2]
            cooked_pot_position = dish_agent_and_cooking_pot_position[2]
            # Only agent at index 0 has the dish
            options = uniform_cost_search(walk_graph, handover_graph, terrain_mtx, dish_agent_positions, 0,
                                          cooked_pot_position)
            for agents_ending_positions_i in options.keys():
                for forward_counter_op, forward_walking_cost, forward_path in options[agents_ending_positions_i]:
                    agents_ending_positions_and_cooked_pot_i = (agents_ending_positions_i[0], agents_ending_positions_i[1], cooked_pot_position)
                    if agents_ending_positions_and_cooked_pot_i not in possible_agent_and_cooked_dished_pot_positions.keys():
                        possible_agent_and_cooked_dished_pot_positions[agents_ending_positions_and_cooked_pot_i] = []
                    possible_agent_and_cooked_dished_pot_positions[agents_ending_positions_and_cooked_pot_i].append([
                        backwards_counter_op + forward_counter_op,
                        backwards_walking_cost + forward_walking_cost + 1,
                        backwards_path + forward_path]
                    )

    if len(possible_agent_and_cooked_dished_pot_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)
    if not silent:
        print("possible positions of agents and cooked dished pot")
        for k in possible_agent_and_cooked_dished_pot_positions:
            print(k, "     ", possible_agent_and_cooked_dished_pot_positions[k])
            print("----")
        print("*************************************")



    # In the end, we need to deliver the meal to the serving point
    # Agent 0, who has the dish, must start the delivery
    serving_locations = get_feature_locations(terrain_mtx, 'S')
    possible_agents_served_positions = {}

    for s_location in serving_locations:
        for agent_and_cooked_dished_pot_position in possible_agent_and_cooked_dished_pot_positions.keys():
            for backwards_counter_op, backwards_walking_cost, backwards_path in possible_agent_and_cooked_dished_pot_positions[
                agent_and_cooked_dished_pot_position]:
                agent_positions = agent_and_cooked_dished_pot_position[:2]

                options = uniform_cost_search(walk_graph, handover_graph, terrain_mtx, agent_positions, 0,
                                              s_location)

                for agents_ending_positions_i in options.keys():
                    for forward_counter_op, forward_walking_cost, forward_path in options[agents_ending_positions_i]:
                        agents_served_positions_i = (agents_ending_positions_i[0], agents_ending_positions_i[1], s_location)
                        if agents_served_positions_i not in possible_agents_served_positions.keys():
                            possible_agents_served_positions[agents_served_positions_i] = []
                        possible_agents_served_positions[agents_served_positions_i].append([
                            backwards_counter_op + forward_counter_op,
                            backwards_walking_cost + forward_walking_cost + 1,
                            backwards_path + forward_path]
                        )

    if len(possible_agents_served_positions) > 0:
        stage_score.append(1)
    else:
        stage_score.append(0)
    if not silent:
        print("possible positions of agents after serving")
        for k in possible_agents_served_positions:
            print(k, "     ", possible_agents_served_positions[k])
            print("----")
        print("*************************************")
    return stage_score

