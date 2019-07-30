import itertools


class Direction(object):
    """
    The four possible directions a player can be facing.
    """
    
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST  = (1, 0)
    WEST  = (-1, 0)
    ALL_DIRECTIONS = INDEX_TO_DIRECTION = [NORTH, SOUTH, EAST, WEST]
    DIRECTION_TO_INDEX = { a:i for i, a in enumerate(INDEX_TO_DIRECTION) }
    OPPOSITE_DIRECTIONS = { NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST }

    @staticmethod
    def get_adjacent_directions(direction):
        """Returns the directions within 90 degrees of the given direction.

        direction: One of the Directions, except not Direction.STAY.
        """
        if direction in [Direction.NORTH, Direction.SOUTH]:
            return [Direction.EAST, Direction.WEST]
        elif direction in [Direction.EAST, Direction.WEST]:
            return [Direction.NORTH, Direction.SOUTH]
        raise ValueError('Invalid direction: %s' % direction)


class Action(object):
    """
    The six actions available in the OvercookedGridworld.

    Includes definitions of the actions as well as utility functions for
    manipulating them or applying them.
    """

    STAY = (0, 0)
    INTERACT = 'interact'
    ALL_ACTIONS = INDEX_TO_ACTION = Direction.INDEX_TO_DIRECTION + [STAY, INTERACT]
    INDEX_TO_ACTION_INDEX_PAIRS = [v for v in itertools.product(range(len(INDEX_TO_ACTION)), repeat=2)]
    ACTION_TO_INDEX = { a:i for i, a in enumerate(INDEX_TO_ACTION) }
    MOTION_ACTIONS = Direction.ALL_DIRECTIONS + [STAY]
    ACTION_TO_CHAR = { 
        Direction.NORTH: "↑",
        Direction.SOUTH: "↓",
        Direction.EAST: "→",
        Direction.WEST: "←",
        STAY: "stay",
        INTERACT: "interact"
    }

    @staticmethod
    def move_in_direction(point, direction):
        """
        Takes a step in the given direction and returns the new point.

        point: Tuple (x, y) representing a point in the x-y plane.
        direction: One of the Directions.
        """
        assert direction in Action.MOTION_ACTIONS
        x, y = point
        dx, dy = direction
        return (x + dx, y + dy)

    @staticmethod
    def determine_action_for_change_in_pos(old_pos, new_pos):
        """Determines an action that will enable intended transition"""
        if old_pos == new_pos:
            return Action.STAY
        new_x, new_y = new_pos
        old_x, old_y = old_pos
        direction = (new_x - old_x, new_y - old_y)
        assert direction in Direction.ALL_DIRECTIONS
        return direction

    @staticmethod
    def to_char(action):
        assert action in Action.ALL_ACTIONS
        return Action.ACTION_TO_CHAR[action]

    @staticmethod
    def joint_action_to_char(joint_action):
        assert all([a in Action.ALL_ACTIONS for a in joint_action])
        return tuple(Action.to_char(a) for a in joint_action)