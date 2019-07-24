import _ from 'lodash';
import assert from 'assert';

export class Direction {
    static move_in_direction(point, direction) {
        /*Takes a step in the given direction and returns the new point.
        point: Tuple (x, y) representing a point in the x-y plane.
        direction: One of the Directions, except not Direction.STAY or
        Direction.SELF_LOOP.*/
        let [x, y] = point;
        let [dx, dy] = direction;
        return [x + dx, y + dy]
    }
}
Direction.NORTH = [0, -1];
Direction.SOUTH = [0, 1];
Direction.EAST = [1, 0];
Direction.WEST = [-1, 0];
Direction.STAY = [0, 0];
Direction.CARDINAL = [
    Direction.NORTH, Direction.SOUTH,
    Direction.EAST, Direction.WEST
];
Direction.INDEX_TO_DIRECTION = [
    Direction.NORTH, Direction.SOUTH,
    Direction.EAST, Direction.WEST, Direction.STAY
];
Direction.DIRECTION_TO_INDEX =
    _.fromPairs(Direction.INDEX_TO_DIRECTION.map((d, i) => {
        return [d, i]
    }));
Direction.ALL_DIRECTIONS = Direction.INDEX_TO_DIRECTION;
Direction.OPPOSITE_DIRECTIONS = _.fromPairs([
    [Direction.NORTH, Direction.SOUTH],
    [Direction.SOUTH, Direction.NORTH],
    [Direction.EAST, Direction.WEST],
    [Direction.WEST, Direction.EAST]
]);
Direction.DIRECTION_TO_NAME = {
    '0,-1': 'NORTH',
    '0,1': 'SOUTH',
    '1,0': 'EAST',
    '-1,0': 'WEST'
}

export class Action {}
Action.INTERACT = "INTERACT";
Action.INDEX_TO_ACTION = _.clone(Direction.INDEX_TO_DIRECTION);
Action.INDEX_TO_ACTION.push(Action.INTERACT);
Action.ACTION_TO_INDEX = _.fromPairs(Action.INDEX_TO_ACTION.map((a, i) => [a, i]));
Action.ALL_ACTIONS = Action.INDEX_TO_ACTION;
Action.MOTION_ACTIONS = Direction.INDEX_TO_DIRECTION;

export class PlayerState {
    constructor({
        position,
        orientation,
        held_object = undefined
    }) {
        assert(Array.isArray(position));
        // assert(orientation in Direction.ALL_DIRECTIONS)
        if (typeof(held_object) !== "undefined") {
            // assert(ObjectState.is_object(held_object));
            assert(_.isEqual(held_object.position, position));
        }
        this.position = position;
        this.orientation = orientation;
        this.held_object = held_object;
    }

    // pos_and_or () {}
    has_object () {
        return typeof(this.held_object) !== 'undefined'
    }
    get_object () {
        assert(this.has_object());
        return this.held_object
    }
    set_object (obj) {
        assert(!this.has_object());
        obj.position = this.position;
        this.held_object = obj;
    }
    remove_object () {
        assert(this.has_object());
        let obj = this.held_object;
        this.held_object = undefined;
        return obj
    }
    update_pos_and_or (new_position, new_orientation) {
        assert(Array.isArray(new_position));
        this.position = new_position;
        this.orientation = new_orientation;
        if (this.has_object()) {
            this.get_object().position = new_position;
        }
    }
    deepcopy () {
        let new_obj;
        if (this.has_object()) {
            new_obj = this.held_object.deepcopy();
        }
        return new PlayerState({
            position: this.position,
            orientation: this.orientation,
            held_object: new_obj
        });
    }
}

export class ObjectState {
    constructor ({
        name,
        position,
        state
    }) {
        //TODO: Use numbers instead of strings for name, and have a dictionary
        //to convert to and from
        this.name = name;
        this.position = position;
        if (name === 'soup') {
            assert(state.length === OvercookedGridworld.num_items_for_soup)
        }
        this.state = state;
    }
    is_valid () {
        if (this.name === 'onion') {
            return typeof(this.state) === 'undefined'
        }
        if (this.name === 'tomato') {
            return typeof(this.state) === 'undefined'
        }
        if (this.name === 'dish') {
            return typeof(this.state) === 'undefined'
        }
        if (this.name === 'soup') {
            let [soup_type, num_items, cook_time] = this.state;
            let valid_soup_type = _.includes(ObjectState.SOUP_TYPES, soup_type);
            let valid_item_num = (1 <= num_items) && (num_items <= 3);
            let valid_cook_time = 0 <= cook_time;
            return valid_soup_type && valid_item_num && valid_cook_time
        }
        return false
    }
    deepcopy () {
        return new ObjectState({
            name: this.name,
            position: this.position,
            state: this.state
        })
    }
}
ObjectState.SOUP_TYPES = ['onion', 'tomato'];

export class OvercookedState {
    constructor ({
        players,
        objects,
        order_list = [],
        pot_explosion=false
    }) {
        // Represents a state in Overcooked.
        // players: List of PlayerStates.
        // objects: Dictionary mapping positions (x, y) to ObjectStates. Does NOT
        //     include objects held by players.
        // Order is important for players but not for objects
        for (let pos in objects) {
            if (!objects.hasOwnProperty(pos)) {continue}
            let obj = objects[pos];
            assert(
                _.isEqual(String(obj.position), String(pos)),
                `${String(obj.position)} !== ${String(pos)}`
                );
        }
        this.players = players.map((p) => {return p.deepcopy()});
        this.objects = objects;
        // assert all([o in OvercookedGridworld.ORDER_TYPES for o in order_list])
        this.order_list = order_list;
        this.pot_explosion = pot_explosion;
    }

    static from_object(obj) {
        obj['players'] = obj['players'].map((p) => {
            if (typeof(p.held_object) !== 'undefined') {
                p.held_object = new ObjectState(p.held_object);
            }
            return p
        });
        obj['players'] = obj['players'].map((p) => {return new PlayerState(p)});
        obj['objects'] = _.mapValues(obj['objects'], (o) => {return new ObjectState(o)});
        return new OvercookedState(obj);
    }

    player_positions () {
        return this.players.map((p) => {return p.position})
    }
    player_orientations () {
        return this.players.map((p) => {return p.orientation})
    }
    has_object (pos) {
        return _.includes(_.keys(this.objects).map(String), String(pos));
    }
    get_object (pos) {
        assert(this.has_object(pos));
        return this.objects[pos]
    }
    add_object (obj, pos) {
        if (typeof(pos) === 'undefined') {
            pos = obj.position
        }
        assert(!this.has_object(pos));
        obj.position = pos;
        this.objects[pos] = obj;
    }
    remove_object(pos) {
        assert(this.has_object(pos));
        let obj = this.objects[pos];
        delete this.objects[pos];
        return obj
    }
    deepcopy() {
        return new OvercookedState({
            players: this.players.map((p) => p.deepcopy()),
            objects: _.fromPairs(_.map(this.objects, (obj, pos) => {
                return [pos, obj.deepcopy()]
            })),
            order_list: this.order_list.map((i) => i),
            pot_explosion: this.pot_explosion
        })
    }

    //static methods
    static from_players_pos_and_or(players_pos_and_or, order_list) {
        return new OvercookedState({
            players: players_pos_and_or.map((params) => {
                params = {position: params[0], orientation: params[1]};
                return new PlayerState(params)
            }),
            objects: {},
            order_list: order_list
        })
    }
    static from_player_positions(player_positions, order_list) {
        let dummy_pos_and_or = player_positions.map((pos) => [pos, Direction.NORTH]);
        return OvercookedState.from_players_pos_and_or(dummy_pos_and_or, order_list);
    }
}

export function dictToState(state_dict) {
    let object_dict = {}
    if (state_dict['objects'].length > 0) {
        state_dict['objects'].forEach(function (item, index) {
            object_dict[item['position']] = dictToObjectState(item)
            })
        }
    state_dict['objects'] = object_dict

    return new OvercookedState({
        players: [dictToPlayerState(state_dict['players'][0]), dictToPlayerState(state_dict['players'][1])], 
        objects: state_dict['objects'], 
        order_list: state_dict['order_list']
    })
}

export function dictToPlayerState(player_dict) {
    if (player_dict['held_object'] == null) {
        player_dict['held_object'] = undefined
    }
    else {
        player_dict['held_object'] = dictToObjectState(player_dict['held_object'])
     }
     return new PlayerState({
        position: player_dict['position'], 
        orientation: player_dict['orientation'], 
        held_object: player_dict['held_object']
     })
    }

export function dictToObjectState(object_dict) {
    if (object_dict['state'] == null) {
        object_dict['state'] = undefined;
    }
    return new ObjectState(
        {name: object_dict['name'],
        position: object_dict['position'], 
        state: object_dict['state']
        })
}


export function lookupActions(actions_arr) {
    let actions = []; 
    actions_arr.forEach(function (item, index) {
        if (item == "interact") {
            item = Action.INTERACT; 
        }
        if (arraysEqual(Direction.STAY, item) || item == "stay") {
            item = Direction.STAY;
        }
        actions.push(item);
    }
        )
    return actions;
    
}

function arraysEqual(a, b) {
    // Stolen from https://stackoverflow.com/questions/3115982/how-to-check-if-two-arrays-are-equal-with-javascript
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; ++i) {
    if (a[i] !== b[i]) return false;
  }
  return true;

}
/*

    Main MDP Class

 */

export class OvercookedGridworld {
    constructor ({
        terrain,
        player_positions,
        explosion_time=Number.MAX_SAFE_INTEGER,
        COOK_TIME = OvercookedGridworld.COOK_TIME,
        DELIVERY_REWARD = OvercookedGridworld.DELIVERY_REWARD,
        num_items_for_soup = OvercookedGridworld.num_items_for_soup,
        always_serve = false //when this is set to a string, its what's always served
    }) {
        this.terrain_mtx = terrain;
        this.terrain_pos_dict = this._get_terrain_type_pos_dict();
        this.start_player_positions = player_positions;
        this.explosion_time = explosion_time;
        this.COOK_TIME = COOK_TIME
        this.DELIVERY_REWARD = DELIVERY_REWARD
        this.always_serve = always_serve;
        this.num_items_for_soup = num_items_for_soup;
    }

    get_start_state (order_list) {
        if (this.always_serve) {
            order_list = [this.always_serve]
        }
        return OvercookedState.from_player_positions(
            this.start_player_positions,
            order_list
        );
    }

    get_actions (state) {
        /* Returns the list of lists of valid actions for 'state'.
        The ith element of the list is the list of valid actions that player i
        can take.
        Note that you can request moves into terrain, which are equivalent to
        STAY. The order in which actions are returned is guaranteed to be
        deterministic, in order to allow agents to implement deterministic
        behavior. */
        this._check_valid_state(state);
        return state.players.map((p, i) => {return this._get_player_actions(state, i)})
    }

    _get_player_actions (state, player_num) {
        return Action.ALL_ACTIONS
    }

    is_terminal ({state}) {
        return (state.order_list.length === 0) || (state.pot_explosion)
    }

    get_transition_states_and_probs ({state, joint_action}) {
        /*Gets information about possible transitions for the action.
        Returns list of (next_state, prob) pairs representing the states
        reachable from 'state' by taking 'action' along with their transition
        probabilities.*/
        let action_sets = this.get_actions(state);
        for (let pi = 0; pi < state.players.length; pi++) {
            let [player, action, action_set] =
                [state.players[pi], joint_action[pi], action_sets[pi]];
            assert(_.includes(action_set.map(String), String(action)))
        }
        let new_state = state.deepcopy();

        assert(_.isEqual(new_state.objects, state.objects),
            `${JSON.stringify(new_state.objects)} !== ${JSON.stringify(state.objects)}`);

        //resolve interacts first
        let reward = this.resolve_interacts(new_state, joint_action);

        assert(_.isEqual(new_state.player_positions().map(String), state.player_positions().map(String)));
        assert(_.isEqual(new_state.player_orientations().map(String), state.player_orientations().map(String)));


        //resolve player movements
        this.resolve_movement(new_state, joint_action);

        //finally, environment effects
        this.step_environment_effects(new_state);
        return [[new_state, 1.0], reward]
    }

    resolve_interacts (new_state, joint_action) {
        /*TODO: Currently if two players both interact with a terrain, we
        resolve player 1's interact first and then player 2's, without doing
        anything like collision checking. Is this okay?*/
        let reward = 0;
        for (let pi = 0; pi < new_state.players.length; pi++) {
            let [player, action] = [new_state.players[pi], joint_action[pi]];
            if (action !== Action.INTERACT) {
                continue
            }

            let [pos, o] = [player.position, player.orientation];
            let i_pos = Direction.move_in_direction(pos, o);
            let terrain_type = this.get_terrain_type_at(i_pos);

            if (terrain_type === 'X') {
                if (player.has_object() && !new_state.has_object(i_pos)) {
                    new_state.add_object(player.remove_object(), i_pos);
                }
                else if (!player.has_object() && new_state.has_object(i_pos)) {
                    player.set_object(new_state.remove_object(i_pos));
                }
            }
            else if (!player.has_object()) {
                if (terrain_type === 'O') {
                    player.set_object(new ObjectState({name: 'onion', position: pos}));
                }
                else if (terrain_type === 'T') {
                    player.set_object(new ObjectState({name: 'tomato', position: pos}));
                }
                else if (terrain_type === 'D') {
                    player.set_object(new ObjectState({name: 'dish', position: pos}));
                }
            }
            else if (player.has_object()) {
                if (terrain_type === 'P') {
                    if ((player.get_object().name === 'dish') && (new_state.has_object(i_pos))) {
                        let obj = new_state.get_object(i_pos);
                        assert(obj.name === 'soup', "Object in pot was not soup");
                        let [temp, num_items, cook_time] = obj.state;
                        if ((num_items === this.num_items_for_soup) && (cook_time >= this.COOK_TIME)) {
                            player.remove_object(); //turnt he dish into the soup
                            player.set_object(new_state.remove_object(i_pos));
                        }
                    }
                    else if (_.includes(['onion', 'tomato'], player.get_object().name)) {
                        let item_type = player.get_object().name;
                        if (!new_state.has_object(i_pos)) {
                            player.remove_object();
                            new_state.add_object(
                                new ObjectState({
                                    name: 'soup',
                                    position: i_pos,
                                    state: [item_type, 1, 0]
                                }),
                                i_pos);
                        }
                        else {
                            let obj = new_state.get_object(i_pos);
                            // assert(obj.name === 'soup', "Object in pot was not soup")
                            let [soup_type, num_items, cook_time] = obj.state;
                            if ((num_items < this.num_items_for_soup) && soup_type === item_type) {
                                player.remove_object();
                                obj.state = [soup_type, num_items + 1, 0];
                            }
                        }
                    }
                }
                else if (terrain_type === 'S') {
                    let obj = player.get_object();
                    if (obj.name === 'soup') {
                        let [soup_type, num_items, cook_time] = obj.state;
                        assert(_.includes(ObjectState.SOUP_TYPES, soup_type));
                        assert(num_items === this.num_items_for_soup &&
                               cook_time >= this.COOK_TIME &&
                               cook_time < this.explosion_time);
                        player.remove_object();

                        //If the delivered soup is the one currently required
                        let current_order = new_state.order_list[0];
                        if ((current_order === 'any') || (soup_type === current_order)) {
                            new_state.order_list = new_state.order_list.slice(1);
                            reward += this.DELIVERY_REWARD;
                        }
                        if (this.always_serve) {
                            new_state.order_list = [this.always_serve, ]
                        }
                    }
                }
            }
        }
        return reward
    }

    resolve_movement (state, joint_action){
        // Resolve player movement and deal with possible collisions
        let [new_positions, new_orientations] =
            this.compute_new_positions_and_orientations({
                old_player_states: state.players,
                joint_action
            });
        for (let pi = 0; pi < state.players.length; pi++) {
            let [player_state, new_pos, new_o] =
                [state.players[pi], new_positions[pi], new_orientations[pi]];
            player_state.update_pos_and_or(new_pos, new_o);
        }
    }

    compute_new_positions_and_orientations ({old_player_states, joint_action}) {
        //Compute new positions and orientations ignoring collisions
        let new_positions = [];
        let old_positions = [];
        let new_orientations = [];
        for (let pi = 0; pi < old_player_states.length; pi++) {
            let p = old_player_states[pi];
            let a = joint_action[pi];
            let [new_pos, new_o] = this._move_if_direction(p.position, p.orientation, a);
            new_positions.push(new_pos);
            old_positions.push(p.position);
            new_orientations.push(new_o);
        }
        new_positions = this._handle_collisions(old_positions, new_positions);
        return [new_positions, new_orientations]
    }

    _handle_collisions(old_positions, new_positions) {
        //only 2 players for nwo
        if (this.is_collision(old_positions, new_positions)) {
            return old_positions
        }
        return new_positions
    }

    is_collision(old_positions, new_positions) {
        let [p1_old, p2_old] = old_positions;
        let [p1_new, p2_new] = new_positions;
        if (_.isEqual(p1_new, p2_new)) {
            return true
        }
        else if (_.isEqual(p1_new, p2_old) && _.isEqual(p1_old, p2_new)) {
            return true
        }
        return false
    }

    _move_if_direction(position, orientation, action) {
        if (!_.includes(Action.MOTION_ACTIONS.map(String), String(action))) {
            return [position, orientation]
        }
        let new_pos = Direction.move_in_direction(position, action);
        let new_orientation;

        if (_.isEqual(Direction.STAY, action)) {
            new_orientation = orientation;
        }
        else {
            new_orientation = action;
        }

        if (!_.includes(this.get_valid_player_positions().map(String), String(new_pos))) {
            return [position, new_orientation]
        }

        return [new_pos, new_orientation]
    }

    get_valid_player_positions () {
        return this.terrain_pos_dict[' '];
    }

    _get_terrain_type_pos_dict () {
        let pos_dict = {};
        for (let y = 0; y < this.terrain_mtx.length; y++) {
            for (let x = 0; x < this.terrain_mtx[y].length; x++) {
                let ttype = this.terrain_mtx[y][x];
                if (!pos_dict.hasOwnProperty(ttype)) {
                    pos_dict[ttype] = [];
                }
                pos_dict[ttype].push([x, y]);
            }
        }
        return pos_dict
    }

    _check_valid_state (state) {
        /*Checks that the state is valid.
        Conditions checked:
        - Players are on free spaces, not terrain
        - Held objects have the same position as the player holding them
        - Non-held objects are on terrain
        - No two players or non-held objects occupy the same position
        - Objects have a valid state (eg. no pot with 4 onions)*/
        let all_objects = _.values(state.objects);
        for (let pi = 0; pi < state.players.length; pi++) {
            let pstate = state.players[pi];
            //Check that players are not on terrain
            let pos = pstate.position;
            assert(_.includes(this.get_valid_player_positions().map(String), String(pos)),
                    JSON.stringify(this.get_valid_player_positions())+" "+pos);

            //check that held obj have the same position
            if (pstate.has_object()) {
                all_objects.push(pstate.held_object);
                assert(_.isEqual(pstate.held_object.position, pstate.position));
            }
        }

        for (let obj_pos in state.objects) {
            if (!state.objects.hasOwnProperty(obj_pos)) {continue}
            obj_pos = str_to_array(obj_pos);
            let obj_state = state.objects[obj_pos];
            //check that the hash key position agrees with the position stored in the object state
            assert(
                _.isEqual(obj_state.position, obj_pos),
                `${obj_state.position}, ${obj_pos}`
            );
            //check that the non-held obj are on terrain
            assert(!_.isEqual(this.get_terrain_type_at(obj_pos), ' '));
        }

        //check that players and non-held objects don't overlap
        let all_pos = state.players.map((p) => p.position);
        let all_objpos = _.values(state.objects).map((o) => o.position);
        all_pos = _.concat(all_pos, all_objpos);
        assert(all_pos.length === _.uniq(all_pos).length, "Overlapping players or objects");

        //check that objects have a valid state
        all_objects.map((o) => assert(o.is_valid(), "Invalid Obj: "+JSON.stringify(o)))
    }

    get_terrain_type_at (pos) {
        let [x, y] = pos;
        return this.terrain_mtx[y][x];
    }

    step_environment_effects(state) {
        for (let pos in state.objects) {
            if (!state.objects.hasOwnProperty(pos)) {
                continue
            }
            let obj = state.objects[pos];
            if (obj.name === 'soup') {
                let [x, y] = obj.position;
                let [soup_type, num_items, cook_time] = obj.state;
                if (
                        (this.terrain_mtx[y][x] === 'P') &&
                        (cook_time < this.explosion_time) &&
                        (num_items === this.num_items_for_soup)) {
                    obj.state = [soup_type, num_items, Math.min(cook_time + 1, this.COOK_TIME)];
                }
                if ((obj.state[2] === this.explosion_time) && (num_items === this.num_items_for_soup)) {
                    state.pot_explosion = true
                }
            }
        }
    }

    static from_grid (grid, params) {
        grid = grid.map((r) => _.map(r, (c) => c));
        let player_pos = [null, null];
        for (let y = 0; y < grid.length; y++) {
            for (let x = 0; x < grid[0].length; x++) {
                let c = grid[y][x];
                if (_.includes(['1', '2'], c)) {
                    grid[y][x] = ' ';
                    assert(player_pos[parseInt(c)-1] === null, "Duplicate player in grid");
                    player_pos[parseInt(c)-1] = [x, y];
                }
            }
        }
        assert(_.every(player_pos), 'A player was missing');
        params = typeof(params) === 'undefined' ? {} : params;
        params = Object.assign({}, params, {
            terrain: grid,
            player_positions: player_pos
        });
        return new OvercookedGridworld(params)
    }
}
OvercookedGridworld.COOK_TIME = 5;
OvercookedGridworld.DELIVERY_REWARD = 20;
OvercookedGridworld.ORDER_TYPES = ObjectState.SOUP_TYPES + ['any'];
OvercookedGridworld.num_items_for_soup = 3; 

let str_to_array = (val) => {
    if (Array.isArray(val)) {
        return val
    }
    return val.split(',').map((i) => parseInt(i))
};

// let assert = function (bool, msg) {
//     if (typeof(msg) === 'undefined') {
//         msg = "Assert Failed";
//     }
//     if (bool) {
//         return
//     }
//     else {
//         console.log(msg);
//         console.trace();
//     }
// };