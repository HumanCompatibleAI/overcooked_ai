import _ from 'lodash';

// HTMLCanvasElement.prototype.getContext = () => {
//     // return whatever getContext has to return
// };

expect.extend({
  toEqualState(received, expected) {
    let pass = _.isEqual(
        JSON.stringify(received),
        JSON.stringify(expected)
    );
    if (!pass) {
        let r = JSON.stringify(received);
        let e = JSON.stringify(expected);
        return {
            message: () =>
                `Received: ${r}\nExpected: ${e}`,
            pass: false
        };
    } else {
      return {
        message: () => "",
        pass: true
      };
    }
    }
});

require("../overcooked-window.js");
//jest exposes a mock window object
let OvercookedMDP = window.Overcooked.OvercookedMDP;
let PlayerState = OvercookedMDP.PlayerState;
let ObjectState = OvercookedMDP.ObjectState;
let Direction = OvercookedMDP.Direction;
let Action = OvercookedMDP.Action;
let OvercookedGridworld = OvercookedMDP.OvercookedGridworld;
let OvercookedState = OvercookedMDP.OvercookedState;

test("Test Start Positions", () => {
    let ogw = OvercookedMDP.OvercookedGridworld.from_grid([
        'XXPXX',
        'O  2O',
        'T1  T',
        'XDPSX'
    ]);
    let s = ogw.get_start_state(['any']);
});

test("Test Transitions and Environment", () => {
    let [n, s, e, w] = Direction.CARDINAL;
    let [stay, interact] = [Direction.STAY, Action.INTERACT];
    let [P, Obj] = [PlayerState, ObjectState];
    let delivery_reward = OvercookedGridworld.DELIVERY_REWARD;

    let mdp = OvercookedMDP.OvercookedGridworld.from_grid([
        'XXPXX',
        'O  2O',
        'T1  T',
        'XDPSX'
    ]);

    let check_transition = ({state, action, expected_state, expected_reward}) => {
        let [[pred_state, prob], reward] = mdp.get_transition_states_and_probs({
            state: state,
            joint_action: action
        });
        expect(prob).toBe(1);
        expect(pred_state).toEqualState(expected_state);
        expect(reward).toBe(expected_reward);
        return pred_state
    };

    let state = mdp.get_start_state(['onion', 'any']);
    state = check_transition({
        state,
        action: [n, e],
        expected_state: new OvercookedState({
            players: [
                new P({position: [1, 1], orientation: n}),
                new P({position: [3, 1], orientation: e})
            ],
            objects: {},
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [w, interact],
        expected_state: new OvercookedState({
            players: [
                new P({position: [1, 1], orientation: w}),
                new P({
                    position: [3, 1],
                    orientation: e,
                    held_object: new Obj({name: 'onion', position: [3, 1]})
                })
            ],
            objects: {},
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [interact, w],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [1, 1],
                    orientation: w,
                    held_object: new Obj({name: 'onion', position: [1, 1]})
                }),
                new P({
                    position: [2, 1],
                    orientation: w,
                    held_object: new Obj({name: 'onion', position: [2, 1]})
                })
            ],
            objects: {},
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [e, n],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [1, 1],
                    orientation: e,
                    held_object: new Obj({name: 'onion', position: [1, 1]})
                }),
                new P({
                    position: [2, 1],
                    orientation: n,
                    held_object: new Obj({name: 'onion', position: [2, 1]})
                })
            ],
            objects: {},
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [stay, interact],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [1, 1],
                    orientation: e,
                    held_object: new Obj({name: 'onion', position: [1, 1]})
                }),
                new P({
                    position: [2, 1],
                    orientation: n
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 1, 0]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [e, e],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [2, 1],
                    orientation: e,
                    held_object: new Obj({name: 'onion', position: [2, 1]})
                }),
                new P({
                    position: [3, 1],
                    orientation: e
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 1, 0]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [n, interact],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [2, 1],
                    orientation: n,
                    held_object: new Obj({name: 'onion', position: [2, 1]})
                }),
                new P({
                    position: [3, 1],
                    orientation: e,
                    held_object: new Obj({name: 'onion', position: [3, 1]})
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 1, 0]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [interact, w],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [2, 1],
                    orientation: n
                }),
                new P({
                    position: [3, 1],
                    orientation: w,
                    held_object: new Obj({name: 'onion', position: [3, 1]})
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 2, 0]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [w, w],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [1, 1],
                    orientation: w
                }),
                new P({
                    position: [2, 1],
                    orientation: w,
                    held_object: new Obj({name: 'onion', position: [2, 1]})
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 2, 0]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [s, n],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [1, 2],
                    orientation: s
                }),
                new P({
                    position: [2, 1],
                    orientation: n,
                    held_object: new Obj({name: 'onion', position: [2, 1]})
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 2, 0]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [interact, interact],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [1, 2],
                    orientation: s,
                    held_object: new Obj({name: 'dish', position: [1, 2]})
                }),
                new P({
                    position: [2, 1],
                    orientation: n
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 3, 1]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [e, s],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [1, 2],
                    orientation: e,
                    held_object: new Obj({name: 'dish', position: [1, 2]})
                }),
                new P({
                    position: [2, 1],
                    orientation: s
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 3, 2]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [e, interact],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [2, 2],
                    orientation: e,
                    held_object: new Obj({name: 'dish', position: [2, 2]})
                }),
                new P({
                    position: [2, 1],
                    orientation: s
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 3, 3]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [n, e],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [2, 1],
                    orientation: n,
                    held_object: new Obj({name: 'dish', position: [2, 1]})
                }),
                new P({
                    position: [3, 1],
                    orientation: e
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 3, 4]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });
    state = check_transition({
        state,
        action: [interact, interact],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [2, 1],
                    orientation: n,
                    held_object: new Obj({name: 'dish', position: [2, 1]})
                }),
                new P({
                    position: [3, 1],
                    orientation: e,
                    held_object: new Obj({name: 'onion', position: [3, 1]})
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 3, 5]
                })
            },
            order_list: ['onion', 'any']
        }),
        expected_reward: 0
    });

    let atraj = [
        [stay, stay], [interact, interact],
        [e, w], [e, s], [s, interact], [s, w], [interact, n], [e, interact],
        [interact, s], [w, w], [n, interact], [interact, interact],
        [s, interact], [interact, interact]
    ];
    let traj = [];
    for (let i = 0; i < (atraj.length - 1); i++) {
        let action = atraj[i];
        let [[pred_state, prob], reward] = mdp.get_transition_states_and_probs({
            state: state,
            joint_action: action
        });
        let step = {
            state: state,
            expected_state: pred_state,
            expected_reward: reward,
            action: action
        }
        traj.push(step);
        state = pred_state;
    }

    check_transition({
        state,
        action: [interact, interact],
        expected_state: new OvercookedState({
            players: [
                new P({
                    position: [2, 2],
                    orientation: s
                }),
                new P({
                    position: [1, 2],
                    orientation: w,
                    held_object: new Obj({name: 'tomato', position: [1, 2]})
                })
            ],
            objects: {
                '2,0': new Obj({
                    name: 'soup',
                    position: [2, 0],
                    state: ['onion', 1, 0]
                }),
                '2,3': new Obj({
                    name: 'soup',
                    position: [2, 3],
                    state: ['tomato', 1, 0]
                }),
            },
            order_list: ['any']
        }),
        expected_reward: 0
    });
})