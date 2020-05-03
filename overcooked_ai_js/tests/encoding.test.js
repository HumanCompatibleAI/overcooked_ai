import _ from 'lodash';
import $ from "jquery";
import { AssertionError } from 'assert';

require("overcooked-window.js");

let OvercookedMDP = window.Overcooked.OvercookedMDP;
let OvercookedGame = window.Overcooked.OvercookedGame.OvercookedGame;

let dictToState = OvercookedMDP.dictToState;

let player_colors = {};
player_colors[0] = 'green';
player_colors[1] = 'blue';

const python_lossless_encoding0_file = '../../common_tests/encoding_tests/lossless_py0.json';
const python_lossless_encoding1_file = '../../common_tests/encoding_tests/lossless_py1.json';
const python_featurization0_file = '../../common_tests/encoding_tests/featurized_py0.json';
const python_featurization1_file = '../../common_tests/encoding_tests/featurized_py1.json';
const original_trajectory_file = '../../common_tests/trajectory_tests/trajs.json';

function losslessEncodingTest(original_trajectory_file, python_encoding_file, player_index) {
	let trajectoryData = require(original_trajectory_file);
	let python_encoding = require(python_encoding_file);

	let mdp_params = trajectoryData.mdp_params[0];
	let game = new OvercookedGame({
		start_grid: [
			"XXPXX",
			"O  2O",
			"X1  X",
			"XDXSX"
		],
		container_id: "overcooked",
		assets_loc: "assets/",
		ANIMATION_DURATION: 200 * .9,
		tileSize: 80,
		COOK_TIME: mdp_params.cook_time,
        DELIVERY_REWARD: mdp_params.delivery_reward,
        num_items_for_soup: mdp_params.num_items_for_soup,
		player_colors: player_colors
	});


	let traj_states = trajectoryData.ep_observations[0];

	for (let i = 0; i < traj_states.length; i++) {
		let curr_state = dictToState(traj_states[i]);
		let [js_encoded_state_w_padding, shape] = game.mdp.lossless_state_encoding(curr_state, player_index, 1);
		let py_encoded_state = python_encoding[i];
		let js_encoded_state = js_encoded_state_w_padding[0];

		let failMessage = "PY did not match JS\nPY:" + JSON.stringify(py_encoded_state) + "\nJS:" + JSON.stringify(js_encoded_state);
		expect(js_encoded_state, failMessage).toEqual(py_encoded_state);
	}
}

test('Encodings should be the same', () => {
	losslessEncodingTest(original_trajectory_file, python_lossless_encoding0_file, 0)
	losslessEncodingTest(original_trajectory_file, python_lossless_encoding1_file, 1)
});

// This is not supported yet

// function featurizationTest(original_trajectory_file, python_encoding_file, player_index) {
// 	let trajectoryData = require(original_trajectory_file);
// 	let python_encoding = require(python_encoding_file);

// 	let mdp_params = trajectoryData.mdp_params[0];
// 	let game = new OvercookedGame({
// 		start_grid: [
// 			"XXPXX",
// 			"O  2O",
// 			"X1  X",
// 			"XDXSX"
// 		],
// 		container_id: "overcooked",
// 		assets_loc: "assets/",
// 		ANIMATION_DURATION: 200 * .9,
// 		tileSize: 80,
// 		COOK_TIME: mdp_params.cook_time,
//         DELIVERY_REWARD: mdp_params.delivery_reward,
//         num_items_for_soup: mdp_params.num_items_for_soup,
// 		player_colors: player_colors
// 	});


// 	let traj_states = trajectoryData.ep_observations[0];

// 	for (let i = 0; i < traj_states.length; i++) {
// 		let curr_state = dictToState(traj_states[i]);
// 		let [js_encoded_state_w_padding, shape] = game.mdp.featurize_state(curr_state, player_index);
// 		let py_encoded_state = python_encoding[i];
// 		let js_encoded_state = js_encoded_state_w_padding[0];

// 		let failMessage = "PY did not match JS\nPY:" + JSON.stringify(py_encoded_state) + "\nJS:" + JSON.stringify(js_encoded_state);
// 		expect(js_encoded_state, failMessage).toEqual(py_encoded_state);
// 	}
// }

// test('Encodings should be the same', () => {
// 	featurizationTest(original_trajectory_file, python_featurization0_file, 0)
// 	featurizationTest(original_trajectory_file, python_featurization1_file, 1)
// });
