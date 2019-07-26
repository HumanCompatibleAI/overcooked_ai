import _ from 'lodash';
import $ from "jquery";

require("overcooked-window.js");

let OvercookedMDP = window.Overcooked.OvercookedMDP;
let OvercookedGame = window.Overcooked.OvercookedGame.OvercookedGame;
let PlayerState = OvercookedMDP.PlayerState;
let ObjectState = OvercookedMDP.ObjectState;
let Direction = OvercookedMDP.Direction;
let Action = OvercookedMDP.Action;
let OvercookedGridworld = OvercookedMDP.OvercookedGridworld;
let OvercookedState = OvercookedMDP.OvercookedState;

let dictToState = OvercookedMDP.dictToState; 
let lookupActions = OvercookedMDP.lookupActions;

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
                `Trajectory State: ${r}\nJS Transition State: ${e}`,
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


let player_colors = {};
	player_colors[0] = 'green';
	player_colors[1] = 'blue';

const testTrajectoryFolder = '../common_tests/trajectory_tests/';
const fs = require('fs');
let testFiles = []; 
fs.readdirSync(testTrajectoryFolder).forEach(function(item, index) {
	// I'm sorry for this, blame the fact that require and all other file system things use different reference points
	testFiles.push('../' + testTrajectoryFolder + item)
})

// to get a test that is known to fail, run with "../../common_tests/failing_traj.json";

function trajectoryTest(trajectoryFile) {
	var trajectoryData = require(trajectoryFile);
	let game = new OvercookedGame({
		        start_grid: [
							    "XXPXX", 
							    "O  2O", 
							    "T1  T", 
							    "XDPSX"
							    ],
		        container_id: "overcooked",
		        assets_loc: "assets/",
		        ANIMATION_DURATION: 200*.9,
		        tileSize: 80,
		        COOK_TIME: 5,
		        explosion_time: Number.MAX_SAFE_INTEGER,
		        DELIVERY_REWARD: 20,
		        player_colors: player_colors
		    });

			let alignedStates = []; 
			let alignedRewards = [];
			let alignedStartingStates = []; 
			let alignedJointActions = [];

			let observations = trajectoryData.ep_observations[0];
		    let actions = trajectoryData.ep_actions[0];
		    let rewards = trajectoryData.ep_rewards[0];

		    let current_state = dictToState(observations[0]);

		    let i = 0; 
		    while (i < observations.length - 2) {
		    	let joint_action = lookupActions(actions[i]); 
		    	let trajectory_reward = rewards[i]
		    	let  [[next_transitioned_state, prob], transitioned_reward] =
		                    game.mdp.get_transition_states_and_probs({
		                        state: current_state,
		                        joint_action: joint_action
		                    }); 
		        let next_trajectory_state = dictToState(observations[i+1]);
		        alignedStartingStates.push(current_state); 
		        alignedJointActions.push(joint_action); 
		        alignedStates.push({'trajectory': next_trajectory_state,  'transitioned': next_transitioned_state}); 
		        alignedRewards.push({'trajectory': trajectory_reward, 'transitioned': transitioned_reward})
		        current_state = next_trajectory_state

		    	i += 1; 
		    }

			alignedStates.forEach(function(item, index) {
				//console.log("Reached state index " + index)
				let failMessage = "State failed to match. Started from " + JSON.stringify(alignedStartingStates[index]) + " and took actions " + alignedJointActions[index]; 
				expect(item['trajectory'], failMessage).toEqualState(item['transitioned'])
			})
			alignedRewards.forEach(function(item, index) {
				let failMessage = "Reward failed to match. Started from " + JSON.stringify(alignedStartingStates[index]) + " and took actions " + alignedJointActions[index];
				expect(item['trajectory'], failMessage).toBe(item['transitioned'])
			})
}
testFiles.forEach(function(testFile, index) {
	console.log("Testing " + testFile)
	test('States and rewards in ' + testFile + ' should be equivalent between python and JS', () => {
	trajectoryTest(testFile)	
});

})






