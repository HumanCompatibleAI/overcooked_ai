import tqdm
import random
import numpy as np
from overcooked_gridworld.mdp.overcooked_mdp import OvercookedGridworld, Action, Direction

class OvercookedEnv(object):
    """An environment containing a single agent that can take actions.

    The environment keeps track of the current state of the agent, and updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(self, mdp, start_state_fn=None, horizon=float('inf'), random_start_pos=False, random_start_objs=False):
        """
        start_state_fn (OvercookedState): function that returns start state, called at each environment reset
        horizon (float): number of steps before the environment returns True to .is_done()
        """
        self.mdp = mdp
        self.horizon = horizon
        self.random_start_pos = random_start_pos
        self.random_start_objs = random_start_objs

        if start_state_fn is None:
            self.start_state_fn = lambda: self.mdp.get_start_state(random_start_pos=random_start_pos, random_start_objs=random_start_objs)
        else:
            self.start_state_fn = start_state_fn

        self.reset()

    @staticmethod
    def from_config(env_config, start_state=None):
        mdp = OvercookedGridworld(env_config["mdp_config"])
        return OvercookedEnv(
            mdp,
            start_state_fn=None,
            horizon=env_config["env_horizon"],
            random_start_pos=env_config["rnd_starting_position"],
            random_start_objs=env_config["rnd_starting_objs"]
        )

    def __repr__(self):
        return self.mdp.state_string(self.state)

    def copy(self):
        return OvercookedEnv(
            mdp=self.mdp,
            start_state_fn=self.start_state_fn,
            horizon=self.horizon,
            random_start_pos=self.random_start_pos,
            random_start_objs=self.random_start_objs
        )

    # def get_current_state(self):
    #     return self.state

    # def get_actions(self, state):
    #     return self.mdp.get_actions(state)

    def step(self, joint_action):
        """Performs a joint action, updating the state and providing a reward.
        
        sparse_reward is the environment sparse reward
        reward_shaping is the component of the reward that is shaped
        """
        assert not self.is_done()
        next_state, sparse_reward, reward_shaping = self.mdp.get_transition_states_and_probs(self.state, joint_action)
        self.cumulative_sparse_rewards += sparse_reward
        self.cumulative_shaped_rewards += reward_shaping
        self.state = next_state
        self.t += 1
        done = self.is_done()
        info = {'shaped_r': reward_shaping}
        if done:
            info['episode'] = {
                'ep_sparse_r': self.cumulative_sparse_rewards,
                'ep_shaped_r': self.cumulative_shaped_rewards,
                'ep_length': self.t
            }
        return (next_state, sparse_reward, done, info)

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        self.state = self.start_state_fn()
        self.cumulative_sparse_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0

    def is_done(self):
        """Returns True if the episode is over and the agent cannot act."""
        return self.t >= self.horizon or self.mdp.is_terminal(self.state)

    @staticmethod
    def execute_plan(mdp, start_state, action_plan, display=False, horizon=np.Inf):
        """Executes action_plan from start_state in mdp and returns resulting state."""
        env = OvercookedEnv(mdp, lambda: start_state, horizon=horizon)
        env.state = start_state
        if display: print("Starting state\n{}".format(env))
        for a in action_plan:
            env.step(a)
            if display: print(env)
            if env.is_done():
                break
        successor_state = env.state
        return successor_state, env.is_done()

    def run_agents(self, agent_pair, display=False, displayEnd=False, final_state=False, joint_actions=False, displayUntil=np.Inf):
        """
        Trajectory returned will a list of state-action pairs (s_t, a_t, r_t, d_t), in which
        the last element will be the last state visited and a None joint action.
        Therefore, there will be t + 1 tuples in the trajectory list.
        """
        trajectory = []
        done = False

        if display: print(self)
        while not done:
            s_t = self.state
            a_t = agent_pair.joint_action(s_t)

            # Break if either agent is out of actions
            if any([a is None for a in a_t]):
                break

            s_tp1, r_t, done, info = self.step(a_t)
            trajectory.append((s_t, a_t, r_t, done))

            if display or (done and displayEnd):
                # Cutting amount to display
                if self.t < displayUntil:
                    print("Timestep: {}\nJoint action: {} \t Reward: {} + shape * {} \n{}".format(self.t, a_t, r_t, info["dense_r"], self))

        # Add final state
        # TODO: Clean up
        if final_state:
            trajectory.append((s_tp1, (None, None), 0, True))
            assert len(trajectory) == self.t + 1, "{} vs {}".format(len(trajectory), self.t)
        else:
            assert len(trajectory) == self.t, "{} vs {}".format(len(trajectory), self.t)
        
        time_taken, tot_sparse_rewards, tot_shaped_rewards = self.t, self.cumulative_sparse_rewards, self.cumulative_shaped_rewards

        # Reset environment
        self.reset()
        agent_pair.reset()
        trajectory = np.array(trajectory)
        return trajectory, time_taken, tot_sparse_rewards, tot_shaped_rewards

    def get_rollouts(self, agent_pair, num_games, display=False, displayEnd=False, processed=False, final_state=False, agent_idx=0, reward_shaping=0.0):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed 
        trajectories.

        Only returns the trajectories for one of the agents (the actions _that_ agent took), 
        namely the one indicated by `agent_idx`.

        Returning excessive information to be able to convert trajectories to any required format 
        (baselines, stable_baselines, etc)
        """
        trajectories = {
            # With shape (n_timesteps, game_len), where game_len might vary across games:
            "ep_observations": [],
            "ep_actions": [],
            "ep_rewards": [], # Individual (dense) reward values
            "ep_dones": [], # Individual done values

            # With shape (n_episodes, ):
            "ep_returns": [], # Sum of dense rewards across each episode
            "ep_returns_sparse": [], # Sum of sparse rewards across each episode
            "ep_lengths": [], # Lengths of each episode
            
            # With shape (1, ):
            # "layout_name", included at the end
        }

        for _ in tqdm.trange(num_games):
            agent_pair.set_mdp(self.mdp)

            trajectory, time_taken, tot_rews_sparse, tot_rews_shaped = self.run_agents(agent_pair, display=display, displayEnd=displayEnd, final_state=final_state)

            obs, actions, rews, dones = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3]
            if processed:
                # NOTE: only actions and observations for agent `agent_idx`
                obs = np.array([self.mdp.preprocess_observation(state)[agent_idx] for state in obs])
                actions = np.array([np.array([Action.ACTION_TO_INDEX[joint_action[agent_idx]]]) for joint_action in actions]).astype(int)

            trajectories["ep_observations"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_returns"].append(tot_rews_sparse + tot_rews_shaped * reward_shaping)
            trajectories["ep_returns_sparse"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)

        print("Avg reward {} (std: {}) over {} games of avg length {}".format(
            np.mean(trajectories["ep_returns"]), np.std(trajectories["ep_returns"]), num_games, np.mean(trajectories["ep_lengths"])))

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}
        trajectories['layout_name'] = self.mdp.layout_name
        return trajectories

    def display_states(self, *states):
        old_state = self.state

        for s in states:
            self.state = s
            print(self)
        
        self.state = old_state

    @staticmethod
    def print_state(mdp, s):
        e = OvercookedEnv(mdp, s)
        print(e)

class VariableOvercookedEnv(OvercookedEnv):
    """Wrapper for Env class which changes mdp at each reset from a mdp_generator function"""

    def __init__(self, mdp_generator_fn, horizon=float('inf')):
        """
        start_state (OvercookedState): what the environemt resets to when calling reset
        horizon (float): number of steps before the environment returns True to .is_done()
        """
        self.mdp_generator_fn = mdp_generator_fn
        self.horizon = horizon
        self.reset()

    def reset(self):
        self.mdp = self.mdp_generator_fn()
        self.start_state = self.mdp.get_start_state()
        self.state = self.start_state
        self.cumulative_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0

import gym
from gym import spaces
class Overcooked(gym.Env):
    """Wrapper for the Env class above that is compatible with gym API
    
    The convention is that all processed observations returned are ordered in such a way
    to be standard input for the main agent policy. The index of the main agent in the mdp 
    is randomized at each reset of the environment, and is kept track of by the self.agent_idx
    attribute.
    
    One can use the switch_player function to change the observation to be in standard 
    format for the secondary agent policy.
    """

    def custom_init(self, base_env, joint_actions=False, featurize_fn=None):
        """
        base_env_fn: a function that when called will return a initialized version of the
                     Env class. Can be called again to reset the environment.
        """
        self.base_env = base_env
        self.joint_actions = joint_actions

        dummy_state = self.base_env.mdp.get_start_state()

        if featurize_fn is None:
            self.featurize_fn = self.base_env.mdp.preprocess_observation
            obs_shape = self.base_env.mdp.preprocess_observation(dummy_state)[0].shape
            high = np.ones(obs_shape) * 5
        else:
            self.featurize_fn = featurize_fn
            obs_shape = featurize_fn(dummy_state)[0].shape
            high = np.ones(obs_shape) * 10 # NOTE: arbitrary right now

        self.observation_space = spaces.Box(high * 0, high, dtype=np.float32)

        if self.joint_actions:
            self.action_space = spaces.Discrete(len(Action.ALL_ACTIONS)**2)
        else:
            self.action_space = spaces.Discrete(len(Action.ALL_ACTIONS))
        self.reset()

    def step(self, action):
        """
        action: 
            (self.agent_idx action, other agent action)
            is a tuple with the action of the primary and secondary agents in index format
        
        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        if self.joint_actions:
            action = Action.INDEX_TO_ACTION_INDEX_PAIRS[action]
        else:
            assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        if self.agent_idx == 0:
            joint_action = (agent_action, other_agent_action)
        else:
            joint_action = (other_agent_action, agent_action)

        next_state, reward, done, info = self.base_env.step(joint_action)
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)
        return (both_agents_ob, (next_state, 1 - self.agent_idx)), reward, done, info

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to 
        complete the task starting at either of the hardcoded positions.
        """
        # If fixed map, reset it, otherwise, generate new one at each reset
        self.base_env.reset()
        self.agent_idx = np.random.choice([0, 1])
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)
        return (both_agents_ob, (self.base_env.state, 1 - self.agent_idx))

    def render(self, mode='human', close=False):
        pass