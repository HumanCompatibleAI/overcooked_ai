import gym
import tqdm
import numpy as np
from overcooked_gridworld.utils import mean_and_std_err, rnd_int_uniform, rnd_uniform
from overcooked_gridworld.mdp.actions import Action
from overcooked_gridworld.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_gridworld.mdp.layout_generator import LayoutGenerator


DEFAULT_ENV_PARAMS = {
    "horizon": 400,
    "random_start_pos": False,
    "random_start_objs_p": False
}

MAX_HORIZON = 10e10

class OvercookedEnv(object):
    """An environment containing a single agent that can take actions.

    The environment keeps track of the current state of the agent, and updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(self, mdp, start_state_fn=None, horizon=MAX_HORIZON, random_start_pos=False, random_start_objs_p=0.0, debug=False):
        """
        # TODO
        start_state_fn (OvercookedState): function that returns start state given an mdp, called at each environment reset
        horizon (float): number of steps before the environment returns True to .is_done()
        """
        if isinstance(mdp, OvercookedGridworld):
            self.mdp_generator_fn = lambda: mdp
        elif callable(mdp) and isinstance(mdp(), OvercookedGridworld):
            self.mdp_generator_fn = mdp
        else:
            raise ValueError("Mdp should be either OvercookedGridworld instance or a generating function")
        
        self.horizon = horizon
        self.start_state_fn = start_state_fn
        self.random_start_pos = random_start_pos
        self.random_start_objs_p = random_start_objs_p
        self.reset()

    def get_start_state(self):
        """Assumes self.mdp has already been set"""
        if self.start_state_fn is not None:
            return self.start_state_fn()
        else:
            return self.mdp.get_start_state(
                random_start_pos=self.random_start_pos, 
                rnd_obj_prob_thresh=self.random_start_objs_p
            )    

    # @staticmethod
    # def from_mdp_params(env_config, start_state=None):
    #     # TODO: decide what the standard is here
    #     mdp = OvercookedGridworld(**env_config["mdp_params"])
    #     return OvercookedEnv(
    #         mdp,
    #         start_state_fn=None,
    #         horizon=env_config["horizon"],
    #         random_start_pos=env_config["rnd_starting_pos"],
    #         random_start_objs_p=env_config["rnd_starting_objs_p"]
    #     )

    def __repr__(self):
        return self.mdp.state_string(self.state)

    def print_state_transition(self, a_t, r_t, info):
        print("Timestep: {}\nJoint action taken: {} \t Reward: {} + shape * {} \n{}".format(
            self.t, tuple(Action.ACTION_TO_CHAR[a] for a in a_t), r_t, info["shaped_r"], self)
        )

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

    def copy(self):
        return OvercookedEnv(
            mdp=self.mdp,
            start_state_fn=self.start_state_fn,
            horizon=self.horizon,
            random_start_pos=self.random_start_pos,
            random_start_objs_p=self.random_start_objs_p
        )

    # TODO: Clean this
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
        self.mdp = self.mdp_generator_fn()
        self.state = self.get_start_state()
        self.cumulative_sparse_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0
        assert not (self.horizon >= MAX_HORIZON and self.state.order_list is None), "Should not have no order list and infinite environment horizon (no terminal states)"

    def is_done(self):
        """Returns True if the episode is over and the agent cannot act."""
        return self.t >= self.horizon or self.mdp.is_terminal(self.state)

    def execute_plan(self, start_state, action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start state in 
        the mdp and returns resulting state."""
        self.state = start_state
        done = False
        if display: print("Starting state\n{}".format(self))
        for a in action_plan:
            self.step(a)
            done = self.is_done()
            if display: print(self)
            if done: break
        successor_state = self.state
        self.reset()
        return successor_state, done

    def run_agents(self, agent_pair, include_final_state=False, display=False, display_until=np.Inf):
        """
        Trajectory returned will a list of state-action pairs (s_t, a_t, r_t, d_t), in which
        the last element will be the last state visited and a None joint action.
        Therefore, there will be t + 1 tuples in the trajectory list.
        """
        assert self.cumulative_sparse_rewards == self.cumulative_shaped_rewards == 0, "Did not reset environment before running agents"
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

            if display and self.t < display_until:
                self.print_state_transition(a_t, r_t, info)

        assert len(trajectory) == self.t, "{} vs {}".format(len(trajectory), self.t)

        # Add final state
        if include_final_state:
            trajectory.append((s_tp1, (None, None), 0, True))

        # Store useful variables before reset
        t, cumul_sparse_r, cumul_shaped_r = self.t, self.cumulative_sparse_rewards, self.cumulative_shaped_rewards

        # Reset environment and agents
        self.reset()
        agent_pair.reset()
        trajectory = np.array(trajectory)
        return trajectory, t, cumul_sparse_r, cumul_shaped_r

    def get_rollouts(self, agent_pair, num_games, display=False, processed=False, final_state=False, agent_idx=0, reward_shaping=0.0):
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
            # "layout_name": Name of the layout, added after array pre-processing
        }

        for _ in tqdm.trange(num_games):
            agent_pair.set_mdp(self.mdp)

            trajectory, time_taken, tot_rews_sparse, tot_rews_shaped = self.run_agents(agent_pair, display=display, include_final_state=final_state)

            obs, actions, rews, dones = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3]

            # TODO: Remove this?
            if processed:
                print("WE ACTUALLY USE THIS")
                # NOTE: only actions and observations for agent `agent_idx`
                # NOTE: In variable MDP envs this self.mdp call is stale as the mdp will have been reset in run_agents
                obs = np.array([self.mdp.lossless_state_encoding(state)[agent_idx] for state in obs])
                actions = np.array([np.array([Action.ACTION_TO_INDEX[joint_action[agent_idx]]]) for joint_action in actions]).astype(int)

            trajectories["ep_observations"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_returns"].append(tot_rews_sparse + tot_rews_shaped * reward_shaping)
            trajectories["ep_returns_sparse"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)

        mu, se = mean_and_std_err(trajectories["ep_returns"])
        print("Avg reward {:.2f} (std: {:.2f}, se: {:.2f}) over {} games of avg length {}".format(
            mu, np.std(trajectories["ep_returns"]), se, num_games, np.mean(trajectories["ep_lengths"]))
        )

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}
        # NOTE: In variable MDP envs this will not work
        trajectories['layout_name'] = self.mdp.layout_name
        return trajectories


class Overcooked(gym.Env):
    """Wrapper for the Env class above that is compatible with gym API
    
    The convention is that all processed observations returned are ordered in such a way
    to be standard input for the main agent policy. The index of the main agent in the mdp 
    is randomized at each reset of the environment, and is kept track of by the self.agent_idx
    attribute.
    """

    def custom_init(self, base_env, featurize_fn):
        """
        # TODO: documentation
        base_env_fn: a function that when called will return a initialized version of the
                     Env class. Can be called again to reset the environment.
        """
        self.base_env = base_env
        self.mdp = base_env.mdp
        self.featurize_fn = featurize_fn
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.reset()

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape) * max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)
        return gym.spaces.Box(high * 0, high, dtype=np.float32)

    def step(self, action):
        """
        action: 
            (self.agent_idx action, other agent action)
            is a tuple with the action of the primary and secondary agents in index format
        
        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
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
        # TODO: maybe extra data to info

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