import json, copy
import numpy as np
from IPython.display import display

from overcooked_ai_py.utils import save_pickle, load_pickle, cumulative_rewards_from_rew_list, save_as_json, load_from_json, mean_and_std_err, append_dictionaries, merge_dictionaries, rm_idx_from_dict, take_indexes_from_dict
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.agents.agent import AgentPair, CoupledPlanningAgent, RandomAgent, GreedyHumanModel
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


class AgentEvaluator(object):
    """
    Class used to get rollouts and evaluate performance of various types of agents.

    TODO: This class currently only fully supports fixed mdps, or variable mdps that can be created with the LayoutGenerator class,
    but might break with other types of variable mdps. Some methods currently assume that the AgentEvaluator can be reconstructed
    from loaded params (which must be pickleable). However, some custom start_state_fns or mdp_generating_fns will not be easily
    pickleable. We should think about possible improvements/what makes most sense to do here.
    """

    def __init__(self, mdp_params_lst, env_params={}, mdp_fn_params_lst=None, force_compute=False, mlp_params_lst=None, debug=False):
        """
        mdp_params_lst (list): a list of params for creation of an OvercookedGridworld instance
            through the `from_layout_name` method
        env_params (dict): params for creation of an OvercookedEnv
        mdp_fn_params (list): a list of params to setup random MDP generation
        force_compute (bool): whether should re-compute MediumLevelPlanner although matching file is found
        mlp_params (dict): params for MediumLevelPlanner
        """
        assert type(mdp_params_lst) is list and len(mdp_params_lst) > 0, \
            "please make sure mdp_params_lst is a list. " \
            "If not, please see where the function is called and make a fake list"
        assert all([type(mdp_params_i) is dict for mdp_params_i in mdp_params_lst]), \
            "all mdp_params must be a dictionary"

        # The main changes here is that we are putting a list of environment in the AgentEvaluator.
        self.mdp_fn_lst = []
        self.env_lst = []
        if mdp_fn_params_lst is None:
            for mdp_params_i in mdp_params_lst:
                mdp = OvercookedGridworld.from_layout_name(**mdp_params_i)
                self.mdp_fn_lst.append(lambda: mdp)
                self.env_lst.append(OvercookedEnv.from_mdp(mdp, **env_params))
        else:
            assert type(mdp_fn_params_lst) is list and len(mdp_fn_params_lst) > 0
            assert all([type(mdp_fn_params_i) is dict for mdp_fn_params_i in mdp_fn_params_lst]), \
                "all mdp_fn_params must be a dictionary"
            for mdp_params_i, mdp_fn_params_i in zip(mdp_params_lst, mdp_fn_params_lst):
                mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_params_i, **mdp_fn_params_i)
                self.mdp_fn_lst.append(mdp_fn)
                self.env_lst.append(OvercookedEnv(mdp_fn, **env_params))
        
        self.force_compute = force_compute
        self.debug = debug
        self.mlp_params_lst = mlp_params_lst if mlp_params_lst is not None else [NO_COUNTERS_PARAMS] * len(mdp_params_lst)
        self._mlp_lst = [None] * len(mdp_params_lst)

    @property
    def mlp_lst(self):
        # each layout will get its own mlp
        for i in range(len(self._mlp_lst)):
            assert not self.env_lst[i].variable_mdp, "Variable mdp is not currently supported for planning"
            if self._mlp_lst[i] is None:
                if self.debug: print("Computing Planner")
                self._mlp_lst[i] = MediumLevelPlanner.from_pickle_or_compute(self.env_lst[i].mdp, self.mlp_params_lst[i], force_compute=self.force_compute)
        return self._mlp_lst

    """
    @property
    def env(self, idx = 0):
        # this function could be used to ensure backwards compatibility with previous APIs that assumes single env
        # but its use in production is strongly discouraged. Could be used for debugging
        return self.env_lst[idx]
        
    @property
    def mlp(self, idx = 0):
        # this function could be used to ensure backwards compatibility with previous APIs that assumes single env
        # but its use in production is strongly discouraged. Could be used for debugging
        assert not self.env_lst[0].variable_mdp
        
        if self._mlp_lst[0] is None:
            if self.debug: print("Computing Planner")
            self._mlp_lst[0] = MediumLevelPlanner.from_pickle_or_compute(self.env_lst[0].mdp, self.mlp_params_lst[0], force_compute=self.force_compute)
        return self._mlp_lst[0]
    """

    def evaluate_random_pair(self, num_games=1, all_actions=True, display=False):
        agent_pair = AgentPair(RandomAgent(all_actions=all_actions), RandomAgent(all_actions=all_actions))
        return self.evaluate_agent_pair(agent_pair, num_games=num_games, display=display)

    def evaluate_human_model_pair(self, num_games=1, display=False):
        a0 = GreedyHumanModel(self.mlp)
        a1 = GreedyHumanModel(self.mlp)
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair, num_games=num_games, display=display)

    def evaluate_optimal_pair(self, num_games, delivery_horizon=2, display=False):
        a0 = CoupledPlanningAgent(self.mlp, delivery_horizon=delivery_horizon)
        a1 = CoupledPlanningAgent(self.mlp, delivery_horizon=delivery_horizon)
        a0.mlp.env = self.env
        a1.mlp.env = self.env
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair, num_games=num_games, display=display)

    def evaluate_one_optimal_one_random(self, num_games, display=True):
        a0 = CoupledPlanningAgent(self.mlp)
        a1 = RandomAgent()
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair, num_games=num_games, display=display)

    def evaluate_one_optimal_one_greedy_human(self, num_games, h_idx=0, display=True):
        h = GreedyHumanModel(self.mlp)
        r = CoupledPlanningAgent(self.mlp)
        agent_pair = AgentPair(h, r) if h_idx == 0 else AgentPair(r, h)
        return self.evaluate_agent_pair(agent_pair, num_games=num_games, display=display)

    def evaluate_agent_pair(self, agent_pair, num_games, game_length=None, start_state_fn=None, metadata_fn=None, metadata_info_fn=None, display=False, info=True):
        # this index has to be 0 because the Agent_Evaluator only has 1 env initiated
        # if you would like to evaluate on a different env using rllib, please modifiy
        # rllib/ -> rllib.py -> get_rllib_eval_function -> _evaluate
        idx = 0
        horizon_env = self.env_lst[idx].copy()
        horizon_env.horizon = self.env_lst[idx].horizon if game_length is None else game_length
        horizon_env.start_state_fn = self.env_lst[idx].start_state_fn if start_state_fn is None else start_state_fn
        horizon_env.reset()
        return horizon_env.get_rollouts(agent_pair, num_games=num_games, display=display, info=info, metadata_fn=metadata_fn, metadata_info_fn=metadata_info_fn)

    def get_agent_pair_trajs(self, a0, a1=None, num_games=100, game_length=None, start_state_fn=None, display=False, info=True):
        """Evaluate agent pair on both indices, and return trajectories by index"""
        if a1 is None:
            ap = AgentPair(a0, a0, allow_duplicate_agents=True)
            trajs_0 = trajs_1 = self.evaluate_agent_pair(ap, num_games=num_games, game_length=game_length, start_state_fn=start_state_fn, display=display, info=info)
        else:
            trajs_0 = self.evaluate_agent_pair(AgentPair(a0, a1), num_games=num_games, game_length=game_length, start_state_fn=start_state_fn, display=display, info=info)
            trajs_1 = self.evaluate_agent_pair(AgentPair(a1, a0), num_games=num_games, game_length=game_length, start_state_fn=start_state_fn, display=display, info=info)
        return trajs_0, trajs_1

    @staticmethod
    def check_trajectories(trajectories, from_json=False):
        """
        Checks that of trajectories are in standard format and are consistent with dynamics of mdp.
        If the trajectories were saves as json, do not check that they have standard traj keys.
        """
        if not from_json:
            AgentEvaluator._check_standard_traj_keys(set(trajectories.keys()))
        AgentEvaluator._check_right_types(trajectories)
        AgentEvaluator._check_trajectories_dynamics(trajectories)
        # TODO: Check shapes?

    @staticmethod
    def _check_standard_traj_keys(traj_keys_set):
        default_traj_keys = OvercookedEnv.DEFAULT_TRAJ_KEYS
        assert traj_keys_set == set(default_traj_keys), "Keys of traj dict did not match standard form.\nMissing keys: {}\nAdditional keys: {}".format(
            [k for k in default_traj_keys if k not in traj_keys_set], [k for k in traj_keys_set if k not in default_traj_keys]
        )
    
    @staticmethod
    def _check_right_types(trajectories):
        for idx in range(len(trajectories["ep_states"])):
            states, actions, rewards = trajectories["ep_states"][idx], trajectories["ep_actions"][idx], trajectories["ep_rewards"][idx]
            mdp_params, env_params = trajectories["mdp_params"][idx], trajectories["env_params"][idx]
            assert all(type(j_a) is tuple for j_a in actions)
            assert all(type(s) is OvercookedState for s in states)
            assert type(mdp_params) is dict
            assert type(env_params) is dict
            # TODO: check that are all lists

    @staticmethod
    def _check_trajectories_dynamics(trajectories):
        if any(env_params["_variable_mdp"] for env_params in trajectories["env_params"]):
            print("Skipping trajectory consistency checking because MDP was recognized as variable. "
                  "Trajectory consistency checking is not yet supported for variable MDPs.")
            return

        _, envs = AgentEvaluator.get_mdps_and_envs_from_trajectories(trajectories)

        for idx in range(len(trajectories["ep_states"])):
            states, actions, rewards = trajectories["ep_states"][idx], trajectories["ep_actions"][idx], trajectories["ep_rewards"][idx]
            simulation_env = envs[idx]

            assert len(states) == len(actions) == len(rewards), "# states {}\t# actions {}\t# rewards {}".format(
                len(states), len(actions), len(rewards)
            )

            # Checking that actions would give rise to same behaviour in current MDP
            for i in range(len(states) - 1):
                curr_state = states[i]
                simulation_env.state = curr_state

                next_state, reward, done, info = simulation_env.step(actions[i])

                assert states[i + 1] == next_state, "States differed (expected vs actual): {}\n\nexpected dict: \t{}\nactual dict: \t{}".format(
                    simulation_env.display_states(states[i + 1], next_state), states[i+1].to_dict(), next_state.to_dict()
                )
                assert rewards[i] == reward, "{} \t {}".format(rewards[i], reward)

    @staticmethod
    def get_mdps_and_envs_from_trajectories(trajectories):
        mdps, envs = [], []
        for idx in range(len(trajectories["ep_lengths"])):
            mdp_params = copy.deepcopy(trajectories["mdp_params"][idx])
            env_params = copy.deepcopy(trajectories["env_params"][idx])
            mdp = OvercookedGridworld.from_layout_name(**mdp_params)
            env = OvercookedEnv.from_mdp(mdp, **env_params)
            mdps.append(mdp)
            envs.append(env)
        return mdps, envs


    ### I/O METHODS ###

    @staticmethod
    def save_trajectories(trajectories, filename):
        AgentEvaluator.check_trajectories(trajectories)
        if any(t["env_params"]["start_state_fn"] is not None for t in trajectories):
            print("Saving trajectories with a custom start state. This can currently "
                  "cause things to break when loading in the trajectories.")
        save_pickle(trajectories, filename)

    @staticmethod
    def load_trajectories(filename):
        trajs = load_pickle(filename)
        AgentEvaluator.check_trajectories(trajs)
        return trajs

    @staticmethod
    def get_joint_traj_in_single_agent_stable_baselines_format(trajs, encoding_fn, save=False, filename=None):
        """
        This requires splitting each trajectory into two, one for each action in the
        joint action.
        """
        trajs = copy.deepcopy(trajs)
        sb_traj_dict_keys = ["actions", "obs", "rewards", "episode_starts", "episode_returns"]
        sb_trajs_dict = { k:[] for k in sb_traj_dict_keys }

        AgentEvaluator.add_observations_to_trajs_in_metadata(trajs, encoding_fn)

        for traj_idx in range(len(trajs["ep_lengths"])):
            # Extract single-agent trajectory for each agent
            # for agent_idx in range(2):
            agent_idx = trajs["metadatas"]["ep_agent_idxs"][traj_idx]
                
            # Getting only actions for current agent index, and processing them to an array 
            # with shape (1, )
            processed_agent_actions = [[Action.ACTION_TO_INDEX[j_a[agent_idx]]] for j_a in trajs["ep_actions"][traj_idx]]
            sb_trajs_dict["actions"].extend(processed_agent_actions)

            agent_obs = [both_agent_obs[agent_idx] for both_agent_obs in trajs["metadatas"]["ep_obs_for_both_agents"][traj_idx]]
            sb_trajs_dict["obs"].extend(agent_obs)

            sb_trajs_dict["rewards"].extend(trajs["ep_rewards"][traj_idx])

            # Converting episode dones to episode starts
            traj_starts = [1 if i == 0 else 0 for i in range(trajs["ep_lengths"][traj_idx])]
            sb_trajs_dict["episode_starts"].extend(traj_starts)
            sb_trajs_dict["episode_returns"].append(trajs["ep_returns"][traj_idx])

        sb_trajs_dict = { k:np.array(v) for k, v in sb_trajs_dict.items() }

        if save:
            assert filename is not None
            np.savez(filename, **sb_trajs_dict)

        return sb_trajs_dict

    @staticmethod
    def save_traj_as_json(trajectory, filename):
        """Saves the `idx`th trajectory as a list of state action pairs"""
        assert set(OvercookedEnv.DEFAULT_TRAJ_KEYS) == set(trajectory.keys()), "{} vs\n{}".format(OvercookedEnv.DEFAULT_TRAJ_KEYS, trajectory.keys())
        AgentEvaluator.check_trajectories(trajectory)
        trajectory = AgentEvaluator.make_trajectories_json_serializable(trajectory)
        save_as_json(trajectory, filename)

    @staticmethod
    def make_trajectories_json_serializable(trajectories):
        """
        Cannot convert np.arrays or special types of ints to JSON.
        This method converts all components of a trajectory to standard types.
        """
        dict_traj = copy.deepcopy(trajectories)
        dict_traj["ep_states"] = [[ob.to_dict() for ob in one_ep_obs] for one_ep_obs in trajectories["ep_states"]]
        for k in dict_traj.keys():
            dict_traj[k] = list(dict_traj[k])
        dict_traj['ep_actions'] = [list(lst) for lst in dict_traj['ep_actions']]
        dict_traj['ep_rewards'] = [list(lst) for lst in dict_traj['ep_rewards']]
        dict_traj['ep_dones'] = [list(lst) for lst in dict_traj['ep_dones']]
        dict_traj['ep_returns'] = [int(val) for val in dict_traj['ep_returns']]
        dict_traj['ep_lengths'] = [int(val) for val in dict_traj['ep_lengths']]

        # NOTE: Currently saving to JSON does not support ep_infos (due to nested np.arrays) or metadata
        del dict_traj['ep_infos']
        del dict_traj['metadatas']
        return dict_traj

    @staticmethod
    def load_traj_from_json(filename):
        traj_dict = load_from_json(filename)
        traj_dict["ep_states"] = [[OvercookedState.from_dict(ob) for ob in curr_ep_obs] for curr_ep_obs in traj_dict["ep_states"]]
        traj_dict["ep_actions"] = [[tuple(tuple(a) if type(a) is list else a for a in j_a) for j_a in ep_acts] for ep_acts in traj_dict["ep_actions"]]
        return traj_dict

    ############################
    # TRAJ MANINPULATION UTILS #
    ############################
    # TODO: add more documentation!

    @staticmethod
    def merge_trajs(trajs_n):
        """
        Takes in multiple trajectory objects and appends all the information into one trajectory object

        [trajs0, trajs1] -> trajs
        """
        metadatas_merged = merge_dictionaries([trajs["metadatas"] for trajs in trajs_n])
        merged_trajs = merge_dictionaries(trajs_n)
        merged_trajs["metadatas"] = metadatas_merged
        return merged_trajs

    @staticmethod
    def remove_traj_idx(trajs, idx):
        # NOTE: MUTATING METHOD for trajs, returns the POPPED IDX
        metadatas = trajs["metadatas"]
        del trajs["metadatas"]
        removed_idx_d = rm_idx_from_dict(trajs, idx)
        removed_idx_metas = rm_idx_from_dict(metadatas, idx)
        trajs["metadatas"] = metadatas
        removed_idx_d["metadatas"] = removed_idx_metas
        return removed_idx_d

    @staticmethod
    def take_traj_indices(trajs, indices):
        # NOTE: non mutating method
        subset_trajs = take_indexes_from_dict(trajs, indices, keys_to_ignore=["metadatas"])
        # TODO: Make metadatas field into additional keys for trajs, rather than having a metadatas field?
        subset_trajs["metadatas"] = take_indexes_from_dict(trajs["metadatas"], indices)
        return subset_trajs

    @staticmethod
    def add_metadata_to_traj(trajs, metadata_fn, input_keys):
        """
        Add an additional metadata entry to the trajectory, based on manipulating 
        the trajectory `input_keys` values
        """
        metadata_fn_input = [trajs[k] for k in input_keys]
        metadata_key, metadata_data = metadata_fn(metadata_fn_input)
        assert metadata_key not in trajs["metadatas"].keys()
        trajs["metadatas"][metadata_key] = metadata_data
        return trajs

    @staticmethod
    def add_observations_to_trajs_in_metadata(trajs, encoding_fn):
        """Adds processed observations (for both agent indices) in the metadatas"""
        def metadata_fn(data):
            traj_ep_states = data[0]
            obs_metadata = []
            for one_traj_states in traj_ep_states:
                obs_metadata.append([encoding_fn(s) for s in one_traj_states])
            return "ep_obs_for_both_agents", obs_metadata
        return AgentEvaluator.add_metadata_to_traj(trajs, metadata_fn, ["ep_states"])


    ### VIZUALIZATION METHODS ###

    @staticmethod
    def interactive_from_traj(trajectories, traj_idx=0, nested_keys_to_print=[]):
        """
        Displays ith trajectory of trajectories (in standard format) 
        interactively in a Jupyter notebook.

        keys_to_print is a list of keys of info to be printed. By default
        states and actions (corresponding to the previous timestep will be printed)
        will be printed.
        """
        from ipywidgets import widgets, interactive_output

        states = trajectories["ep_states"][traj_idx]
        joint_actions = trajectories["ep_actions"][traj_idx]
        
        other_info = {}
        for nested_k in nested_keys_to_print:
            inner_data = trajectories
            for k in nested_k:
                inner_data = inner_data[k]
            inner_data = inner_data[traj_idx]
            
            assert np.array(inner_data).shape == np.array(states).shape, "{} vs {}".format(np.array(inner_data).shape, np.array(states).shape)
            other_info[k] = inner_data

        cumulative_rewards = cumulative_rewards_from_rew_list(trajectories["ep_rewards"][traj_idx])
        mdp_params = trajectories["mdp_params"][traj_idx]
        env_params = trajectories["env_params"][traj_idx]
        env = AgentEvaluator(mdp_params, env_params=env_params).env

        def update(t = 1.0):
            traj_timestep = int(t)
            env.state = states[traj_timestep]
            joint_action = joint_actions[traj_timestep - 1] if traj_timestep > 0 else (Action.STAY, Action.STAY)
            print(env)
            print("Joint Action: {} \t Score: {}".format(Action.joint_action_to_char(joint_action), cumulative_rewards[t]))

            for k, data in other_info.items():
                print("{}: {}".format(k, data[traj_timestep]))

        t = widgets.IntSlider(min=0, max=len(states) - 1, step=1, value=0)
        out = interactive_output(update, {'t': t})
        display(out, t)

    # EVENTS VISUALIZATION METHODS #
    
    @staticmethod
    def events_visualization(trajs, traj_index):
        # TODO
        pass
