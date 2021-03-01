import copy, json
import numpy as np

from overcooked_ai_py.utils import save_pickle, load_pickle, cumulative_rewards_from_rew_list, save_as_json, \
    load_from_json, merge_dictionaries, rm_idx_from_dict, take_indexes_from_dict, is_iterable, NumpyArrayEncoder
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent, GreedyHumanModel
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action, OvercookedState, Recipe
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.visualization.extract_events import extract_events
from overcooked_ai_py.visualization.visualization_utils import run_html_in_ipython, run_html_in_web, \
    create_chart_html, DEFAULT_EVENT_CHART_SETTINGS


class AgentEvaluator(object):
    """
    Class used to get rollouts and evaluate performance of various types of agents.

    TODO: This class currently only fully supports fixed mdps, or variable mdps that can be created with the LayoutGenerator class,
    but might break with other types of variable mdps. Some methods currently assume that the AgentEvaluator can be reconstructed
    from loaded params (which must be pickleable). However, some custom start_state_fns or mdp_generating_fns will not be easily
    pickleable. We should think about possible improvements/what makes most sense to do here.
    """

    def __init__(self, env_params, mdp_fn, force_compute=False, mlam_params=NO_COUNTERS_PARAMS, debug=False):
        """
        env_params (dict): params for creation of an OvercookedEnv
        mdp_fn (callable function): a function that can be used to create mdp
        force_compute (bool): whether should re-compute MediumLevelActionManager although matching file is found
        mlam_params (dict): the parameters for mlam, the MediumLevelActionManager
        debug (bool): whether to display debugging information on init
        """
        assert callable(mdp_fn), "mdp generating function must be a callable function"
        if env_params.get("mlam_params") is None:
            env_params = copy.deepcopy(env_params)
            env_params["mlam_params"] = mlam_params
        self.mdp_fn = mdp_fn
        self.env = OvercookedEnv(self.mdp_fn, **env_params)
        self.force_compute = force_compute

    @staticmethod
    def from_mdp_params_infinite(mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None, force_compute=False,
                        mlam_params=NO_COUNTERS_PARAMS, debug=False):
        """
        mdp_params (dict): params for creation of an OvercookedGridworld instance through the `from_layout_name` method
        outer_shape: the outer shape of environment
        mdp_params_schedule_fn: the schedule for varying mdp params
        Information for the rest of params please refer to the __init__ method above

        Infinitely generate mdp using the naive mdp_fn
        """
        assert outer_shape is not None, "outer_shape needs to be defined for variable mdp"
        assert "num_mdp" in env_params and np.isinf(env_params["num_mdp"]), \
            "num_mdp needs to be specified and infinite"
        mdp_fn_naive = LayoutGenerator.mdp_gen_fn_from_dict(mdp_params, outer_shape, mdp_params_schedule_fn)
        return AgentEvaluator(env_params, mdp_fn_naive, force_compute, mlam_params, debug)

    @staticmethod
    def from_mdp_params_finite(mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None, force_compute=False,
                        mlam_params=NO_COUNTERS_PARAMS, debug=False):
        """
        mdp_params (dict): params for creation of an OvercookedGridworld instance through the `from_layout_name` method
        outer_shape: the outer shape of environment
        mdp_params_schedule_fn: the schedule for varying mdp params
        Information for the rest of params please refer to the __init__ method above

        Generate a finite list of mdp (mdp_lst) using the naive mdp_fn, and then use the from_mdp_lst to generate
        the AgentEvaluator
        """
        assert outer_shape is not None, "outer_shape needs to be defined for variable mdp"
        assert "num_mdp" in env_params and not np.isinf(env_params["num_mdp"]), \
            "num_mdp needs to be specified and finite"
        mdp_fn_naive = LayoutGenerator.mdp_gen_fn_from_dict(mdp_params, outer_shape, mdp_params_schedule_fn)
        # finite mdp, random choice
        num_mdp = env_params['num_mdp']
        assert type(num_mdp) == int and num_mdp > 0, "invalid number of mdp: " + str(num_mdp)
        mdp_lst = [mdp_fn_naive() for _ in range(num_mdp)]
        return AgentEvaluator.from_mdp_lst(mdp_lst=mdp_lst, env_params=env_params,
                                           force_compute=force_compute, mlam_params=mlam_params, debug=debug)

    @staticmethod
    def from_mdp(mdp, env_params, force_compute=False, mlam_params=NO_COUNTERS_PARAMS, debug=False):
        """
        mdp (OvercookedGridworld): the mdp that we want the AgentEvaluator to always generate
        Information for the rest of params please refer to the __init__ method above
        """
        assert type(mdp) == OvercookedGridworld, "mdp must be a OvercookedGridworld object"
        mdp_fn = lambda _ignored: mdp
        return AgentEvaluator(env_params, mdp_fn, force_compute, mlam_params, debug)

    @staticmethod
    def from_layout_name(mdp_params, env_params, force_compute=False, mlam_params=NO_COUNTERS_PARAMS, debug=False):
        """
        mdp_params (dict): params for creation of an OvercookedGridworld instance through the `from_layout_name` method
        Information for the rest of params please refer to the __init__ method above
        """
        assert type(mdp_params) is dict and "layout_name" in mdp_params
        mdp = OvercookedGridworld.from_layout_name(**mdp_params)
        return AgentEvaluator.from_mdp(mdp, env_params, force_compute, mlam_params, debug)

    @staticmethod
    def from_mdp_lst(mdp_lst, env_params, sampling_freq=None, force_compute=False, mlam_params=NO_COUNTERS_PARAMS, debug=False):
        """
        mdp_lst (list): a list of mdp (OvercookedGridworld) we would like to
        sampling_freq (list): a list of number that signify the sampling frequency of each mdp in the mdp_lst
        Information for the rest of params please refer to the __init__ method above
        """
        assert is_iterable(mdp_lst), "mdp_lst must be a list"
        assert all([type(mdp) == OvercookedGridworld for mdp in mdp_lst]), "some mdps are not OvercookedGridworld objects"

        if sampling_freq is None:
            sampling_freq = np.ones(len(mdp_lst)) /len(mdp_lst)

        mdp_fn = lambda _ignored: np.random.choice(mdp_lst, p=sampling_freq)
        return AgentEvaluator(env_params, mdp_fn, force_compute, mlam_params, debug)

    def evaluate_random_pair(self, num_games=1, all_actions=True, display=False, native_eval=False):
        agent_pair = AgentPair(RandomAgent(all_actions=all_actions), RandomAgent(all_actions=all_actions))
        return self.evaluate_agent_pair(agent_pair, num_games=num_games, display=display, native_eval=native_eval)

    def evaluate_human_model_pair(self, num_games=1, display=False, native_eval=False):
        a0 = GreedyHumanModel(self.env.mlam)
        a1 = GreedyHumanModel(self.env.mlam)
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair, num_games=num_games, display=display, native_eval=native_eval)

    def evaluate_agent_pair(self, agent_pair, num_games, game_length=None, start_state_fn=None, metadata_fn=None, metadata_info_fn=None, display=False, dir=None,
                            display_phi=False, info=True, native_eval=False):
        # this index has to be 0 because the Agent_Evaluator only has 1 env initiated
        # if you would like to evaluate on a different env using rllib, please modify
        # rllib/ -> rllib.py -> get_rllib_eval_function -> _evaluate

        # native eval: using self.env in evaluation instead of creating a copy
        # this is particularly helpful with variable MDP, where we want to make sure
        # the mdp used in evaluation is the same as the native self.env.mdp
        if native_eval:
            return self.env.get_rollouts(agent_pair, num_games=num_games, display=display, dir=dir, display_phi=display_phi,
                                         info=info, metadata_fn=metadata_fn, metadata_info_fn=metadata_info_fn)
        else:
            horizon_env = self.env.copy()
            horizon_env.horizon = self.env.horizon if game_length is None else game_length
            horizon_env.start_state_fn = self.env.start_state_fn if start_state_fn is None else start_state_fn
            horizon_env.reset()
            return horizon_env.get_rollouts(agent_pair, num_games=num_games, display=display, dir=dir, display_phi=display_phi,
                                            info=info, metadata_fn=metadata_fn, metadata_info_fn=metadata_info_fn)

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
            if states[0].orders_list.is_adding_orders and len(states[0].orders_list.orders_to_add) > 1:
                print("Skipping trajectory consistency checking for idx %i because orders are added at random" % idx)
                return
            simulation_env = envs[idx]

            assert len(states) == len(actions) == len(rewards), "# states {}\t# actions {}\t# rewards {}".format(
                len(states), len(actions), len(rewards)
            )

            # Checking that actions would give rise to same behaviour in current MDP
            for i in range(len(states) - 1):
                curr_state = states[i]
                simulation_env.state = curr_state

                next_state, reward, done, info = simulation_env.step(actions[i])

                assert states[i + 1].ids_independent_equal(next_state), "States differed (expected vs actual): {}\n\nexpected dict: \t{}\nactual dict: \t{}".format(
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
        if any(env_params["start_state_fn"] is not None for env_params in trajectories["env_params"]):
            print("Saving trajectories with a custom start state. This can currently "
                  "cause things to break when loading in the trajectories.")
        save_pickle(trajectories, filename)


    @staticmethod
    def load_trajectories(filename):
        AgentEvaluator._configure_recipe_if_needed()
        trajs = load_pickle(filename)
        AgentEvaluator.check_trajectories(trajs)
        return trajs

    @staticmethod
    def save_traj_as_json(trajectories, filename):
        assert set(OvercookedEnv.DEFAULT_TRAJ_KEYS) == set(trajectories.keys()), "{} vs\n{}".format(OvercookedEnv.DEFAULT_TRAJ_KEYS, trajectories.keys())
        AgentEvaluator.check_trajectories(trajectories)
        trajectories = AgentEvaluator.make_trajectories_json_serializable(trajectories)
        save_as_json(trajectories, filename)

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
        # NOTE: Currently saving to JSON does not support metadata
        dict_traj['ep_infos'] = json.loads(json.dumps(dict_traj['ep_infos'], cls=NumpyArrayEncoder))
        del dict_traj['metadatas']
        return dict_traj

    @staticmethod
    def load_traj_from_json(filename):
        AgentEvaluator._configure_recipe_if_needed()
        traj_dict = load_from_json(filename)
        return AgentEvaluator.load_traj_from_json_obj(traj_dict)

    @staticmethod
    def load_traj_from_json_obj(obj):
        # currently ep_infos is not changed back to numpy arrays/tuples from lists
        traj_dict = copy.deepcopy(obj)
        traj_dict["ep_states"] = [[OvercookedState.from_dict(ob) for ob in curr_ep_obs] for curr_ep_obs in traj_dict["ep_states"]]
        traj_dict["ep_actions"] = [[tuple(tuple(a) if type(a) is list else a for a in j_a) for j_a in ep_acts] for ep_acts in traj_dict["ep_actions"]]
        return traj_dict
    
    @staticmethod
    def _configure_recipe_if_needed(config={}):
        """
        method that configures recipe to avoid ValueError when recipe is created before configuring class
        recipe is created on reading trajectory because recipe is part of the order that is part of the state that is part of the treajectory
        """
        if not Recipe._configured:
            print("Configuring recipe from AgentEvaluator class with config:", config)
            Recipe.configure(config)

    ############################
    # TRAJ MANIPULATION UTILS #
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

    # EVENTS VISUALIZATION METHODS #
        
    @staticmethod
    def events_visualization(trajs, traj_index=0, ipython=False, chart_settings=None):
        """
        Displays chart with visualization of events (when items pickups happened etc.)
        ipython - chooses between opening chart
        in default browser or below ipython cell (for ipython=True)
        chart_settings - json with various chart settings that overwrites default ones
        """
        settings = DEFAULT_EVENT_CHART_SETTINGS.copy()
        settings.update(chart_settings or {})

        if settings.get("show_cumulative_data"):
            events = extract_events(trajs, traj_index, settings["cumulative_events_description"])
        else: # no need to add cumulative data that won't be shown
            events = extract_events(trajs, traj_index)

        html = create_chart_html(events, settings)
        if ipython:
            run_html_in_ipython(html)
        else:
            run_html_in_web(html)
