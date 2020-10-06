from overcooked_ai_py.agents.agent import AgentPair, RandomAgent, GreedyHumanModel, add_random_play_to_agent
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
import itertools, copy
import numpy as np

DEFAULT_ENV_PARAMS = {"horizon": 1000}


def _get_pairs_scores(agent_evaluator, pairs, num_games_per_pair=1):
    return list(itertools.chain(list(agent_evaluator.evaluate_agent_pair(pair, num_games=num_games_per_pair)["ep_returns"] for pair in pairs)))

def non_self_play_difficulty(mdp, agents=None, num_games_per_pair=1,
                              num_games_per_non_self_play_pair=None,
                              num_games_per_self_play_pair=None,
                              env_params=None):
    if num_games_per_non_self_play_pair is None:
        num_games_per_non_self_play_pair = num_games_per_pair
    if num_games_per_self_play_pair is None:
        num_games_per_self_play_pair = num_games_per_pair
    if env_params is None:
        env_params = copy.deepcopy(DEFAULT_ENV_PARAMS)
    
    agent_eval = AgentEvaluator.from_mdp(mdp, env_params=env_params)

    if agents is None:
        # TODO: needs better agents - at least one that does any real coordination
        agents = [RandomAgent(all_actions=True), GreedyHumanModel(agent_eval.env.mlam)]

    assert len(agents) > 1
    self_play_pairs = [AgentPair(agent, agent, allow_duplicate_agents=True) for agent in agents]
    non_self_play_pairs = [AgentPair(agent1, agent2) for agent1, agent2 in itertools.combinations(agents, 2)]
    
    self_play_scores = _get_pairs_scores(agent_eval, self_play_pairs, num_games_per_self_play_pair)
    non_self_play_scores = _get_pairs_scores(agent_eval, non_self_play_pairs, num_games_per_non_self_play_pair)

    result = np.mean(self_play_scores)/np.mean(non_self_play_scores)
    if np.isnan(result) or np.isinf(result):
        return None
    else:
        return result



def random_plays_difficulty(mdp, agent_pairs=None, num_games_per_pair=1,
                                   num_games_per_non_randomized_pair=None,
                                   num_games_per_randomized_pair=None,
                                   random_plays_fraction=0.1, does_random_interactions=True,
                                   env_params=None):
    
    if num_games_per_non_randomized_pair is None:
        num_games_per_non_randomized_pair = num_games_per_pair
    if num_games_per_randomized_pair is None:
        num_games_per_randomized_pair = num_games_per_pair
    if env_params is None:
        env_params = copy.deepcopy(DEFAULT_ENV_PARAMS)

    agent_eval = AgentEvaluator.from_mdp(mdp, env_params=env_params)
    
    if agent_pairs is None:
        # TODO: needs better agents - at least one that does any real coordination
        agent_pairs = [AgentPair(GreedyHumanModel(agent_eval.env.mlam), GreedyHumanModel(agent_eval.env.mlam))]
    assert len(agent_pairs) > 0
    non_randomized_agent_pairs = agent_pairs
    randomized_agent_pairs = [AgentPair(*[add_random_play_to_agent(agent, random_plays_fraction, does_random_interactions=does_random_interactions)
                               for agent in pair.agents], allow_duplicate_agents=True) for pair in agent_pairs]
    
    non_randomized_play_scores = _get_pairs_scores(agent_eval, non_randomized_agent_pairs, num_games_per_non_randomized_pair)
    randomized_play_scores = _get_pairs_scores(agent_eval, randomized_agent_pairs, num_games_per_randomized_pair)
    result =  np.mean(non_randomized_play_scores)/np.mean(randomized_play_scores)
    if np.isnan(result) or np.isinf(result):
        return None
    else:
        return result
