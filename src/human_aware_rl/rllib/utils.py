import inspect

import numpy as np

from overcooked_ai_py.agents.benchmarking import AgentEvaluator


def softmax(logits):
    e_x = np.exp(logits.T - np.max(logits))
    return (e_x / np.sum(e_x, axis=0)).T


def get_base_env(
    mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None
):
    ae = get_base_ae(
        mdp_params, env_params, outer_shape, mdp_params_schedule_fn
    )
    return ae.env


def get_base_mlam(
    mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None
):
    ae = get_base_ae(
        mdp_params, env_params, outer_shape, mdp_params_schedule_fn
    )
    return ae.mlam


def get_base_ae(
    mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None
):
    """
    mdp_params: one set of fixed mdp parameter used by the enviroment
    env_params: env parameters (horizon, etc)
    outer_shape: outer shape of the environment
    mdp_params_schedule_fn: the schedule for varying mdp params

    return: the base agent evaluator
    """
    assert (
        mdp_params == None or mdp_params_schedule_fn == None
    ), "either of the two has to be null"
    if type(mdp_params) == dict and "layout_name" in mdp_params:
        ae = AgentEvaluator.from_layout_name(
            mdp_params=mdp_params, env_params=env_params
        )
    elif "num_mdp" in env_params:
        if np.isinf(env_params["num_mdp"]):
            ae = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_params,
                env_params=env_params,
                outer_shape=outer_shape,
                mdp_params_schedule_fn=mdp_params_schedule_fn,
            )
        else:
            ae = AgentEvaluator.from_mdp_params_finite(
                mdp_params=mdp_params,
                env_params=env_params,
                outer_shape=outer_shape,
                mdp_params_schedule_fn=mdp_params_schedule_fn,
            )
    else:
        # should not reach this case
        raise NotImplementedError()
    return ae


# Returns the required arguments as inspect.Parameter objects in a list
def get_required_arguments(fn):
    required = []
    params = inspect.signature(fn).parameters.values()
    for param in params:
        if (
            param.default == inspect.Parameter.empty
            and param.kind == param.POSITIONAL_OR_KEYWORD
        ):
            required.append(param)
    return required


def iterable_equal(a, b):
    if hasattr(a, "__iter__") != hasattr(b, "__iter__"):
        return False
    if not hasattr(a, "__iter__"):
        return a == b

    if len(a) != len(b):
        return False

    for elem_a, elem_b in zip(a, b):
        if not iterable_equal(elem_a, elem_b):
            return False

    return True
