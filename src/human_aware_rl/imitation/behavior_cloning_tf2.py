import copy
import os
import pickle

import numpy as np
import tensorflow as tf
from ray.rllib.policy import Policy as RllibPolicy
from tensorflow import keras
from tensorflow.compat.v1.keras.backend import get_session, set_session

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.process_dataframes import (
    get_human_human_trajectories,
    get_trajs_from_data,
)
from human_aware_rl.rllib.rllib import (
    RlLibAgent,
    evaluate,
    get_base_ae,
    softmax,
)
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN
from human_aware_rl.utils import get_flattened_keys, recursive_dict_update
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

#################
# Configuration #
#################

BC_SAVE_DIR = os.path.join(DATA_DIR, "bc_runs")

DEFAULT_DATA_PARAMS = {
    "layouts": ["cramped_room"],
    "check_trajectories": False,
    "featurize_states": True,
    "data_path": CLEAN_2019_HUMAN_DATA_TRAIN,
}

DEFAULT_MLP_PARAMS = {
    # Number of fully connected layers to use in our network
    "num_layers": 2,
    # Each int represents a layer of that hidden size
    "net_arch": [64, 64],
}

DEFAULT_TRAINING_PARAMS = {
    "epochs": 100,
    "validation_split": 0.15,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "use_class_weights": False,
}

DEFAULT_EVALUATION_PARAMS = {
    "ep_length": 400,
    "num_games": 1,
    "display": False,
}

DEFAULT_BC_PARAMS = {
    "eager": True,
    "use_lstm": False,
    "cell_size": 256,
    "data_params": DEFAULT_DATA_PARAMS,
    "mdp_params": {"layout_name": "cramped_room", "old_dynamics": False},
    "env_params": DEFAULT_ENV_PARAMS,
    "mdp_fn_params": {},
    "mlp_params": DEFAULT_MLP_PARAMS,
    "training_params": DEFAULT_TRAINING_PARAMS,
    "evaluation_params": DEFAULT_EVALUATION_PARAMS,
    "action_shape": (len(Action.ALL_ACTIONS),),
}

# Boolean indicating whether all param dependencies have been loaded. Used to prevent re-loading unceccesarily
_params_initalized = False


def _get_base_ae(bc_params):
    return get_base_ae(bc_params["mdp_params"], bc_params["env_params"])


def _get_observation_shape(bc_params):
    """
    Helper function for creating a dummy environment from "mdp_params" and "env_params" specified
    in bc_params and returning the shape of the observation space
    """
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    dummy_state = base_env.mdp.get_standard_start_state()
    obs_shape = base_env.featurize_state_mdp(dummy_state)[0].shape
    return obs_shape


# For lazily loading the default params. Prevents loading on every import of this module
def get_bc_params(**args_to_override):
    """
    Loads default bc params defined globally. For each key in args_to_override, overrides the default with the
    value specified for that key. Recursively checks all children. If key not found, creates new top level parameter.

    Note: Even though children can share keys, for simplicity, we enforce the condition that all keys at all levels must be distict
    """
    global _params_initalized, DEFAULT_BC_PARAMS
    if not _params_initalized:
        DEFAULT_BC_PARAMS["observation_shape"] = _get_observation_shape(
            DEFAULT_BC_PARAMS
        )
        _params_initalized = False
    params = copy.deepcopy(DEFAULT_BC_PARAMS)

    for arg, val in args_to_override.items():
        updated = recursive_dict_update(params, arg, val)
        if not updated:
            print(
                "WARNING, no value for specified bc argument {} found in schema. Adding as top level parameter".format(
                    arg
                )
            )

    all_keys = get_flattened_keys(params)
    if len(all_keys) != len(set(all_keys)):
        raise ValueError(
            "Every key at every level must be distict for BC params!"
        )

    return params


##############
# Model code #
##############


class LstmStateResetCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()


def _pad(sequences, maxlen=None, default=0):
    if not maxlen:
        maxlen = max([len(seq) for seq in sequences])
    for seq in sequences:
        pad_len = maxlen - len(seq)
        seq.extend([default] * pad_len)
    return sequences


def load_data(bc_params, verbose=False):
    processed_trajs = get_human_human_trajectories(
        **bc_params["data_params"], silent=not verbose
    )
    inputs, targets = (
        processed_trajs["ep_states"],
        processed_trajs["ep_actions"],
    )

    if bc_params["use_lstm"]:
        seq_lens = np.array([len(seq) for seq in inputs])
        seq_padded = _pad(
            inputs,
            default=np.zeros(
                (
                    len(
                        inputs[0][0],
                    )
                )
            ),
        )
        targets_padded = _pad(targets, default=np.zeros(1))
        seq_t = np.dstack(seq_padded).transpose((2, 0, 1))
        targets_t = np.dstack(targets_padded).transpose((2, 0, 1))
        return seq_t, seq_lens, targets_t
    else:
        return np.vstack(inputs), None, np.vstack(targets)


def build_bc_model(use_lstm=True, eager=False, **kwargs):
    if not eager:
        tf.compat.v1.disable_eager_execution()
    if use_lstm:
        return _build_lstm_model(**kwargs)
    else:
        return _build_model(**kwargs)


def train_bc_model(model_dir, bc_params, verbose=False):
    inputs, seq_lens, targets = load_data(bc_params, verbose)

    training_params = bc_params["training_params"]

    if training_params["use_class_weights"]:
        # Get class counts, and use these to compute balanced class weights
        classes, counts = np.unique(targets.flatten(), return_counts=True)
        weights = sum(counts) / counts
        class_weights = dict(zip(classes, weights))
    else:
        # Default is uniform class weights
        class_weights = None

    # Retrieve un-initialized keras model
    model = build_bc_model(
        **bc_params, max_seq_len=np.max(seq_lens), verbose=verbose
    )

    # Initialize the model
    # Note: have to use lists for multi-output model support and not dicts because of tensorlfow 2.0.0 bug
    if bc_params["use_lstm"]:
        loss = [
            keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            None,
            None,
        ]
        metrics = [["sparse_categorical_accuracy"], [], []]
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["sparse_categorical_accuracy"]
    model.compile(
        optimizer=keras.optimizers.Adam(training_params["learning_rate"]),
        loss=loss,
        metrics=metrics,
    )

    # Customize our training loop with callbacks
    callbacks = [
        # Early terminate training if loss doesn't improve for "patience" epochs
        keras.callbacks.EarlyStopping(monitor="loss", patience=20),
        # Reduce lr by "factor" after "patience" epochs of no improvement in loss
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss", patience=3, factor=0.1
        ),
        # Log all metrics model was compiled with to tensorboard every epoch
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, "logs"), write_graph=False
        ),
        # Save checkpoints of the models at the end of every epoch (saving only the best one so far)
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "checkpoints"),
            monitor="loss",
            save_best_only=True,
        ),
    ]

    ## Actually train our model

    # Create input dict for both models
    N = inputs.shape[0]
    inputs = {"Overcooked_observation": inputs}
    targets = {"logits": targets}

    # Inputs unique to lstm model
    if bc_params["use_lstm"]:
        inputs["seq_in"] = seq_lens
        inputs["hidden_in"] = np.zeros((N, bc_params["cell_size"]))
        inputs["memory_in"] = np.zeros((N, bc_params["cell_size"]))

    # Batch size doesn't include time dimension (seq_len) so it should be smaller for rnn model
    batch_size = 1 if bc_params["use_lstm"] else training_params["batch_size"]
    model.fit(
        inputs,
        targets,
        callbacks=callbacks,
        batch_size=batch_size,
        epochs=training_params["epochs"],
        validation_split=training_params["validation_split"],
        class_weight=class_weights,
        verbose=2 if verbose else 0,
    )

    # Save the model
    save_bc_model(model_dir, model, bc_params, verbose=verbose)

    return model


def save_bc_model(model_dir, model, bc_params, verbose=False):
    """
    Saves the specified model under the directory model_dir. This creates three items

        assets/         stores information essential to reconstructing the context and tf graph
        variables/      stores the model's trainable weights
        saved_model.pd  the saved state of the model object

    Additionally, saves a pickled dictionary containing all the parameters used to construct this model
    at model_dir/metadata.pickle
    """
    if verbose:
        print("Saving bc model at ", model_dir)
    model.save(model_dir, save_format="tf")
    with open(os.path.join(model_dir, "metadata.pickle"), "wb") as f:
        pickle.dump(bc_params, f)


def load_bc_model(model_dir, verbose=False):
    """
    Returns the model instance (including all compilation data like optimizer state) and a dictionary of parameters
    used to create the model
    """
    if verbose:
        print("Loading bc model from ", model_dir)
    model = keras.models.load_model(model_dir, custom_objects={"tf": tf})
    with open(os.path.join(model_dir, "metadata.pickle"), "rb") as f:
        bc_params = pickle.load(f)
    return model, bc_params


def evaluate_bc_model(model, bc_params, verbose=False):
    """
    Creates an AgentPair object containing two instances of BC Agents, whose policies are specified by `model`. Runs
    a rollout using AgentEvaluator class in an environment specified by bc_params

    Arguments

        - model (tf.keras.Model)        A function that maps featurized overcooked states to action logits
        - bc_params (dict)              Specifies the environemnt in which to evaluate the agent (i.e. layout, reward_shaping_param)
                                            as well as the configuration for the rollout (rollout_length)

    Returns

        - reward (int)                  Total sparse reward achieved by AgentPair during rollout
    """
    evaluation_params = bc_params["evaluation_params"]
    mdp_params = bc_params["mdp_params"]

    # Get reference to state encoding function used by bc agents, with compatible signature
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env

    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)

    # Wrap Keras models in rllib policies
    agent_0_policy = BehaviorCloningPolicy.from_model(
        model, bc_params, stochastic=True
    )
    agent_1_policy = BehaviorCloningPolicy.from_model(
        model, bc_params, stochastic=True
    )

    # Compute the results of the rollout(s)
    results = evaluate(
        eval_params=evaluation_params,
        mdp_params=mdp_params,
        outer_shape=None,
        agent_0_policy=agent_0_policy,
        agent_1_policy=agent_1_policy,
        agent_0_featurize_fn=featurize_fn,
        agent_1_featurize_fn=featurize_fn,
        verbose=verbose,
    )

    # Compute the average sparse return obtained in each rollout
    reward = np.mean(results["ep_returns"])
    return reward


def _build_model(observation_shape, action_shape, mlp_params, **kwargs):
    ## Inputs
    inputs = keras.Input(
        shape=observation_shape, name="Overcooked_observation"
    )
    x = inputs

    ## Build fully connected layers
    assert (
        len(mlp_params["net_arch"]) == mlp_params["num_layers"]
    ), "Invalid Fully Connected params"

    for i in range(mlp_params["num_layers"]):
        units = mlp_params["net_arch"][i]
        x = keras.layers.Dense(
            units, activation="relu", name="fc_{0}".format(i)
        )(x)

    ## output layer
    logits = keras.layers.Dense(action_shape[0], name="logits")(x)

    return keras.Model(inputs=inputs, outputs=logits)


def _build_lstm_model(
    observation_shape,
    action_shape,
    mlp_params,
    cell_size,
    max_seq_len=20,
    **kwargs
):
    ## Inputs
    obs_in = keras.Input(
        shape=(None, *observation_shape), name="Overcooked_observation"
    )
    seq_in = keras.Input(shape=(), name="seq_in", dtype=tf.int32)
    h_in = keras.Input(shape=(cell_size,), name="hidden_in")
    c_in = keras.Input(shape=(cell_size,), name="memory_in")
    x = obs_in

    ## Build fully connected layers
    assert (
        len(mlp_params["net_arch"]) == mlp_params["num_layers"]
    ), "Invalid Fully Connected params"

    for i in range(mlp_params["num_layers"]):
        units = mlp_params["net_arch"][i]
        x = keras.layers.TimeDistributed(
            keras.layers.Dense(
                units, activation="relu", name="fc_{0}".format(i)
            )
        )(x)

    mask = keras.layers.Lambda(
        lambda x: tf.sequence_mask(x, maxlen=max_seq_len)
    )(seq_in)

    ## LSTM layer
    lstm_out, h_out, c_out = keras.layers.LSTM(
        cell_size,
        return_sequences=True,
        return_state=True,
        stateful=False,
        name="lstm",
    )(inputs=x, mask=mask, initial_state=[h_in, c_in])

    ## output layer
    logits = keras.layers.TimeDistributed(
        keras.layers.Dense(action_shape[0]), name="logits"
    )(lstm_out)

    return keras.Model(
        inputs=[obs_in, seq_in, h_in, c_in], outputs=[logits, h_out, c_out]
    )


################
# Rllib Policy #
################


class NullContextManager:
    """
    No-op context manager that does nothing
    """

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class TfContextManager:
    """
    Properly sets the execution graph and session of the keras backend given a "session" object as input

    Used for isolating tf execution in graph mode. Do not use with eager models or with eager mode on
    """

    def __init__(self, session):
        self.session = session

    def __enter__(self):
        self.ctx = self.session.graph.as_default()
        self.ctx.__enter__()
        set_session(self.session)

    def __exit__(self, *args):
        self.ctx.__exit__(*args)


class BehaviorCloningPolicy(RllibPolicy):
    def __init__(self, observation_space, action_space, config):
        """
        RLLib compatible constructor for initializing a behavior cloning model

        observation_space (gym.Space|tuple)     Shape of the featurized observations
        action_space (gym.space|tuple)          Shape of the action space (len(Action.All_ACTIONS),)
        config (dict)                           Dictionary of relavant bc params
            - model_dir (str)                   Path to pickled keras.Model used to map observations to action logits
            - stochastic (bool)                 Whether action should return logit argmax or sample over distribution
            - bc_model (keras.Model)            Pointer to loaded policy model. Overrides model_dir
            - bc_params (dict)                  Dictionary of parameters used to train model. Required if "model" is present
            - eager (bool)                      Whether the model should run in eager (or graph) mode. Overrides bc_params['eager'] if present
        """
        super(BehaviorCloningPolicy, self).__init__(
            observation_space, action_space, config
        )

        if "bc_model" in config and config["bc_model"]:
            assert (
                "bc_params" in config
            ), "must specify params in addition to model"
            assert issubclass(
                type(config["bc_model"]), keras.Model
            ), "model must be of type keras.Model"
            model, bc_params = config["bc_model"], config["bc_params"]
        else:
            assert (
                "model_dir" in config
            ), "must specify model directory if model not specified"
            model, bc_params = load_bc_model(config["model_dir"])

        # Save the session that the model was loaded into so it is available at inference time if necessary
        self._sess = get_session()
        self._setup_shapes()

        # Basic check to make sure model dimensions match
        assert self.observation_shape == bc_params["observation_shape"]
        assert self.action_shape == bc_params["action_shape"]

        self.model = model
        self.stochastic = config["stochastic"]
        self.use_lstm = bc_params["use_lstm"]
        self.cell_size = bc_params["cell_size"]
        self.eager = (
            config["eager"] if "eager" in config else bc_params["eager"]
        )
        self.context = self._create_execution_context()

    def _setup_shapes(self):
        # This is here to make the class compatible with both tuples or gym.Space objs for the spaces
        # Note: action_space = (len(Action.ALL_ACTIONS,)) is technically NOT the action space shape, which would be () since actions are scalars
        self.observation_shape = (
            self.observation_space
            if type(self.observation_space) == tuple
            else self.observation_space.shape
        )
        self.action_shape = (
            self.action_space
            if type(self.action_space) == tuple
            else (self.action_space.n,)
        )

    @classmethod
    def from_model_dir(cls, model_dir, stochastic=True):
        model, bc_params = load_bc_model(model_dir)
        config = {
            "bc_model": model,
            "bc_params": bc_params,
            "stochastic": stochastic,
        }
        return cls(
            bc_params["observation_shape"], bc_params["action_shape"], config
        )

    @classmethod
    def from_model(cls, model, bc_params, stochastic=True):
        config = {
            "bc_model": model,
            "bc_params": bc_params,
            "stochastic": stochastic,
        }
        return cls(
            bc_params["observation_shape"], bc_params["action_shape"], config
        )

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        """
        Computes sampled actions for each of the corresponding OvercookedEnv states in obs_batch

        Args:
            obs_batch (np.array): batch of pre-process (lossless state encoded) observations

        Returns:
            actions (list|np.array): batch of output actions shape [BATCH_SIZE, ACTION_SHAPE]
            state_outs (list): only necessary for rnn hidden states
            infos (dict): dictionary of extra feature batches { "action_dist_inputs" : [BATCH_SIZE, ...] }
        """
        # Cast to np.array if list (no-op if already np.array)
        obs_batch = np.array(obs_batch)

        # Run the model
        with self.context:
            action_logits, states = self._forward(obs_batch, state_batches)

        # Softmax in numpy to convert logits to probabilities
        action_probs = softmax(action_logits)
        if self.stochastic:
            # Sample according to action_probs for each row in the output
            actions = np.array(
                [
                    np.random.choice(self.action_shape[0], p=action_probs[i])
                    for i in range(len(action_probs))
                ]
            )
        else:
            actions = np.argmax(action_logits, axis=1)

        return actions, states, {"action_dist_inputs": action_logits}

    def get_initial_state(self):
        """
        Returns the initial hidden and memory states for the model if it is recursive

        Note, this shadows the rllib.Model.get_initial_state function, but had to be added here as
        keras does not allow mixins in custom model classes

        Also note, either this function or self.model.get_initial_state (if it exists) must be called at
        start of an episode
        """
        if self.use_lstm:
            return [
                np.zeros(
                    self.cell_size,
                ),
                np.zeros(
                    self.cell_size,
                ),
            ]
        return []

    def get_weights(self):
        """
        No-op to keep rllib from breaking, won't be necessary in future rllib releases
        """
        pass

    def set_weights(self, weights):
        """
        No-op to keep rllib from breaking
        """
        pass

    def learn_on_batch(self, samples):
        """
        Static policy requires no learning
        """
        return {}

    def _forward(self, obs_batch, state_batches):
        if self.use_lstm:
            obs_batch = np.expand_dims(obs_batch, 1)
            seq_lens = np.ones(len(obs_batch))
            model_out = self.model.predict(
                [obs_batch, seq_lens] + state_batches
            )
            logits, states = model_out[0], model_out[1:]
            logits = logits.reshape((logits.shape[0], -1))
            return logits, states
        else:
            return self.model.predict(obs_batch, verbose=0), []

    def _create_execution_context(self):
        """
        Creates a private execution context for the model

        Necessary if using with rllib in order to isolate this policy model from others
        """
        if self.eager:
            return NullContextManager()
        return TfContextManager(self._sess)


if __name__ == "__main__":
    params = get_bc_params()
    model = train_bc_model(
        os.path.join(BC_SAVE_DIR, "default"), params, verbose=True
    )
    # Evaluate our model's performance in a rollout
    evaluate_bc_model(model, params)
