import numpy as np
import tensorflow as tf
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


class RllibPPOModel(TFModelV2):
    """
    Model that will map environment states to action probabilities. Will be shared across agents
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs
    ):
        super(RllibPPOModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        # params we got to pass in from the call to "run"
        custom_params = model_config["custom_model_config"]

        ## Parse custom network params
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]
        d2rl = custom_params["D2RL"]
        assert type(d2rl) == bool

        ## Create graph of custom network. It will under a shared tf scope such that all agents
        ## use the same model
        self.inputs = tf.keras.Input(
            shape=obs_space.shape, name="observations"
        )
        out = self.inputs

        # Apply initial conv layer with a larger kenel (why?)
        if num_convs > 0:
            y = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.leaky_relu,
                name="conv_initial",
            )
            out = y(out)

        # Apply remaining conv layers, if any
        for i in range(0, num_convs - 1):
            padding = "same" if i < num_convs - 2 else "valid"
            out = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i),
            )(out)

        # Apply dense hidden layers, if any
        conv_out = tf.keras.layers.Flatten()(out)
        out = conv_out
        for i in range(num_hidden_layers):
            if i > 0 and d2rl:
                out = tf.keras.layers.Concatenate()([out, conv_out])
            out = tf.keras.layers.Dense(size_hidden_layers)(out)
            out = tf.keras.layers.LeakyReLU()(out)

        # Linear last layer for action distribution logits
        layer_out = tf.keras.layers.Dense(self.num_outputs)(out)

        # Linear last layer for value function branch of model
        value_out = tf.keras.layers.Dense(1)(out)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def forward(self, input_dict, state=None, seq_lens=None):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class RllibLSTMPPOModel(RecurrentNetwork):
    """
    Model that will map encoded environment observations to action logits

                                                         |_______|
                                                     /-> | value |
             ___________     _________     ________ /    |_______|
    state -> | conv_net | -> | fc_net | -> | lstm |
             |__________|    |________|    |______| \\    |_______________|
                                           /    \\   \\-> | action_logits |
                                          h_in   c_in     |_______________|
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs
    ):
        super(RllibLSTMPPOModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        # params we passed in from rllib client
        custom_params = model_config["custom_model_config"]

        ## Parse custom network params
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]
        cell_size = custom_params["CELL_SIZE"]

        ### Create graph of the model ###
        flattened_dim = np.prod(obs_space.shape)

        # Need an extra batch dimension (None) for time dimension
        flattened_obs_inputs = tf.keras.Input(
            shape=(None, flattened_dim), name="input"
        )
        lstm_h_in = tf.keras.Input(shape=(cell_size,), name="h_in")
        lstm_c_in = tf.keras.Input(shape=(cell_size,), name="c_in")
        seq_in = tf.keras.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Restore initial observation shape
        obs_inputs = tf.keras.layers.Reshape(
            target_shape=(-1, *obs_space.shape)
        )(flattened_obs_inputs)
        out = obs_inputs

        ## Initial "vision" network

        # Apply initial conv layer with a larger kenel (why?)
        if num_convs > 0:
            out = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.leaky_relu,
                    name="conv_initial",
                )
            )(out)

        # Apply remaining conv layers, if any
        for i in range(0, num_convs - 1):
            padding = "same" if i < num_convs - 2 else "valid"
            out = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=[3, 3],
                    padding=padding,
                    activation=tf.nn.leaky_relu,
                    name="conv_{}".format(i),
                )
            )(out)

        # Flatten spatial features
        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(out)

        # Apply dense hidden layers, if any
        for i in range(num_hidden_layers):
            out = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=size_hidden_layers,
                    activation=tf.nn.leaky_relu,
                    name="fc_{0}".format(i),
                )
            )(out)

        ## LSTM network
        lstm_out, h_out, c_out = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=out,
            mask=tf.sequence_mask(seq_in),
            initial_state=[lstm_h_in, lstm_c_in],
        )

        # Linear last layer for action distribution logits
        layer_out = tf.keras.layers.Dense(self.num_outputs, name="logits")(
            lstm_out
        )

        # Linear last layer for value function branch of model
        value_out = tf.keras.layers.Dense(1, name="values")(lstm_out)

        self.cell_size = cell_size
        self.base_model = tf.keras.Model(
            inputs=[flattened_obs_inputs, seq_in, lstm_h_in, lstm_c_in],
            outputs=[layer_out, value_out, h_out, c_out],
        )

    def forward_rnn(self, inputs, state, seq_lens):
        """
        Run the forward pass of the model

        Arguments:
            inputs: np.array of shape [BATCH, T, obs_shape]
            state:  list of np.arrays [h_in, c_in] each of shape [BATCH, self.cell_size]
            seq_lens: np.array of shape [BATCH] where the ith element is the length of the ith sequence

        Output:
            model_out: tensor of shape [BATCH, T, self.num_outputs] representing action logits
            state: list of tensors [h_out, c_out] each of shape [BATCH, self.cell_size]
        """
        model_out, self._value_out, h_out, c_out = self.base_model(
            [inputs, seq_lens, state]
        )

        return model_out, [h_out, c_out]

    def value_function(self):
        """
        Returns a tensor of shape [BATCH * T] representing the value function for the most recent forward pass
        """
        return tf.reshape(self._value_out, [-1])

    def get_initial_state(self):
        """
        Returns the initial hidden state for the LSTM
        """
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]
