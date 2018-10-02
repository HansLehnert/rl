import tensorflow as tf


class AC_Network():
    def __init__(self, name, n_out, global_net=None, optimizer=None):
        self.name = name
        self.optimizer = optimizer

        with tf.variable_scope(self.name):
            self._create_shared_network()
            self._create_ac_network(n_out)

            if global_net is not None:
                self._create_update_op(n_out, global_net)
                self._create_copy_op(global_net)

        # Empty RNN states definitions for networks that don't use RNN
        # NOTE: Move to get function?
        self.rnn_initial_state = []
        self.rnn_state_in = []
        self.rnn_state_out = []

    def _create_shared_network(self):
        """Defines input processing part of the network.

        Override to change network architecture."""

        # Image input
        # TODO: Allow size configuration
        self.input = tf.placeholder(
            shape=[None, 64, 64, 3], dtype=tf.uint8, name='input_')

        input_normalized = tf.to_float(self.input) / 255.0

        # Convolutional layers
        conv1 = tf.layers.conv2d(
            input_normalized, 16, 8, 4, activation=tf.nn.relu, name='conv1')
        conv2 = tf.layers.conv2d(
            conv1, 32, 4, 2, activation=tf.nn.relu, name='conv2')

        # Fully connected layer
        dense1 = tf.layers.dense(tf.layers.flatten(conv2), 256, name="fc1")

        # LSTM
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(256)
        self.rnn_initial_state = lstm_cell.zero_state(1, tf.float32)
        self.rnn_state_in = (
            tf.placeholder(
                tf.float32, self.rnn_initial_state.c.shape, name='rnn_c'),
            tf.placeholder(
                tf.float32, self.rnn_initial_state.h.shape, name='rnn_h')
        )
        lstm_out, lstm_state = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=tf.expand_dims(dense1, 0),
            initial_state=tf.contrib.rnn.LSTMStateTuple(*self.rnn_state_in),
            scope='lstm1'
        )
        self.rnn_state_out = lstm_state

        self.shared_output = lstm_out

    def _create_ac_network(self, n_out):
        """Create network layers for policy and value outputs."""

        self.policy = tf.layers.dense(
            inputs=self.shared_output,
            units=n_out,
            activation=tf.nn.softmax,
            name='policy'
        )

        self.value = tf.layers.dense(
            inputs=self.shared_output, units=1, name='value')

    def _create_update_op(self, n_out, global_net):
        """Create training operations."""

        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name='actions')
        self.target_value = tf.placeholder(
            shape=[None], dtype=tf.float32, name='target_value')
        self.advantages = tf.placeholder(
            shape=[None], dtype=tf.float32, name='advantages')

        actions_onehot = tf.one_hot(self.actions, n_out, dtype=tf.float32)

        responsible_outputs = tf.reduce_sum(self.policy * actions_onehot, 2)
        self.selected_policy = responsible_outputs

        # Loss functions
        self.value_loss = tf.reduce_sum(
            tf.square(self.target_value - tf.reshape(self.value, [-1])))
        self.entropy_loss = -tf.reduce_sum(self.policy * tf.log(self.policy))
        self.policy_loss = tf.reduce_sum(
            tf.log(responsible_outputs) * self.advantages)
        self.loss = (
            0.5 * self.value_loss
            - self.policy_loss
            - 0.01 * self.entropy_loss
        )

        # Get gradients from local network using local losses
        local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        self.gradients = self.optimizer.compute_gradients(
            self.loss, local_vars)

        # Apply local gradients to global network
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, global_net.name)
        grads_and_vars = zip([x[0] for x in self.gradients], global_vars)
        self.apply_grads = self.optimizer.apply_gradients(grads_and_vars)

    def _create_copy_op(self, global_net):
        """Create operation to copy variables from global network."""

        local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, global_net.name)

        self.copy_params = []
        for local_var, global_var in zip(local_vars, global_vars):
            assign = local_var.assign(global_var)
            self.copy_params.append(assign)
