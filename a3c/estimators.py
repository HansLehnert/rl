import tensorflow as tf
import numpy as np


class AC_Network:
    def __init__(
            self, n_out, optimizer, input_dim=None, name='', add_summary=True,
            **kwargs):
        self.name = name
        self.optimizer = optimizer
        self.add_summary = add_summary

        if input_dim is None:
            self.input_dim = [84, 84, 3]
        else:
            self.input_dim = input_dim

        # Empty RNN states definitions for networks that don't use RNN
        # NOTE: Move to get function?
        self.rnn_initial_state = []
        self.rnn_state_in = []
        self.rnn_state_out = []

        # Create network
        with tf.variable_scope(self.name, '', reuse=tf.AUTO_REUSE):
            self._create_shared_network()
            self._create_ac_network(n_out)
            self._create_gradient_op(n_out)

            self.summaries = tf.summary.merge_all(
                tf.GraphKeys.SUMMARIES, self.name)
            if self.summaries is None:
                self.summaries = tf.no_op()

    def _create_shared_network(self):
        """Defines input processing part of the network.

        Override to change network architecture."""

        # Image input
        # TODO: Allow size configuration
        self.input = tf.placeholder(
            shape=[None, None] + self.input_dim, dtype=tf.uint8, name='input_')
        input_shape = tf.shape(self.input)
        batch_size = input_shape[0]
        time_size = input_shape[1]

        input_normalized = tf.to_float(self.input) * 2 / 255.0 - 1

        # Convolutional layers
        conv1 = tf.layers.conv3d(
            inputs=input_normalized,
            filters=16,
            kernel_size=(1, 8, 8),
            strides=(1, 4, 4),
            activation=tf.nn.elu,
            name='conv1'
        )
        conv2 = tf.layers.conv3d(
            inputs=conv1,
            filters=32,
            kernel_size=(1, 4, 4),
            strides=(1, 2, 2),
            activation=tf.nn.elu,
            name='conv2'
        )

        flattened_conv = tf.reshape(
            tensor=conv2,
            shape=[batch_size, time_size, np.prod(conv2.shape[2:])],
            name='flatten'
        )

        # Fully connected layer
        dense = tf.layers.dense(
            flattened_conv, 256, name="fc1", activation=tf.nn.elu)

        # LSTM
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(256)
        lstm_state_size = lstm_cell.state_size
        self.rnn_initial_state = lstm_cell.zero_state(1, tf.float32)
        self.rnn_state_in = (
            tf.placeholder(
                tf.float32, [None, lstm_state_size.c], name='rnn_c'),
            tf.placeholder(
                tf.float32, [None, lstm_state_size.h], name='rnn_h')
        )
        initial_state = tf.contrib.rnn.LSTMStateTuple(*self.rnn_state_in)
        lstm_out, lstm_state = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=dense,
            initial_state=initial_state,
            scope='lstm1'
        )
        self.rnn_state_out = lstm_state

        self.shared_output = lstm_out
        # self.shared_output = tf.expand_dims(dense1, 0)

        # Create summaries
        if self.add_summary:
            # Weights
            conv1_kernel = tf.transpose(
                tf.squeeze(tf.get_variable('conv1/kernel')), [3, 0, 1, 2])
            tf.summary.image('Kernel/Conv1', conv1_kernel, max_outputs=16)

            # tf.summary.scalar(
            #     'Weights/Conv1', tf.reduce_sum(tf.abs(conv1_kernel)))

            # conv2_kernel = tf.get_variable('conv2/kernel')
            # tf.summary.scalar(
            #     'Weights/Conv2', tf.reduce_sum(tf.abs(conv2_kernel)))

            # dense1_kernel = tf.get_variable('fc1/kernel')
            # tf.summary.scalar(
            #     'Weights/Dense', tf.reduce_sum(tf.abs(dense1_kernel)))

            # # Activations
            # activation_vars = [
            #     (conv1, 'Conv1'), (conv1, 'Conv2'), (dense1, 'Dense')
            # ]
            # for tensor, name in activation_vars:
            #     zero = tf.constant(0, dtype=tf.float32)
            #     activation = tf.cast(tf.less_equal(tensor, zero), tf.float32)
            #     activation = tf.reduce_sum(activation)
            #     activation /= tf.cast(tf.size(tensor), tf.float32)
            #     tf.summary.scalar('Zeros/{}'.format(name), activation)

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

    def _create_gradient_op(self, n_out):
        """Create training operations."""

        self.actions = tf.placeholder(
            shape=[None, None], dtype=tf.int32, name='actions')
        self.target_value = tf.placeholder(
            shape=[None, None], dtype=tf.float32, name='target_value')
        self.advantages = tf.placeholder(
            shape=[None, None], dtype=tf.float32, name='advantages')

        actions_onehot = tf.one_hot(self.actions, n_out, dtype=tf.float32)

        responsible_outputs = tf.reduce_sum(self.policy * actions_onehot, 2)
        self.selected_policy = responsible_outputs

        # Loss functions
        value_loss = tf.reduce_mean(
            tf.square(self.target_value - tf.squeeze(self.value)))
        entropy_loss = -tf.reduce_mean(self.policy * tf.log(self.policy))
        policy_loss = tf.reduce_mean(
            tf.log(responsible_outputs) * self.advantages)
        loss = (
            0.5 * value_loss
            - policy_loss
            - 0.0005 * entropy_loss
        )

        # Get gradients from local network using local losses
        self.local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        gradients = self.optimizer.compute_gradients(loss, self.local_vars)
        gradients = [g[0] for g in gradients]
        self.gradients_out, self.gradient_norm = tf.clip_by_global_norm(
            gradients, 50)

        self.gradients_in = []
        for tensor in self.gradients_out:
            self.gradients_in.append(
                tf.placeholder(tensor.dtype, tensor.shape))

        self.apply_gradients = self.optimizer.apply_gradients(
            zip(self.gradients_in, self.local_vars))

        # Load global variables
        self.assign = []
        self.vars_in = []
        for var in self.local_vars:
            placeholder = tf.placeholder(var.dtype, var.shape)
            assign_op = tf.assign(var, placeholder)
            self.vars_in.append(placeholder)
            self.assign.append(assign_op)

        # Create summaries
        if self.add_summary:
            tf.summary.scalar('Loss/Value', value_loss,)
            tf.summary.scalar('Loss/Policy', policy_loss,)
            tf.summary.scalar('Loss/Entropy', entropy_loss,)
            tf.summary.scalar('Loss/Total', loss,)

            tf.summary.scalar('Perf/Value', tf.reduce_mean(self.value))
            tf.summary.scalar(
                'Perf/Policy', tf.reduce_mean(responsible_outputs))
