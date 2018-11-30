import tensorflow as tf


class AC_Network:
    def __init__(
            self,
            n_out,
            optimizer,
            input_dim=None,
            add_summary=True,
            prediction_loss=False,
            reward_feedback=False,
            visual_depth=1,
            temporal_stride=1,
            kernel=None,
            name='',
    ):
        self.name = name
        self.optimizer = optimizer
        self.add_summary = add_summary
        self.n_out = n_out

        # Default balues
        if input_dim is None:
            input_dim = [84, 84, 3]

        # Empty RNN states definitions for networks that don't use RNN
        # NOTE: Move to get function?
        self.rnn_initial_state = []
        self.rnn_state_in = []
        self.rnn_state_out = []

        self.inputs = []

        # Create network
        with tf.variable_scope(self.name, '', reuse=tf.AUTO_REUSE):
            self._create_shared_network(
                input_dim,
                reward_feedback,
                visual_depth,
                temporal_stride,
                kernel
            )
            self._create_ac_network()
            self._create_loss(prediction_loss)
            self._create_gradient_op()
            self._merge_summaries()

    def _create_shared_network(
            self,
            input_dim,
            reward_feedback,
            visual_depth,
            temporal_stride,
            kernel,
    ):
        """Defines input processing part of the network.

        Override to change network architecture."""

        # Inputs
        visual_input = tf.placeholder(
            shape=[None] + input_dim, dtype=tf.uint8, name='visual_input')
        self.inputs.append(visual_input)

        if reward_feedback:
            reward_input = tf.placeholder(
                shape=[None], dtype=tf.float32, name='reward_input')
            self.inputs.append(reward_input)

        with tf.variable_scope('image_normalization'):
            # Normalization
            visual_input = tf.to_float(visual_input) * 2 / 255.0 - 1
            visual_input = tf.expand_dims(visual_input, 0)

        # Convolutional layers
        if kernel is not None:
            with tf.variable_scope('visual_filter'):
                spatial_kernels = tf.constant(
                    kernel['spatial'],
                    dtype=tf.float32,
                    name='spatial_kernel')
                temporal_kernels = tf.constant(
                    kernel['temporal'],
                    dtype=tf.float32,
                    name='temporal_kernel')
                n_kernel = spatial_kernels.shape[-1]
                temporal_kernels = tf.split(
                    temporal_kernels, n_kernel, axis=-1)

                visual_input = tf.transpose(visual_input, (1, 4, 2, 3, 0))
                spatial_filter = tf.nn.conv3d(
                    input=visual_input,
                    filter=spatial_kernels,
                    strides=(1, 1, 4, 4, 1),
                    padding='SAME',
                    name='visual_conv'
                )
                spatial_filter = tf.transpose(spatial_filter, (1, 0, 2, 3, 4))
                spatial_filter = tf.split(spatial_filter, n_kernel, axis=-1)

                visual_input = []
                for spatial, temporal in zip(spatial_filter, temporal_kernels):
                    visual_input.append(
                        tf.nn.conv3d(
                            input=spatial,
                            filter=temporal,
                            strides=(1, temporal_stride, 1, 1, 1),
                            padding='VALID',
                            name='temporal_conv'
                        )
                    )

                visual_input = tf.concat(visual_input, 0)
                visual_input = tf.transpose(visual_input, [4, 1, 2, 3, 0])
            conv1 = visual_input
        else:
            conv1 = tf.layers.conv3d(
                inputs=visual_input,
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

        flattened_conv = tf.layers.flatten(tf.squeeze(conv2, axis=[0]))

        # Fully connected layer
        dense = tf.layers.dense(
            flattened_conv, 256, name="fc1", activation=tf.nn.elu)
        self.internal_representation = dense

        if reward_feedback:
            lstm_input = tf.concat(
                [dense, tf.expand_dims(reward_input, -1)], 1)
        else:
            lstm_input = dense

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
            inputs=tf.expand_dims(lstm_input, 0),
            initial_state=initial_state,
            scope='lstm1'
        )
        self.rnn_state_out = lstm_state

        self.shared_output = lstm_out
        # self.shared_output = tf.expand_dims(dense1, 0)

        # Create summaries
        if self.add_summary:
            # Weights
            if kernel is None:
                conv1_kernel = tf.transpose(
                    tf.reduce_mean(
                        tf.squeeze(tf.get_variable('conv1/kernel')),
                        keepdims=True,
                        axis=2
                    ),
                    [3, 0, 1, 2]
                )
                tf.summary.image(
                    'Conv1Spatial', conv1_kernel, max_outputs=16)
            else:
                input_channels = tf.transpose(
                    visual_input[:, 0, ...], (3, 1, 2, 0))
                tf.summary.image(
                    'VisualInput', input_channels, max_outputs=6)

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

    def _create_ac_network(self):
        """Create network layers for policy and value outputs."""

        self.policy = tf.layers.dense(
            inputs=self.shared_output,
            units=self.n_out,
            activation=tf.nn.softmax,
            name='policy'
        )

        self.value = tf.layers.dense(
            inputs=self.shared_output, units=1, name='value')

    def _create_loss(self, prediction_loss):
        """Create loss computation operations."""
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name='actions')
        self.target_value = tf.placeholder(
            shape=[None], dtype=tf.float32, name='target_value')
        self.advantages = tf.placeholder(
            shape=[None], dtype=tf.float32, name='advantages')

        self.actions_onehot = tf.one_hot(
            self.actions, self.n_out, dtype=tf.float32)

        responsible_outputs = tf.reduce_sum(
            self.policy * self.actions_onehot, 2)
        self.selected_policy = responsible_outputs

        # Loss functions
        value_loss = tf.reduce_mean(
            tf.square(self.target_value - tf.squeeze(self.value)))
        entropy_loss = -tf.reduce_mean(self.policy * tf.log(self.policy))
        policy_loss = tf.reduce_mean(
            tf.log(responsible_outputs) * self.advantages)
        self.loss = (
            0.5 * value_loss
            - policy_loss
            - 0.0005 * entropy_loss
        )

        if prediction_loss:
            # State prediction
            prediction_input = tf.concat(
                [self.internal_representation, self.actions_onehot], 1)
            state_prediction = tf.layers.dense(
                prediction_input,
                256,
                name='state_prediction',
                activation=tf.nn.elu)

            prediction_error = (
                state_prediction[:-1, :]
                - self.internal_representation[1:, :]
            )
            state_loss = tf.reduce_mean(tf.square(prediction_error))

            # Inverse kinematics
            dense = tf.layers.dense(
                tf.concat(
                    [self.internal_representation[:-1],
                        self.internal_representation[:-1]],
                    1),
                64
            )

            predicted_action = tf.layers.dense(dense, self.n_out)

            kinematics_loss = tf.losses.softmax_cross_entropy(
                self.actions_onehot[:-1, :], predicted_action)

            self.loss += 0.01 * state_loss + 0.005 * kinematics_loss

        # Create summaries
        if self.add_summary:
            tf.summary.scalar('Loss/Value', value_loss)
            tf.summary.scalar('Loss/Policy', policy_loss)
            tf.summary.scalar('Loss/Entropy', entropy_loss)
            tf.summary.scalar('Loss/Total', self.loss)

            if prediction_loss:
                tf.summary.scalar('Loss/StatePrediction', state_loss)
                tf.summary.scalar('Loss/ActionPrediction', kinematics_loss)

            tf.summary.scalar('Perf/Value', tf.reduce_mean(self.value))
            tf.summary.scalar(
                'Perf/Policy', tf.reduce_mean(responsible_outputs))

    def _create_gradient_op(self):
        """Create training operations."""

        # Get gradients from local network using local losses
        self.local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        gradients = self.optimizer.compute_gradients(
            self.loss, self.local_vars)
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

    def _merge_summaries(self):
        self.summaries = tf.summary.merge_all(
            tf.GraphKeys.SUMMARIES, self.name)
        if self.summaries is None:
            self.summaries = tf.no_op()
