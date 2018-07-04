import tensorflow as tf
import numpy as np
import trajectory
import functools

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
# import matplotlib.pyplot as plt


def generate_place_cells(n, x1, y1, x2, y2):
    """Generate place cells positions."""
    place = np.vstack([
        np.random.uniform(x1, x2, (1, n)),
        np.random.uniform(y1, y2, (1, n))])

    return place.T


def generate_head_cells(n):
    """Generate head cell position."""
    return np.random.uniform(-np.pi, np.pi, (n, 1))


def row_normalize(a):
    """Normalize each row of matrix a and return the resulting matrix."""
    for row in a:
        acc = np.sum(a)
        if acc != 0:
            row /= acc
    return a


def generate_input(
        batch_size, place, head, scene, sigma=0.1, kappa=20, length=100):
    """Create function for generating random walk input data."""
    scene = trajectory.generate_square_scene(2.2)

    features = []
    labels = []

    for i in range(batch_size):
        # Generate feature vectors from random walk
        initial_position = np.random.uniform(0.1, 2.1, (2,))
        initial_rotation = np.random.uniform(-np.pi, np.pi)
        data = trajectory.generate_path(
            initial_position,
            initial_rotation,
            scene)

        features.append(np.vstack([
            data['disp_speed'][0:length],
            np.sin(data['turn_speed'][0:length]),
            np.cos(data['turn_speed'][0:length])]).T)

        # Compute activations based on position
        place_activations = np.zeros((length, place.shape[0]))
        for n, cell in enumerate(place):
            act = data['position'][0:length, :] - cell
            act = np.array([np.dot(v, v) for v in act])
            act = np.exp(- act / 2 / sigma ** 2)
            place_activations[:, n] = act
        place_activations = row_normalize(place_activations)

        head_activations = np.zeros((length, head.size))
        for n, cell in enumerate(head):
            act = np.exp(kappa * np.cos(data['rotation'][0:length] - cell))
            head_activations[:, n] = act
        head_activations = row_normalize(head_activations)

        labels.append(np.hstack([place_activations, head_activations]))

    features = np.array(features)
    labels = np.array(labels)

    return (features, labels)


def cross_entropy_loss(
        labels,
        predictions,
        weights=1.0,
        scope=None,
        loss_collection=ops.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Implements a cross entropy loss function.

    Based on tensorflow's log_loss implementation.
    """

    values = (predictions, labels, weights)
    with ops.name_scope(scope, "cross_entropy_loss", values) as scope:
        labels = math_ops.to_float(labels)
        predictions = math_ops.to_float(predictions)
        losses = -math_ops.multiply(labels, math_ops.log(predictions))
        return tf.losses.compute_weighted_loss(
            losses, weights, scope, loss_collection, reduction=reduction)


def model_grid_network(features, labels, mode, params):
    """Create the tensorflow model for the grid cell network."""
    # Input layer has speed and angle vector components as input
    features = tf.cast(features, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.float32)

    # Single layer RNN
    initial_c = tf.layers.dense(labels[:, 0, :], 128, name='initial_c')
    initial_h = tf.layers.dense(labels[:, 0, :], 128, name='initial_h')
    # print(initial_c)
    # cell = tf.contrib.rnn.BasicLSTMCell(128)
    # print(list([[10, s] for s in cell.state_size]))

    lstm_output, lstm_state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.BasicLSTMCell(128),
        inputs=features,
        initial_state=tf.contrib.rnn.LSTMStateTuple(initial_c, initial_h))

    # Single layer fully connected with 0.5 dropout.
    # Should exhibit grid like activity
    linear = tf.layers.dense(lstm_output, 512)
    dropout = tf.layers.dropout(linear, rate=0.5)

    # Output layer
    output_layer = tf.layers.dense(dropout, 268, activation=tf.nn.softmax)

    loss = cross_entropy_loss(labels, output_layer)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=0.00001,
            momentum=0.9)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        pass

    if mode == tf.estimator.ModeKeys.PREDICT:
        pass


if __name__ == '__main__':
    # Generate trajectories
    scene = trajectory.generate_square_scene(2.2)

    # Generate place and head cells distribution
    place_cells = generate_place_cells(256, 0, 0, 2.2, 2.2)
    head_cells = generate_head_cells(12)

    # data = generate_input(1, place_cells, head_cells)()
    # plt.scatter(
    #     *data[2][0]['position'].T, c=np.reshape(data[1][0][:, 0], [-1]))
    # plt.scatter(*place_cells.T)
    # plt.show()

    tf.logging.set_verbosity(tf.logging.INFO)

    estimator = tf.estimator.Estimator(
        model_fn=model_grid_network,
        params={},

    )

    estimator.train(
        input_fn=functools.partial(
            generate_input, 10, place_cells, head_cells, scene),
        max_steps=100000
    )
