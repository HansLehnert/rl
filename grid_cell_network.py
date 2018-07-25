import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import trajectory
import functools
import argparse
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
# import matplotlib.pyplot as plt


def generate_place_cells(n, x1, y1, x2, y2, seed=0):
    """Generate place cells positions."""
    np.random.seed(seed)

    place = np.vstack([
        np.random.uniform(x1, x2, (1, n)),
        np.random.uniform(y1, y2, (1, n))])

    return place.T


def generate_head_cells(n, seed=1):
    """Generate head cell position."""
    np.random.seed(seed)

    return np.random.uniform(-np.pi, np.pi, (n, 1))


def row_normalize(a):
    """Normalize each row of matrix a and return the resulting matrix."""
    for row in a:
        acc = np.sum(row)
        if acc != 0:
            row /= acc
    return a


def generate_input(
        batch_size,
        place,
        head,
        scene,
        sigma=0.1,
        kappa=20,
        length=100):
    """Create function for generating random walk input data.

    batch_size: number of trajectories to generate for each input dataset
    place: place cells locations (used for calculating the groundtruth)
    head: head cells locations (used for calculating the groundtruth)
    scene: description of the scene
    sigma: place cell scale factor
    kappa: head cell scale factor
    length: amount of steps for each trajectory
    """
    features = {'speed': [], 'position': []}
    labels = []

    # np.random.seed(None)

    for i in range(batch_size):
        # Generate feature vectors from random walk
        initial_position = np.random.uniform(0.1, 2.1, (2,))
        initial_rotation = np.random.uniform(-np.pi, np.pi)
        data = trajectory.generate_path(
            initial_position,
            initial_rotation,
            scene)

        features['speed'].append(np.vstack([
            data['disp_speed'][0:length],
            np.sin(data['turn_speed'][0:length]),
            np.cos(data['turn_speed'][0:length])]).T)
        features['position'].append(data['position'][:length, :])

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

    features = {k: np.array(features[k]) for k in features}
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


def model_grid_network(features, labels, mode):
    """Create the tensorflow model for the grid cell network."""
    # Input layer has speed and angle vector components as input
    features = {k: tf.cast(features[k], dtype=tf.float32) for k in features}
    labels = tf.cast(labels, dtype=tf.float32)

    # Single layer RNN
    initial_c = tf.layers.dense(
        labels[:, 0, :], 128, use_bias=False, name='initial_c')
    initial_h = tf.layers.dense(
        labels[:, 0, :], 128, use_bias=False, name='initial_h')

    lstm_output, lstm_state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.BasicLSTMCell(128),
        inputs=features['speed'],
        initial_state=tf.contrib.rnn.LSTMStateTuple(initial_c, initial_h))

    # Single layer fully connected with 0.5 dropout.
    # Should exhibit grid like activity
    linear = tf.layers.dense(
        lstm_output,
        512, name='grid_code',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00001))
    dropout = tf.layers.dropout(linear, rate=0.5)

    # Output layer
    place_output = tf.layers.dense(
        dropout, 256, activation=tf.nn.softmax, name='place_output')
    head_output = tf.layers.dense(
        dropout, 12, activation=tf.nn.softmax, name='head_output')

    output = tf.concat([place_output, head_output], 2)

    loss = cross_entropy_loss(labels, output)

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=0.00001,
        momentum=0.9)

    train_op = optimizer.minimize(
        loss, global_step=tf.train.get_global_step())

    estimator = tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op,
        predictions={
            'position': features['position'],
            'place_cell': output,
            'grid_code': linear,
            'labels': labels})

    return estimator


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='commands', dest='command')
    # subparsers.required = True

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('max_steps', type=int)

    predict_parser = subparsers.add_parser('predict')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    # Create scene
    scene = trajectory.generate_square_scene(2.2)

    # Generate place and head cells distribution
    place_cells = generate_place_cells(256, 0, 0, 2.2, 2.2)
    head_cells = generate_head_cells(12)

    # Create estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_grid_network,
        model_dir='./model',
    )

    if args.command == 'train':
        estimator.train(
            input_fn=functools.partial(
                generate_input, 10, place_cells, head_cells, scene),
            max_steps=args.max_steps
        )
    elif args.command == 'predict':
        os.system("rm ./fig/*")

        # For this to work the predict() function of the Estimator class
        # was modified to also pass the labels to the model_fn (which needs
        # them for RNN state initialization). In the future, the data
        # generation function should be modified to avoid using the 'labels'
        # data for prediction.
        predictions = estimator.predict(
            input_fn=functools.partial(
                generate_input, 1, place_cells, head_cells, scene))

        for x in predictions:
            for i in range(x['place_cell'].shape[1]):
                if (np.all(x['labels'][:, i] < 0.05)
                        and np.all(x['place_cell'][:, i] < 0.05)):
                    continue

                plt.clf()

                if i < 256:
                    plt.subplot(121)
                    plt.scatter(place_cells[:, 0], place_cells[:, 1])
                    plt.scatter(place_cells[i, 0], place_cells[i, 1])
                    plt.plot(x['position'][:, 0], x['position'][:, 1], 'r')

                    plt.subplot(122)

                plt.plot(x['labels'][:, i])
                plt.plot(x['place_cell'][:, i])
                plt.ylim(0, 1)
                plt.savefig('fig/{}.png'.format(i))

            break
    else:
        features, labels = generate_input(1, place_cells, head_cells, scene)
        print(labels.shape)

        plt.plot(np.sum(labels[0], 1))
        plt.show()
