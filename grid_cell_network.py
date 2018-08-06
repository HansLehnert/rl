import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
import functools
import argparse
import os
import trajectory
import input_fn

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
        sigma=0.1,
        kappa=20,
        peek=0.05,
        **kwargs):
    """Create function for generating random walk input data.

    batch_size: number of trajectories to generate for each input dataset
    place: place cells locations (used for calculating the groundtruth)
    head: head cells locations (used for calculating the groundtruth)
    scene: description of the scene
    sigma: place cell scale factor
    kappa: head cell scale factor
    """
    features = {'speed': [], 'position': [], 'peek': []}
    labels = []

    np.random.seed(None)

    for i in range(batch_size):
        # Generate feature vectors from random walk
        # initial_position = np.array([1.1, 1.1])
        # initial_rotation = 0
        initial_position = np.random.uniform(0.1, 2.1, (2,))
        initial_rotation = np.random.uniform(-np.pi, np.pi)

        data = trajectory.generate_path(
            initial_position,
            initial_rotation,
            **kwargs)

        steps = data['position'].shape[0]

        features['speed'].append(np.vstack([
            data['disp_speed'],
            np.sin(data['turn_speed'] * 0.02),
            np.cos(data['turn_speed'] * 0.02)]).T)
        features['position'].append(data['position'])

        # Compute activations based on position
        place_activations = np.zeros((steps, place.shape[0]))
        for n, cell in enumerate(place):
            act = data['position'] - cell
            act = np.array([np.dot(v, v) for v in act])
            act = np.exp(- act / 2 / sigma ** 2)
            place_activations[:, n] = act
        place_activations = row_normalize(place_activations)

        head_activations = np.zeros((steps, head.size))
        for n, cell in enumerate(head):
            act = np.exp(kappa * np.cos(data['rotation'] - cell))
            head_activations[:, n] = act
        head_activations = row_normalize(head_activations)

        activations = np.hstack([place_activations, head_activations])
        labels.append(activations)

        activation_peek = np.zeros(activations.shape)
        # activation_peek = np.zeros(activations.shape + np.array([0, 1]))
        for t in range(activation_peek.shape[0]):
            if t == 0 or np.random.rand() < peek:
                # activation_peek[t, -1] = 1
                activation_peek[t, :] = activations[t, :] / peek

        features['peek'].append(activation_peek)

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
    # features = {k: tf.cast(features[k], dtype=tf.float32) for k in features}
    # labels = tf.cast(labels, dtype=tf.float32)

    # input_layer = tf.concat([features['speed'], features['peek']], 2)
    # input_layer = features['speed']

    # Single layer RNN
    # initial_c = tf.layers.dense(
    #     features['peek'][:, 0, :], 128, use_bias=False, name='initial_c')
    # initial_h = tf.layers.dense(
    #     features['peek'][:, 0, :], 128, use_bias=False, name='initial_h')
    cell = tf.contrib.rnn.BasicLSTMCell(128)

    lstm_output, lstm_state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=features,
        # initial_state=tf.contrib.rnn.LSTMStateTuple(initial_c, initial_h))
        initial_state=cell.zero_state(features.shape[0], tf.float32))

    # Single layer fully connected with 0.5 dropout.
    # Should exhibit grid like activity
    linear = tf.layers.dense(
        lstm_output,
        512, name='grid_code',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
    dropout = tf.layers.dropout(linear, rate=0.5)

    # Output layer
    place_output = tf.layers.dense(
        # dropout, 256, activation=tf.nn.softmax, name='place_output')
        dropout, 256, name='place_output')
    head_output = tf.layers.dense(
        # dropout, 12, activation=tf.nn.softmax, name='head_output')
        dropout, 12, name='head_output')

    # output = tf.concat([place_output, head_output], 2)

    # loss = cross_entropy_loss(labels, output)
    loss = tf.losses.softmax_cross_entropy(labels[:, :, :256], place_output)
    loss += tf.losses.softmax_cross_entropy(labels[:, :, -12:], head_output)

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=1e-5,
        momentum=0.9)

    gradients = optimizer.compute_gradients(loss)

    clipped_gradients = [(tf.clip_by_value(g, -1e-5, 1e-5), v) for g, v in gradients]

    train_op = optimizer.apply_gradients(
        clipped_gradients, global_step=tf.train.get_global_step())
    # train_op = optimizer.mini(
    #     loss, global_step=tf.train.get_global_step())

    estimator = tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op,
        predictions={
            # 'position': features['position'],
            'place_cell': tf.nn.softmax(place_output),
            'head_cell': tf.nn.softmax(head_output),
            'grid_code': linear,
            'labels': labels})

    return estimator


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./model')
    subparsers = parser.add_subparsers(title='commands', dest='command')
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
        model_dir=args.model_dir,
    )

    if args.command == 'train':
        estimator.train(
            input_fn=functools.partial(
                input_fn.generate_dataset,
                batch_size=10,
                place=place_cells,
                head=head_cells,
                scene=scene,
                time=2),
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
                generate_input,
                batch_size=10,
                place=place_cells,
                head=head_cells,
                scene=scene,
                time=2))

        position = None
        grid_code = None

        for n, x in enumerate(predictions):
            if n == 0:
                position = x['position']
                grid_code = x['grid_code']
            else:
                if n >= 10:
                    break

                position = np.vstack([position, x['position']])
                grid_code = np.vstack([grid_code, x['grid_code']])
                continue

            for i in range(x['place_cell'].shape[1]):
                if (np.all(x['labels'][:, i] < 0.05)
                        and np.all(x['place_cell'][:, i] < 0.05)):
                    continue

                plt.clf()

                plt.subplot(121)
                plt.scatter(place_cells[:, 0], place_cells[:, 1])
                plt.scatter(place_cells[i, 0], place_cells[i, 1])
                plt.plot(x['position'][:, 0], x['position'][:, 1], 'r')

                plt.subplot(122)
                plt.plot(x['labels'][:, i])
                plt.plot(x['place_cell'][:, i])
                plt.ylim(0, 1)
                plt.savefig('fig/place_{}.png'.format(i))

            for i in range(x['head_cell'].shape[1]):
                if (np.all(x['labels'][:, i + 256] < 0.05)
                        and np.all(x['place_cell'][:, i] < 0.05)):
                    continue

                plt.clf()

                plt.plot(x['labels'][:, i + 256])
                plt.plot(x['head_cell'][:, i])
                plt.ylim(0, 1)
                plt.savefig('fig/head_{}.png'.format(i))

        print(position[10::100, :])
        grid_x, grid_y = np.mgrid[0:2.2:0.01, 0:2.2:0.01]
        for i in range(grid_code.shape[1]):
            z = scipy.interpolate.griddata(
                position, grid_code[:, i], (grid_x, grid_y))

            plt.clf()
            plt.imshow(z, extent=(0, 2.2, 0, 2.2))
            plt.savefig('fig/grid_{}.png'.format(i))

            break
