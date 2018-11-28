import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import functools
import copy
import argparse
import os

import trajectory
import input_fn


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


def model_grid_network(features, labels, mode):
    """Create the tensorflow model for the grid cell network."""
    cell = tf.contrib.rnn.BasicLSTMCell(128)

    lstm_output, lstm_state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=features,
        initial_state=cell.zero_state(features.shape[0], tf.float32))

    # Single layer fully connected with 0.5 dropout.
    # Should exhibit grid like activity
    linear = tf.layers.dense(
        lstm_output,
        512, name='grid_code',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
    dropout = tf.layers.dropout(
        linear,
        rate=0.5,
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Output layer
    place_output = tf.layers.dense(
        dropout, 256, name='place_output')
    head_output = tf.layers.dense(
        dropout, 12, name='head_output')

    loss = tf.losses.softmax_cross_entropy(labels[:, :, :256], place_output)
    loss += tf.losses.softmax_cross_entropy(labels[:, :, -12:], head_output)

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=1e-5,
        momentum=0.9)

    gradients = optimizer.compute_gradients(loss)

    clipped_gradients = [
        (tf.clip_by_value(g, -1e-5, 1e-5), v) for g, v in gradients]

    train_op = optimizer.apply_gradients(
        clipped_gradients, global_step=tf.train.get_global_step())

    estimator = tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op,
        predictions={
            'place_cell': tf.nn.softmax(place_output),
            'head_cell': tf.nn.softmax(head_output),
            'grid_code': linear})

    return estimator


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./model')
    subparsers = parser.add_subparsers(title='commands', dest='command')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('max_steps', type=int)
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('max_steps', type=int)
    args = parser.parse_args()

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
        tf.logging.set_verbosity(tf.logging.INFO)

        estimator.train(
            input_fn=functools.partial(
                input_fn.create_dataset,
                batch_size=10,
                place=place_cells,
                head=head_cells,
                scene=scene,
                time=2),
            max_steps=args.max_steps)
    elif args.command == 'predict':
        os.system("rm ./fig/*")

        rand_state = np.random.RandomState()
        input_args = {
            'time': 2,
            'place': place_cells,
            'head': head_cells,
            'scene': scene}

        predictions = estimator.predict(
            input_fn=functools.partial(
                input_fn.create_dataset,
                batch_size=1,
                n_data=args.max_steps,
                random=copy.copy(rand_state),
                **input_args))

        n_bins = 64

        activation_acc = np.zeros([512, n_bins, n_bins])
        activation_count = np.zeros(activation_acc.shape)

        print('Generating and evaluating data...')
        for n, x in enumerate(predictions):
            if n % 100 == 0:
                print('{}/{}'.format(n, args.max_steps))

            data = input_fn.generate_activations(
                random=rand_state,
                **input_args)

            if n == 0:
                for i in range(x['place_cell'].shape[1]):
                    if (np.all(data['place_activations'][:, i] < 0.05) and
                            np.all(x['place_cell'][:, i] < 0.05)):
                        continue

                    plt.clf()

                    plt.subplot(121)
                    plt.scatter(place_cells[:, 0], place_cells[:, 1])
                    plt.scatter(place_cells[i, 0], place_cells[i, 1])
                    plt.plot(
                        data['position'][:, 0], data['position'][:, 1], 'r')

                    plt.subplot(122)
                    plt.plot(data['place_activations'][:, i])
                    plt.plot(x['place_cell'][:, i])
                    plt.ylim(0, 1)
                    plt.savefig('fig/place_{}.png'.format(i))

                for i in range(x['head_cell'].shape[1]):
                    if (np.all(data['head_activations'][:, i] < 0.05) and
                            np.all(x['place_cell'][:, i] < 0.05)):
                        continue

                    plt.clf()

                    plt.plot(data['head_activations'][:, i])
                    plt.plot(x['head_cell'][:, i])
                    plt.ylim(0, 1)
                    plt.savefig('fig/head_{}.png'.format(i))

            # Accumulate data for activation maps
            for i in range(x['grid_code'].shape[0]):
                bin_x = int(data['position'][i, 0] * n_bins / 2.2)
                bin_y = int(data['position'][i, 1] * n_bins / 2.2)

                for j in range(x['grid_code'].shape[1]):
                    activation_acc[j, bin_x, bin_y] += x['grid_code'][i, j]
                    activation_count[j, bin_x, bin_y] += 1

        print('Plotting...')

        activation_maps = activation_acc / activation_count
        activation_maps = np.nan_to_num(activation_maps)

        for i in range(0, activation_maps.shape[0], 8):
            plt.clf()

            for j in range(8):
                plt.subplot(4, 4, 2 * j + 1)
                plt.imshow(
                    activation_maps[i + j],
                    extent=(0, 2.2, 0, 2.2),
                    cmap='jet')

                conv = scipy.signal.correlate2d(
                    activation_maps[i + j],
                    activation_maps[i + j])

                plt.subplot(4, 4, 2 * j + 2)
                plt.imshow(conv, cmap='jet')

            plt.savefig('fig/grid_{}.png'.format(i))
            print('{}/{}'.format(i, activation_maps.shape[0]))
