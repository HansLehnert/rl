import os
import sys
import tensorflow as tf
import threading
import multiprocessing
import argparse
import functools

import estimators
from env.maze import MazeEnvironment as Environment
from worker import AC_Worker


class Network(estimators.AC_Network):
    def _create_shared_network(self):
        # Image input
        # TODO: Allow size configuration
        self.input = tf.placeholder(
            shape=[None, 128, 128, 3], dtype=tf.uint8, name='input_')

        input_normalized = tf.to_float(self.input) / 255.0

        # Convolutional layers
        conv1 = tf.layers.conv2d(
            input_normalized, 16, 8, 4, activation=tf.nn.elu, name='conv1')
        conv2 = tf.layers.conv2d(
            conv1, 32, 4, 2, activation=tf.nn.elu, name='conv2')

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

        # Create summaries
        if self.add_summary:
            # Weights
            conv1_kernel = tf.transpose(
                tf.get_variable('conv1/kernel'), [3, 0, 1, 2])
            tf.summary.scalar(
                'Weights/Conv1', tf.reduce_sum(tf.abs(conv1_kernel)))
            tf.summary.image('Kernel/Conv1', conv1_kernel, max_outputs=20)

            conv2_kernel = tf.get_variable('conv2/kernel')
            tf.summary.scalar(
                'Weights/Conv2', tf.reduce_sum(tf.abs(conv2_kernel)))

            dense1_kernel = tf.get_variable('fc1/kernel')
            tf.summary.scalar(
                'Weights/Dense', tf.reduce_sum(tf.abs(dense1_kernel)))


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./model_maze/')
parser.add_argument('--t_max', type=int, default=100)
parser.add_argument('-n', type=int)
parser.add_argument('--viewport', action='store_true')
args = parser.parse_args(sys.argv[1:])

model_dir = args.model_dir
checkpoint_dir = os.path.join(model_dir, 'checkpoint')

# Set the number of workers
n_workers = args.n
if n_workers is None:
    n_workers = multiprocessing.cpu_count()

summary_writer = tf.summary.FileWriter(model_dir)

# with tf.device('/cpu:0'):
if True:
    global_step = tf.train.get_or_create_global_step()

    # Global network, to be updated by worker threads
    global_net = Network('global', len(Environment.ACTIONS))

    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-5, 0.99, 0.95, 1e-2)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0)

    # Create workers graphs
    workers = []
    for worker_id in range(n_workers):
        worker_name = 'worker_{}'.format(worker_id)

        enable_viewport = False
        worker_summary = None
        if worker_id == 0:
            enable_viewport = args.viewport
            worker_summary = summary_writer

        worker_net = Network(
            name=worker_name,
            n_out=len(Environment.ACTIONS),
            global_net=global_net,
            optimizer=optimizer,
            add_summary=worker_summary is not None)

        worker = AC_Worker(
            name=worker_name,
            env=Environment(plot=enable_viewport, num_walls=0),
            global_net=global_net,
            local_net=worker_net,
            saver=saver,
            checkpoint_dir=checkpoint_dir,
            optimizer=optimizer,
            summary_writer=worker_summary,
        )
        workers.append(worker)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    summary_writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    # Restore checkpoints
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoints is not None:
        saver.recover_last_checkpoints(checkpoints.all_model_checkpoint_paths)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, latest_checkpoint)
        print('Restored checkpoint {}'.format(latest_checkpoint))

    # Start worker threads
    worker_threads = []
    for n, worker in reversed(list(enumerate(workers))):
        worker_fn = functools.partial(worker.run, sess, coord, args.t_max)

        if n != 0:
            thread = threading.Thread(target=worker_fn)
            thread.start()
            worker_threads.append(thread)
        else:
            worker_fn()

    # Wait for all workers to finish
    coord.join(worker_threads)
