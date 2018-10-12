import os
import sys
import tensorflow as tf
import threading
import multiprocessing
import argparse
import functools

import estimators
from env.lab import LabEnvironment as Environment
from worker import AC_Worker


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('level', nargs='?', default='nav_corridor')
parser.add_argument('--model_dir', default='./model_lab/')
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
    global_net = estimators.AC_Network('global', len(Environment.ACTIONS))

    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(
        1e-5, 0.99, 0.95, 1e-2, use_locking=True)

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

        worker_net = estimators.AC_Network(
            name=worker_name,
            n_out=len(Environment.ACTIONS),
            global_net=global_net,
            optimizer=optimizer,
            add_summary=worker_summary is not None)

        worker = AC_Worker(
            name=worker_name,
            env=Environment(level=args.level, plot=enable_viewport),
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
