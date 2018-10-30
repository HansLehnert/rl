import os
import sys
import tensorflow as tf
import multiprocessing
import argparse
import functools

import estimators
from env.lab import LabEnvironment as Environment
from worker import AC_Worker
from learner import Learner


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('level', nargs='?', default='nav_maze_static_01')
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
    n_workers = multiprocessing.cpu_count() - 1

# Multiprocess queues
parameter_queue = multiprocessing.Queue()
gradient_queue = multiprocessing.Queue()

# with tf.device('/cpu:0'):
if True:
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(
        1e-5, 0.99, 0.95, 1e-2, use_locking=True)

    # Global network, to be updated by worker threads
    net = estimators.AC_Network('global', len(Environment.ACTIONS), optimizer)

    # Create workers graphs
    workers = []
    for worker_id in range(n_workers):
        worker_name = 'worker_{}'.format(worker_id)

        enable_viewport = False
        # worker_summary = None
        if worker_id == 0:
            enable_viewport = args.viewport
            # worker_summary = summary_writer

        worker = AC_Worker(
            name=worker_name,
            env_class=Environment,
            env_params={
                'level': args.level,
                'plot': enable_viewport,
            },
            net=net,
            param_queue=parameter_queue,
            grad_queue=gradient_queue,
        )
        workers.append(worker)


    # coord = tf.train.Coordinator()

    # Start worker threads
    worker_threads = []
    for n, worker in reversed(list(enumerate(workers))):
        worker_fn = functools.partial(worker.run, args.t_max)
        process = multiprocessing.Process(target=worker_fn)
        process.start()
        worker_threads.append(process)

    learner = Learner(net, parameter_queue, gradient_queue, model_dir)
    learner.run(len(worker_threads))

    # Wait for all workers to finish
    # coord.join(worker_threads)
