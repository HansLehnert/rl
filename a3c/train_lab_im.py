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
parser.add_argument('level', nargs='?', default='nav_maze_random_goal_02')
parser.add_argument('-n', '--n-workers', type=int, dest='n')
parser.add_argument('--model-dir', default='./model_im/', dest='model_dir')
parser.add_argument('--t_max', type=int, default=100)
parser.add_argument('--viewport', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument(
    '--train-steps', type=int, default=10**8, dest='train_steps')
args = parser.parse_args(sys.argv[1:])

model_dir = args.model_dir
checkpoint_dir = os.path.join(model_dir, 'checkpoint')

# Set the number of workers
if args.test:
    n_workers = 1
else:
    n_workers = args.n
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

# Multiprocess queues
parameter_queue = multiprocessing.Queue()
gradient_queue = multiprocessing.Queue()

# Optimizer
optimizer = tf.train.RMSPropOptimizer(
    1e-5, 0.99, 0.95, 1e-2, use_locking=True)

# Global network, to be updated by worker threads
net = estimators.AC_Network(
    len(Environment.ACTIONS), optimizer, initialize=False)

with net.variable_scope():
    net._create_shared_network()
    net._create_ac_network()
    net._create_loss()

    # Add IM loss
    prediction_input = tf.concat(
        [net.internal_representation, net.actions_onehot], 1)
    state_prediction = tf.layers.dense(
        prediction_input,
        256,
        name='state_prediction',
        activation=tf.nn.elu)

    prediction_error = (
        state_prediction[:-1, :] - net.internal_representation[1:, :]
    )
    prediction_loss = tf.reduce_mean(tf.square(prediction_error))

    tf.summary.scalar('Loss/StatePrediction', prediction_loss)

    # Inverse kinematics loss
    dense = tf.layers.dense(
        tf.concat(
            [net.internal_representation[:-1],
                net.internal_representation[:-1]],
            1),
        64
    )

    predicted_action = tf.layers.dense(
        dense, len(Environment.ACTIONS)
    )

    kinematics_loss = tf.losses.softmax_cross_entropy(
        net.actions_onehot[:-1, :], predicted_action
    )

    net.loss += 0.001 * prediction_loss + 0.005 * kinematics_loss

    tf.summary.scalar('Loss/ActionPrediction', kinematics_loss)

    net._create_gradient_op()
    net._merge_summaries()

# Create workers graphs
workers = []
for worker_id in range(n_workers):
    worker_name = 'worker_{:02}'.format(worker_id)

    enable_viewport = False
    worker_summary = None
    if worker_id == 0:
        enable_viewport = args.viewport
        if not args.test:
            worker_summary = model_dir

    worker = AC_Worker(
        name=worker_name,
        env_class=Environment,
        env_params={
            'level': args.level,
            'plot': enable_viewport,
        },
        net=net,
        reward_clipping=1,
        param_queue=parameter_queue,
        grad_queue=gradient_queue,
        model_dir=worker_summary,
    )
    workers.append(worker)

# Start worker threads
worker_threads = []
for n, worker in reversed(list(enumerate(workers))):
    worker_fn = functools.partial(worker.run, args.t_max)
    process = multiprocessing.Process(target=worker_fn)
    process.start()
    worker_threads.append(process)

learner = Learner(
    net,
    parameter_queue,
    gradient_queue,
    model_dir,
    args.train_steps,
    n_workers,
    learn=(not args.test)
)

learner.run()
