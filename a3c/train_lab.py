import sys
import os
import multiprocessing
import argparse
import functools
import tensorflow as tf
import numpy as np

import estimators
from env.lab import LabEnvironment as Environment
from worker import AC_Worker
from learner import Learner


def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'level', nargs='?', default='nav_maze_random_goal_02',
        help='deepmindlab level used by the environment')
    parser.add_argument(
        '-n', '--n-workers', type=int, dest='n',
        help='number of worker processes')
    parser.add_argument(
        '--model-dir', default='models/lab',
        help='directory to store the trained model')
    parser.add_argument(
        '--t_max', type=int, default=50)
    parser.add_argument(
        '--viewport', action='store_true',
        help='open a viewport for the environment')
    parser.add_argument(
        '--test', action='store_true', help='disable training')
    parser.add_argument(
        '--train-steps', type=int, default=10**8,
        help='maximum number of training steps')
    parser.add_argument(
        '--beholder', action='store_true', help='enable tensorboard beholder')
    parser.add_argument(
        '--prediction', action='store_true', help='include prediction loss')
    parser.add_argument(
        '--reward-feedback', action='store_true',
        help='add reward feedback into the network')
    parser.add_argument(
        '--temporal-filter', action='store_true')
    parser.add_argument(
        '--color', default='rgb', choices=['rgb', 'lab'],
        help='color space for the environment observations')
    parser.add_argument(
        '--kernel')

    args = parser.parse_args(argv)

    model_dir = args.model_dir
    if not model_dir.endswith(os.sep):
        model_dir += os.sep
    os.makedirs(model_dir, exist_ok=True)

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

    # Load kernel
    if args.kernel is not None:
        kernel = np.load(args.kernel)[()]
    else:
        kernel = None

    # Global network, to be updated by worker threads
    net = estimators.AC_Network(
        n_out=len(Environment.ACTIONS),
        optimizer=optimizer,
        reward_feedback=args.reward_feedback,
        prediction_loss=args.prediction,
        visual_depth=(8 if args.temporal_filter else 1),
        temporal_stride=(4 if args.temporal_filter else 1),
        kernel=kernel,
    )

    # Create workers graphs
    workers = []
    for worker_id in range(n_workers):
        worker_name = 'worker_{:02}'.format(worker_id)

        if worker_id == 0:
            enable_viewport = args.viewport
        else:
            enable_viewport = False

        if not args.test:
            worker_summary = model_dir
        else:
            worker_summary = None

        worker = AC_Worker(
            name=worker_name,
            env_class=Environment,
            env_params={
                'level': args.level,
                'reward_feedback': args.reward_feedback,
                'skip_repeat_frames': not args.temporal_filter,
                'color_space': args.color,
                'plot': enable_viewport,
            },
            net=net,
            reward_clipping=1,
            param_queue=parameter_queue,
            grad_queue=gradient_queue,
            model_dir=worker_summary,
            state_buffer=(1 if args.temporal_filter else 0, 0),
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
        learn=not args.test,
        beholder=args.beholder,
    )

    learner.run()


if __name__ == '__main__':
    main(sys.argv[1:])
