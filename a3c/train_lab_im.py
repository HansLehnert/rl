import sys
import os
import tensorflow as tf
import multiprocessing
import argparse
import functools

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
        '--model-dir', default='model_im/', dest='model_dir',
        help='directory to store the trained model')
    parser.add_argument(
        '--t_max', type=int, default=100)
    parser.add_argument(
        '--viewport', action='store_true',
        help='open a viewport for the environment')
    parser.add_argument(
        '--test', action='store_true', help='disable training')
    parser.add_argument(
        '--train-steps', type=int, default=10**8, dest='train_steps',
        help='maximum number of training steps')
    parser.add_argument(
        '--beholder', action='store_true', help='enable tensorboard beholder')
    args = parser.parse_args(argv)

    model_dir = os.path.join('models', args.model_dir)
    if not model_dir.endswith(os.sep):
        model_dir += os.sep

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
        n_out=len(Environment.ACTIONS),
        optimizer=optimizer,
        reward_feedback=True,
        prediction_loss=True,
        visual_depth=1,
    )

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
                'reward_feedback': True,
                'plot': enable_viewport,
            },
            net=net,
            reward_clipping=1,
            param_queue=parameter_queue,
            grad_queue=gradient_queue,
            model_dir=worker_summary,
            state_buffer=(7, 0),
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
        learn=(not args.test),
        beholder=args.beholder,
    )

    learner.run()


if __name__ == '__main__':
    main(sys.argv[1:])
