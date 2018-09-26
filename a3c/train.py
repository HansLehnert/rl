import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import threading
import multiprocessing
import argparse
import functools
import deepmind_lab

from estimators import ValueEstimator, PolicyEstimator
from worker import Worker

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('level', nargs='?', default='nav_maze_random_goal_01')
parser.add_argument('--model_dir', default='./model_a3c')
parser.add_argument('--t_max', type=int, default=5)
parser.add_argument('-n', type=int)
args = parser.parse_args(sys.argv[1:])

# tf.flags.DEFINE_integer("max_global_steps", None,
# "Stop training after this many steps in the environment.
# Defaults to running indefinitely.")
# tf.flags.DEFINE_integer(
# "eval_every", 300, "Evaluate the policy every N seconds")


class LabEnvironment():
    def __init__(
        self,
        level,
        plot=False
    ):
        self.lab = deepmind_lab.Lab(
            level,
            ['RGB_INTERLEAVED'],
            {'fps': '30', 'width': '64', 'height': '64'}
        )

        self.actions = (
            'MOVE_FORWARD',
            'MOVE_BACKWARD',
            'LOOK_LEFT',
            'LOOK_RIGHT',
            'STRAFE_LEFT',
            'STRAFE_RIGHT'
        )

        self.plot = plot
        if self.plot:
            plt.figure()
            self.viewport = plt.imshow(np.zeros((40, 60)))
            plt.show(block=False)

        self.reset()

    def reset(self):
        self.lab.reset()
        self.last_reward = 0
        self.step()

    def step(self, action=None):
        action_vec = np.zeros((7,), dtype=np.intc)

        if action == 'MOVE_FORWARD':
            action_vec[3] = 1
        elif action == 'MOVE_BACKWARD':
            action_vec[3] = -1
        elif action == 'LOOK_LEFT':
            action_vec[0] = -40
        elif action == 'LOOK_RIGHT':
            action_vec[0] = 40
        elif action == 'STRAFE_LEFT':
            action_vec[2] = -1
        elif action == 'STRAFE_RIGHT':
            action_vec[2] = 1

        self.last_reward = self.lab.step(action_vec)

        if self.lab.is_running():
            self.observations = self.lab.observations()

            if self.plot:
                self.viewport.set_data(self.observations['RGB_INTERLEAVED'])
                plt.draw()
                plt.pause(1e-5)

    def state(self):
        return self.observations['RGB_INTERLEAVED']

    def reward(self):
        return self.last_reward

    def done(self):
        return not self.lab.is_running()


# Depending on the game we may have a limited action space
valid_actions = list(range(6))

# Set the number of workers
n_workers = args.n
if n_workers is None:
    n_workers = multiprocessing.cpu_count()

model_dir = args.model_dir

# Optionally empty model directory
checkpoint_dir = os.path.join(model_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

summary_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train'))

with tf.device('/cpu:0'):
    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Global step iterator
    global_counter = itertools.count()

    # Global policy and value nets
    with tf.variable_scope('global'):
        policy_net = PolicyEstimator(num_outputs=len(valid_actions))
        value_net = ValueEstimator(reuse=True)

    # Create worker graphs
    workers = []
    for worker_id in range(n_workers):
        # We only write summaries in one of the workers because they're
        # pretty much identical and writing them on all workers
        # would be a waste of space
        worker_summary_writer = None
        if worker_id == 0:
            worker_summary_writer = summary_writer

        worker = Worker(
            name='worker_{}'.format(worker_id),
            env=LabEnvironment(args.level, plot=worker_id == 0),
            policy_net=policy_net,
            value_net=value_net,
            global_counter=global_counter,
            discount_factor=0.99,
            summary_writer=worker_summary_writer)
        workers.append(worker)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)

    # Used to occasionally save videos for our policy net
    # and write episode rewards to Tensorboard
    # pe = PolicyMonitor(
    #     env=make_env(wrap=False),
    #     policy_net=policy_net,
    #     summary_writer=summary_writer,
    #     saver=saver)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Start worker threads
    worker_threads = []
    for worker in workers[1:]:
        worker_fn = functools.partial(worker.run, sess, coord, args.t_max)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)

    workers[0].run(sess, coord, args.t_max)

    # Start a thread for policy eval task
    # monitor_thread = threading.Thread(
    #     target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
    # monitor_thread.start()

    # Wait for all workers to finish
    coord.join(worker_threads)
