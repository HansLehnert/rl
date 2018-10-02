import os
import collections
import numpy as np
import tensorflow as tf

Transition = collections.namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'value', 'next_state', 'done'])


class AC_Worker():
    def __init__(
            self, name, env, global_net, local_net, optimizer,
            discount_factor=0.99,
            saver=None,
            checkpoint_dir=None,
            summary_writer=None,
    ):
        self.name = name
        self.env = env
        self.discount_factor = discount_factor
        self.saver = saver
        self.summary_writer = summary_writer
        self.checkpoint_dir = checkpoint_dir

        self.net = local_net

        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope(self.name, auxiliary_name_scope=False):
            self.increase_global_step = tf.assign_add(
                tf.train.get_global_step(), 1)

    def run(self, sess, coord, max_steps):
        episode_count = 0
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.net.copy_params)

                transitions = []

                # Start episode
                self.env.reset()
                state = self.env.state()
                rnn_state = sess.run(self.net.rnn_initial_state)
                batch_rnn_state = rnn_state

                episode_total_reward = 0
                episode_positive_reward = 0
                episode_negative_reward = 0

                while not self.env.done():
                    policy_eval, value_eval, rnn_state = sess.run(
                        fetches=[
                            self.net.policy,
                            self.net.value,
                            self.net.rnn_state_out
                        ],
                        feed_dict={
                            self.net.input: [state],
                            **dict(zip(self.net.rnn_state_in, rnn_state))
                        }
                    )

                    policy_eval = np.squeeze(policy_eval) + 1e-6
                    value_eval = value_eval[0, 0, 0]

                    # Compute next action based on policy
                    action = np.random.choice(
                        len(self.env.ACTIONS), p=policy_eval)
                    self.env.step(self.env.ACTIONS[action])

                    next_state = self.env.state()
                    reward = self.env.reward()
                    done = self.env.done()

                    episode_total_reward += reward
                    if reward > 0:
                        episode_positive_reward += reward
                        print('{}: {}'.format(self.name, reward))
                    else:
                        episode_negative_reward += reward

                    # Store transitions
                    transitions.append(Transition(
                        state, action, reward, value_eval, next_state, done))

                    # Update global network
                    if len(transitions) >= max_steps or done:
                        if done:
                            bootstrap_value = 0
                        else:
                            feed_dict = {
                                self.net.input: [transitions[-1].next_state],
                                **dict(zip(self.net.rnn_state_in, rnn_state))
                            }

                            bootstrap_value = sess.run(
                                self.net.value,
                                feed_dict=feed_dict
                            )
                            bootstrap_value = bootstrap_value[0, 0, 0]

                        self.train(
                            transitions,
                            sess,
                            bootstrap_value,
                            batch_rnn_state)

                        sess.run(self.net.copy_params)
                        transitions = []
                        batch_rnn_state = rnn_state

                    state = next_state

                episode_count += 1

                # Log episode data
                if self.summary_writer is not None:
                    summary = tf.Summary()
                    summary.value.add(
                        tag='Perf/RewardTotal',
                        simple_value=episode_total_reward
                    )
                    summary.value.add(
                        tag='Perf/RewardPositive',
                        simple_value=episode_positive_reward
                    )
                    summary.value.add(
                        tag='Perf/RewardNegative',
                        simple_value=episode_negative_reward
                    )

                    global_step = sess.run(self.global_step)
                    self.summary_writer.add_summary(summary, global_step)
                    self.summary_writer.flush()

    def train(self, transitions, sess, bootstrap_value, rnn_state):
        reward = bootstrap_value

        target_values = []
        advantages = []
        actions = [x.action for x in transitions]
        states = [x.state for x in transitions]

        for transition in transitions[::-1]:
            reward = transition.reward + self.discount_factor * reward
            advantage = reward - transition.value

            target_values.append(reward)
            advantages.append(advantage)

        target_values.reverse()
        advantages.reverse()

        results = sess.run(
            fetches=[
                self.net.value_loss,
                self.net.policy_loss,
                self.net.entropy_loss,
                self.net.loss,
                self.net.selected_policy,
                self.net.apply_grads,
            ],
            feed_dict={
                self.net.advantages: advantages,
                self.net.target_value: target_values,
                self.net.actions: actions,
                self.net.input: states,
                **dict(zip(self.net.rnn_state_in, rnn_state))
            }
        )

        global_step = sess.run(self.increase_global_step)

        # Save model
        if global_step % 100 == 0 and self.saver is not None:
            print('Saving checkpoint on step {}'.format(global_step))
            checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint')
            self.saver.save(sess, checkpoint_path, global_step=global_step)

        # Write summaries
        if self.summary_writer is not None:
            mean_value = np.mean(np.array([x.value for x in transitions]))
            mean_policy = np.mean(results[4])

            summary = tf.Summary()
            summary.value.add(tag='Perf/Value', simple_value=mean_value)
            summary.value.add(tag='Perf/Policy', simple_value=mean_policy)
            summary.value.add(tag='Loss/Value', simple_value=results[0])
            summary.value.add(tag='Loss/Policy', simple_value=results[1])
            summary.value.add(tag='Loss/Entropy', simple_value=results[2])
            summary.value.add(tag='Loss/Total', simple_value=results[3])

            self.summary_writer.add_summary(summary, global_step)
            self.summary_writer.flush()
