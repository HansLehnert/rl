import collections
import os.path
import numpy as np
import tensorflow as tf

Transition = collections.namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'value', 'next_state', 'done'])


class AC_Worker:
    def __init__(
            self, name, net, env_class, env_params, param_queue, grad_queue,
            discount_factor=0.99, batch_size=1, model_dir=None, **kwargs
    ):
        self.name = name
        self.param_queue = param_queue
        self.grad_queue = grad_queue
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.model_dir = model_dir

        self.env_class = env_class
        self.env_params = env_params
        self.net = net

        self._step = 0

    def run(self, max_steps):
        # Create environemnt
        self.env = self.env_class(**(self.env_params))

        # Create summary writer
        if self.model_dir is not None:
            summary_dir = os.path.join(self.model_dir, self.name)
            self.summary_writer = tf.summary.FileWriter(summary_dir)
        else:
            self.summary_writer = None

        # Create session
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        self.session = tf.Session(config=session_config)
        self.session.run(tf.global_variables_initializer())

        self.update_vars()

        batch = []

        while True:
            transitions = []

            # Start episode
            self.env.reset()
            state = self.env.state()
            rnn_state = self.session.run(self.net.rnn_initial_state)
            batch_rnn_state = rnn_state

            episode_total_reward = 0
            episode_positive_reward = 0
            episode_negative_reward = 0

            while not self.env.done():
                policy_eval, value_eval, rnn_state = self.session.run(
                    fetches=[
                        self.net.policy,
                        self.net.value,
                        self.net.rnn_state_out
                    ],
                    feed_dict={
                        self.net.input: [[state]],
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
                            self.net.input: [[transitions[-1].next_state]],
                            **dict(zip(self.net.rnn_state_in, rnn_state))
                        }

                        bootstrap_value = self.session.run(
                            self.net.value,
                            feed_dict=feed_dict
                        )
                        bootstrap_value = bootstrap_value[0, 0, 0]

                    batch.append(
                        (transitions, bootstrap_value, batch_rnn_state))
                    transitions = []
                    batch_rnn_state = rnn_state

                    if len(batch) >= self.batch_size:
                        self.train(batch)
                        self.update_vars()

                        batch = []

                state = next_state

            print('[{}] Episode Reward: {}'.format(
                self.name, episode_total_reward))

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

                self.summary_writer.add_summary(summary, self._step)
                self.summary_writer.flush()

    def train(self, batch):
        total_steps = 0

        batch_target_values = []
        batch_advantages = []
        batch_actions = []
        batch_states = []
        batch_rnn_state = None

        for transitions, bootstrap_value, rnn_state in batch:
            total_steps += len(transitions)

            batch_actions.append([x.action for x in transitions])
            batch_states.append([x.state for x in transitions])

            reward = bootstrap_value

            target_values = []
            advantages = []

            for transition in reversed(transitions):
                reward = transition.reward + self.discount_factor * reward
                advantage = reward - transition.value

                target_values.append(reward)
                advantages.append(advantage)

            target_values.reverse()
            advantages.reverse()

            batch_target_values.append(target_values)
            batch_advantages.append(advantages)

            if batch_rnn_state is None:
                batch_rnn_state = []
                for state in rnn_state:
                    batch_rnn_state.append(state)
            else:
                for i in range(len(batch_rnn_state)):
                    batch_rnn_state[i] = np.vstack(
                        (batch_rnn_state[i], rnn_state[i]))

        gradients, summaries = self.session.run(
            fetches=[
                self.net.gradients_out,
                self.net.summaries,
            ],
            feed_dict={
                self.net.target_value: batch_target_values,
                self.net.advantages: batch_advantages,
                self.net.actions: batch_actions,
                self.net.input: batch_states,
                **dict(zip(self.net.rnn_state_in, batch_rnn_state))
            }
        )

        self.grad_queue.put((gradients, total_steps))

        # Write summaries
        if self.summary_writer is not None:
            if summaries is not None:
                self.summary_writer.add_summary(summaries, self._step)

            self.summary_writer.flush()

    def update_vars(self):

        self._step, params = self.param_queue.get()

        self.session.run(
            self.net.assign,
            feed_dict=dict(zip(self.net.vars_in, params))
        )
