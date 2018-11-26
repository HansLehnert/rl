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
            discount_factor=0.99, reward_clipping=None, batch_size=1,
            model_dir=None, **kwargs
    ):
        self.name = name
        self.param_queue = param_queue
        self.grad_queue = grad_queue
        self.discount_factor = discount_factor
        self.reward_clipping = reward_clipping
        self.batch_size = batch_size
        self.model_dir = model_dir

        self.env_class = env_class
        self.env_params = env_params
        self.net = net

        self._step = 0

    def run(self, max_steps):
        self.active = True

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

        while self.active:
            transitions = []

            # Start episode
            self.env.reset()
            state = self.env.state()
            rnn_state = self.session.run(self.net.rnn_initial_state)
            initial_rnn_state = rnn_state

            episode_total_reward = 0
            episode_positive_reward = 0
            episode_negative_reward = 0

            while not self.env.done() and self.active:
                policy_eval, value_eval, rnn_state = self.session.run(
                    fetches=[
                        self.net.policy,
                        self.net.value,
                        self.net.rnn_state_out
                    ],
                    feed_dict={
                        **dict(zip(self.net.rnn_state_in, rnn_state)),
                        **dict(zip(self.net.inputs, self.group_states(
                            [state]))),
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
                            **dict(zip(self.net.rnn_state_in, rnn_state)),
                            **dict(zip(self.net.inputs, self.group_states(
                                [transitions[-1].next_state]))),
                        }

                        bootstrap_value = self.session.run(
                            self.net.value,
                            feed_dict=feed_dict
                        )
                        bootstrap_value = bootstrap_value[0, 0, 0]

                    self.train(transitions, bootstrap_value, initial_rnn_state)
                    self.update_vars()

                    initial_rnn_state = rnn_state
                    transitions = []

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

        print('[{}] Ending'.format(self.name))

    def train(self, transitions, bootstrap_value, rnn_state):
        actions = [x.action for x in transitions]
        states = [x.state for x in transitions]

        reward = bootstrap_value

        target_values = []
        advantages = []

        for transition in reversed(transitions):
            clipped_reward = transition.reward
            if self.reward_clipping is not None:
                clipped_reward = np.clip(
                    clipped_reward,
                    -self.reward_clipping,
                    self.reward_clipping)

            reward = clipped_reward + self.discount_factor * reward
            advantage = reward - transition.value

            target_values.append(reward)
            advantages.append(advantage)

        target_values.reverse()
        advantages.reverse()

        gradients, summaries = self.session.run(
            fetches=[
                self.net.gradients_out,
                self.net.summaries,
            ],
            feed_dict={
                self.net.target_value: target_values,
                self.net.advantages: advantages,
                self.net.actions: actions,
                **dict(zip(self.net.inputs, self.group_states(states))),
                **dict(zip(self.net.rnn_state_in, rnn_state))
            }
        )

        self.grad_queue.put((gradients, len(transitions)))

        # Write summaries
        if self.summary_writer is not None:
            if summaries is not None:
                self.summary_writer.add_summary(summaries, self._step)

            self.summary_writer.flush()

    def update_vars(self):
        response = self.param_queue.get()

        if isinstance(response, str) and response == 'end':
            self.active = False
        else:
            self._step, params = response

            self.session.run(
                self.net.assign,
                feed_dict=dict(zip(self.net.vars_in, params))
            )

    def group_states(self, states):
        result = [[s[i] for s in states] for i in range(len(states[0]))]
        return result
