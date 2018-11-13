import tensorflow as tf
import datetime
import os.path


class Learner:
    def __init__(
            self, net, param_queue, grad_queue, model_dir, max_steps,
            learn=True, **kwargs
    ):
        self.net = net
        self.param_queue = param_queue
        self.grad_queue = grad_queue
        self.max_steps = max_steps
        self.learn = learn

        self.model_dir = model_dir
        self.summary_dir = os.path.join(model_dir, 'learner')

        self.last_time = None

    def run(self, n_workers):
        # Create session
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        self.session = tf.Session(config=session_config)

        # Step counters
        self.global_step_counter = tf.train.get_or_create_global_step()
        increase_global_step = self.global_step_counter.assign_add(1)

        self.env_step_counter = tf.Variable(0, dtype=tf.int32, name='env_step')
        env_step_increment = tf.placeholder(tf.int32)
        increase_env_step = self.env_step_counter.assign_add(
            env_step_increment)

        self.session.run(tf.global_variables_initializer())

        # Load variables from checkpoint
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0)
        checkpoints = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoints is not None:
            self.saver.recover_last_checkpoints(
                checkpoints.all_model_checkpoint_paths)
            latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
            self.saver.restore(self.session, latest_checkpoint)
            print('Restored checkpoint {}'.format(latest_checkpoint))
        step, env_step = self.session.run(
            [self.global_step_counter, self.env_step_counter])

        # Create summary writer
        if self.summary_dir is not None and self.learn:
            self.summary_writer = tf.summary.FileWriter(self.summary_dir)
            self.summary_writer.add_graph(self.session.graph)
        else:
            self.summary_writer = None

        # Send the initial parameters to the worker threads
        local_vars = self.session.run(self.net.local_vars)
        for i in range(n_workers):
            self.param_queue.put((step, local_vars))

        # Update loop
        while env_step < self.max_steps:
            # Get grads
            grads, grad_steps = self.grad_queue.get()

            step = self.session.run(increase_global_step)
            env_step = self.session.run(
                increase_env_step, feed_dict={env_step_increment: grad_steps})

            # Update network
            if self.learn:
                self.session.run(
                    self.net.apply_gradients,
                    dict(zip(self.net.gradients_in, grads))
                )

            # Send parameters back
            local_vars = self.session.run(self.net.local_vars)
            self.param_queue.put((step, local_vars))

            # Save model
            if self.learn and step % 1000 == 0:
                print('Saving checkpoint on step {}'.format(step))
                self.saver.save(self.session, self.model_dir, global_step=step)

                time = datetime.datetime.now()
                if self.last_time is not None:
                    elapsed_time = (time - self.last_time).total_seconds()

                    summary = tf.Summary()
                    summary.value.add(
                        tag='Info/StepsPerMinute',
                        simple_value=(1000 / elapsed_time * 60)
                    )
                    summary.value.add(
                        tag='Info/EnvironmentSteps',
                        simple_value=env_step
                    )

                    self.summary_writer.add_summary(summary, step)
                    self.summary_writer.flush()
                self.last_time = time

        for i in range(n_workers):
            self.param_queue.put('end')
