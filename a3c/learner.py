import tensorflow as tf
import tensorboard.plugins.beholder
import datetime
import os.path


class Learner:
    def __init__(
            self, net, param_queue, grad_queue, model_dir, max_steps,
            n_workers, learn=True, beholder=False, **kwargs
    ):
        self.net = net
        self.param_queue = param_queue
        self.grad_queue = grad_queue
        self.max_steps = max_steps
        self.n_workers = n_workers
        self.learn = learn
        self.enable_beholder = beholder

        self.model_dir = model_dir
        self.summary_dir = os.path.join(model_dir, 'learner')

    def run(self):
        # Create session
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        session = tf.Session(config=session_config)

        # Tensorboard beholder
        if self.enable_beholder:
            beholder = tensorboard.plugins.beholder.Beholder(self.model_dir)

        # Step counters
        global_step_counter = tf.train.get_or_create_global_step()
        increase_global_step_op = global_step_counter.assign_add(1)

        env_step_counter = tf.Variable(0, dtype=tf.int32, name='env_step')
        env_step_increment = tf.placeholder(tf.int32)
        increase_env_step_op = env_step_counter.assign_add(env_step_increment)

        session.run(tf.global_variables_initializer())

        # Load variables from checkpoint
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0)
        checkpoints = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoints is not None:
            saver.recover_last_checkpoints(
                checkpoints.all_model_checkpoint_paths)
            latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
            saver.restore(session, latest_checkpoint)
            print('Restored checkpoint {}'.format(latest_checkpoint))
        step, env_step = session.run([global_step_counter, env_step_counter])

        # Create summary writer
        if self.summary_dir is not None and self.learn:
            summary_writer = tf.summary.FileWriter(self.summary_dir)
            summary_writer.add_graph(session.graph)
        else:
            summary_writer = None

        # Send the initial parameters to the worker threads
        local_vars = session.run(self.net.local_vars)
        for i in range(self.n_workers):
            self.param_queue.put((env_step, local_vars))

        last_time = None

        # Update loop
        while env_step < self.max_steps:
            # Get grads
            grads, grad_steps = self.grad_queue.get()

            # Increase step counters
            step = session.run(increase_global_step_op)
            env_step = session.run(
                increase_env_step_op,
                feed_dict={env_step_increment: grad_steps}
            )

            # Update network
            if self.learn:
                session.run(
                    self.net.apply_gradients,
                    dict(zip(self.net.gradients_in, grads))
                )

            # Send parameters back
            local_vars = session.run(self.net.local_vars)
            self.param_queue.put((env_step, local_vars))

            if self.enable_beholder:
                beholder.update(session)

            # Save model
            if self.learn and step % 1000 == 0:
                print('Saving checkpoint on step {}'.format(step))
                saver.save(session, self.model_dir, global_step=step)

                time = datetime.datetime.now()
                if last_time is not None:
                    elapsed_time = (time - last_time).total_seconds()

                    summary = tf.Summary()
                    summary.value.add(
                        tag='Info/StepsPerMinute',
                        simple_value=(1000 / elapsed_time * 60)
                    )
                    summary.value.add(
                        tag='Info/EnvironmentSteps',
                        simple_value=env_step
                    )

                    summary_writer.add_summary(summary, step)
                    summary_writer.flush()
                last_time = time

        self.stop()

    def stop(self):
        for i in range(self.n_workers):
            self.param_queue.put('end')
