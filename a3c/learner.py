import tensorflow as tf
import datetime
import os.path


class Learner:
    def __init__(self, net, param_queue, grad_queue, model_dir, **kwargs):
        self.param_queue = param_queue
        self.grad_queue = grad_queue
        self.net = net
        self.model_dir = model_dir
        self.summary_dir = os.path.join(model_dir, 'learner')

        self.last_time = None

    def run(self, n_workers):
        # Create session
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        self.session = tf.Session(config=session_config)
        self.global_step = tf.train.get_or_create_global_step()
        self.session.run(tf.global_variables_initializer())
        self.increase_global_step = self.global_step.assign_add(1)

        # Load variables from checkpoint
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0)
        checkpoints = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoints is not None:
            self.saver.recover_last_checkpoints(
                checkpoints.all_model_checkpoint_paths)
            latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
            self.saver.restore(self.session, latest_checkpoint)
            print('Restored checkpoint {}'.format(latest_checkpoint))

        # Create summary writer
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)
        self.summary_writer.add_graph(self.session.graph)

        # Send the initial parameters to the worker threads
        local_vars = self.session.run(self.net.local_vars)
        for i in range(n_workers):
            self.param_queue.put(local_vars)

        # Update loop
        while True:
            # Get grads
            grads = self.grad_queue.get()

            # Update network
            self.session.run(
                self.net.apply_gradients,
                dict(zip(self.net.gradients_in, grads))
            )

            # Sen paramenters back
            local_vars = self.session.run(self.net.local_vars)
            self.param_queue.put(local_vars)

            step = self.session.run(self.increase_global_step)

            # Save model
            if step % 100 == 0:
                print('Saving checkpoint on step {}'.format(step))
                self.saver.save(self.session, self.model_dir, global_step=step)

                time = datetime.datetime.now()
                if self.last_time is not None:
                    elapsed_time = (time - self.last_time).total_seconds()

                    summary = tf.Summary()
                    summary.value.add(
                        tag='Info/StepsPerMinute',
                        simple_value=(100 / elapsed_time * 60)
                    )

                    self.summary_writer.add_summary(summary, step)
                    self.summary_writer.flush()
                self.last_time = time
