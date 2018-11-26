import deepmind_lab
import numpy as np
import matplotlib.pyplot as plt


class LabEnvironment():
    ACTIONS = (
        'MOVE_FORWARD',
        'MOVE_BACKWARD',
        'LOOK_LEFT',
        'LOOK_RIGHT',
        'STRAFE_LEFT',
        'STRAFE_RIGHT'
    )

    def __init__(
        self,
        level,
        reward_feedback=False,
        plot=False
    ):
        self.reward_feedback = reward_feedback

        self.lab = deepmind_lab.Lab(
            level,
            ['RGB_INTERLEAVED', 'DEBUG.POS.TRANS'],
            {'fps': '30', 'width': '84', 'height': '84'}
        )

        self.plot = plot
        if self.plot:
            plt.figure('viewport')
            self.viewport = plt.imshow(np.zeros((84, 84)))
            plt.figure('map')
            self.points, = plt.plot([], [], c='g')
            self.trajectory, = plt.plot([], [])
            plt.axis('equal')
            plt.show(block=False)

        self.reset()

    def reset(self):
        self.lab.reset()
        self._reward = 0
        self.step()

        if self.plot:
            plt.figure('map')
            plt.cla()
            self.points, = plt.plot([], [], ls='', c='g', marker='o')
            self.trajectory, = plt.plot([], [])

    def step(self, action=None):
        action_vec = np.zeros((7,), dtype=np.intc)

        if action == 'MOVE_FORWARD':
            action_vec[3] = 1
        elif action == 'MOVE_BACKWARD':
            action_vec[3] = -1
        elif action == 'LOOK_LEFT':
            action_vec[0] = -30
        elif action == 'LOOK_RIGHT':
            action_vec[0] = 30
        elif action == 'STRAFE_LEFT':
            action_vec[2] = -1
        elif action == 'STRAFE_RIGHT':
            action_vec[2] = 1

        if not self.plot:
            self._reward = self.lab.step(action_vec, 4)

            if self.lab.is_running():
                self.observations = self.lab.observations()
        else:
            self._reward = 0

            for i in range(4):
                step_reward = self.lab.step(action_vec, 1)
                self._reward += step_reward

                if not self.lab.is_running():
                    break

                self.observations = self.lab.observations()

                # Update viewport
                self.viewport.set_data(self.observations['RGB_INTERLEAVED'])

                # Update trajectory
                position = self.observations['DEBUG.POS.TRANS']
                self.trajectory.set_xdata(
                    np.append(self.trajectory.get_xdata(), position[0]))
                self.trajectory.set_ydata(
                    np.append(self.trajectory.get_ydata(), position[1]))

                if step_reward == 10:
                    self.trajectory.set_linestyle('--')
                    self.trajectory, = plt.plot([], [])
                elif step_reward != 0:
                    self.points.set_xdata(
                        np.append(self.points.get_xdata(), position[0]))
                    self.points.set_ydata(
                        np.append(self.points.get_ydata(), position[1]))

                axes = plt.gca()
                axes.relim()
                axes.autoscale_view()

                plt.draw()
                plt.pause(1e-5)

    def state(self):
        state = [self.observations['RGB_INTERLEAVED']]
        if self.reward_feedback:
            state.append(self._reward)

        return tuple(state)

    def reward(self):
        return self._reward

    def done(self):
        return not self.lab.is_running()
