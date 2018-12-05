import deepmind_lab
import numpy as np
import matplotlib.pyplot as plt
import skimage.color


class LabEnvironment():
    ACTIONS = (
        'MOVE_FORWARD',
        'MOVE_BACKWARD',
        'WALK_AND_ROTATE_LEFT',
        'WALK_AND_ROTATE_RIGHT',
        'LOOK_LEFT',
        'LOOK_RIGHT',
        'STRAFE_LEFT',
        'STRAFE_RIGHT'
    )

    def __init__(
        self,
        level,
        action_repeat=4,
        skip_repeat_frames=True,
        reward_feedback=False,
        color_space='rgb',
        plot=False,
    ):
        self.action_repeat = action_repeat
        self.skip_repeat_frames = skip_repeat_frames
        self.reward_feedback = reward_feedback
        self.color_space = color_space

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
        if action == 'MOVE_FORWARD':
            action_vec = (0, 0, 0, 1, 0, 0, 0)
        elif action == 'MOVE_BACKWARD':
            action_vec = (0, 0, 0, -1, 0, 0, 0)
        elif action == 'LOOK_LEFT':
            action_vec = (-30, 0, 0, 0, 0, 0, 0)
        elif action == 'LOOK_RIGHT':
            action_vec = (30, 0, 0, 0, 0, 0, 0)
        elif action == 'STRAFE_LEFT':
            action_vec = (0, 0, -1, 0, 0, 0, 0)
        elif action == 'STRAFE_RIGHT':
            action_vec = (0, 0, 1, 0, 0, 0, 0)
        elif action == 'WALK_AND_ROTATE_LEFT':
            action_vec = (-20, 0, 0, 1, 0, 0, 0)
        elif action == 'WALK_AND_ROTATE_RIGHT':
            action_vec = (20, 0, 0, 1, 0, 0, 0)
        else:
            action_vec = (0, 0, 0, 0, 0, 0, 0)
        action_vec = np.array(action_vec, dtype=np.intc)

        if not self.plot and self.skip_repeat_frames:
            self._reward = self.lab.step(action_vec, self.action_repeat)

            if self.lab.is_running():
                self._frames = [self.lab.observations()['RGB_INTERLEAVED']]
        else:
            self._reward = 0
            self._frames = []

            for i in range(self.action_repeat):
                step_reward = self.lab.step(action_vec, 1)
                self._reward += step_reward

                if not self.lab.is_running():
                    break

                observations = self.lab.observations()
                if not self.skip_repeat_frames or i == self.action_repeat - 1:
                    self._frames.append(observations['RGB_INTERLEAVED'])

                if self.plot:
                    # Update viewport
                    self.viewport.set_data(observations['RGB_INTERLEAVED'])

                    # Update trajectory
                    position = observations['DEBUG.POS.TRANS']
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

        # Convert color space
        if self.color_space == 'lab':
            for i in range(len(self._frames)):
                self._frames[i] = skimage.color.rgb2lab(self._frames[i])
                self._frames[i] = self._frames[i].astype(int)
                self._frames[i][..., 1:] += 128
                self._frames[i][..., 0] *= 255
                self._frames[i][..., 0] //= 100
                # self._frames[i] = self._frames[i].astype(int)

        if len(self._frames) > 0:
            self._frames = np.stack(self._frames, 0)

    @property
    def state(self):
        result = []

        result.append(self._frames)

        if self.reward_feedback:
            result.append(np.array([self._reward]))

        return tuple(result)

    @property
    def reward(self):
        return self._reward

    @property
    def done(self):
        return not self.lab.is_running()


if __name__ == '__main__':
    lab = LabEnvironment('nav_maze_static_01', plot=True, color_space='lab')

    plt.figure()
    plt.show(block=False)

    lab.reset()
    while not lab.done:
        lab.step(np.random.choice(lab.ACTIONS))

        plt.clf()
        plt.subplot(2, 2, 1)
        plt.imshow(np.squeeze(lab.state[0]))
        plt.subplot(2, 2, 2)
        plt.imshow(lab.state[0][0, :, :, 0])
        plt.subplot(2, 2, 3)
        plt.imshow(lab.state[0][0, :, :, 1])
        plt.subplot(2, 2, 4)
        plt.imshow(lab.state[0][0, :, :, 2])
        plt.draw()
        plt.pause(1e-5)
