import deepmind_lab
import numpy as np
import matplotlib.pyplot as plt


class LabEnvironment():
    ACTIONS = (
        'MOVE_FORWARD',
        'MOVE_BACKWARD',
        'LOOK_LEFT',
        'LOOK_RIGHT',
        # 'STRAFE_LEFT',
        # 'STRAFE_RIGHT'
    )

    def __init__(
        self,
        level,
        plot=False
    ):
        self.lab = deepmind_lab.Lab(
            level,
            ['RGB_INTERLEAVED'],
            {'fps': '30', 'width': '84', 'height': '84'}
        )

        self.plot = plot
        if self.plot:
            plt.figure()
            self.viewport = plt.imshow(np.zeros((84, 84)))
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
            action_vec[0] = -15
        elif action == 'LOOK_RIGHT':
            action_vec[0] = 15
        elif action == 'STRAFE_LEFT':
            action_vec[2] = -1
        elif action == 'STRAFE_RIGHT':
            action_vec[2] = 1

        self.last_reward = self.lab.step(action_vec, 4)

        if self.lab.is_running():
            self.observations = self.lab.observations()

            if self.plot:
                self.viewport.set_data(self.observations['RGB_INTERLEAVED'])
                plt.draw()
                plt.pause(1e-5)

    def state(self):
        return self.observations['RGB_INTERLEAVED']

    def reward(self):
        return np.clip(self.last_reward, -1, 1)

    def done(self):
        return not self.lab.is_running()
