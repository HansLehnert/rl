import numpy as np
import matplotlib.pyplot as plt


class SimpleEnvironment():
    ACTIONS = (
        'MOVE_UP',
        'MOVE_DOWN',
        'MOVE_LEFT',
        'MOVE_RIGHT',
        # 'STRAFE_LEFT',
        # 'STRAFE_RIGHT'
    )

    def __init__(self, plot=False, max_steps=100, n_goals=1):
        self.max_steps = max_steps
        self.n_goals = n_goals

        self.size = 8
        self.plot = plot
        if self.plot:
            plt.figure()
            self.viewport = plt.imshow(np.zeros((self.size, self.size)))
            plt.show(block=False)

        self.reset()

    def reset(self):
        self.player_pos = np.random.randint(0, self.size, (2,))
        self.goal_pos = np.random.randint(0, self.size, (self.n_goals, 2))

        self.last_reward = 0
        self.t = 0

    def step(self, action=None):
        self.last_reward = 0

        if action == 'MOVE_UP':
            self.player_pos[0] -= 1
        elif action == 'MOVE_DOWN':
            self.player_pos[0] += 1
        elif action == 'MOVE_LEFT':
            self.player_pos[1] -= 1
        elif action == 'MOVE_RIGHT':
            self.player_pos[1] += 1

        # Check if player is out of bounds
        for i in range(self.player_pos.size):
            if self.player_pos[i] < 0:
                self.player_pos[i] = 0
                self.last_reward = -1
            elif self.player_pos[i] >= self.size:
                self.player_pos[i] = self.size - 1
                self.last_reward = -1

        # Check if player arrived at goal:
        for i in range(self.n_goals):
            if np.all(self.player_pos == self.goal_pos[i]):
                self.last_reward = 5
                self.goal_pos[i] = np.random.randint(0, self.size, (2,))

        self.t += 1

        if self.plot:
            self.viewport.set_data(self.state())
            plt.draw()
            plt.pause(1e-5)

    def state(self):
        state_img = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        state_img[self.player_pos[0], self.player_pos[1], 0] = 255
        state_img[self.goal_pos[:, 0], self.goal_pos[:, 1], 1] = 255

        return state_img

    def reward(self):
        return self.last_reward

    def done(self):
        return self.t >= self.max_steps
