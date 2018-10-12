import numpy as np
import matplotlib.pyplot as plt


class MazeEnvironment():
    ACTIONS = (
        'MOVE_UP',
        'MOVE_DOWN',
        'MOVE_LEFT',
        'MOVE_RIGHT',
    )

    def __init__(self, plot=False, num_walls=3):
        self.size = 16
        self.max_steps = 400
        self.n_walls = num_walls
        self.plot = plot

        if self.plot:
            plt.figure(0)
            self.viewport = plt.imshow(np.zeros((128, 128)))
            plt.show(block=False)

        # Sprites
        self.sprite_1 = np.zeros((16, 16, 3), dtype=np.uint8)
        for i in range(16):
            for j in range(16):
                r = (i - 7.5) ** 2 + (j - 7.5) ** 2
                if r <= 49:
                    self.sprite_1[i, j, :] = r * 255 / 49

        self.sprite_2 = np.zeros((16, 16, 3), dtype=np.uint8)
        self.sprite_2[range(2, 14), range(2, 14), :] = 255
        self.sprite_2[range(2, 14), range(13, 1, -1), :] = 255

        self.reset()

    def reset(self):
        # Create walls
        self.walls = []
        for i in range(self.n_walls):
            wall = []
            wall.append(np.random.choice(('v', 'h')))
            wall.append(np.random.randint(self.size))
            wall += list(np.sort(np.random.randint(0, self.size, (2,))))
            self.walls.append(tuple(wall))

        # Spawn player
        self.player_pos = None
        while (self.player_pos is None
                or self.__check_wall_collision(self.player_pos)):
            self.player_pos = np.random.randint(0, self.size - 1, (2,))

        self.goal_pos = np.random.randint(0, self.size - 1, (2,))

        # Randomize sprites
        if np.random.randint(2) == 1:
            self.player_sprite = self.sprite_1
            self.goal_sprite = self.sprite_2
        else:
            self.player_sprite = self.sprite_2
            self.goal_sprite = self.sprite_1

        self.state_image = None
        self.t = 0

    def step(self, action):
        self.t += 1
        self.last_reward = 0
        self.state_image = None

        new_pos = np.copy(self.player_pos)

        if action == 'MOVE_UP':
            new_pos[0] -= 1
        elif action == 'MOVE_DOWN':
            new_pos[0] += 1
        elif action == 'MOVE_LEFT':
            new_pos[1] -= 1
        elif action == 'MOVE_RIGHT':
            new_pos[1] += 1
        else:
            raise ValueError('Invalid action {}'.format(action))

        # Keep player inside bounds
        for i in range(new_pos.size):
            if new_pos[i] < 0:
                new_pos[i] = 0
            elif new_pos[i] >= self.size - 1:
                new_pos[i] = self.size - 2

        if not self.__check_wall_collision(new_pos):
            self.player_pos = new_pos

        # Check if player arrives at goal
        # if np.all(np.abs(self.goal_pos - self.player_pos) <= 1):
        if _check_collision(self.player_pos, (2, 2), self.goal_pos, (2, 2)):
            self.last_reward = 1
            self.goal_pos = np.random.randint(0, self.size - 1, (2,))

        if self.plot:
            self.viewport.set_data(self.state())
            plt.draw()
            plt.pause(1e-5)

    def state(self):
        if self.state_image is None:
            self.state_image = self.__draw_state()
        return self.state_image

    def reward(self):
        return self.last_reward

    def done(self):
        return self.t >= self.max_steps

    def __draw_state(self):
        # state = np.zeros((self.size * 8,) * 2 + (3,), dtype=np.uint8)
        state = np.random.randint(
            0, 255, (self.size * 8,) * 2 + (3,), dtype=np.uint8)

        # Draw player
        x = self.player_pos[0] * 8
        y = self.player_pos[1] * 8
        state[x:x + 16, y:y + 16, :] |= self.player_sprite

        # Draw goal
        x = self.goal_pos[0] * 8
        y = self.goal_pos[1] * 8
        state[x:x + 16, y:y + 16, :] |= self.goal_sprite

        for wall in self.walls:
            if wall[0] == 'h':
                x = wall[2] * 8
                y = wall[1] * 8
                width = (wall[3] - wall[2] + 1) * 8
                height = 8
            else:
                x = wall[1] * 8
                y = wall[2] * 8
                width = 8
                height = (wall[3] - wall[2] + 1) * 8
            state[x:x + width, y:y + height, :] = 255

        return state

    def __check_wall_collision(self, pos):
        for wall in self.walls:
            if wall[0] == 'h':
                wall_pos = (wall[2], wall[1])
                wall_size = (wall[3] - wall[2] + 1, 1)
            else:
                wall_pos = (wall[1], wall[2])
                wall_size = (1, wall[3] - wall[2] + 1)
            if _check_collision(pos, (2, 2), wall_pos, wall_size):
                return True
        return False


def _check_collision(pos_1, size_1, pos_2, size_2):
    for c in range(pos_1.size):
        if (((pos_1[c] <= pos_2[c]) and (pos_2[c] - pos_1[c] < size_1[c]))
            or ((pos_1[c] > pos_2[c]) and (pos_1[c] - pos_2[c] < size_2[c]))):
            continue
        break
    else:
        return True
    return False


if __name__ == '__main__':
    env = MazeEnvironment(plot=True)

    while plt.fignum_exists(0):
        if env.player_pos[0] < env.goal_pos[0]:
            env.step('MOVE_DOWN')
        elif env.player_pos[0] > env.goal_pos[0]:
            env.step('MOVE_UP')
        elif env.player_pos[1] < env.goal_pos[1]:
            env.step('MOVE_RIGHT')
        else:
            env.step('MOVE_LEFT')
        plt.pause(0.01)
