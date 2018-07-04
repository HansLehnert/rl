"""Generate random walk trajectories based on mouse model."""

import numpy as np
import matplotlib.pyplot as plt


def generate_path(
        start_position,
        start_rotation,
        scene,
        time=15,
        time_delta=0.02,
        perimeter=0.03):
    """Generate a 2D random walk.

    scene: a list of vector pairs describing the segments which form the
    environment
    """
    steps = int(time / time_delta)

    pos = np.zeros((steps, 2))
    pos[0] = start_position

    rot = np.zeros((steps))
    rot[0] = start_rotation

    vel_disp = np.zeros((steps,))
    vel_turn = np.zeros((steps,))

    for i in range(0, steps - 1):
        # Check if the current position lies in the perimeter
        dist_vec = closest_wall(pos[i], scene) - pos[i]
        dist_wall = np.linalg.norm(dist_vec)
        angle_wall = (np.arctan2(dist_vec[1], dist_vec[0]) - rot[i])
        angle_wall = (angle_wall + np.pi) % (2 * np.pi) - np.pi

        # Compute speed and turn depending on
        vel_turn[i + 1] = np.random.normal(scale=(330 * np.pi / 180))
        if dist_wall < perimeter and np.abs(angle_wall) < np.pi / 2:
            vel_disp[i + 1] = vel_disp[i] * 0.75
            vel_turn[i + 1] -= (
                (np.pi / 2 - np.abs(angle_wall)) * np.sign(angle_wall)
                / time_delta)
        else:
            vel_disp[i + 1] = np.random.rayleigh(0.13)

        # Update position
        head_vec = np.array((np.cos(rot[i - 1]), np.sin(rot[i - 1])))
        pos[i + 1] = pos[i] + head_vec * vel_disp[i + 1] * time_delta
        rot[i + 1] = rot[i] + vel_turn[i + 1] * time_delta

    result = {
        'position': pos,
        'rotation': rot,
        'disp_speed': vel_disp,
        'turn_speed': vel_turn}

    return result


def closest_wall(position, scene):
    """Find the closest point of the scene."""
    closest = None
    min_dist = np.inf
    for a, b in scene:
        pa = position - a
        ba = b - a

        x = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1)
        point = a + x * ba

        dist = np.linalg.norm(position - point)
        if dist < min_dist:
            min_dist = dist
            closest = point
    return closest


def draw_scene(scene, *args, **kwargs):
    """Plot the scene in the currently active figure."""
    for a, b in scene:
        plt.plot((a[0], b[0]), (a[1], b[1]), *args, **kwargs)


def generate_square_scene(size):
    """Generate a simple scene with four walls."""
    scene = [
        (np.array((0.0, 0.0)), np.array((size, 0.0))),
        (np.array((0.0, 0.0)), np.array((0.0, size))),
        (np.array((size, size)), np.array((size, 0.0))),
        (np.array((size, size)), np.array((0.0, size)))]

    return scene


if __name__ == '__main__':
    scene = generate_square_scene(2.2)
    scene += [(np.array([0.4, 1.1]), np.array([1.8, 1.1]))]

    path = generate_path(
        np.random.uniform(0.1, 2.1, (2,)),
        np.random.uniform(-np.pi, np.pi),
        scene,
        200)

    draw_scene(scene, 'k')
    plt.plot(path['position'][:, 0], path['position'][:, 1])

    plt.show()
