"""Generate random walk trajectories based on mouse model."""

import numpy as np
import matplotlib.pyplot as plt


def generate_path(
        start_position,
        start_rotation,
        scene,
        time=15,
        time_delta=0.2,
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

    vel = np.zeros((steps,))

    for i in range(0, steps - 1):
        # Check if the current position lies in the perimeter
        dist_vec = closest_wall(pos[i], scene) - pos[i]
        dist_wall = np.linalg.norm(dist_vec)
        angle_wall = (np.arctan2(*dist_vec) - rot[i]) % (2 * np.pi)

        # Compute speed and turn depending on
        turn = np.random.normal(scale=(330 * np.pi / 180)) * time_delta
        if dist_wall < perimeter and np.abs(angle_wall) < np.pi / 2:
            vel[i + 1] = vel[i] * 0.75
            turn += angle_wall
        else:
            vel[i + 1] = np.random.rayleigh(0.13)

        # Update position
        head_vec = np.array((np.cos(rot[i - 1]), np.sin(rot[i - 1])))
        pos[i + 1] = pos[i] + head_vec * vel[i + 1] * time_delta
        rot[i + 1] = rot[i] + turn

    return {'position': pos}


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


if __name__ == '__main__':
    scene = (
        (np.array((0.0, 0.0)), np.array((2.2, 0.0))),
        (np.array((0.0, 0.0)), np.array((0.0, 2.2))),
        (np.array((2.2, 2.2)), np.array((2.2, 0.0))),
        (np.array((2.2, 2.2)), np.array((0.0, 2.2))),
    )

    plt.figure(1)
    draw_scene(scene, 'k')

    path = generate_path((1.1, 1.1), 0, scene, 100, 0.02)

    plt.figure(1)
    plt.plot(path['position'][:, 0], path['position'][:, 1])

    plt.show()
