import tensorflow as tf
import numpy as np
import trajectory


def __row_normalize(a):
    """Normalize each row of matrix a and return the resulting matrix."""
    for row in a:
        acc = np.sum(row)
        if acc != 0:
            row /= acc
    return a


def generate_dataset(
        batch_size,
        place,
        head,
        n_data=None,
        sigma=0.1,
        kappa=20,
        peek=0.05,
        **kwargs):
    """Create function for generating random walk input data.

    batch_size: number of trajectories to generate for each input dataset
    place: place cells locations (used for calculating the groundtruth)
    head: head cells locations (used for calculating the groundtruth)
    scene: description of the scene
    sigma: place cell scale factor
    kappa: head cell scale factor
    """

    def gen():
        i = 0
        while n_data is None or i < n_data:
            i += 1

            # Generate feature vectors from random walk
            np.random.seed(None)
            initial_position = np.random.uniform(0.1, 2.1, (2,))
            initial_rotation = np.random.uniform(-np.pi, np.pi)

            data = trajectory.generate_path(
                initial_position,
                initial_rotation,
                **kwargs)

            steps = data['position'].shape[0]

            # Compute activations based on position
            place_activations = np.zeros((steps, place.shape[0]))
            for n, cell in enumerate(place):
                act = data['position'] - cell
                act = np.array([np.dot(v, v) for v in act])
                act = np.exp(- act / 2 / sigma ** 2)
                place_activations[:, n] = act
            place_activations = __row_normalize(place_activations)

            head_activations = np.zeros((steps, head.size))
            for n, cell in enumerate(head):
                act = np.exp(kappa * np.cos(data['rotation'] - cell))
                head_activations[:, n] = act
            head_activations = __row_normalize(head_activations)

            labels = np.hstack([
                place_activations,
                head_activations])

            activation_peek = np.zeros(labels.shape)
            for t in range(activation_peek.shape[0]):
                if t == 0 or np.random.rand() < peek:
                    activation_peek[t, :] = labels[t, :] / peek

            features = np.hstack([
                data['disp_speed'][:, None],
                np.sin(data['turn_speed'][:, None] * 0.02),
                np.cos(data['turn_speed'][:, None] * 0.02),
                activation_peek])

            yield (features, labels)

    dataset = tf.data.Dataset.from_generator(
        gen,
        (tf.float32, tf.float32),
        (tf.TensorShape([100, place.shape[0] + head.shape[0] + 3]),
            tf.TensorShape([100, place.shape[0] + head.shape[0]])))

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    return dataset


if __name__ == '__main__':
    # Create scene
    scene = trajectory.generate_square_scene(1)

    # Generate place and head cells distribution
    place_cells = np.array([[0, 0]])
    head_cells = np.array([0])

    dataset = generate_dataset(
        batch_size=10,
        place=place_cells,
        head=head_cells,
        scene=scene,
        time=2,
    )

    print(dataset)
