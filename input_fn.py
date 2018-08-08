import tensorflow as tf
import numpy as np
import trajectory


def generate_activations(
        place,
        head,
        n_data=None,
        sigma=0.1,
        kappa=20,
        peek=0.05,
        random=None,
        **kwargs):
    """Create function for generating random walk input data.

    place: place cells locations (used for calculating the groundtruth)
    head: head cells locations (used for calculating the groundtruth)
    scene: description of the scene
    sigma: place cell scale factor
    kappa: head cell scale factor
    """
    if random is None:
        random = np.random.RandomState()

    # Generate feature vectors from random walk
    initial_position = random.uniform(0.1, 2.1, (2,))
    initial_rotation = random.uniform(-np.pi, np.pi)

    data = trajectory.generate_path(
        initial_position,
        initial_rotation,
        random=random,
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
    data['place_activations'] = place_activations

    head_activations = np.zeros((steps, head.size))
    for n, cell in enumerate(head):
        act = np.exp(kappa * np.cos(data['rotation'] - cell))
        head_activations[:, n] = act
    head_activations = __row_normalize(head_activations)
    data['head_activations'] = head_activations

    peek_time = np.append([1], random.binomial(1, peek, steps - 1))
    data['peek_time'] = peek_time

    place_peek = np.zeros([steps, place.shape[0]])
    head_peek = np.zeros([steps, head.shape[0]])
    for t in np.where(peek_time):
        place_peek[t, :] = place_activations[t, :]
        head_peek[t, :] = head_activations[t, :]

    data['place_peek'] = place_peek
    data['head_peek'] = head_peek

    return data


def create_dataset(
        batch_size,
        place,
        head,
        n_data=None,
        time=2,
        time_delta=0.02,
        **kwargs):
    """Create a dataset that generates head and place activations.

    Returns a tensorflow dataset that generates random walk patterns and
    computes activation of provided head and place cells.
    """

    def gen_wrapper():
        i = 0
        while n_data is None or i < n_data:
            i += 1

            data = generate_activations(
                place=place,
                head=head,
                time=time,
                time_delta=time_delta,
                **kwargs)

            labels = np.hstack([
                data['place_activations'],
                data['head_activations']])

            features = np.hstack([
                data['disp_speed'][:, None],
                np.sin(data['turn_speed'][:, None] * 0.02),
                np.cos(data['turn_speed'][:, None] * 0.02),
                data['place_peek'],
                data['head_peek']])

            yield (features, labels)

    steps = int(time / time_delta)

    dataset = tf.data.Dataset.from_generator(
        gen_wrapper,
        (tf.float32, tf.float32),
        (tf.TensorShape([steps, place.shape[0] + head.shape[0] + 3]),
            tf.TensorShape([steps, place.shape[0] + head.shape[0]])))

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    return dataset


def __row_normalize(a):
    """Normalize each row of matrix a and return the resulting matrix."""
    for row in a:
        acc = np.sum(row)
        if acc != 0:
            row /= acc
    return a


if __name__ == '__main__':
    # Create scene
    scene = trajectory.generate_square_scene(1)

    # Generate place and head cells distribution
    place_cells = np.array([[0, 0]])
    head_cells = np.array([0])

    dataset = create_dataset(
        batch_size=10,
        place=place_cells,
        head=head_cells,
        scene=scene,
        time=2,
    )

    print(dataset)
