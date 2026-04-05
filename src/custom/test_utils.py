import pytest
import numpy as np
import math

from filter import Filter
from custom.generation import gen_many_parallel
from utils import normalise


def test_normalise() -> None:
    """Tests the normalisation and denormalisation of images and labels."""
    # Create the image and label to test on
    E_MAX_BOUNDS = (0.1, 5)
    T_P_BOUNDS = (0.05, 2)
    N_PARTICLES_BOUNDS = (10**7, 10**10)
    N_MACROPARTICLES = int(1e5)
    SCINT_THICKNESS = None
    BASE_UNIT = [[9e-2, 4e-2, 2e-2], [9e-3, 4e-3, 2e-3], [1e-3, 0.5e-3, 0.2e-3]]
    filter_object = Filter(BASE_UNIT, 20, (1, 1))
    N_DATA = 2
    N_WORKERS = 1
    CALIBRATED_MAX = 4095

    images_and_labels = gen_many_parallel(
        E_MAX_BOUNDS,
        T_P_BOUNDS,
        N_PARTICLES_BOUNDS,
        N_MACROPARTICLES,
        SCINT_THICKNESS,
        np.array(filter_object.filter),
        filter_object.map,
        N_DATA,
        N_WORKERS,
        pixel_calibration=CALIBRATED_MAX,
    )

    combined_images = images_and_labels["combined_images"]
    combined_labels = images_and_labels["combined_labels"]

    normalised_images, normalised_labels, inverse_scale = normalise(
        combined_images, combined_labels
    )

    EPSILON = 1e-9
    # Normalised image pixels should be between 0 and 1
    assert min(normalised_images[0].flatten()) >= 0
    assert max(normalised_images[0].flatten()) <= 1 + EPSILON

    # Labels should also be between 0 and 1
    assert min(normalised_labels[0]) >= 0
    assert max(normalised_labels[0]) <= 1

    # Restoring labels and checking that they are the same as the initial labels
    restored_labels = inverse_scale(normalised_labels)
    for combined_label, restored_label in zip(combined_labels, restored_labels):
        for combined_val, restored_val in zip(combined_label, restored_label):
            assert math.isclose(combined_val, restored_val)
