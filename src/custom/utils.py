from typing import List
import numpy as np


def calibrate_images(images: List[np.ndarray], calibrated_max: int) -> List[np.ndarray]:
    """
    Calibrates a list of images to between a desired range based on the global
    maximum pixel value of the images.
    """
    global_max = max(image.max() for image in images)
    scaling_factor = calibrated_max / global_max

    calibrated_images = np.array([np.round(image * scaling_factor) for image in images])
    return calibrated_images
