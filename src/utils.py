"""Utility functions."""

import os
import numpy as np
from typing import List
from sklearn.preprocessing import MinMaxScaler


class DynamicMinMaxScaler:
    """
    Based off sklearn MinMaxScaler but dynamically handles batches of 2x2 matrices.
    """

    def __init__(self) -> None:
        self.scaler: MinMaxScaler = MinMaxScaler()

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        """Normalises the images to between 0 and 1."""
        flat_size = self.get_flat_size(images)

        flat_images = images.reshape(-1, flat_size)
        norm_flat_images = self.scaler.fit_transform(flat_images)
        norm_images = norm_flat_images.reshape(-1, *images.shape[1:])
        return norm_images

    def inverse_transform(self, images: np.ndarray) -> np.ndarray:
        """Inverse transforms the images to redeem their original magnitude."""
        flat_size = self.get_flat_size(images)

        flat_images = images.reshape(-1, flat_size)
        norm_restored_images = self.scaler.inverse_transform(flat_images)
        restored_images = norm_restored_images.reshape(-1, *images.shape[1:])
        return restored_images

    def get_flat_size(self, images: np.ndarray) -> int:
        """Determines the flat size of the image."""
        flat_size = 1
        for i in range(1, len(images.shape)):
            flat_size *= images.shape[i]
        return flat_size


def rolling_windows(thicknesses: List[float], window_size: int, step_size: int, num_windows: int) -> List[List[float]]:

    # Check that it is possible
    if window_size + (num_windows - 1) * step_size != len(thicknesses):
        raise Exception("Wrong numbers")
    start = 0
    arr = []
    for _ in range(num_windows):
        arr.append(thicknesses[start : start + window_size])
        start += step_size
    return arr


def round_to_closest(source_list: List[float], target_list: List[float]) -> List[float]:
    """
    Rounds each number in source_list to the closest number in target_list.
    """
    rounded_list = []
    for number in source_list:
        closest_number = min(target_list, key=lambda x: abs(x - number))
        rounded_list.append(closest_number)
    return rounded_list


def exponential_dist(a: float, b: float, input: float | np.ndarray) -> float | np.ndarray:
    return a * b**input


def diffs_to_vals(start: float, diffs: np.ndarray) -> np.ndarray:
    """
    Given a start value and an array of differences, returns a series of values by
    subtracting the difference from each subsequent element.
    """
    abs_arr = [start]
    for diff in diffs:
        end = len(abs_arr) - 1
        abs_arr.append(abs_arr[end] - diff)

    return np.array(abs_arr)


def set_lower_bnd(arr: np.ndarray, lower_bnd: float) -> np.ndarray:
    """Ensure that each element is at minimum the lower bound."""
    out = []
    for elem in arr:
        out.append(max(elem, lower_bnd))
    return np.array(out)


def create_output_dirs() -> None:
    """
    Creates a directory structure in the root of the project for storing job related
    files such as labels, neural networks, optuna studies and synthetic images:

    logs/
    output/
    ├── labels/
    ├── nn_models/
    ├── optuna_studies/
    └── synthetic_images/
    """
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # Adding logs dir
    if not os.path.exists(f"{current_dir}/../logs"):
        os.mkdir(f"{current_dir}/../logs")

    # Adding output dir and subdirs
    if not os.path.exists(f"{current_dir}/../output"):
        os.mkdir(f"{current_dir}/../output")

    folders = ["labels", "nn_models", "optuna_studies", "synthetic_images"]
    for folder in folders:
        if not os.path.exists(f"{current_dir}/../output/{folder}"):
            os.mkdir(f"{current_dir}/../output/{folder}")
