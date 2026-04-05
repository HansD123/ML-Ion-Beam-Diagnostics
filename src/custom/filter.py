import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from typing import List, Tuple


# Functions
def scale_matrix_dimensions(matrix: List[List[float]], scale_factor: int) -> List[List[float]]:
    """
    Scales a matrix uniformly by an integer scale factor. Scaled in both the x
    and y direction by the same factor.
    """
    scaled_rows = [row for row in matrix for _ in range(scale_factor)]

    # Scale columns
    scaled_matrix = [
        [element for element in row for _ in range(scale_factor)] for row in scaled_rows
    ]
    return scaled_matrix


def repeat_units_xy(matrix: List[List[float]], x_units: int, y_units: int) -> List[List[float]]:
    """Repeats the units of a matrix in the x and y direction."""
    repeated_rows = [row * x_units for row in matrix]
    repeated_matrix = repeated_rows * y_units

    return repeated_matrix


# Classes
class Filter:
    """Class that simulates a filter of repeat units containing varying thicknesses."""

    def __init__(
        self,
        base_unit: List[List],
        repeat_unit_size: int,
        repeat_unit_dimensions: Tuple[int, int],
    ) -> None:
        """
        base_unit: Square shape list containing thicknesses of a repeat unit.
        repeat_unit_size: The size of the base units width and height.
        repeat_unit_dimensions: The dimensions of the filter in y, x directions in terms of the number of repeat units.
        """

        if len(base_unit) != len(base_unit[0]):
            raise RuntimeError("Repeat unit width and height must be the same.")

        # Obtain a map from filter number to thickness
        self.map: dict[int, float] = {}
        filter_no = 1
        for i in range(len(base_unit)):
            for j in range(len(base_unit)):
                self.map[filter_no] = base_unit[i][j]
                filter_no += 1

        # Replacing repeat unit thickness with filter number
        self.repeat_unit = deepcopy(base_unit)
        filter_no = 1
        for i in range(len(self.repeat_unit)):
            for j in range(len(self.repeat_unit)):
                self.repeat_unit[i][j] = filter_no
                filter_no += 1

        # Uniformly scaling the repeat unit of filter numbers by the desired number
        self.repeat_unit = scale_matrix_dimensions(self.repeat_unit, repeat_unit_size)

        # Generating the filter based off the repeat unit
        self.filter = repeat_units_xy(
            self.repeat_unit, repeat_unit_dimensions[1], repeat_unit_dimensions[0]
        )

        self.dimensions: Tuple[int, int] = (len(self.filter), len(self.filter[0]))

    def display(self) -> None:
        """Method for visualising the filter and its pixel dimensions."""
        thickness_filter = [
            [self.map[element] for element in row] for row in self.filter
        ]
        plt.figure(figsize=(8, 6), dpi=100)
        ax = plt.gca()
        cax = plt.imshow(thickness_filter, cmap="Greys", interpolation="nearest")
        plt.colorbar(cax, label="Thickness")  # Add a label to the colorbar
        filter_dimensions = f"{self.dimensions[0]} x {self.dimensions[1]}"
        ax.set_title(
            f"Filter Thickness Visualization - Dimensions: {filter_dimensions}"
        )
        plt.show()
