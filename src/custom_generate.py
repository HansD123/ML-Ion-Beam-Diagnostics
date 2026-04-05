"""Generating custom synthetic images and labels and writing to a pickle file."""

import pickle
import numpy as np
import os

import custom.filter as fil
import custom.generation as dg
import utils

# Get the current file directory
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

output_dir = f"{current_dir}/../output/synthetic_images"
output_filename = "custom_images_labels.pickle"
output_path = f"{output_dir}/{output_filename}"


def main() -> None:
    utils.create_output_dirs()
    if not os.path.isdir(output_dir):
        raise Exception(f"{output_dir} directory is not found")

    # Data params
    E_MAX_BOUNDS = (0.1, 5)
    T_P_BOUNDS = (0.05, 2)
    N_PARTICLES_BOUNDS = (1e7, 1e10)
    N_MACROPARTICLES = int(1e5)
    MAX_PIXEL = 4095
    ADD_ELECTRONS = True

    # Filter
    BASE_UNIT = [
        [0.004758472174937324, 0.0037335975777684615, 0.003641319252854112],
        [0.0034866912858844638, 0.002903644369313124, 0.0025435081569248875],
        [0.0006230103002935051, 0.0002645738586726377, 1e-07],
    ]
    filter = fil.Filter(BASE_UNIT, 10, (1, 1))

    # Generating the data
    N_IMAGES = 30000
    N_WORKERS = 125
    output = dg.gen_many_parallel(
        E_MAX_BOUNDS,
        T_P_BOUNDS,
        N_PARTICLES_BOUNDS,
        N_MACROPARTICLES,
        np.array(filter.filter),
        filter.map,
        N_IMAGES,
        N_WORKERS,
        add_electrons=ADD_ELECTRONS,
        pixel_calibration=MAX_PIXEL,
    )

    # Writing to a pickle file
    with open(output_path, "wb") as file:
        pickle.dump(output, file)


if __name__ == "__main__":
    main()
