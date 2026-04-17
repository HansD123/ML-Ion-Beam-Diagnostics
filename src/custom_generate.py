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
    ADD_ELECTRONS = False

    # Filter
    BASE_UNIT = [
        [85.4e-6, 40.1e-6, 18.8e-6],
        [8.9e-6,  4.2e-6,  2.0e-6],
        [0.9e-6,  0.4e-6,  0.2e-6],
    ]
    filter = fil.Filter(BASE_UNIT, 10, (1, 1))

    # Generating the data
    N_IMAGES = 5000
    N_WORKERS = 8
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
