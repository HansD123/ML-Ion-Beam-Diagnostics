"""Generating synthetic images and labels with BDSIM and writing to a pickle file."""

import pickle
import os

from bdsim.generation import BDSIMGenerator
import utils

# Get the current directory path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

output_dir = f"{current_dir}/../output/synthetic_images"
output_filename = "bdsim_images_labels.pickle"
output_path = f"{output_dir}/{output_filename}"


def main() -> None:
    utils.create_output_dirs()
    if not os.path.isdir(output_dir):
        raise Exception(f"{output_dir} directory is not found")

    # Constants
    N_IMAGES = 50000
    N_WORKERS = 125

    # Data params
    JOB_ID = "generate"
    N_RANGE_PROTON = (1e7, 1e10)
    E_MAX_RANGE_PROTON = (0.1, 5)
    T_P_RANGE_PROTON = (0.05, 2)
    N_RANGE_ELECTRON = (1e7, 1e10)
    T_P_RANGE_ELECTRON = (0.05, 2)
    N_MACROPARTICLES = 25000  # 25000 electrons/protons
    FILTER_SIZE = 0.001
    FILTER_CENTRE = (0, 0, 0.05)
    FILTER_ARRAY = [
        0.004758472174937324,
        0.0037335975777684615,
        0.003641319252854112,
        0.0034866912858844638,
        0.002903644369313124,
        0.0025435081569248875,
        0.0006230103002935051,
        0.0002645738586726377,
        1e-07,
    ]
    SC_THICKNESS = 20e-6
    PIXEL_NO = 30
    CLEAR_FILES = False

    dataset = BDSIMGenerator(
        E_MAX_RANGE_PROTON,
        T_P_RANGE_PROTON,
        N_RANGE_PROTON,
        JOB_ID,
        N_MACROPARTICLES,
        N_IMAGES,
        N_WORKERS,
        FILTER_SIZE,
        FILTER_CENTRE,
        FILTER_ARRAY,
        SC_THICKNESS,
        PIXEL_NO,
        Tp_range_electrons=T_P_RANGE_ELECTRON,
        N0_range_electrons=N_RANGE_ELECTRON,
        clear_files=CLEAR_FILES,
    )

    output = dataset.gen_many_parallel()
    # Writing to a pickle file
    with open(output_path, "wb") as file:
        pickle.dump(output, file)


if __name__ == "__main__":
    main()
