import os
import glob
from typing import List

# Get the current directory path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

def create_output_dirs() -> None:
    """
    Adds folders for storing temporary BDSIM files required for generating BDSIM
    scripts.

    /
    ├── bdsim_outputs/
    ├── bdsim_particle_files/
    └── bdsim_scripts/
    """
    folders = ["bdsim_outputs", "bdsim_particle_files", "bdsim_scripts"]
    for folder in folders:
        if not os.path.exists(f"{current_dir}/{folder}"):
            os.mkdir(f"{current_dir}/{folder}")

def clear_bdsim_scripts() -> None:
    """
    Warning: this function clears all .gmad files in the bdsim_scripts directory.
    Do not use if there is useful files in this directory, as this will all be deleted.
    """
    directory = f"{current_dir}/bdsim_scripts"
    file_pattern = os.path.join(directory, "*.gmad")
    root_files = glob.glob(file_pattern)
    for file_path in root_files:
        try:
            os.remove(file_path)
        except Exception as e:
            raise Exception(f"Error: {e}")
        
def clear_bdsim_particle_files() -> None:
    """
    Warning: this function clears all .dat files in the bdsim_particle_files directory.
    Do not use if there is useful files in this directory, as this will all be deleted.
    """
    directory = f"{current_dir}/bdsim_particle_files"
    file_pattern = os.path.join(directory, "*.dat")
    root_files = glob.glob(file_pattern)
    for file_path in root_files:
        try:
            os.remove(file_path)
        except Exception as e:
            raise Exception(f"Error: {e}")
        

def clear_bdsim_outputs() -> None:
    """
    Warning: this function clears all .root files in the bdsim_outputs directory.
    Do not use if there is useful files in this directory, as this will all be deleted.
    """
    directory = f"{current_dir}/bdsim_outputs"
    file_pattern = os.path.join(directory, "*.root")
    root_files = glob.glob(file_pattern)
    for file_path in root_files:
        try:
            os.remove(file_path)
        except Exception as e:
            raise Exception(f"Error: {e}")
        
def generate_script(
    job_ID: str,
    job_number: int,
    filter_array: List[float],
    scintillator_thickness: float,
    filter_width: float,
    pixel_size: int,
    particle_type: str,
) -> None:
    """
    Fills out the gmad file Beam_original according to the parameters inputted and outputs the file to
    bdsim_scripts, to be used in bdsm
    filter_array must be in m (and must currently be of length 9)
    scintillator_thickness is in m
    filter_width in m
    """
    if len(filter_array) != 9:
        raise Exception("Error: Only a 9 filter configuration is currently supported")

    if particle_type != "proton" and particle_type != "electron":
        raise Exception('Error: Particle type must be either "proton" or "electron"')

    with open(f"{current_dir}/bdsim_scripts_original/Beam_original.gmad", "r") as file:
        script = file.readlines()

    # horizontalWidth alu1
    line_10 = script[9]
    script[9] = line_10[:64] + str(filter_width) + line_10[64:]

    # length alu1
    line_10 = script[9]
    script[9] = line_10[:18] + str(filter_array[0]) + line_10[18:]

    # horizontalWidth alu2
    line_11 = script[10]
    script[10] = line_11[:64] + str(filter_width) + line_11[64:]

    # length alu2
    line_11 = script[10]
    script[10] = line_11[:18] + str(filter_array[1]) + line_11[18:]

    # horizontalWidth alu3
    line_12 = script[11]
    script[11] = line_12[:64] + str(filter_width) + line_12[64:]

    # length alu3
    line_12 = script[11]
    script[11] = line_12[:18] + str(filter_array[2]) + line_12[18:]

    # horizontalWidth alu4
    line_13 = script[12]
    script[12] = line_13[:64] + str(filter_width) + line_13[64:]

    # length alu4
    line_13 = script[12]
    script[12] = line_13[:18] + str(filter_array[3]) + line_13[18:]

    # horizontalWidth alu5
    line_14 = script[13]
    script[13] = line_14[:64] + str(filter_width) + line_14[64:]

    # length alu5
    line_14 = script[13]
    script[13] = line_14[:18] + str(filter_array[4]) + line_14[18:]

    # horizontalWidth alu6
    line_15 = script[14]
    script[14] = line_15[:64] + str(filter_width) + line_15[64:]

    # length alu6
    line_15 = script[14]
    script[14] = line_15[:18] + str(filter_array[5]) + line_15[18:]

    # horizontalWidth alu7
    line_16 = script[15]
    script[15] = line_16[:64] + str(filter_width) + line_16[64:]

    # length alu7
    line_16 = script[15]
    script[15] = line_16[:18] + str(filter_array[6]) + line_16[18:]

    # horizontalWidth alu8
    line_17 = script[16]
    script[16] = line_17[:64] + str(filter_width) + line_17[64:]

    # length alu8
    line_17 = script[16]
    script[16] = line_17[:18] + str(filter_array[7]) + line_17[18:]

    # horizontalWidth alu9
    line_18 = script[17]
    script[17] = line_18[:64] + str(filter_width) + line_18[64:]

    # length alu9
    line_18 = script[17]
    script[17] = line_18[:18] + str(filter_array[8]) + line_18[18:]

    # horizontalWidth sci
    line_21 = script[20]
    script[20] = line_21[:80] + str(3 * filter_width) + line_21[80:]

    # length sci
    line_21 = script[20]
    script[20] = line_21[:17] + str(scintillator_thickness) + line_21[17:]

    # z pl1
    line_25 = script[24]
    script[24] = (
        line_25[:56] + str(filter_array[4] - filter_array[0] / 2) + line_25[56:]
    )

    # y pl1
    line_25 = script[24]
    script[24] = line_25[:49] + str(filter_width) + line_25[49:]

    # x pl1
    line_25 = script[24]
    script[24] = line_25[:42] + str(filter_width) + line_25[42:]

    # z pl2
    line_26 = script[25]
    script[25] = (
        line_26[:49] + str(filter_array[4] - filter_array[1] / 2) + line_26[49:]
    )

    # y pl2
    line_26 = script[25]
    script[25] = line_26[:42] + str(filter_width) + line_26[42:]

    # z pl3
    line_27 = script[26]
    script[26] = (
        line_27[:56] + str(filter_array[4] - filter_array[2] / 2) + line_27[56:]
    )

    # y pl3
    line_27 = script[26]
    script[26] = line_27[:49] + str(filter_width) + line_27[49:]

    # x pl3
    line_27 = script[26]
    script[26] = line_27[:42] + str(-filter_width) + line_27[42:]

    # z pl4
    line_28 = script[27]
    script[27] = (
        line_28[:49] + str(filter_array[4] - filter_array[3] / 2) + line_28[49:]
    )

    # x pl4
    line_28 = script[27]
    script[27] = line_28[:42] + str(filter_width) + line_28[42:]

    # z pl6
    line_29 = script[28]
    script[28] = (
        line_29[:49] + str(filter_array[4] - filter_array[5] / 2) + line_29[49:]
    )

    # x pl6
    line_29 = script[28]
    script[28] = line_29[:42] + str(-filter_width) + line_29[42:]

    # z pl7
    line_30 = script[29]
    script[29] = (
        line_30[:56] + str(filter_array[4] - filter_array[6] / 2) + line_30[56:]
    )

    # y pl7
    line_30 = script[29]
    script[29] = line_30[:49] + str(-filter_width) + line_30[49:]

    # x pl7
    line_30 = script[29]
    script[29] = line_30[:42] + str(filter_width) + line_30[42:]

    # z pl8
    line_31 = script[30]
    script[30] = (
        line_31[:49] + str(filter_array[4] - filter_array[7] / 2) + line_31[49:]
    )

    # y pl8
    line_31 = script[30]
    script[30] = line_31[:42] + str(-filter_width) + line_31[42:]

    # z pl9
    line_32 = script[31]
    script[31] = (
        line_32[:56] + str(filter_array[4] - filter_array[8] / 2) + line_32[56:]
    )

    # y pl9
    line_32 = script[31]
    script[31] = line_32[:49] + str(-filter_width) + line_32[49:]

    # x pl9
    line_32 = script[31]
    script[31] = line_32[:42] + str(-filter_width) + line_32[42:]

    # detector ysize
    line_38 = script[37]
    script[37] = line_38[:92] + str(3 * filter_width) + line_38[92:]

    # detector xsize
    line_38 = script[37]
    script[37] = line_38[:80] + str(3 * filter_width) + line_38[80:]

    # pixel number ny
    line_38 = script[37]
    script[37] = line_38[:62] + str(pixel_size) + line_38[62:]

    # pixel number nx
    line_38 = script[37]
    script[37] = line_38[:55] + str(pixel_size) + line_38[55:]

    # detector z
    line_39 = script[38]
    script[38] = (
        line_39[:20] + str(filter_array[4] + scintillator_thickness / 2) + line_39[20:]
    )

    # detector zsize
    line_39 = script[38]
    script[38] = line_39[:12] + str(scintillator_thickness) + line_39[12:]

    # beam particle
    line_43 = script[42]
    if particle_type == "proton":
        script[42] = line_43[:18] + "proton" + line_43[18:]
    else:
        script[42] = line_43[:18] + "electron" + line_43[18:]

    # job number beam
    line_46 = script[45]
    script[45] = line_46[:46] + str(job_number) + line_46[46:]

    # job ID beam
    line_46 = script[45]
    script[45] = line_46[:45] + job_ID + line_46[45:]

    # particle_type beam
    line_46 = script[45]
    if particle_type == "proton":
        script[45] = line_46[:44] + "proton" + line_46[44:]
    else:
        script[45] = line_46[:44] + "electron" + line_46[44:]

    with open(
        f"{current_dir}/bdsim_scripts/"
        + particle_type
        + "beam_"
        + job_ID
        + "_"
        + str(job_number)
        + ".gmad",
        "w",
    ) as file:
        file.writelines(script)