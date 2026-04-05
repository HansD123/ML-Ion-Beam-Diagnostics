import numpy as np
import dill
from typing import Dict, List, Tuple
import random
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import multiprocessing
import os


"""
Process generates two images, one with electrons affects included, and one without them. This should be used
to test the effectiveness of the neural network in removing electron effects from the images.
"""

# Get the directory containing this module
module_dir = os.path.dirname(__file__)

# Constructing paths relative to this module
al_spline_path = os.path.join(
    module_dir, "splines/pickles/al_proton_remaining_spline.pkl"
)
sc_spline_path = os.path.join(
    module_dir, "splines/pickles/sc_proton_deposited_spline.pkl"
)

al_electron_spline_path = os.path.join(
    module_dir, "splines/pickles/al_electron_remaining_spline.pkl"
)
sc_electron_spline_path = os.path.join(
    module_dir, "splines/pickles/sc_electron_deposited_spline.pkl"
)


# Defining Splines and Energy functions
with open(al_spline_path, "rb") as f:
    """
    Param: tuple = (Initial Energy, Thickness of Al)
    Returns: Energy Remaining
    """
    al_remaining_spline = dill.load(f)

with open(sc_spline_path, "rb") as f:
    """
    Param: tuple = (Initial Energy)
    Returns: Energy Deposited
    """
    sc_deposited_spline = dill.load(f)

with open(al_electron_spline_path, "rb") as f:
    """
    Param: tuple = (Initial Energy, Thickness of Al)
    Returns: Energy Remaining
    """
    al_remaining_electron_spline = dill.load(f)

with open(sc_electron_spline_path, "rb") as f:
    """
    Param: tuple = (Initial Energy)
    Returns: Energy Deposited
    """
    sc_deposited_electron_spline = dill.load(f)


def energy_after_al(initial_energy: float, thickness: float) -> float:
    """
    Determines the energy remaining from the initial energy and thickness of
    aluminium.
    """
    M_TO_CM = 100 # Convert from m to cm
    thickness_cm = thickness*M_TO_CM
    return float(al_remaining_spline(initial_energy, thickness_cm)[0][0])


def energy_after_al_electron(initial_energy: float, thickness: float) -> float:
    """
    Determines the energy remaining from the initial energy and thickness of
    aluminium.
    """
    M_TO_CM = 100 # Convert from m to cm
    thickness_cm = thickness*M_TO_CM
    return float(al_remaining_electron_spline(initial_energy, thickness_cm)[0][0])


def energy_deposited(initial_energy: float) -> float:
    """
    Determines the energy deposited from the initial energy assuming
    Scintillator thickness: 2e-3 m.
    """
    return sc_deposited_spline(initial_energy)


def energy_deposited_electron(initial_energy: float) -> float:
    """
    Determines the energy deposited from the initial energy assuming
    Scintillator thickness: 2e-3 m.
    """
    return sc_deposited_electron_spline(initial_energy)


# Data Generation Functions
def gen_params(
    E_max_bounds: Tuple[float, float],
    T_p_bounds: Tuple[float, float],
    macroparticles_bounds: Tuple[float, float],
) -> Tuple[float, float, float]:
    """
    Generates three parameters: Energy, temperature and number of macroparticles
    sampled from a uniform distribution between set bounds.
    """

    E_max = np.random.uniform(E_max_bounds[0], E_max_bounds[1])
    T_p = np.random.uniform(T_p_bounds[0], T_p_bounds[1])
    N0 = np.random.uniform(macroparticles_bounds[0], macroparticles_bounds[1])

    return E_max, T_p, N0


def gen_electron_params(
    T_p_bounds: Tuple[float, float], macroparticles_bounds: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Generates two parameters: Temperature and number of macroparticles sampled
    from a uniform distribution between set bounds.
    """
    T_p = np.random.uniform(T_p_bounds[0], T_p_bounds[1])
    N0 = np.random.uniform(macroparticles_bounds[0], macroparticles_bounds[1])

    return T_p, N0


def gen_energies(n_macroparticles: int, E_max: float, T_p: float) -> np.ndarray:
    """
    Samples energy values from the theoretical probability distributions modelling the
    energies of particles in an ion beam.
    """

    energies = []
    count = 0  # Tracking number of valid energies
    while count < n_macroparticles:
        deviate = np.random.random()
        energy = -T_p * np.log(1 - deviate * (1 - np.exp(-E_max / T_p)))
        if energy > E_max:
            pass
        # elif np.isnan(energy):
        #     pass
        else:
            energies.append(energy)
            count += 1
    return np.array(energies)


def gen_energies_electrons(n_macroparticles: int, T_p: float) -> np.ndarray:
    """
    Samples energy values from the theoretical probability distributions modelling the
    energies of electrons in an ion beam. This assumes there is no maximum energy parameter
    for the electron beam.
    """
    energies = []
    for i in range(n_macroparticles):
        deviate = np.random.random()
        energy = -T_p * np.log(1 - deviate)
        if energy > 10:
            energy = 10
        energies.append(energy)
    return np.array(energies)


def gen_energies_lists(
    n_deviates: int, n_filters: int, E_max: float, T_p: float
) -> List[np.ndarray]:
    """
    Generates a random set of energies and assigns each energy to the list associated with
    the filter number it lands on.
    """
    energies = gen_energies(n_deviates, E_max, T_p)
    energies_lists = [[] for _ in range(n_filters)]

    # Allocating energies to the arrays in the list
    for energy in energies:
        rand_index = int(np.random.randint(0, n_filters))
        energies_lists[rand_index].append(round(energy, 4))
    return [np.array(i) for i in energies_lists]


def gen_energies_electrons_lists(n_deviates: int, n_filters: int, T_p: float) -> List[np.ndarray]:
    """
    Generates a random set of energies and assigns each energy to the list associated
    with the filter number it lands on (for electrons).
    """
    energies = gen_energies_electrons(n_deviates, T_p)
    energies_lists = [[] for _ in range(n_filters)]

    # Allocating energies to the arrays in the list
    for energy in energies:
        rand_index = int(np.random.randint(0, n_filters))
        energies_lists[rand_index].append(round(energy, 4))
    return [np.array(i) for i in energies_lists]


def make_filter_map(thicknesses: List[float]) -> Dict[int, float]:
    """
    Returns a dictionary as follows:
    filter number: thickness
    """

    filter_map = {}
    for i in range(len(thicknesses)):
        filter_map[i] = thicknesses[i]
    return filter_map


# Proton energy functions:


def energies_after_filter(
    energies_list: List[np.ndarray], filter_map: dict
) -> List[np.ndarray]:
    """
    Determines the remaining energies of the particles based on the thickness off
    the thicknesses in the aluminium filters.

    Returns remaining energies in the same shape as the input.
    """
    if len(energies_list) != len(filter_map):
        raise RuntimeError("Size of energies_list must be the same as filter_map.")

    energies_after_list = []
    for i in range(len(energies_list)):
        filter_no = i + 1
        energies_arr = []
        for j in range(len(energies_list[i])):
            energy_after = energy_after_al(energies_list[i][j], filter_map[filter_no])
            energies_arr.append(energy_after)
        energies_after_list.append(np.array(energies_arr))

    return energies_after_list


def energies_deposited(energies_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Determines the list of energies depositied for each filter based on the initial
    energy and the thickness of the scintillator.
    """

    energies_deposited_list = []
    for i in range(len(energies_list)):
        energies_arr = []
        for j in range(len(energies_list[i])):
            energy_after = energy_deposited(energies_list[i][j])
            energies_arr.append(energy_after)
        energies_deposited_list.append(np.array(energies_arr))

    return energies_deposited_list


def allocate_energies(deposits_list: List[np.array], filter: np.ndarray) -> np.ndarray:
    """Randomly distributes the deposited energies to the necessary filters."""

    cumulative_energies = np.zeros_like(np.array(filter), dtype=np.float64)
    for i in range(len(deposits_list)):
        filter_no = i + 1
        result = np.where(filter == filter_no)
        indices = list(zip(result[0], result[1]))

        # Adding the energy depositedto a position corresponding to the associated
        # filter
        for j in range(len(deposits_list[i])):
            rand_index = random.choice(indices)
            cumulative_energies[rand_index] += deposits_list[i][j]

    return cumulative_energies


# Electron energy functions:
def electron_energies_after_filter(
    energies_list: List[np.ndarray], filter_map: dict
) -> List[np.ndarray]:
    """
    Determines the remaining energies of the particles based on the thickness off
    the thicknesses in the aluminium filters.
    Returns remaining energies in the same shape as the input.
    """
    if len(energies_list) != len(filter_map):
        raise RuntimeError("Size of energies_list must be the same as filter_map.")
    energies_after_list = []
    for i in range(len(energies_list)):
        filter_no = i + 1
        energies_arr = []
        for j in range(len(energies_list[i])):
            energy_after = energy_after_al_electron(
                energies_list[i][j], filter_map[filter_no]
            )
            energies_arr.append(energy_after)
        energies_after_list.append(np.array(energies_arr))
    return energies_after_list


def electron_energies_deposited(energies_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Determines the list of energies depositied for each filter based on the initial
    energy and the thickness of the scintillator.
    """
    energies_deposited_list = []
    for i in range(len(energies_list)):
        energies_arr = []
        for j in range(len(energies_list[i])):
            energy_after = energy_deposited_electron(energies_list[i][j])
            energies_arr.append(energy_after)
        energies_deposited_list.append(np.array(energies_arr))
    return energies_deposited_list


def smooth_squares(arr: np.ndarray, n_filters: int) -> np.ndarray:
    """
    perform gaussian filter on each square and then on the entire array
    arr : 2D np.array image
    n_filters : number of filters in the image (must be square number)
    """

    if int(np.sqrt(n_filters)) != (np.sqrt(n_filters)):
        raise Exception("n_filters must be a square number")

    # create empty return array which will be populated
    return_arr = np.empty(arr.shape)

    # get the number of squares per side i.e, sqrt(n_filters)
    length = int(np.sqrt(n_filters))

    # define one step which is total length / n_filters
    step = int(arr.shape[0] // length)

    # define the array to iterate through which is just integer multiples of 'step'
    iter_array = [step * i for i in range(1, length + 1)]

    # iterate along y axis
    for i in iter_array:

        # iterate along x axis
        for j in iter_array:

            # set that 'square' = gaussian filtered version of the image
            return_arr[i - step : i, j - step : j] = gaussian_filter(
                arr[i - step : i, j - step : j], 1
            )

    # then gaussian blur the whole image
    return_arr = gaussian_filter(return_arr, 1)
    return return_arr


def gen_single_data(
    e_max_bounds: Tuple[float, float],
    t_p_bounds: Tuple[float, float],
    n_particle_bounds: Tuple[float, float],
    n_macroparticles: int,
    filter: np.ndarray,
    filter_map: Dict[int, float],
    add_electrons: bool,
) -> Tuple[np.ndarray, tuple]:
    """
    Generates a simulated data image of the energies deposited in the scintillator
    by the protons and another image for the electrons. Smooths each image separately
    and then combines the images together, returning both the combined image and
    the proton only image. Electron and proton parameters are generated separately from one another.
    However the bounds for the parameters are the same. Use the same number of simulated macroparticles
    for both the protons and electrons.
    """

    n_filters = len(filter_map)

    # Proton image generation:
    # 1) Three random parameters sampled uniformly in the bounds
    eMaxProton, tempProton, numberParticlesProton = gen_params(
        e_max_bounds, t_p_bounds, n_particle_bounds
    )
    # 2) List of energy arrays with each array corresponding to a specific
    # filter
    initial_e_list = gen_energies_lists(
        n_macroparticles, n_filters, eMaxProton, tempProton
    )
    e_max_observed = np.max([np.max(i) for i in initial_e_list])
    label = (e_max_observed, tempProton, numberParticlesProton)
    # 3) Energies remaining after passing through the aluminium filters
    e_list_after_filter = energies_after_filter(initial_e_list, filter_map)
    # 4) Energies deposited in the scintillator
    e_deposited_list = energies_deposited(e_list_after_filter)
    # 5) Allocating energies to positions on the grid and scaling according to the
    # number of particles
    e_image = allocate_energies(e_deposited_list, filter)
    e_image *= numberParticlesProton / n_macroparticles
    # 6) Appyling smooth_squares function
    e_image = smooth_squares(e_image, n_filters=n_filters)

    # Optional electron image generation:
    if add_electrons:
        tempElectron, numberParticlesElectron = gen_electron_params(
            t_p_bounds, n_particle_bounds
        )

        initialEnergiesElectrons = gen_energies_electrons_lists(
            n_macroparticles, n_filters, tempElectron
        )
        label = label + (tempElectron, numberParticlesElectron)

        energyListAfterFilterElectron = electron_energies_after_filter(
            initialEnergiesElectrons, filter_map
        )

        energyDepositedListElectron = electron_energies_deposited(
            energyListAfterFilterElectron
        )

        eImageElectron = allocate_energies(energyDepositedListElectron, filter)
        eImageElectron *= numberParticlesElectron / n_macroparticles

        eImageElectron = smooth_squares(eImageElectron, n_filters=n_filters)

        # Combining
        e_image += eImageElectron

    return e_image, label


def gen_many_data(
    e_max_bounds: Tuple[float, float],
    t_p_bounds: Tuple[float, float],
    n_particle_bounds: Tuple[float, float],
    n_macroparticles: int,
    filter: np.ndarray,
    filter_map: Dict[int, float],
    n_data: int,
    random_seed: int,
    add_electrons: bool,
) -> Dict[str, List]:
    """Generates a set of synthetic data."""
    # Reseeding the random number based on time
    np.random.seed(random_seed)

    images_and_labels = []
    for _ in tqdm(range(n_data)):
        image_label = gen_single_data(
            e_max_bounds,
            t_p_bounds,
            n_particle_bounds,
            n_macroparticles,
            filter,
            filter_map,
            add_electrons,
        )  # Adjusted to call worker directly
        images_and_labels.append(image_label)

    images = [item[0] for item in images_and_labels]
    labels = [item[1] for item in images_and_labels]

    return {
        "images": images,
        "labels": labels,
    }


def divide_and_distribute(number: int, divisor: int) -> List[int]:
    if divisor <= 0:
        return "Error: Divisor must be greater than 0"

    quotient, remainder = divmod(number, divisor)
    result_list = [quotient] * divisor

    # Distribute the remainder across the list, one by one until exhausted
    for i in range(remainder):
        result_list[i] += 1

    return result_list


def calibrate_images(images: List[np.ndarray], calibrated_max: int) -> List[np.ndarray]:
    """
    Calibrates a list of images to between a desired range based on the global
    maximum pixel value of the images.
    """
    global_max = max(image.max() for image in images)
    scaling_factor = calibrated_max / global_max

    calibrated_images = np.array([np.round(image * scaling_factor) for image in images])
    return calibrated_images


def gen_many_parallel(
    e_max_bounds: Tuple[float, float],
    t_p_bounds: Tuple[float, float],
    n_particle_bounds: Tuple[float, float],
    n_macroparticles: int,
    filter: np.ndarray,
    filter_map: Dict[int, float],
    n_data: int,
    n_workers: int = 1,
    add_electrons: bool = False,
    pixel_calibration: int | None = None,
) -> Dict[str, List]:
    """Generates images and labels through multiple parallel processes."""

    args_list = []
    number_split = divide_and_distribute(n_data, n_workers)
    for number in number_split:
        # Generating a random seed for each set of arguments
        random_seed = np.random.randint(0, 2**31 - 1)
        args = (
            e_max_bounds,
            t_p_bounds,
            n_particle_bounds,
            n_macroparticles,
            filter,
            filter_map,
            number,
            random_seed,
            add_electrons,
        )
        args_list.append(args)

    # Data generation in parallel
    with multiprocessing.Pool() as pool:
        results = pool.starmap(gen_many_data, args_list)

    # Combining the results
    images = []
    labels = []
    for result in results:
        images += result["images"]
        labels += result["labels"]

    # Calibrating the pixels to the desired value
    if pixel_calibration != None:
        images = calibrate_images(images, pixel_calibration)

    return {
        "images": images,
        "labels": labels,
    }
