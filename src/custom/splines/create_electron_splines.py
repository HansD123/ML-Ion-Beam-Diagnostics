import csv
from typing import Tuple
import numpy as np
from scipy.interpolate import CubicSpline
import dill
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm


csv_file_path_aluminium = "deposit_tables/al_electron_energies.csv"
csv_file_path_scintillator = "deposit_tables/sc_electron_energies.csv"


def create_al_electron_spline(filepath: str) -> None:
    """
    Reads the aluminum energies csv table downloaded from NIST ESTAR website and fits the table to
    a spline. Two arrays: incoming_al and deposited_al are created to hold the tabular data, which are then passed into the
    scipy CubicSpline function to create the splines. Dill is used to store the splines as pkl objects.
    """
    incoming_al = np.array([])
    deposited_al = np.array([])

    with open(filepath, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            incoming_al = np.append(incoming_al, float((row[0])))
            deposited_al = np.append(deposited_al, float((row[1])))

    spline_al = CubicSpline(incoming_al, deposited_al)

    with open("pickles/al_electron_spline.pkl", "wb") as f:
        dill.dump(spline_al, f)


def create_sc_electron_spline(filepath: str) -> None:
    """
    Reads the scintillator energies csv table downloaded from NIST ESTAR website and fits the table to
    a spline. Two arrays: incoming_sc and deposited_sc are created to hold the tabular data, which are then passed into the
    scipy CubicSpline function to create the splines. Dill is used to store the splines as pkl objects.
    """
    incoming_sc = np.array([])
    deposited_sc = np.array([])

    with open(filepath, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            incoming_sc = np.append(incoming_sc, float((row[0])))
            deposited_sc = np.append(deposited_sc, float((row[1])))

    spline_al = CubicSpline(incoming_sc, deposited_sc)

    with open("pickles/sc_electron_spline.pkl", "wb") as f:
        dill.dump(spline_al, f)


def create_al_2d_electron_spline(energy_range: Tuple[float, float], thickness_range: Tuple[float, float]) -> None:
    """
    Loads in aluminium spline, and then defines a function to calculate the energy reamining after an electron
    passes through a specific thickness of aluminium. Uses the range of energies and thicknesses supplied to
    create a grid of the energy remaining after passing through the aluminium. Turns this grid into a 2D spline using
    scipy RectBivariate and then stores the spline as a pickle file.
    """
    with open("pickles/al_electron_spline.pkl", "rb") as f:
        spline_al = dill.load(f)

    def calc_stopping_aluminium(
        initial_energy: float, steps: int, thickness: float
    ) -> float:
        """
        Models the electrons path through the material as a series of steps to determine the energy remaining
        """
        h = thickness / steps
        i = 0
        # energy_lost = 0
        energy_remaining = initial_energy
        while i < steps and energy_remaining > 0:
            energy_remaining -= spline_al(energy_remaining) * 2.71 * h
            i += 1
        if energy_remaining < 0:
            return 0
        return energy_remaining

    energies = np.linspace(energy_range[0], energy_range[1], 1200)
    thicknesses = np.logspace(thickness_range[0], thickness_range[1], 400)

    energy_remaining = np.empty((1200, 400))

    for i in tqdm(range(1200)):
        for j in range(400):
            energy_remaining[i][j] = calc_stopping_aluminium(
                energies[i], 1000, thicknesses[j]
            )

    al_energy_remaining_spline = RectBivariateSpline(
        energies, thicknesses, energy_remaining, kx=3, ky=3
    )

    with open("pickles/al_electron_remaining_spline.pkl", "wb") as f:
        dill.dump(al_energy_remaining_spline, f)


def create_sc_electron_deposited_spline(
    energy_range: Tuple[float, float], scintillator_thickness: float
) -> None:
    """
    Loads in scintillator EStar spline and defines a function to determine the energy deposited in
    the scintillator. Using the scintillator thickness provided, loops through and calculates the energy deposited for various incoming energies,
    which are then combined using scipy CubicSpline, which is stored as a pickle file.
    """
    with open("pickles/sc_electron_spline.pkl", "rb") as f:
        spline_sc = dill.load(f)

    def calc_stopping_scintillator(
        initial_energy: float, steps: int, thickness: float
    ) -> float:
        """
        Models the protons path through the material as a series of steps to determine the energy deposited
        """
        h = thickness / steps
        i = 0
        energy_remaining = initial_energy
        while i < steps and energy_remaining > 0:
            energy_remaining -= spline_sc(energy_remaining) * 0.9383 * h
            i += 1
        if energy_remaining < 0:
            return initial_energy
        else:
            return initial_energy - energy_remaining

    energies = np.linspace(energy_range[0], energy_range[1], 1200)
    energy_deposited = np.empty(1200)

    for i in tqdm(range(1200)):
        energy_deposited[i] = calc_stopping_scintillator(
            energies[i], 1000, scintillator_thickness
        )

    sc_energy_deposited_spline = CubicSpline(energies, energy_deposited)

    with open("pickles/sc_electron_deposited_spline.pkl", "wb") as f:
        dill.dump(sc_energy_deposited_spline, f)


create_al_electron_spline(csv_file_path_aluminium)
create_sc_electron_spline(csv_file_path_scintillator)
create_al_2d_electron_spline((0, 10), (-5, 0))
create_sc_electron_deposited_spline((0, 10), 2e-3)
