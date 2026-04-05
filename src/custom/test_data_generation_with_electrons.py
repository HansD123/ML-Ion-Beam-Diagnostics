import pytest
import numpy as np

from custom.generation import (
    energy_after_al,
    energy_after_al_electron,
    energy_deposited,
    energy_deposited_electron,
    gen_params,
    gen_electron_params,
    gen_energies,
    gen_energies_electrons,
    gen_energies_lists,
    gen_energies_electrons_lists,
    gen_single_data,
    gen_many_parallel,
)
from filter import Filter


class TestSplines:
    """
    Tests the splines used for determining the energy remaining/deposited when
    protons/electrons pass through through the aluminium/splin.
    """

    def test_energy_after_al(self) -> None:
        """Proton energy after passing through aluminium."""
        INITIAL_ENERGY = 3  # MeV
        THICKNESS = 1e-3  # m

        energy_after = energy_after_al(INITIAL_ENERGY, THICKNESS)

        assert energy_after >= 0
        assert type(energy_after) == float

    def test_energy_after_al_electron(self) -> None:
        """Electron energy after passing through aluminium."""
        INITIAL_ENERGY = 3  # MeV
        THICKNESS = 1e-3  # m

        energy_after = energy_after_al_electron(INITIAL_ENERGY, THICKNESS)

        assert energy_after >= 0
        assert type(energy_after) == float

    def test_energy_deposited(self) -> None:
        """Proton energy deposited in the scintillator."""
        INITIAL_ENERGY = 3  # MeV

        energy_after = energy_deposited(INITIAL_ENERGY)

        assert energy_after >= 0

    def test_energy_deposited_electron(self) -> None:
        """Electron energy deposited in the scintillator."""
        INITIAL_ENERGY = 5  # MeV

        energy_after = energy_deposited_electron(INITIAL_ENERGY)

        assert energy_after >= 0


class TestParameterGeneration:
    """Tests the generation of beam parameters for both protons and electrons."""

    def test_gen_params(self) -> None:
        """Protons"""
        E_MAX_BOUNDS = (0.1, 5)  # MeV
        T_P_BOUNDS = (0.05, 2)  # MeV
        N_PARTICLES_BOUNDS = (10**7, 10**10)

        e_max, t_p, n_0 = gen_params(E_MAX_BOUNDS, T_P_BOUNDS, N_PARTICLES_BOUNDS)

        assert E_MAX_BOUNDS[0] <= e_max <= E_MAX_BOUNDS[1]
        assert T_P_BOUNDS[0] <= t_p <= T_P_BOUNDS[1]
        assert N_PARTICLES_BOUNDS[0] <= n_0 <= N_PARTICLES_BOUNDS[1]

    def test_gen_electron_params(self) -> None:
        """Electrons"""
        E_MAX_BOUNDS = (0.1, 5)
        T_P_BOUNDS = (0.05, 2)

        e_max, t_p = gen_electron_params(E_MAX_BOUNDS, T_P_BOUNDS)

        assert E_MAX_BOUNDS[0] <= e_max <= E_MAX_BOUNDS[1]
        assert T_P_BOUNDS[0] <= t_p <= T_P_BOUNDS[1]


class TestGenEnergies:
    """Tests the generation of a list of energies for both protons and electrons."""

    def test_gen_energies(self) -> None:
        """Protons"""
        E_MAX = 1.0
        T_P = 1.0
        N_MACROPARTICLES = 10**5
        proton_energy_list = gen_energies(N_MACROPARTICLES, E_MAX, T_P)

        assert len(proton_energy_list) == N_MACROPARTICLES
        assert max(proton_energy_list) <= E_MAX
        assert min(proton_energy_list) >= 0

    def test_gen_energies_electrons(self) -> None:
        """Electrons"""
        T_P = 1.0
        N_MACROPARTICLES = 10**5
        electron_energy_list = gen_energies_electrons(N_MACROPARTICLES, T_P)

        assert len(electron_energy_list) == N_MACROPARTICLES
        assert min(electron_energy_list) >= 0


class TestGenEnergyLists:
    """
    Tests the generation of a list of energies allocated to sublists with each
    sublist corresponding to a filter. Tests the generation of both proton and
    electron lists.
    """

    def test_gen_energies_list(self) -> None:
        E_MAX = 1.0
        T_P = 1.0
        N_MACROPARTICLES = 10**5
        N_FILTERS = 9
        proton_energy_list = gen_energies_lists(N_MACROPARTICLES, N_FILTERS, E_MAX, T_P)

        assert len(proton_energy_list) == N_FILTERS
        assert sum(len(sublist) for sublist in proton_energy_list) == N_MACROPARTICLES

    def test_gen_energies_electrons_list(self) -> None:
        T_P = 1.0
        N_MACROPARTICLES = 10**5
        N_FILTERS = 9
        electron_energy_list = gen_energies_electrons_lists(
            N_MACROPARTICLES, N_FILTERS, T_P
        )

        assert len(electron_energy_list) == N_FILTERS
        assert sum(len(sublist) for sublist in electron_energy_list) == N_MACROPARTICLES


class TestDataGeneration:
    """Tests functions associated with synthetic image generation."""

    def test_gen_single_data_uncalibrated(self) -> None:
        """
        Tests the generation of a single proton image and image that includes the electron
        contribution. The image is not calibrated.
        """
        E_MAX_BOUNDS = (0.1, 5)
        T_P_BOUNDS = (0.05, 2)
        N_PARTICLES_BOUNDS = (10**7, 10**10)
        N_MACROPARTICLES = int(1e5)
        SCINT_THICKNESS = None
        BASE_UNIT = [[90e-6, 40e-6, 20e-6], [9e-6, 4e-6, 2e-6], [1e-6, 0.5e-6, 0.2e-6]]
        filter_object = Filter(BASE_UNIT, 20, (1, 1))

        image_and_label = gen_single_data(
            E_MAX_BOUNDS,
            T_P_BOUNDS,
            N_PARTICLES_BOUNDS,
            N_MACROPARTICLES,
            SCINT_THICKNESS,
            np.array(filter_object.filter),
            filter_object.map,
        )

        proton_image, proton_label, combined_image, combined_label = image_and_label

        # Testing shape of the output data
        assert len(proton_label) == 3
        assert len(combined_label) == 5

        assert len(proton_image[0]) == 60
        assert len(proton_image[1]) == 60
        assert len(combined_image[0]) == 60
        assert len(combined_image[1]) == 60

        # Testing that all of the pixel energy deposits are positive
        assert min(proton_image.flatten()) >= 0
        assert min(combined_image.flatten()) >= 0

        # Ensuring that the energy deposited of the electron/proton image is greater than
        # that of the proton image alone
        assert np.mean(combined_image.flatten()) >= np.mean(proton_image.flatten())

    def test_gen_single_data_calibrated(self) -> None:
        """
        Tests the generation of a synthetic image with pixels calibrated to between
        0-4095.
        """
        E_MAX_BOUNDS = (0.1, 5)
        T_P_BOUNDS = (0.05, 2)
        N_PARTICLES_BOUNDS = (10**7, 10**10)
        N_MACROPARTICLES = int(1e5)
        SCINT_THICKNESS = None
        BASE_UNIT = [[90e-6, 40e-6, 20e-6], [9e-6, 4e-6, 2e-6], [1e-6, 0.5e-6, 0.2e-6]]
        filter_object = Filter(BASE_UNIT, 20, (1, 1))
        N_DATA = 1
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

        proton_image = images_and_labels["proton_images"][0]
        proton_label = images_and_labels["proton_labels"][0]
        combined_image = images_and_labels["combined_images"][0]
        combined_label = images_and_labels["combined_labels"][0]

        # Testing shape of the output data
        assert len(proton_label) == 3
        assert len(combined_label) == 5

        assert len(proton_image[0]) == 60
        assert len(proton_image[1]) == 60
        assert len(combined_image[0]) == 60
        assert len(combined_image[1]) == 60

        # Testing that all of the pixel energy deposits are positive
        assert min(proton_image.flatten()) >= 0
        assert min(combined_image.flatten()) >= 0

        # Ensuring that the maximum pixel value is at most 4095
        assert max(proton_image.flatten()) <= 4095
        assert max(combined_image.flatten()) <= 4095

        # Ensuring that the energy deposited of the electron/proton image is greater than
        # that of the proton image alone
        assert np.mean(combined_image.flatten()) >= np.mean(proton_image.flatten())
