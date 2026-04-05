import bdsim.bdsim_utilities as utils
import numpy as np
import os
import random
import subprocess
import uproot
from scipy.ndimage import gaussian_filter
from typing import List, Dict
from tqdm import tqdm
import multiprocessing

# Get the current directory path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)


class BDSIMGenerator:
    def __init__(
        self,
        Emax_bounds_proton: tuple,
        Tp_bounds_proton: tuple,
        N0_bounds_proton: tuple,
        Job_name: str,
        Number_macroparticles: int,
        N_data: int,
        N_workers: int,
        Filter_size: float,
        centre: tuple,
        Filter_array: list,
        scintillator_thickness: float,
        image_pixel_width: int,
        Tp_range_electrons: tuple | None = None,
        N0_range_electrons: tuple | None = None,
        clear_files: bool = True,
    ) -> None:
        """
        Initialises BDSIMGenerator:
        class with the following parameters:
        Emax_bounds_proton: tuple containing the minimum and maximum proton energy values
        Tp_bounds_proton: tuple containing the minimum and maximum proton temperature values
        N0_bounds_proton: tuple containing the minimum and maximum proton number values
        Job_name: name of the job
        Number_macroparticles: number of macroparticles
        N_data: number of images
        N_workers: number of workers
        Filter_size: filter size
        centre: beam origin
        Filter_array: filter array
        scintillator_thickness: scintillator thickness
        Tp_range_electrons: tuple containing the minimum and maximum electron temperature values
        N0_range_electrons: tuple containing the minimum and maximum electron number values
        clear_files: boolean value to clear files
        """
        # Creating directories for temporary BDSIM file outputs
        utils.create_output_dirs()

        self.E_bound_proton = Emax_bounds_proton
        self.Tp_bound_proton = Tp_bounds_proton
        self.N0_bound_proton = N0_bounds_proton
        self.Job_name = Job_name
        self.Number_macroparticles = Number_macroparticles
        self.N_data = N_data
        self.N_workers = N_workers
        self.Filter_size = Filter_size
        self.centre = centre
        self.Filter_array = Filter_array
        self.scintillator_thickness = scintillator_thickness
        self.pixel_number = image_pixel_width
        self.Tp_range_electrons = Tp_range_electrons
        self.N0_range_electrons = N0_range_electrons
        self.clear_files = clear_files
        self.add_electrons = False
        if self.Tp_range_electrons and self.N0_range_electrons:
            self.add_electrons = True

    def generate_energies(
        self,
        T: float,
        E_max: float = 0,
        particle_type: str = "proton",
    ) -> np.ndarray:
        """
        Generates energies from specified beam distribution
        E_max (MeV) is the maximum energy of the beam (irrelevant for electron beams)
        T (MeV) is the temperature of the beam
        particle_type ("electron" or "proton") is the type of particle being generated
        """
        proton_rest_mass = 938.272  # MeV as BDSIM requires particles rest mass energy not kinetic energy
        electron_rest_mass = (
            0.511  # MeV as BDSIM requires particles rest mass energy not kinetic energy
        )

        if len(self.centre) == 2:
            raise Exception(
                "Error: Length of centre tuple must be 3 (for 3 dimensions)"
            )
        # Generating energies array:
        energies = []
        if particle_type == "proton":
            count = 0
            while count < self.Number_macroparticles:
                deviate = np.random.random()
                energy = -T * np.log(1 - deviate * (1 - np.exp(-E_max / T)))
                if energy > E_max:
                    pass
                else:
                    energies.append(energy + proton_rest_mass)
                    count += 1
            energies = np.array(energies)
        elif particle_type == "electron":
            for i in range(self.Number_macroparticles):
                deviate = np.random.random()
                energy = -T * np.log(1 - deviate)
                energies.append(energy + electron_rest_mass)
            energies = np.array(energies)
        else:
            raise Exception(
                'Error: particle type must be one of "electron" or "proton"'
            )
        return energies

    def generate_momenta(self) -> List[List[float]]:
        """
        Generates x and y momenta from specified beam distribution
        """
        thetaxmin = np.arctan((self.centre[0] - 1.5*self.Filter_size) / self.centre[2])
        thetaxmax = np.arctan((self.centre[0] + 1.5*self.Filter_size) / self.centre[2])
        thetaymin = np.arctan((self.centre[1] - 1.5*self.Filter_size) / self.centre[2])
        thetaymax = np.arctan((self.centre[1] + 1.5*self.Filter_size) / self.centre[2])
        momenta = []
        for i in range(self.Number_macroparticles):
            xp = np.random.uniform(thetaxmin, thetaxmax)
            yp = np.random.uniform(thetaymin, thetaymax)
            momenta.append([xp,yp])
        return momenta

    def generate_particle_file(
        self, T: float, Job_number: str, E: float = 0, particle_type: str = "proton"
    ) -> float:
        """
        Generates a particle .dat file for BDSIM
        """
        proton_rest_mass = 938.272
        energies = self.generate_energies(T, E, particle_type)
        momenta = self.generate_momenta()
        particles_array = []
        for i in range(self.Number_macroparticles):
            particles_array.append(
                [
                    self.centre[0],
                    self.centre[1],
                    -self.centre[2],
                    energies[i],
                    momenta[i][0],
                    momenta[i][1],
                ]
            )
        filepath = (
            f"{current_dir}/bdsim_particle_files/"
            + particle_type
            + "_"
            + self.Job_name
            + "_"
            + str(Job_number)
            + ".dat"
        )

        with open(filepath, "w") as file:
            for row in particles_array:
                line = " ".join(map(str, row))
                file.write(line + "\n")

        return np.max(energies) - proton_rest_mass

    def generate_parameters_loop(self, number_of_jobs: int, start: int) -> np.ndarray:
        """
        Loops through the parameter ranges and calls the function generate_particle_file for a series of parameters that are
        randomly generated between the two ranges, for the number of times specified by number_of_jobs
        This is performed for both protons and electrons separately, and creates two separate files, one for proton and one for electron
        Number of macropartiles is the number of protons or electrons simulated (ie total number of particles simulated is double number of macroparticles)
        """
        parameters = []

        for i in range(number_of_jobs):
            row = []
            # Randonly generating proton parameters
            E_max = random.uniform(
                self.E_bound_proton[0], self.E_bound_proton[1]
            )  # Defining parameters for each job
            row.append(E_max)
            T_p_proton = random.uniform(
                self.Tp_bound_proton[0], self.Tp_bound_proton[1]
            )
            row.append(T_p_proton)

            expo = random.uniform(self.N0_bound_proton[0], self.N0_bound_proton[1])
            row.append(expo)
            # Creating proton beam file
            Emax_actual = self.generate_particle_file(
                T_p_proton, start + i, E_max, "proton"
            )
            row[0] = Emax_actual
            # Randomly generating electron parameters
            if self.add_electrons:
                T_p_electron = random.uniform(
                    self.Tp_range_electrons[0], self.Tp_range_electrons[1]
                )
                row.append(T_p_electron)
                expo = random.uniform(
                    self.N0_range_electrons[0], self.N0_range_electrons[1]
                )
                row.append(expo)
                self.generate_particle_file(
                    T_p_electron, start + i, particle_type="electron"
                )
            parameters.append(row)
        return np.array(parameters)

    def run_single_bdsim_script(
        self,
        particle_type: str,
        number: int,
    ) -> None:
        """
        Runs one full simulation in BDSIM, producing an output file in bdsim_outputs
        """
        if particle_type != "proton" and particle_type != "electron":
            raise Exception(
                'Error: Particle type must be either "proton" or "electron"'
            )

        try:
            result = subprocess.run(
                [
                    "bash",
                    f"{current_dir}/run_bdsim_single.sh",
                    particle_type.replace('"', "'"),
                    self.Job_name.replace('"', "'"),
                    str(self.Number_macroparticles).replace('"', "'"),
                    str(number).replace('"', "'"),
                    str(current_dir).replace('"', "'"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error {e.stderr} (return code: {e.returncode})")
        except Exception as e:
            raise Exception(f"Error: {e}")

    def gen_script(self, Job_number: int, particle_type: str) -> None:
        """
        Generates a script for BDSIM
        """
        utils.generate_script(
            self.Job_name,
            Job_number,
            self.Filter_array,
            self.scintillator_thickness,
            self.Filter_size,
            self.pixel_number,
            particle_type,
        )

    def merge_histograms(
        self, number_of_jobs: int, particle_type: str, start: int
    ) -> None:
        """
        Merges .root file histograms together to create the correct histograms to extract
        data from.
        """

        if particle_type != "proton" and particle_type != "electron":
            raise Exception(
                'Error: Particle type must be either "proton" or "electron"'
            )

        try:
            result = subprocess.run(
                [
                    "bash",
                    f"{current_dir}/merge_histo.sh",
                    particle_type.replace('"', "'"),
                    self.Job_name.replace('"', "'"),
                    str(number_of_jobs + start - 1).replace('"', "'"),
                    str(start).replace('"', "'"),
                    str(current_dir).replace('"', "'"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error {e.stderr} (return code: {e.returncode})")
        except Exception as e:
            raise Exception(f"Error: {e}")

    def extract_grids(
        self, number_of_jobs: int, particle_type: str, start: int
    ) -> List[np.ndarray]:
        """
        Converts the root histograms into list and outputs them to folder
        bdsim_grids
        """

        if particle_type != "proton" and particle_type != "electron":
            raise Exception(
                'Error: Particle type must be either "proton" or "electron"'
            )

        data_list = []
        for i in range(number_of_jobs):
            file = uproot.open(
                f"{current_dir}/bdsim_outputs/"
                + particle_type
                + "_output_"
                + self.Job_name
                + "_"
                + str(start + i)
                + ".root"
            )
            tree = file["Event"]
            hist3D = tree["MergedHistograms/detector-denergy"]
            hist_np = hist3D.to_numpy()
            counts_3d = hist_np[0]
            counts_2d = (
                np.sum(counts_3d, axis=2)
                if counts_3d.shape[2] > 1
                else counts_3d[:, :, 0]
            )
            data_list.append(np.transpose(counts_2d))

        return data_list

    def gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a guassian blur to the image. Sets sigma to be a thrid of the radius, ie
        the number of pixels that are significant to the blur.
        """
        radius = 2  # assuming that the scintillator resolution is twice the pixel size of the images
        sigma = radius / 3
        return_arr = np.empty(image.shape)

        # get the number of squares per side i.e, sqrt(n_filters)
        length = 3

        # define one step which is total length / n_filters
        step = 10

        # define the array to iterate through which is just integer multiples of 'step'
        iter_array = [step * i for i in range(1, length + 1)]

        # iterate along y axis
        for i in iter_array:

            # iterate along x axis
            for j in iter_array:

                # set that 'square' = gaussian filtered version of the image
                return_arr[i - step : i, j - step : j] = gaussian_filter(
                    image[i - step : i, j - step : j], sigma
                )

        # then gaussian blur the whole image
        return_arr = gaussian_filter(return_arr, sigma)
        return return_arr

    def generate_proton_image(
        self, job_number: int, Emax: float, Tp: float
    ) -> List[np.ndarray]:
        """
        Generates an image using protons
        """
        self.gen_script(job_number, "proton")
        self.run_single_bdsim_script("proton", job_number)
        self.merge_histograms(1, "proton", job_number)
        data_proton = self.extract_grids(1, "proton", job_number)
        return data_proton

    def generate_electron_image(self, job_number: int, Tp: float) -> List[np.ndarray]:
        """
        Generates an image using electrons
        """
        self.gen_script(job_number, "electron")
        self.run_single_bdsim_script("electron", job_number)
        self.merge_histograms(1, "electron", job_number)
        data_electron = self.extract_grids(1, "electron", job_number)
        return data_electron

    def generate_single_image(self, job_number: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a single image, either with electrons included or without them included
        """

        parameters = self.generate_parameters_loop(1, job_number)
        data_proton = self.generate_proton_image(
            job_number, parameters[0][0], parameters[0][1]
        )
        multiplication_factor = [parameters[0][2] / self.Number_macroparticles]
        if self.add_electrons:
            data_electron = self.generate_electron_image(job_number, parameters[0][3])
            multiplication_factor.append(parameters[0][4] / self.Number_macroparticles)
        images = []
        for i in range(len(data_proton)):
            data_proton[i] = data_proton[i] * multiplication_factor[0]
            if self.add_electrons:
                data_electron[i] = data_electron[i] * multiplication_factor[1]
                images.append(self.gaussian_blur(data_proton[i] + data_electron[i]))
            else:
                images.append(self.gaussian_blur(data_proton[i]))
        return images[0], parameters[0]

    def gen_many_data(
        self, number_of_jobs: int, random_seed: int, start: int
    ) -> Dict[str, List]:
        # Reseeding the random number based on time
        np.random.seed(random_seed)

        images_and_labels = []
        for i in tqdm(range(number_of_jobs)):
            output = self.generate_single_image(start + i)
            images_and_labels.append((output[0], output[1]))

        images = [item[0] for item in images_and_labels]
        labels = [item[1] for item in images_and_labels]

        return {"images": images, "labels": labels}

    def divide_and_distribute(self, number: int, divisor: int) -> List[int]:
        if divisor <= 0:
            return "Error: Divisor must be greater than 0"

        quotient, remainder = divmod(number, divisor)
        result_list = [quotient] * divisor

        # Distribute the remainder across the list, one by one until exhausted
        for i in range(remainder):
            result_list[i] += 1

        return result_list

    def gen_many_parallel(self) -> Dict[str, List]:

        args_list = []
        number_split = self.divide_and_distribute(self.N_data, self.N_workers)
        start = 0
        for number in number_split:
            # Generating a random seed for each set of arguments
            random_seed = np.random.randint(0, 2**31 - 1)
            args = (number, random_seed, start)
            args_list.append(args)
            start += number

        # Data generation in parallel
        with multiprocessing.Pool() as pool:
            results = pool.starmap(self.gen_many_data, args_list)

        # Combining the results
        images = []
        labels = []
        for result in results:
            images += result["images"]
            labels += result["labels"]

        if self.clear_files:
            utils.clear_bdsim_outputs()
            utils.clear_bdsim_particle_files()
            utils.clear_bdsim_scripts()

        return {"images": images, "labels": labels}
