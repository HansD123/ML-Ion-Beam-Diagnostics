
# Machine Learning Guided Diagnostics for Laser-Driven Ion Beams

Link to report: https://github.com/arantruslove/ML-Ion-Beam-Diagnostics/blob/main/Aran_Truslove_ML_Diagnostics_Report.pdf

## Overview
This project explores the application of machine learning, specifically convolutional neural networks (CNNs), to rapidly diagnose key parameters of laser-driven ion beams. By using synthetic data generation and Bayesian optimization, the project aims to predict three essential beam parameters: maximum proton energy (Emax), proton spectra temperature (Tp), and the number of protons in the beam (N0). The developed methods are intended to improve the efficiency and accuracy of beam diagnostics in advanced laser-driven ion acceleration systems.

## Project Objectives
- **Synthetic Data Generation:** Develop a Monte Carlo simulation to generate synthetic images of proton beams passing through a 3x3 grid of aluminium filters and a scintillator.
- **Neural Network Training:** Train a CNN on the synthetic dataset to predict beam parameters from CCD-captured images.
- **Bayesian Optimization:** Optimise both the filter thicknesses used in the data generation process and the architecture of the CNN to enhance prediction accuracy.

## Methodology
1. **Data Generation:** 
   - Simulated 100,000 synthetic images using a model that accounts for proton energy distributions and energy deposition in filters and scintillators.
   - Applied Gaussian noise and normalization techniques to mimic real-world data collection.

2. **Neural Network:**
   - Implemented a CNN using Keras, optimised through mean-squared error loss minimization.
   - Trained on 75,000 images and validated on 25,000 images.

3. **Bayesian Optimization:**
   - Used Optuna to optimise filter thickness configurations and CNN architectural parameters, improving model accuracy significantly.

## Results
- **Performance Improvements:** Bayesian optimization led to a 40% reduction in prediction error when optimizing filter thicknesses and an additional 10% improvement when optimizing the CNN architecture.
- **Prediction Accuracy:** The optimised CNN achieved mean relative absolute errors (MRAE) of 7% for Emax, 13% for Tp, and 4% for N0.
- **Speed:** The optimised CNN's prediction time averaged 7.3 ms, making it suitable for real-time applications in high-frequency laser systems.

## UROP Work
- **BDSIM:** Integrated BDSIM, a wrapper for Geant4, to produce highly reliable and accurate simulations of the passage of ions through materials.
- **Electrons**: Modelled the addition of electrons in both the custom Python and BDSIM simulations.
- **Containerisation**: Implemented Docker along with Apptainer to allow the BDSIM integrated code to operate seamlessly on different platforms including on HPC clusters.
- **Classifier Network**: Developed a CNN to classify images based on how accurately the regression CNN is expected to predict beam parameters for each image

## How to Use Code
- **Dockerfile**: Dockerfile for installing BDSIM/Geant 4/Python on Linux machine to use for simulations (ie on HPC)
- **src/bdsim**: Stores logic for BDSIM simulator
- **src/custom**: stores logic for custom Python simulation
- **src/bdsim_generate**: script to generate bdsim simulation images
- **src/bdsim_op**: script to perform bayesian optimisation (optuna) for BDSIM simulations
- **src/custom_electron_op**: script to perform bayesian optimisation (optuna) for custom Python simulations with electron filters
- **src/custom_generate**: script to generate custom Python simulation images
- **src/custom_op**: script to perform bayesian optimisation (optuna) for custom Python simulations
- **src/custom_proton_op**: script to perform bayesian optimisation (optuna) for custom Python simulations with proton filters
- **src/ml**: trains NN of set of images

## Conclusion
The project demonstrates the potential of machine learning to significantly improve the speed and accuracy of diagnostics in laser-driven ion beam systems. The optimised CNN model shows promising results for real-time applications, paving the way for further research and development, particularly with regard to the use of Bayesian optimisation to optimise filter parameters.
