# Bayesian optimised design of a differentially filtered particle diagnostic for laser-driven ion sources

Code accompanying the paper: *Bayesian optimised design of differentially filtered spatial and spectral particle diagnostic for laser-driven ion source* — Deol, Truslove, Hussain, Najmudin, Dover.

The goal is to find the optimal set of filter thicknesses for a PROBIES diagnostic. The optimiser iterates over candidate filter configurations, generates synthetic scintillator images for each, trains a CNN to predict beam parameters from those images, and uses the validation loss to guide the next set of candidates. The final filter configuration is the one that gives the lowest validation loss across all trials.

---

## Setup

The code is designed to run inside a Docker container that provides BDSIM/Geant4 and Python 3.12. Build the image with:

```bash
docker build -t ml-ion-beam .
```

The container is based on `jairhul/centos7-geant4v10.7.2.3-jai-environment` (CentOS 7 + Geant4 v10.7.2.3 + BDSIM). Python packages are installed from `requirements.txt` at build time.

---

## Workflow

There are two simulation backends: a fast custom Python implementation and a high-fidelity BDSIM/Geant4 implementation. The custom version is faster but ignores scattering; BDSIM is slower but physically accurate and was used for final results in the paper.

### 1. Generate synthetic images

```bash
python src/custom_generate.py   # fast, for development
python src/bdsim_generate.py    # high-fidelity, for final results
```

Images are saved to `output/synthetic_images/` as pickle files.

### 2. Run Bayesian optimisation

Each script below runs Optuna over many trials, training a CNN per trial and saving the best filter configuration and model. Filter thicknesses are distributed as an exponential function `tf(n) = a*b^n`, where `a` and `b` are the parameters searched by Optuna.

| Script | Description |
|---|---|
| `src/custom_electron_op.py` | Optimise 3 electron filters with 6 proton filters fixed |
| `src/custom_proton_op.py` | Optimise 6 proton filters with 3 electron filters fixed |

The two scripts are run in sequence: first `custom_electron_op.py`, then `custom_proton_op.py` with the resulting electron filter thicknesses fixed.

```bash
python src/custom_electron_op.py
python src/custom_proton_op.py
```

Optuna studies are stored in `output/optuna_studies/` as SQLite databases, which can be inspected with the Optuna dashboard:

```bash
optuna-dashboard sqlite:///output/optuna_studies/new_custom_op_electrons_filters_jan18.db
```

### 3. Train final model

Once the optimal filter configuration is known, train a larger final CNN on a bigger dataset:

```bash
python src/ml.py
```

This reads from `data/` and writes the trained model and label predictions to `output/`.

---

## Repository structure

```
src/
├── bdsim/              # BDSIM simulation wrapper and utilities
├── custom/             # Custom Python simulation (no scattering)
│   └── splines/        # Stopping power tables for Al (protons + electrons)
├── machine_learning/   # CNN architecture and training loop
├── analysis/           # Post-hoc analysis of results
├── custom_electron_op.py  # Optimise 3 electron filters
├── custom_proton_op.py    # Optimise 6 proton filters
├── *_generate.py          # Image generation entry points
└── ml.py                  # Final model training
output/                 # Generated at runtime (models, images, studies)
```