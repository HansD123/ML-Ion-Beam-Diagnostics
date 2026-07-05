"""
Generates proton energies, distributes them across a grid keeping the energies
per cell, pads each cell to the length of the longest entry, and saves the
result as torch tensors.
"""

import os
import numpy as np
import torch
from tqdm import tqdm

import custom.generation as dg
import utils

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = f"{current_dir}/../output/synthetic_images"
output_filename = "custom_energies.pt"
output_path = f"{output_dir}/{output_filename}"


def distribute_energies(energies: np.ndarray, dimensions: tuple) -> list:
    """
    Uniformly distributes energies onto a grid, keeping the list of energies that
    land on each cell (not summed).
    """
    rows, cols = dimensions
    grid = [[[] for _ in range(cols)] for _ in range(rows)]
    row_idx = np.random.randint(0, rows, size=len(energies))
    col_idx = np.random.randint(0, cols, size=len(energies))
    for r, c, e in zip(row_idx, col_idx, energies):
        grid[r][c].append(e)
    return grid


def pad_grids(grids: list, pad_value: float = 0.0) -> tuple:
    """
    Pads a batch of ragged energy grids to the longest cell across the whole
    batch. Returns padded and mask tensors of shape (n, rows, cols, k).
    """
    n = len(grids)
    rows, cols = len(grids[0]), len(grids[0][0])
    k = max((len(cell) for g in grids for row in g for cell in row), default=0)

    padded = torch.full((n, rows, cols, k), pad_value, dtype=torch.float32)
    mask = torch.zeros((n, rows, cols, k), dtype=torch.bool)
    for i, g in enumerate(grids):
        for r in range(rows):
            for c in range(cols):
                cell = g[r][c]
                if cell:
                    padded[i, r, c, : len(cell)] = torch.tensor(cell, dtype=torch.float32)
                    mask[i, r, c, : len(cell)] = True
    return padded, mask


def generate_energies(
    n_samples: int,
    n_macroparticles: int,
    grid_dimensions: tuple,
    e_max_bounds: tuple,
    t_p_bounds: tuple,
    n_particles_bounds: tuple,
) -> tuple:
    """
    Generates a batch of padded energy grids and their labels. Returns padded and
    mask tensors of shape (n_samples, rows, cols, k) and a (n_samples, 3) labels
    tensor of (E_max, T_p, N_p).
    """
    grids = []
    labels = []
    for _ in tqdm(range(n_samples)):
        e_max, t_p, n0 = dg.gen_params(e_max_bounds, t_p_bounds, n_particles_bounds)
        energies = dg.gen_energies(n_macroparticles, e_max, t_p)
        grids.append(distribute_energies(energies, grid_dimensions))
        labels.append((e_max, t_p, n0))

    padded, mask = pad_grids(grids)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, mask, labels


def main() -> None:
    # Data params
    E_MAX_BOUNDS = (0.1, 5)
    T_P_BOUNDS = (0.05, 2)
    N_PARTICLES_BOUNDS = (1e7, 1e10)
    N_MACROPARTICLES = 25000
    GRID_DIMENSIONS = (30, 30)  # matches Filter(BASE_UNIT, 10, (1, 1)) in custom_generate.py
    N_SAMPLES = 100

    utils.create_output_dirs()

    padded, mask, labels = generate_energies(
        N_SAMPLES,
        N_MACROPARTICLES,
        GRID_DIMENSIONS,
        E_MAX_BOUNDS,
        T_P_BOUNDS,
        N_PARTICLES_BOUNDS,
    )

    torch.save({"energies": padded, "mask": mask, "labels": labels}, output_path)
    print(f"Saved energies tensor {tuple(padded.shape)} to {output_path}")


if __name__ == "__main__":
    main()
