"""
Torch reimplementation of create_proton_splines.py, where the cubic splines are
implemented as torch ops so they can be differentiated with autograd.
"""

import csv
import os
from typing import Tuple

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_ALUMINIUM = os.path.join(_HERE, "deposit_tables/al_proton_energies.csv")
CSV_SCINTILLATOR = os.path.join(_HERE, "deposit_tables/sc_proton_energies.csv")

RHO_AL = 2.71      # aluminium density (g/cm^3)
RHO_PVT = 0.9383   # PVT scintillator density (g/cm^3); should later correct to 1.032

DTYPE = torch.float64


def _thomas_batched(
    sub: torch.Tensor, diag: torch.Tensor, sup: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """
    Solves a tridiagonal system with shared coefficients for a batch of right hand
    sides. sub, diag and sup have shape (k,); rhs has shape (S, k). Written
    functionally so autograd can flow through it.
    """
    k = diag.shape[0]
    cp = [sup[0] / diag[0]]
    dp = [rhs[:, 0] / diag[0]]
    for i in range(1, k):
        denom = diag[i] - sub[i] * cp[i - 1]
        cp.append(sup[i] / denom if i < k - 1 else torch.zeros_like(denom))
        dp.append((rhs[:, i] - sub[i] * dp[i - 1]) / denom)
    x = [None] * k
    x[k - 1] = dp[k - 1]
    for i in range(k - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return torch.stack(x, dim=1)


def _natural_second_derivs(knots: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Returns the natural cubic spline second derivatives through (knots, values),
    with knots of shape (n,) and batched values of shape (S, n).
    """
    h = knots[1:] - knots[:-1]
    dy = (values[:, 1:] - values[:, :-1]) / h
    rhs = 6.0 * (dy[:, 1:] - dy[:, :-1])
    diag = 2.0 * (h[:-1] + h[1:])
    m_interior = _thomas_batched(h[:-1], diag, h[1:], rhs)
    zeros = torch.zeros(values.shape[0], 1, dtype=values.dtype)
    return torch.cat([zeros, m_interior, zeros], dim=1)


class TorchCubicSpline1D:
    """Natural cubic spline over 1D knots, differentiable w.r.t. the query."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x.to(DTYPE)
        self.y = y.to(DTYPE)
        self.m = _natural_second_derivs(self.x, self.y.unsqueeze(0)).squeeze(0)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=DTYPE)
        shape = t.shape
        tf = t.reshape(-1).clamp(self.x[0], self.x[-1])
        idx = (torch.searchsorted(self.x, tf, right=True) - 1).clamp(0, self.x.shape[0] - 2)
        xi, xi1 = self.x[idx], self.x[idx + 1]
        h = xi1 - xi
        a = (xi1 - tf) / h
        b = (tf - xi) / h
        out = (
            a * self.y[idx]
            + b * self.y[idx + 1]
            + ((a ** 3 - a) * self.m[idx] + (b ** 3 - b) * self.m[idx + 1]) * (h ** 2) / 6.0
        )
        return out.reshape(shape)

    def state(self) -> dict:
        return {"kind": "1d", "x": self.x, "y": self.y}

    @classmethod
    def load(cls, path: str) -> "TorchCubicSpline1D":
        s = torch.load(path, weights_only=True)
        return cls(s["x"], s["y"])


class TorchCubicSpline2D:
    """
    Tensor-product natural cubic spline over a regular grid, with x of shape (nx,),
    y of shape (ny,) and Z of shape (nx, ny). Differentiable w.r.t. both queries.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, Z: torch.Tensor):
        self.x = x.to(DTYPE)
        self.y = y.to(DTYPE)
        self.Z = Z.to(DTYPE)
        # Precompute the second-derivative arrays once (natural BC in both axes):
        # along x, along y, and the mixed term. Evaluation is then pure gather +
        # arithmetic with no per-call solve.
        self.Mx = _natural_second_derivs(self.x, self.Z.t().contiguous()).t().contiguous()
        self.My = _natural_second_derivs(self.y, self.Z)
        self.Mxy = _natural_second_derivs(self.y, self.Mx)

    def __call__(self, xq: torch.Tensor, yq: torch.Tensor) -> torch.Tensor:
        xq = torch.as_tensor(xq, dtype=DTYPE)
        yq = torch.as_tensor(yq, dtype=DTYPE)
        shape = torch.broadcast_shapes(xq.shape, yq.shape)
        xf = xq.expand(shape).reshape(-1).clamp(self.x[0], self.x[-1])
        yf = yq.expand(shape).reshape(-1).clamp(self.y[0], self.y[-1])

        ix = (torch.searchsorted(self.x, xf, right=True) - 1).clamp(0, self.x.shape[0] - 2)
        iy = (torch.searchsorted(self.y, yf, right=True) - 1).clamp(0, self.y.shape[0] - 2)

        hx = self.x[ix + 1] - self.x[ix]
        a = (self.x[ix + 1] - xf) / hx
        b = (xf - self.x[ix]) / hx
        ca = (a ** 3 - a) * hx ** 2 / 6.0
        cb = (b ** 3 - b) * hx ** 2 / 6.0

        hy = self.y[iy + 1] - self.y[iy]
        ay = (self.y[iy + 1] - yf) / hy
        by = (yf - self.y[iy]) / hy
        cay = (ay ** 3 - ay) * hy ** 2 / 6.0
        cby = (by ** 3 - by) * hy ** 2 / 6.0

        def corners(m):
            return m[ix, iy], m[ix + 1, iy], m[ix, iy + 1], m[ix + 1, iy + 1]

        z00, z10, z01, z11 = corners(self.Z)
        mx00, mx10, mx01, mx11 = corners(self.Mx)
        my00, my10, my01, my11 = corners(self.My)
        mxy00, mxy10, mxy01, mxy11 = corners(self.Mxy)

        # Interpolate along x: the value and its y-second-derivative at y[j], y[j+1].
        g_j = a * z00 + b * z10 + ca * mx00 + cb * mx10
        g_j1 = a * z01 + b * z11 + ca * mx01 + cb * mx11
        gyy_j = a * my00 + b * my10 + ca * mxy00 + cb * mxy10
        gyy_j1 = a * my01 + b * my11 + ca * mxy01 + cb * mxy11

        # Interpolate along y using those as the y-second-derivatives.
        out = ay * g_j + by * g_j1 + cay * gyy_j + cby * gyy_j1
        return out.reshape(shape)

    def state(self) -> dict:
        return {"kind": "2d", "x": self.x, "y": self.y, "Z": self.Z}

    @classmethod
    def load(cls, path: str) -> "TorchCubicSpline2D":
        s = torch.load(path, weights_only=True)
        return cls(s["x"], s["y"], s["Z"])


def _read_table(filepath: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reads a two-column (incoming energy, mass stopping power) NIST CSV."""
    incoming, stopping = [], []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            incoming.append(float(row[0]))
            stopping.append(float(row[1]))
    return torch.tensor(incoming, dtype=DTYPE), torch.tensor(stopping, dtype=DTYPE)


def _march_remaining(
    stopping: TorchCubicSpline1D, E: torch.Tensor, thickness: torch.Tensor, rho: float, steps: int
) -> torch.Tensor:
    """
    Step-integrates the remaining energy as particles pass through the material,
    freezing particles once they reach zero energy.
    """
    h = thickness / steps
    E = E.clone()
    for _ in range(steps):
        active = E > 0
        E = torch.where(active, E - stopping(E.clamp_min(stopping.x[0])) * rho * h, E)
    return E.clamp_min(0.0)


def build_al_remaining_spline(
    energy_range: Tuple[float, float],
    thickness_range: Tuple[float, float],
    n_energy: int = 1200,
    n_thickness: int = 400,
    steps: int = 1000,
) -> TorchCubicSpline2D:
    """Builds the 2D spline of remaining proton energy after a thickness of aluminium."""
    e_in, s_in = _read_table(CSV_ALUMINIUM)
    stopping_al = TorchCubicSpline1D(e_in, s_in)

    energies = torch.linspace(energy_range[0], energy_range[1], n_energy, dtype=DTYPE)
    thicknesses = torch.logspace(thickness_range[0], thickness_range[1], n_thickness, dtype=DTYPE)

    grid_E = energies[:, None].expand(n_energy, n_thickness)
    grid_t = thicknesses[None, :].expand(n_energy, n_thickness)
    remaining = _march_remaining(stopping_al, grid_E, grid_t, RHO_AL, steps)

    return TorchCubicSpline2D(energies, thicknesses, remaining)


def build_sc_deposited_spline(
    energy_range: Tuple[float, float],
    scintillator_thickness: float,
    n_energy: int = 1200,
    steps: int = 1000,
) -> TorchCubicSpline1D:
    """Builds the 1D spline of energy deposited by a proton in the scintillator."""
    e_in, s_in = _read_table(CSV_SCINTILLATOR)
    stopping_sc = TorchCubicSpline1D(e_in, s_in)

    energies = torch.linspace(energy_range[0], energy_range[1], n_energy, dtype=DTYPE)
    thickness = torch.full_like(energies, scintillator_thickness)
    remaining = _march_remaining(stopping_sc, energies, thickness, RHO_PVT, steps)

    return TorchCubicSpline1D(energies, energies - remaining)


def main() -> None:
    pickles = os.path.join(_HERE, "pickles")
    os.makedirs(pickles, exist_ok=True)

    al = build_al_remaining_spline((0, 5), (-5, 0))
    torch.save(al.state(), os.path.join(pickles, "al_proton_remaining_spline_torch.pt"))

    sc = build_sc_deposited_spline((0, 5), 2e-3)  # 2e-3 cm = 20 um
    torch.save(sc.state(), os.path.join(pickles, "sc_proton_deposited_spline_torch.pt"))


if __name__ == "__main__":
    main()
