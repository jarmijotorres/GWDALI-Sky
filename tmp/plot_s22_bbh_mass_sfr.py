#!/usr/bin/env python3
"""Visualise S22 BBH host weights in stellar mass and SFR.

Metallicity is deliberately not plotted or sampled here.  It is already
encoded in the selected S22 metallicity prescription (M57 = MZR or M54 =
FMR).  The third panel mixes the conditional host distributions over
0 <= z <= 1 using the BBH redshift factor psi(z), not the absolute S22 rate.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from skysim_gw.rates import P_BBH, psi  # noqa: E402


DATA_DIR = Path.home() / "Documents/Data/MergerRateDensityModels"
OUTPUT = Path(__file__).with_name("s22_bbh_mstar_sfr_psi_z0_1.png")

# Exact redshifts supplied with the data release for the low-z EAGLE outputs.
LOW_Z_SNAPSHOTS = {
    28: 0.0,
    27: 0.10063853859901428,
    26: 0.18270987272262573,
    25: 0.27090108394622803,
    24: 0.36566856503486633,
    23: 0.5031073093414307,
    22: 0.615189790725708,
    21: 0.7356296181678772,
    20: 0.8650505542755127,
    19: 1.0041216611862183,
}

MODEL = "M57"  # MZR; change to M54 to use the FMR.
ALPHA = "A3"  # Central common-envelope-efficiency scenario.
MASS_EDGES = np.linspace(7.0, 12.0, 51)
SFR_EDGES = np.linspace(-4.0, 2.5, 53)


def load_snapshot(snapshot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log Mstar, log SFR, and n_GW for one S22 host catalogue."""
    path = DATA_DIR / (
        f"HG_{MODEL}_{ALPHA}_GSSI_v2_BBHs_{snapshot}_12Z_50_No_0.14_2.txt"
    )
    values = np.loadtxt(path)
    return np.log10(values[:, 0]), np.log10(values[:, 1]), values[:, 3]


def weighted_grid(
    log_mass: np.ndarray, log_sfr: np.ndarray, rate: np.ndarray
) -> np.ndarray:
    """Rate-weighted host probability in each (log Mstar, log SFR) cell."""
    grid, _, _ = np.histogram2d(
        log_mass, log_sfr, bins=(MASS_EDGES, SFR_EDGES), weights=rate
    )
    return grid / grid.sum()


def draw_panel(axis, grid: np.ndarray, title: str):
    image = axis.pcolormesh(
        MASS_EDGES, SFR_EDGES, grid.T, shading="auto",
        cmap="magma", norm=LogNorm(vmin=1e-6, vmax=1e-1),
    )
    axis.set(title=title, xlabel=r"$\log_{10}(M_\star/M_\odot)$")
    axis.grid(alpha=0.12, lw=0.5)
    return image


def main() -> None:
    grids = {}
    mixture = np.zeros((len(MASS_EDGES) - 1, len(SFR_EDGES) - 1))
    # Snapshot 19 is z=1.004; use it as the available z=1 endpoint when
    # approximating the requested closed interval 0 <= z <= 1.
    redshift = np.minimum(np.array(list(LOW_Z_SNAPSHOTS.values())), 1.0)
    # Trapezoid coefficients make the discrete snapshot set an approximation
    # to integral_0^1 p(host | z) psi(z) dz.
    dz = np.gradient(redshift)

    for coefficient, (snapshot, z) in zip(dz, LOW_Z_SNAPSHOTS.items()):
        log_mass, log_sfr, rate = load_snapshot(snapshot)
        grid = weighted_grid(log_mass, log_sfr, rate)
        grids[snapshot] = grid
        mixture += coefficient * psi(z, P_BBH) * grid
    mixture /= mixture.sum()

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharex=True, sharey=True)
    image = draw_panel(axes[0], grids[28], r"S22 conditional hosts: $z=0$")
    draw_panel(axes[1], grids[19], r"S22 conditional hosts: $z\simeq1$")
    draw_panel(axes[2], mixture, r"$0\leq z\leq1$, mixed with $\psi(z)$")
    axes[0].set_ylabel(r"$\log_{10}({\rm SFR}/M_\odot\,{\rm yr}^{-1})$")
    colorbar = fig.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_label("BBH host probability per plotted cell")
    fig.suptitle("Santoliquido et al. BBH hosts: M57 (MZR), $\\alpha_{\\rm CE}=3$")
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
