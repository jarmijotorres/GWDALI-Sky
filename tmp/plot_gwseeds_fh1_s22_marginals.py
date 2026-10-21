#!/usr/bin/env python3
"""Compare fh1 Menote Seed marginals with metallicity-marginalized S22 BBHs.

The first two panels use the same estimator as
``plot_gwseeds_bbh_rate_marginals.py`` on every fh1 Seed realisation.  S22 is
then projected onto the shared mass/redshift axes and total-rate matched over
0 <= z <= 1; this makes the comparison one of host-distribution shape.  The
last panel is the S22-only SFR marginal because Menote has no SFR coordinate.
"""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from skysim_gw.rates import rate_bbh  # noqa: E402
from skysim_gw.s22 import S22BBHEmpiricalModel  # noqa: E402


SEEDS_DIR = Path.home() / "Documents/Data/GWseeds/Seeds"
S22_DIR = Path.home() / "Documents/Data/MergerRateDensityModels"
OUTPUT = Path(__file__).with_name("gwseeds_fh1_s22_marginals.png")


def seed_marginals():
    """Return the fh1 reference and one reconstruction per Seed realization."""
    paths = sorted(SEEDS_DIR.glob("gw_hosts_fh1_r*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No fh1 seed files found in {SEEDS_DIR}")
    mass_lines, z_lines = [], []
    for path in paths:
        with h5py.File(path, "r") as handle:
            zi, zf = handle["bins/zi"][:], handle["bins/zf"][:]
            mass_edges = handle["bins/log_stellar_mass_edges"][:]
            selected, expected = handle["assigned/BBH"][:], handle["expected/BBH"][:]
        keep = zf <= 1.0 + 1.0e-12
        zi, zf, selected, expected = zi[keep], zf[keep], selected[keep], expected[keep]
        z_center = 0.5 * (zi + zf)
        mass_center = 0.5 * (mass_edges[:-1] + mass_edges[1:])
        model_cells = rate_bbh(z_center[:, None], mass_center[None, :])
        estimate = np.divide(selected * model_cells, expected, out=np.zeros_like(model_cells), where=expected > 0)
        mass_lines.append((estimate * np.diff(np.r_[zi, zf[-1]])[:, None]).sum(axis=0))
        z_lines.append((estimate * np.diff(mass_edges)[None, :]).sum(axis=1))

    dz = zf - zi
    reference_mass = (model_cells * dz[:, None]).sum(axis=0)
    reference_z = (model_cells * np.diff(mass_edges)[None, :]).sum(axis=1)
    return mass_edges, zi, zf, reference_mass, reference_z, np.asarray(mass_lines), np.asarray(z_lines)


def s22_marginals(mass_edges, zi, zf, target_total):
    """Project the interpolated S22 M57/A3 model and match Menote's total."""
    model = S22BBHEmpiricalModel(S22_DIR, metallicity_relation="M57", common_envelope="A3")
    # Integrate using the Seed z-bin centres, so the comparison has identical
    # redshift quadrature.  ``host_rate_grid`` is per (mass,SFR) bin.
    z_center = 0.5 * (zi + zf)
    dz = zf - zi
    grids = np.asarray([model.host_rate_grid(z) for z in z_center])
    mass_width = np.diff(model.mass_edges)
    sfr_width = np.diff(model.sfr_edges)
    s22_z = grids.sum(axis=(1, 2))
    s22_mass_native = (grids.sum(axis=2) * dz[:, None]).sum(axis=0) / mass_width
    s22_sfr = (grids.sum(axis=1) * dz[:, None]).sum(axis=0) / sfr_width

    # Rebin the mass marginal onto the exact Seed bins by integrating density.
    native_center = 0.5 * (model.mass_edges[:-1] + model.mass_edges[1:])
    seed_center = 0.5 * (mass_edges[:-1] + mass_edges[1:])
    s22_mass = np.interp(seed_center, native_center, s22_mass_native, left=0.0, right=0.0)
    s22_total = np.sum(s22_z * (zf - zi))
    scale = target_total / s22_total
    return scale * s22_mass, scale * s22_z, scale * s22_sfr, model.sfr_edges, scale


def main() -> None:
    mass_edges, zi, zf, menote_mass, menote_z, seed_mass, seed_z = seed_marginals()
    menote_total = np.sum(menote_z * (zf - zi))
    s22_mass, s22_z, s22_sfr, sfr_edges, scale = s22_marginals(mass_edges, zi, zf, menote_total)

    fig, (ax_mass, ax_z, ax_sfr) = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    for line in seed_mass:
        ax_mass.step(mass_edges[:-1], line, where="post", color="0.65", alpha=0.45, lw=0.9)
    for line in seed_z:
        ax_z.step(zi, line, where="post", color="0.65", alpha=0.45, lw=0.9)
    ax_mass.step(mass_edges[:-1], np.median(seed_mass, axis=0), where="post", color="tab:blue", lw=1.7, label="fh1 Seeds (median)")
    ax_z.step(zi, np.median(seed_z, axis=0), where="post", color="tab:blue", lw=1.7, label="fh1 Seeds (median)")
    ax_mass.plot(0.5 * (mass_edges[:-1] + mass_edges[1:]), menote_mass, color="black", lw=2.4, label="Menote fh1")
    ax_z.plot(0.5 * (zi + zf), menote_z, color="black", lw=2.4, label="Menote fh1")
    ax_mass.plot(0.5 * (mass_edges[:-1] + mass_edges[1:]), s22_mass, color="tab:red", lw=2.2, label="S22 M57/A3 (total matched)")
    ax_z.plot(0.5 * (zi + zf), s22_z, color="tab:red", lw=2.2, label="S22 M57/A3 (total matched)")
    ax_sfr.step(sfr_edges[:-1], s22_sfr, where="post", color="tab:red", lw=2.2, label="S22 M57/A3 (total matched)")

    ax_mass.set(xlabel=r"$\log_{10}(M_\star/M_\odot)$", ylabel=r"$\int R\,dz$ [Gpc$^{-3}$ yr$^{-1}$ dex$^{-1}$]", title="Stellar-mass marginal")
    ax_z.set(xlabel="redshift z", ylabel=r"$R(z)$ [Gpc$^{-3}$ yr$^{-1}$]", title="Redshift marginal")
    ax_sfr.set(xlabel=r"$\log_{10}({\rm SFR}/M_\odot\,{\rm yr}^{-1})$", ylabel=r"$\int R\,dz$ [Gpc$^{-3}$ yr$^{-1}$ dex$^{-1}$]", title="SFR marginal (S22 only)")
    for axis in (ax_mass, ax_z, ax_sfr):
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"BBH host-rate marginals over 0 ≤ z ≤ 1; S22 scale factor = {scale:.3g}")
    fig.savefig(OUTPUT, dpi=220)
    print(OUTPUT)


if __name__ == "__main__":
    main()
