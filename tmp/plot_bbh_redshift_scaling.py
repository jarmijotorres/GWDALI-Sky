#!/usr/bin/env python3
"""Tabulate and plot the BBH redshift weighting for 0 <= z <= 2.

The histogram is the observer-frame relative event distribution used for
sampling: psi(z) * dVc/(dz dOmega) / (1 + z).  It deliberately excludes
survey area, observing time, and detection efficiency because all are either
constant or belong to a later selection model.  The saved CDF can be sampled
with ``np.interp(u, cdf, z_edges)`` for uniform ``u`` in [0, 1].
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from skysim_gw.rates import P_BBH, differential_comoving_volume, psi  # noqa: E402


Z_EDGES = np.linspace(0.0, 2.0, 101)
OUTPUT = Path(__file__).with_name("bbh_redshift_scaling_z0_2.png")
TABLE_OUTPUT = Path(__file__).with_name("bbh_redshift_scaling_z0_2.npz")


def main() -> None:
    z_center = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
    dz = np.diff(Z_EDGES)
    psi_value = psi(z_center, P_BBH)
    # Units before normalization: Gpc^3 sr^-1, times the arbitrary psi scale.
    observer_weight = psi_value * differential_comoving_volume(z_center) / (1.0 + z_center)
    probability = observer_weight * dz
    probability /= probability.sum()
    cdf = np.concatenate(([0.0], np.cumsum(probability)))
    cdf[-1] = 1.0  # Avoid a rounding-level interpolation endpoint error.

    np.savez_compressed(
        TABLE_OUTPUT,
        z_edges=Z_EDGES,
        z_center=z_center,
        dz=dz,
        psi=psi_value,
        observer_weight=observer_weight,
        probability=probability,
        cdf=cdf,
        description=(
            "Relative observer-frame BBH redshift distribution: "
            "psi(z) * dVc/(dz dOmega) / (1+z), normalized on 0 <= z <= 2."
        ),
    )

    fig, axis = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    width = 0.92 * dz
    bars = axis.bar(
        z_center, probability, width=width, color="tab:purple", alpha=0.82,
        label=r"Expected observer-frame probability per $\Delta z=0.02$ bin",
    )
    axis.set(
        xlim=(0.0, 2.0), xlabel="redshift z", ylabel="probability per bin",
        title=r"BBH redshift sampling, $0\leq z\leq2$",
    )
    axis.grid(axis="y", alpha=0.22)

    psi_axis = axis.twinx()
    psi_axis.plot(z_center, psi_value, color="black", lw=2.0, label=r"$\psi(z)$")
    psi_axis.set_ylabel(r"$\psi(z)$ [relative source-frame rate scale]")
    axis.legend([bars, psi_axis.lines[0]], [bars.get_label(), psi_axis.lines[0].get_label()],
                frameon=False, loc="upper left")
    fig.savefig(OUTPUT, dpi=220)
    print(OUTPUT)
    print(TABLE_OUTPUT)
    print(f"mean_z={np.sum(z_center * probability):.4f}")


if __name__ == "__main__":
    main()
