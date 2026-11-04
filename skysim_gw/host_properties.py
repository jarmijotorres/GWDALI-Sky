"""Derived SkySim5000 host properties used by GW seed assignment.

This module intentionally contains only the production SED-to-SFR and FMR
calculation needed by the NERSC assignment stage.  Keeping it in ``skysim_gw``
means MPI ranks do not depend on a separately installed analysis repository.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


SED_UV_COLUMN = "sed_1246_306_disk_no_host_extinction"
AV_COLUMN = "A_v_disk"
STELLAR_MASS_COLUMN = "stellar_mass"
L_SUN_ERG_S = 3.846e33
CALZETTI_A1600_OVER_AV = 10.33 / 4.05
FMR_MU_MIN = 8.7
FMR_MU_MAX = 11.2


def derive_host_properties(
    data: Mapping[str, np.ndarray],
    *,
    sed_column: str = SED_UV_COLUMN,
    av_column: str = AV_COLUMN,
    stellar_mass_column: str = STELLAR_MASS_COLUMN,
    sed_bin_width_angstrom: float = 306.0,
    sfr_calibration: str = "kennicutt2012",
    fmr_alpha: float = 0.32,
) -> dict[str, np.ndarray]:
    """Derive dust-corrected UV SFR and Mannucci et al. (2010) FMR.

    The SkySim SED chunk is treated as luminosity density in
    ``L_sun / Angstrom``.  The metallicity output is gas-phase
    ``12 + log10(O/H)``; it is not a metal mass fraction or raw S22 logZ.
    Invalid/non-positive SED values remain NaN and no SFR floor is imposed.
    """
    required = (sed_column, av_column, stellar_mass_column)
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"Missing host-property inputs: {missing}")
    if not np.isfinite(sed_bin_width_angstrom) or sed_bin_width_angstrom <= 0:
        raise ValueError("sed_bin_width_angstrom must be finite and positive")
    if sfr_calibration != "kennicutt2012":
        raise ValueError("Only sfr_calibration='kennicutt2012' is supported")
    if not np.isfinite(fmr_alpha):
        raise ValueError("fmr_alpha must be finite")

    sed_flux = np.asarray(data[sed_column], dtype=np.float64)
    av = np.asarray(data[av_column], dtype=np.float64)
    stellar_mass = np.asarray(data[stellar_mass_column], dtype=np.float64)
    if not (sed_flux.shape == av.shape == stellar_mass.shape):
        raise ValueError("SED flux, A_v, and stellar mass must have equal shapes")

    uv_luminosity = sed_flux * sed_bin_width_angstrom * L_SUN_ERG_S
    valid_sfr = (
        np.isfinite(uv_luminosity) & (uv_luminosity > 0.0)
        & np.isfinite(av)
    )
    log_sfr_observed = np.full(sed_flux.shape, np.nan)
    log_sfr_observed[valid_sfr] = (
        np.log10(uv_luminosity[valid_sfr]) - 43.35
    )
    sfr_observed = np.full(sed_flux.shape, np.nan)
    sfr_observed[valid_sfr] = 10.0 ** log_sfr_observed[valid_sfr]

    attenuation_1600 = np.clip(av, 0.0, 10.0) * CALZETTI_A1600_OVER_AV
    log_sfr_corrected = np.full(sed_flux.shape, np.nan)
    log_sfr_corrected[valid_sfr] = (
        log_sfr_observed[valid_sfr]
        + 0.4 * attenuation_1600[valid_sfr]
    )
    sfr_corrected = np.full(sed_flux.shape, np.nan)
    sfr_corrected[valid_sfr] = 10.0 ** log_sfr_corrected[valid_sfr]

    log_mass = np.full(stellar_mass.shape, np.nan)
    valid_mass = np.isfinite(stellar_mass) & (stellar_mass > 0.0)
    log_mass[valid_mass] = np.log10(stellar_mass[valid_mass])
    fmr_valid = valid_mass & np.isfinite(log_sfr_corrected)
    fmr_mu = np.full(stellar_mass.shape, np.nan)
    fmr_mu[fmr_valid] = (
        log_mass[fmr_valid] - fmr_alpha * log_sfr_corrected[fmr_valid]
    )
    offset = fmr_mu - 10.0
    metallicity = np.full(stellar_mass.shape, np.nan)
    metallicity[fmr_valid] = (
        8.90 + 0.37 * offset[fmr_valid]
        - 0.14 * offset[fmr_valid] ** 2
        - 0.19 * offset[fmr_valid] ** 3
    )
    fmr_extrapolated = fmr_valid & (
        (fmr_mu < FMR_MU_MIN) | (fmr_mu > FMR_MU_MAX)
    )

    return {
        "uv_luminosity_erg_s": uv_luminosity,
        "log_sfr_uv_observed": log_sfr_observed,
        "sfr_uv_observed": sfr_observed,
        "log_sfr_uv_corrected": log_sfr_corrected,
        "sfr_uv_corrected": sfr_corrected,
        "metallicity_fmr_12logoh": metallicity,
        "fmr_mu": fmr_mu,
        "fmr_valid": fmr_valid,
        "fmr_extrapolated": fmr_extrapolated,
    }


__all__ = ["derive_host_properties"]
