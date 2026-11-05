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
SED_AB_LNU_UNIT_ERG_S_HZ = 4.4659e20
C_ANGSTROM_S = 2.99792458e18
CALZETTI_A1600_OVER_AV = 10.33 / 4.05
FMR_MU_MIN = 8.3
FMR_MU_MAX = 11.5
FMR_LOG_SFR_MIN = -4.0
FMR_LOG_SFR_MAX = 2.5
S22_SOLAR_METALLICITY = 0.0153
S22_SOLAR_OXYGEN_ABUNDANCE = 8.76


def derive_host_properties(
    data: Mapping[str, np.ndarray],
    *,
    sed_column: str = SED_UV_COLUMN,
    av_column: str = AV_COLUMN,
    stellar_mass_column: str = STELLAR_MASS_COLUMN,
    sed_start_angstrom: float = 1246.0,
    sed_bin_width_angstrom: float = 306.0,
    apply_host_extinction_correction: bool = False,
    sfr_calibration: str = "kennicutt2012",
    fmr_alpha: float = 0.32,
) -> dict[str, np.ndarray]:
    """Derive UV SFR and the Mannucci et al. (2011) equation-2 FMR.

    SkySim SED chunks are integrated rest-frame AB luminosities in units of
    ``4.4659e13 W Hz^-1``.  The default input is already corrected for host
    extinction, so no additional ``A_V`` correction is applied unless
    ``apply_host_extinction_correction=True``.  The metallicity output is
    gas-phase ``12 + log10(O/H)``; it is not a metal mass fraction or raw S22
    logZ. Invalid/non-positive SED values remain NaN and no SFR floor is
    imposed.
    """
    required = [sed_column, stellar_mass_column]
    if apply_host_extinction_correction:
        required.append(av_column)
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"Missing host-property inputs: {missing}")
    if not np.isfinite(sed_start_angstrom) or sed_start_angstrom <= 0:
        raise ValueError("sed_start_angstrom must be finite and positive")
    if not np.isfinite(sed_bin_width_angstrom) or sed_bin_width_angstrom <= 0:
        raise ValueError("sed_bin_width_angstrom must be finite and positive")
    if sfr_calibration != "kennicutt2012":
        raise ValueError("Only sfr_calibration='kennicutt2012' is supported")
    if not np.isfinite(fmr_alpha):
        raise ValueError("fmr_alpha must be finite")

    sed_flux = np.asarray(data[sed_column], dtype=np.float64)
    stellar_mass = np.asarray(data[stellar_mass_column], dtype=np.float64)
    if sed_flux.shape != stellar_mass.shape:
        raise ValueError("SED luminosity and stellar mass must have equal shapes")
    av = None
    if apply_host_extinction_correction:
        av = np.asarray(data[av_column], dtype=np.float64)
        if av.shape != sed_flux.shape:
            raise ValueError("SED luminosity and A_v must have equal shapes")

    central_wavelength = sed_start_angstrom + 0.5 * sed_bin_width_angstrom
    frequency_hz = C_ANGSTROM_S / central_wavelength
    uv_luminosity = sed_flux * SED_AB_LNU_UNIT_ERG_S_HZ * frequency_hz
    valid_sfr = np.isfinite(uv_luminosity) & (uv_luminosity > 0.0)
    log_sfr_observed = np.full(sed_flux.shape, np.nan)
    log_sfr_observed[valid_sfr] = (
        np.log10(uv_luminosity[valid_sfr]) - 43.35
    )
    sfr_observed = np.full(sed_flux.shape, np.nan)
    sfr_observed[valid_sfr] = 10.0 ** log_sfr_observed[valid_sfr]

    log_sfr_corrected = log_sfr_observed.copy()
    if apply_host_extinction_correction:
        valid_attenuation = valid_sfr & np.isfinite(av)
        log_sfr_corrected[valid_sfr & ~np.isfinite(av)] = np.nan
        attenuation_1600 = np.clip(av, 0.0, 10.0) * CALZETTI_A1600_OVER_AV
        log_sfr_corrected[valid_attenuation] += (
            0.4 * attenuation_1600[valid_attenuation]
        )
    sfr_corrected = np.full(sed_flux.shape, np.nan)
    sfr_corrected[valid_sfr] = 10.0 ** log_sfr_corrected[valid_sfr]

    log_mass = np.full(stellar_mass.shape, np.nan)
    valid_mass = np.isfinite(stellar_mass) & (stellar_mass > 0.0)
    log_mass[valid_mass] = np.log10(stellar_mass[valid_mass])
    fmr_valid = valid_mass & np.isfinite(log_sfr_corrected)
    log_sfr_fmr = np.full(stellar_mass.shape, np.nan)
    log_sfr_fmr[fmr_valid] = np.clip(
        log_sfr_corrected[fmr_valid], FMR_LOG_SFR_MIN, FMR_LOG_SFR_MAX
    )
    fmr_sfr_clipped = fmr_valid & (
        log_sfr_corrected != log_sfr_fmr
    )
    fmr_mu = np.full(stellar_mass.shape, np.nan)
    fmr_mu[fmr_valid] = (
        log_mass[fmr_valid] - fmr_alpha * log_sfr_fmr[fmr_valid]
    )
    mass_offset = log_mass - 10.0
    high_mu = fmr_valid & (fmr_mu >= 9.5)
    low_mu = fmr_valid & ~high_mu
    metallicity_12logoh = np.full(stellar_mass.shape, np.nan)
    metallicity_12logoh[high_mu] = (
        8.90
        + 0.37 * mass_offset[high_mu]
        - 0.14 * log_sfr_fmr[high_mu]
        - 0.19 * mass_offset[high_mu] ** 2
        + 0.12 * mass_offset[high_mu] * log_sfr_fmr[high_mu]
        - 0.054 * log_sfr_fmr[high_mu] ** 2
    )
    metallicity_12logoh[low_mu] = (
        8.93 + 0.51 * (fmr_mu[low_mu] - 10.0)
    )
    metallicity_log10_z = (
        np.log10(S22_SOLAR_METALLICITY)
        + metallicity_12logoh
        - S22_SOLAR_OXYGEN_ABUNDANCE
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
        "log_sfr_fmr": log_sfr_fmr,
        "fmr_sfr_clipped": fmr_sfr_clipped,
        "metallicity_fmr_12logoh": metallicity_12logoh,
        "metallicity_fmr_log10_z": metallicity_log10_z,
        "fmr_mu": fmr_mu,
        "fmr_valid": fmr_valid,
        "fmr_extrapolated": fmr_extrapolated,
    }


__all__ = ["derive_host_properties"]
