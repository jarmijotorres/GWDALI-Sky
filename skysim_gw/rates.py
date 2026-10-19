"""Merger-rate density models and expected event-count utilities.

The rate templates return source-frame merger-rate densities in
Gpc^-3 yr^-1 dex^-1. Expected event counts additionally depend on survey
solid angle, comoving volume, redshift-bin width, mass-bin width, observing
time, and the source-to-observer time-dilation factor 1/(1+z).

"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM


# Menote et al. (2025), Table 1 parameters supplied by the analysis.
P_BBH = {
    "N_R": 7.27,
    "A": 0.09,
    "B": 5.28,
    "C": 4.19,
    "K1": 4.73,
    "K2": 0.99,
    "M1": 1.46e7,
    "M2": 4.30e9,
    "C1": -3.4e-4,
    "C2": -9.53e-5,
    "G": 2.52,
    "Gc": 10.69,
    "Gsig": 0.41,
    "Gd": 0.10,
    "nu": 0.07,
}

P_BHNS = {
    "N_R": 2.78,
    "A": 0.82,
    "B": 4.55,
    "C": 4.55,
    "K1": 5.38,
    "K2": 1.03,
    "M1": 1.17e7,
    "M2": 4.95e9,
    "C1": 1.75e-2,
    "C2": -1.19e-2,
    "G": 1.67,
    "Gc": 10.68,
    "Gsig": 0.34,
    "Gd": 0.10,
    "nu": 0.04,
}

P_BNS = {
    "N_R": 541.7,
    "A": 1.17,
    "B": 4.95,
    "C": 2.73,
    "nu_m": 11.68,
    "K": 7.8e-4,
    "Lc": -0.029,
    "sigma_R": 0.22,
}

# Outer Rim / CosmoDC2 cosmology used by the supplied calculation.
OUTER_RIM_COSMOLOGY = FlatLambdaCDM(H0=71.0, Om0=0.265)


def _return_scalar_if_scalar(value, *inputs):
    """Return float for scalar inputs and ndarray otherwise."""
    array = np.asarray(value)
    if all(np.ndim(item) == 0 for item in inputs):
        return float(array)
    return array


def psi(z, parameters: Mapping[str, float]):
    """Universal source-frame redshift factor."""
    z_array = np.asarray(z, dtype=np.float64)
    value = (
        parameters["N_R"]
        * (1.0 + z_array) ** parameters["A"]
        / (
            1.0
            + ((1.0 + z_array) / parameters["C"]) ** parameters["B"]
        )
    )
    return _return_scalar_if_scalar(value, z)


def heaviside_term(z, log_mass, parameters: Mapping[str, float]):
    """Mass-dependent smooth Heaviside component for BBH/BHNS."""
    z_array = np.asarray(z, dtype=np.float64)
    log_mass_array = np.asarray(log_mass, dtype=np.float64)

    first = np.tanh(
        parameters["K1"]
        * (log_mass_array - np.log10(parameters["M1"]))
    )
    second_argument = parameters["K2"] * (
        log_mass_array
        - np.log10(parameters["M2"])
        - parameters["C1"] * z_array
        - parameters["C2"] * z_array**2
    )
    value = first - np.tanh(second_argument)
    return _return_scalar_if_scalar(value, z, log_mass)


def gaussian_term(z, log_mass, parameters: Mapping[str, float]):
    """Mass-dependent Gaussian component for BBH/BHNS."""
    z_array = np.asarray(z, dtype=np.float64)
    log_mass_array = np.asarray(log_mass, dtype=np.float64)
    center = parameters["Gc"] - parameters["nu"] * z_array**2
    exponent = (
        -0.5 * ((log_mass_array - center) / parameters["Gsig"]) ** 2
        - parameters["Gd"] * z_array**2
    )
    value = parameters["G"] * np.exp(exponent)
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bbh_bhns(z, log_mass, parameters: Mapping[str, float]):
    """BBH/BHNS source-frame rate density [Gpc^-3 yr^-1 dex^-1]."""
    value = psi(z, parameters) * (
        heaviside_term(z, log_mass, parameters)
        + gaussian_term(z, log_mass, parameters)
    )
    value = np.where(np.isfinite(value), np.maximum(value, 0.0), 0.0)
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bbh(z, log_mass):
    """BBH source-frame rate density [Gpc^-3 yr^-1 dex^-1]."""
    return rate_bbh_bhns(z, log_mass, P_BBH)


def rate_bbh_menote_heaviside_scaled(z, log_mass, heaviside_scale=1.0):
    """Menote BBH with a scaled Heaviside component.

    A scale of zero leaves the Menote Gaussian component alone; a scale of
    one is exactly :func:`rate_bbh`. Both terms retain Menote's ``psi(z)``
    redshift evolution.
    """
    if not np.isfinite(heaviside_scale) or heaviside_scale < 0.0:
        raise ValueError("heaviside_scale must be finite and non-negative")
    value = psi(z, P_BBH) * (
        heaviside_scale * heaviside_term(z, log_mass, P_BBH)
        + gaussian_term(z, log_mass, P_BBH)
    )
    value = np.where(np.isfinite(value), np.maximum(value, 0.0), 0.0)
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bhns(z, log_mass):
    """BHNS source-frame rate density [Gpc^-3 yr^-1 dex^-1]."""
    return rate_bbh_bhns(z, log_mass, P_BHNS)


def rate_bns(z, log_mass, parameters: Mapping[str, float] = P_BNS):
    """BNS source-frame rate density [Gpc^-3 yr^-1 dex^-1]."""
    z_array = np.asarray(z, dtype=np.float64)
    log_mass_array = np.asarray(log_mass, dtype=np.float64)

    nu_z = parameters["nu_m"] / (1.0 + parameters["K"] * z_array**2)
    ratio = nu_z - log_mass_array

    # Evaluate log10 only where its argument is positive. This vectorizes the
    # scalar guard used in the original notebook without generating warnings.
    log_ratio = np.full(np.broadcast(z_array, log_mass_array).shape, np.nan)
    ratio_broadcast = np.broadcast_to(ratio, log_ratio.shape)
    positive = ratio_broadcast > 0.0
    np.log10(ratio_broadcast, out=log_ratio, where=positive)

    argument = log_ratio - parameters["Lc"]
    exponent = -0.5 * (argument / parameters["sigma_R"]) ** 2
    value = psi(z_array, parameters) * np.exp(exponent)
    value = np.where(positive & np.isfinite(value), np.maximum(value, 0.0), 0.0)
    return _return_scalar_if_scalar(value, z, log_mass)


RATE_FUNCTIONS = {
    "BBH": rate_bbh,
    "BNS": rate_bns,
    "BHNS": rate_bhns,
}


# Simple BBH alternatives.  Each density model is normalized so its integral
# across 7 <= log10(Mstar/Msun) <= 12.5 at z=1 equals the Menote BBH integral.
# This keeps total BBH abundance fixed while comparing host-mass preferences.
#
# A20a is Artale et al. (2020a), Table 3, EAGLE100. Its fits predict the
# per-galaxy rate n_GW [Gyr^-1], not a volumetric density. We expose those
# fits for future host weighting and use their mass dependence below.
BBH_MODEL_LOG_MASS_MIN = 7.0
BBH_MODEL_LOG_MASS_MAX = 12.5
BBH_MODEL_Z_REFERENCE = 1.0
A20A_BBH_Z1_1D = {"alpha_1": 0.813, "alpha_2": -3.845}
A20A_BBH_Z1_3D = {
    "gamma_1": 1.134,
    "gamma_2": -0.172,
    "gamma_3": -0.839,
    "gamma_4": -8.338,
}
# These are for rendering a one-dimensional projection of the 3D fit only.
# Actual host weighting should pass each galaxy's SFR and absolute Z fraction.
A20A_COMPARISON_SSFR_YR = 1.0e-9
A20A_COMPARISON_METALLICITY = 0.02


def a20a_bbh_n_gw_1d(log_mass):
    """A20a BBH z=1 Fit 1D: n_GW in Gyr^-1 per galaxy."""
    log_mass_array = np.asarray(log_mass, dtype=np.float64)
    value = 10.0 ** (
        A20A_BBH_Z1_1D["alpha_1"] * log_mass_array
        + A20A_BBH_Z1_1D["alpha_2"]
    )
    return _return_scalar_if_scalar(value, log_mass)


def a20a_bbh_n_gw_3d(log_mass, sfr_msun_per_year, metallicity):
    """A20a BBH z=1 Fit 3D: n_GW in Gyr^-1 per galaxy.

    ``metallicity`` is the absolute mass fraction Z used by the published fit,
    not Z/Zsun. Invalid or non-positive SFR/Z inputs return zero.
    """
    log_mass_array = np.asarray(log_mass, dtype=np.float64)
    sfr_array = np.asarray(sfr_msun_per_year, dtype=np.float64)
    metallicity_array = np.asarray(metallicity, dtype=np.float64)
    shape = np.broadcast(log_mass_array, sfr_array, metallicity_array).shape
    log_sfr = np.full(shape, np.nan)
    log_metallicity = np.full(shape, np.nan)
    sfr_broadcast = np.broadcast_to(sfr_array, shape)
    metallicity_broadcast = np.broadcast_to(metallicity_array, shape)
    np.log10(sfr_broadcast, out=log_sfr, where=sfr_broadcast > 0.0)
    np.log10(metallicity_broadcast, out=log_metallicity,
             where=metallicity_broadcast > 0.0)
    parameters = A20A_BBH_Z1_3D
    exponent = (
        parameters["gamma_1"] * np.broadcast_to(log_mass_array, shape)
        + parameters["gamma_2"] * log_sfr
        + parameters["gamma_3"] * log_metallicity
        + parameters["gamma_4"]
    )
    value = np.where(np.isfinite(exponent), 10.0 ** exponent, 0.0)
    return _return_scalar_if_scalar(value, log_mass, sfr_msun_per_year, metallicity)


def madau_dickinson_sfrd(z):
    """Madau-Dickinson (2014) cosmic SFR density [Msun yr^-1 Mpc^-3]."""
    z_array = np.asarray(z, dtype=np.float64)
    value = 0.015 * (1.0 + z_array) ** 2.7 / (
        1.0 + ((1.0 + z_array) / 2.9) ** 5.6
    )
    value = np.where(np.isfinite(value) & (value >= 0.0), value, 0.0)
    return _return_scalar_if_scalar(value, z)


def _gaussian_shape(log_mass, mean, sigma):
    return np.exp(-0.5 * ((np.asarray(log_mass, dtype=np.float64) - mean) / sigma) ** 2)


def _flat_shape(log_mass):
    return np.ones_like(np.asarray(log_mass, dtype=np.float64))


def _a20a_1d_shape(log_mass):
    return np.asarray(a20a_bbh_n_gw_1d(log_mass), dtype=np.float64)


def _a20a_3d_comparison_shape(log_mass):
    log_mass_array = np.asarray(log_mass, dtype=np.float64)
    return np.asarray(
        a20a_bbh_n_gw_3d(
            log_mass_array,
            A20A_COMPARISON_SSFR_YR * 10.0 ** log_mass_array,
            A20A_COMPARISON_METALLICITY,
        ),
        dtype=np.float64,
    )


def _gaussian_mass_shape(log_mass):
    return _gaussian_shape(log_mass, mean=10.7, sigma=0.4)


def _bimodal_mass_shape(log_mass):
    return (
        0.65 * _gaussian_shape(log_mass, mean=8.7, sigma=0.6)
        + 0.35 * _gaussian_shape(log_mass, mean=10.7, sigma=0.4)
    )


def _reference_bbh_integral():
    mass_grid = np.linspace(BBH_MODEL_LOG_MASS_MIN, BBH_MODEL_LOG_MASS_MAX, 4097)
    return float(np.trapz(rate_bbh(BBH_MODEL_Z_REFERENCE, mass_grid), mass_grid))


BBH_REFERENCE_INTEGRAL_Z1 = _reference_bbh_integral()
_MENOTE_PEAK_MASS_GRID = np.linspace(
    BBH_MODEL_LOG_MASS_MIN, BBH_MODEL_LOG_MASS_MAX, 4097
)
MENOTE_BBH_PEAK_RATE_Z1 = float(np.max(
    rate_bbh(BBH_MODEL_Z_REFERENCE, _MENOTE_PEAK_MASS_GRID)
))
MENOTE_GAUSSIAN_MEAN_LOG_MASS = P_BBH["Gc"]
# A little narrower than the Menote Gaussian component (Gsig=0.41 dex), so
# its high-mass decline follows the full Menote profile more closely.
MENOTE_GAUSSIAN_SIGMA_DEX = 0.38
BIMODAL_LOW_MEAN_LOG_MASS = 8.7
BIMODAL_LOW_SIGMA_DEX = 0.6
BIMODAL_LOW_PEAK_FRACTION = 0.5
BIMODAL_LOW10_PEAK_FRACTION = 0.1
_BBH_SHAPES = {
    "flat_bbh_fixed_z": _flat_shape,
    "gaussian_bbh_fixed_z": _gaussian_mass_shape,
    "bimodal_bbh_fixed_z": _bimodal_mass_shape,
    "a20a_bbh_1d_fixed_z": _a20a_1d_shape,
    "a20a_bbh_3d_fixed_z": _a20a_3d_comparison_shape,
    "a20a_bbh_3d_madau_sfrd": _a20a_3d_comparison_shape,
}
_BBH_SHAPE_NORMALIZATIONS = {}
for _model_name, _shape_function in _BBH_SHAPES.items():
    _mass_grid = np.linspace(BBH_MODEL_LOG_MASS_MIN, BBH_MODEL_LOG_MASS_MAX, 4097)
    _integral = float(np.trapz(_shape_function(_mass_grid), _mass_grid))
    if not np.isfinite(_integral) or _integral <= 0.0:
        raise ValueError(f"Invalid BBH comparison shape: {_model_name}")
    _BBH_SHAPE_NORMALIZATIONS[_model_name] = BBH_REFERENCE_INTEGRAL_Z1 / _integral
del _integral, _mass_grid, _model_name, _shape_function


def _matched_bbh_density(z, log_mass, shape_name, redshift_factor=None):
    factor = 1.0 if redshift_factor is None else redshift_factor(z)
    value = _BBH_SHAPE_NORMALIZATIONS[shape_name] * _BBH_SHAPES[shape_name](log_mass) * factor
    value = np.where(np.isfinite(value), np.maximum(value, 0.0), 0.0)
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bbh_flat_fixed_z(z, log_mass):
    return _matched_bbh_density(z, log_mass, "flat_bbh_fixed_z")


def _rate_bbh_gaussian_toy_fixed_z(z, log_mass):
    return _matched_bbh_density(z, log_mass, "gaussian_bbh_fixed_z")


def _rate_bbh_bimodal_matched_total_legacy(z, log_mass):
    return _matched_bbh_density(z, log_mass, "bimodal_bbh_fixed_z")


def rate_bbh_a20a_1d_fixed_z(z, log_mass):
    return _matched_bbh_density(z, log_mass, "a20a_bbh_1d_fixed_z")


def rate_bbh_a20a_3d_fixed_z(z, log_mass):
    return _matched_bbh_density(z, log_mass, "a20a_bbh_3d_fixed_z")


def _madau_sfrd_relative_to_z1(z):
    return np.asarray(madau_dickinson_sfrd(z)) / madau_dickinson_sfrd(BBH_MODEL_Z_REFERENCE)


def rate_bbh_a20a_3d_madau_sfrd(z, log_mass):
    return _matched_bbh_density(
        z, log_mass, "a20a_bbh_3d_madau_sfrd", _madau_sfrd_relative_to_z1
    )


def _menote_a20a1d_capped_shape(log_mass):
    """Parameter-free low-mass cap: min(Menote, A20a-1D), both at z=1."""
    log_mass_array = np.asarray(log_mass, dtype=np.float64)
    return np.minimum(
        rate_bbh(BBH_MODEL_Z_REFERENCE, log_mass_array),
        rate_bbh_a20a_1d_fixed_z(BBH_MODEL_Z_REFERENCE, log_mass_array),
    )


_CAPPED_MASS_GRID = np.linspace(
    BBH_MODEL_LOG_MASS_MIN, BBH_MODEL_LOG_MASS_MAX, 4097
)
_CAPPED_RATE_GRID = _menote_a20a1d_capped_shape(_CAPPED_MASS_GRID)
BBH_A20A_CAPPED_INTEGRAL_Z1 = float(np.trapz(_CAPPED_RATE_GRID, _CAPPED_MASS_GRID))
BBH_A20A_CAPPED_MATCH_FACTOR = (
    BBH_REFERENCE_INTEGRAL_Z1 / BBH_A20A_CAPPED_INTEGRAL_Z1
)
BBH_A20A_CAPPED_MEAN_LOG_MASS = float(
    np.trapz(_CAPPED_MASS_GRID * _CAPPED_RATE_GRID, _CAPPED_MASS_GRID)
    / BBH_A20A_CAPPED_INTEGRAL_Z1
)
BBH_A20A_CAPPED_SIGMA_LOG_MASS = float(np.sqrt(
    np.trapz(
        (_CAPPED_MASS_GRID - BBH_A20A_CAPPED_MEAN_LOG_MASS) ** 2
        * _CAPPED_RATE_GRID,
        _CAPPED_MASS_GRID,
    ) / BBH_A20A_CAPPED_INTEGRAL_Z1
))
_CAPPED_GAUSSIAN_UNIT_GRID = _gaussian_shape(
    _CAPPED_MASS_GRID,
    BBH_A20A_CAPPED_MEAN_LOG_MASS,
    BBH_A20A_CAPPED_SIGMA_LOG_MASS,
)
_CAPPED_GAUSSIAN_AMPLITUDE = BBH_A20A_CAPPED_INTEGRAL_Z1 / float(
    np.trapz(_CAPPED_GAUSSIAN_UNIT_GRID, _CAPPED_MASS_GRID)
)
del _CAPPED_GAUSSIAN_UNIT_GRID


def rate_bbh_menote_a20a1d_capped(z, log_mass):
    """Menote BBH density capped by the matched A20a 1D density at z=1.

    This intentionally lowers the total rate: it never raises a Menote value.
    """
    value = _menote_a20a1d_capped_shape(log_mass)
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bbh_menote_a20a1d_capped_matched_total(z, log_mass):
    """Capped BBH shape rescaled to the Menote z=1 total (sensitivity test)."""
    value = BBH_A20A_CAPPED_MATCH_FACTOR * _menote_a20a1d_capped_shape(log_mass)
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bbh_gaussian_fixed_z(
    z,
    log_mass,
    peak_amplitude=MENOTE_BBH_PEAK_RATE_Z1,
    sigma_dex=MENOTE_GAUSSIAN_SIGMA_DEX,
):
    """Narrow Menote-centred Gaussian BBH comparison density.

    ``peak_amplitude`` is in Gpc^-3 yr^-1 dex^-1 and is deliberately exposed
    as a model parameter. Its default is the full Menote BBH peak at z=1.
    """
    if peak_amplitude < 0.0 or sigma_dex <= 0.0:
        raise ValueError("peak_amplitude must be non-negative and sigma_dex positive")
    value = peak_amplitude * _gaussian_shape(
        log_mass, MENOTE_GAUSSIAN_MEAN_LOG_MASS, sigma_dex
    )
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bbh_a20a_lowmass_gaussian_fixed_z(
    z,
    log_mass,
    peak_amplitude=MENOTE_BBH_PEAK_RATE_Z1,
    sigma_dex=MENOTE_GAUSSIAN_SIGMA_DEX,
):
    """A20a low-mass power law continuously joined to a narrow Gaussian.

    Below the Menote Gaussian centre, this is A20a's z=1 1D power law in
    stellar mass, normalized to equal ``peak_amplitude`` at the join. Above
    the join, it is the narrow Gaussian. Thus the low-mass branch is a line
    in log(R)-log(Mstar) space, rather than a Gaussian tail.
    """
    if peak_amplitude < 0.0 or sigma_dex <= 0.0:
        raise ValueError("peak_amplitude must be non-negative and sigma_dex positive")
    log_mass_array = np.asarray(log_mass, dtype=np.float64)
    a20a_at_join = a20a_bbh_n_gw_1d(MENOTE_GAUSSIAN_MEAN_LOG_MASS)
    low_mass_value = peak_amplitude * (
        np.asarray(a20a_bbh_n_gw_1d(log_mass_array)) / a20a_at_join
    )
    high_mass_value = peak_amplitude * _gaussian_shape(
        log_mass_array, MENOTE_GAUSSIAN_MEAN_LOG_MASS, sigma_dex
    )
    value = np.where(
        log_mass_array < MENOTE_GAUSSIAN_MEAN_LOG_MASS,
        low_mass_value,
        high_mass_value,
    )
    value = np.where(np.isfinite(value), np.maximum(value, 0.0), 0.0)
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bbh_bimodal_fixed_z(
    z,
    log_mass,
    low_peak_amplitude=None,
    low_sigma_dex=BIMODAL_LOW_SIGMA_DEX,
    high_peak_amplitude=MENOTE_BBH_PEAK_RATE_Z1,
    high_sigma_dex=MENOTE_GAUSSIAN_SIGMA_DEX,
):
    """Bimodal BBH comparison density with a configurable low-mass peak.

    The default low-mass peak is half the Menote BBH z=1 peak. All amplitudes
    use Gpc^-3 yr^-1 dex^-1 and are intentionally not total-rate matched.
    """
    if low_peak_amplitude is None:
        low_peak_amplitude = BIMODAL_LOW_PEAK_FRACTION * MENOTE_BBH_PEAK_RATE_Z1
    if (
        low_peak_amplitude < 0.0
        or high_peak_amplitude < 0.0
        or low_sigma_dex <= 0.0
        or high_sigma_dex <= 0.0
    ):
        raise ValueError("Bimodal amplitudes must be non-negative and widths positive")
    value = (
        low_peak_amplitude * _gaussian_shape(
            log_mass, BIMODAL_LOW_MEAN_LOG_MASS, low_sigma_dex
        )
        + high_peak_amplitude * _gaussian_shape(
            log_mass, MENOTE_GAUSSIAN_MEAN_LOG_MASS, high_sigma_dex
        )
    )
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bbh_bimodal_low10_fixed_z(
    z,
    log_mass,
    low_peak_amplitude=None,
    low_sigma_dex=BIMODAL_LOW_SIGMA_DEX,
    high_peak_amplitude=MENOTE_BBH_PEAK_RATE_Z1,
    high_sigma_dex=MENOTE_GAUSSIAN_SIGMA_DEX,
):
    """Bimodal model whose default low-mass peak is 10% of Menote's peak."""
    if low_peak_amplitude is None:
        low_peak_amplitude = BIMODAL_LOW10_PEAK_FRACTION * MENOTE_BBH_PEAK_RATE_Z1
    return rate_bbh_bimodal_fixed_z(
        z,
        log_mass,
        low_peak_amplitude=low_peak_amplitude,
        low_sigma_dex=low_sigma_dex,
        high_peak_amplitude=high_peak_amplitude,
        high_sigma_dex=high_sigma_dex,
    )


BBH_RATE_MODELS = {
    "menote_bbh": rate_bbh,
    "menote_heaviside_scaled_bbh": rate_bbh_menote_heaviside_scaled,
    "flat_bbh_fixed_z": rate_bbh_flat_fixed_z,
    "gaussian_bbh_fixed_z": rate_bbh_gaussian_fixed_z,
    "bimodal_bbh_fixed_z": rate_bbh_bimodal_fixed_z,
    "bimodal_bbh_low10_fixed_z": rate_bbh_bimodal_low10_fixed_z,
}


def rate_functions_for_model(model_name, model_parameters=None):
    """Select a BBH model while retaining Menote BNS/BHNS templates."""
    if model_name not in BBH_RATE_MODELS:
        choices = ", ".join(sorted(BBH_RATE_MODELS))
        raise ValueError(f"Unknown model_name {model_name!r}; choose from: {choices}")
    parameters = {} if model_parameters is None else dict(model_parameters)
    permitted = {
        "menote_heaviside_scaled_bbh": {"heaviside_scale"},
        "gaussian_bbh_fixed_z": {"peak_amplitude", "sigma_dex"},
        "bimodal_bbh_fixed_z": {
            "low_peak_amplitude",
            "low_sigma_dex",
            "high_peak_amplitude",
            "high_sigma_dex",
        },
        "bimodal_bbh_low10_fixed_z": {
            "low_peak_amplitude",
            "low_sigma_dex",
            "high_peak_amplitude",
            "high_sigma_dex",
        },
    }
    unknown = set(parameters) - permitted.get(model_name, set())
    if unknown:
        raise ValueError(
            f"Unsupported parameters for {model_name}: {sorted(unknown)}"
        )
    functions = RATE_FUNCTIONS.copy()
    functions["BBH"] = partial(BBH_RATE_MODELS[model_name], **parameters)
    return functions


def bbh_model_metadata(model_name, model_parameters=None):
    """Return serializable provenance for the chosen BBH model."""
    parameters = {} if model_parameters is None else dict(model_parameters)
    rate_functions_for_model(model_name, parameters)
    return {
        "model_name": model_name,
        "model_parameters": parameters,
        "normalization": (
            "menote_integral_matched" if model_name == "flat_bbh_fixed_z"
            else "direct_amplitudes"
        ),
        "reference_redshift": BBH_MODEL_Z_REFERENCE,
        "mass_range_log10_msun": [BBH_MODEL_LOG_MASS_MIN, BBH_MODEL_LOG_MASS_MAX],
        "reference_integral": BBH_REFERENCE_INTEGRAL_Z1,
        "a20a_bbh_z1_1d": A20A_BBH_Z1_1D,
        "a20a_bbh_z1_3d": A20A_BBH_Z1_3D,
        "a20a_3d_comparison_ssfr_yr": A20A_COMPARISON_SSFR_YR,
        "a20a_3d_comparison_metallicity": A20A_COMPARISON_METALLICITY,
        "gaussian_defaults": {
            "mean_log10_mstar": MENOTE_GAUSSIAN_MEAN_LOG_MASS,
            "sigma_dex": MENOTE_GAUSSIAN_SIGMA_DEX,
            "peak_amplitude": MENOTE_BBH_PEAK_RATE_Z1,
        },
        "menote_heaviside_scaled": {
            "heaviside_scale": "0 = Gaussian-only; 1 = original Menote BBH",
        },
        "bimodal_defaults": {
            "low_mean_log10_mstar": BIMODAL_LOW_MEAN_LOG_MASS,
            "low_sigma_dex": BIMODAL_LOW_SIGMA_DEX,
            "low_peak_fraction_of_menote_peak": BIMODAL_LOW_PEAK_FRACTION,
            "high_mean_log10_mstar": MENOTE_GAUSSIAN_MEAN_LOG_MASS,
            "high_sigma_dex": MENOTE_GAUSSIAN_SIGMA_DEX,
        },
        "bimodal_low10_default_low_peak_fraction_of_menote_peak": (
            BIMODAL_LOW10_PEAK_FRACTION
        ),
    }


def solid_angle_sr(area_deg2):
    """Convert square degrees to steradians."""
    area = np.asarray(area_deg2, dtype=np.float64)
    if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("area_deg2 must be positive and finite")
    value = area * (np.pi / 180.0) ** 2
    return _return_scalar_if_scalar(value, area_deg2)


def differential_comoving_volume(z, cosmology=OUTER_RIM_COSMOLOGY):
    """Return dV_c/(dz dOmega) in Gpc^3 sr^-1."""
    value_mpc3 = cosmology.differential_comoving_volume(z).to_value(
        u.Mpc**3 / u.sr
    )
    value = np.asarray(value_mpc3) * 1.0e-9
    return _return_scalar_if_scalar(value, z)


def expected_events_in_bin(
    rate_density,
    z_center,
    dz,
    dlog_mass,
    area_deg2,
    observing_years,
    cosmology=OUTER_RIM_COSMOLOGY,
):
    """Expected observer-frame events in one (z, logM) bin.

    ``rate_density`` must be in Gpc^-3 source-frame yr^-1 dex^-1.
    """
    dz_array = np.asarray(dz, dtype=np.float64)
    mass_width_array = np.asarray(dlog_mass, dtype=np.float64)
    time_array = np.asarray(observing_years, dtype=np.float64)
    if (
        np.any(dz_array <= 0.0)
        or np.any(mass_width_array <= 0.0)
        or np.any(time_array < 0.0)
    ):
        raise ValueError("dz and dlog_mass must be positive; time non-negative")

    volume_factor = (
        differential_comoving_volume(z_center, cosmology)
        * solid_angle_sr(area_deg2)
        * dz_array
    )
    expected = (
        np.asarray(rate_density, dtype=np.float64)
        / (1.0 + z_center)
        * volume_factor
        * mass_width_array
        * time_array
    )
    expected = np.where(
        np.isfinite(expected), np.maximum(expected, 0.0), 0.0
    )
    return _return_scalar_if_scalar(
        expected, rate_density, z_center, dz, dlog_mass, observing_years
    )


def expected_events_for_slice(
    rate_function: Callable,
    z_initial,
    z_final,
    log_mass_edges,
    area_deg2,
    observing_years,
    cosmology=OUTER_RIM_COSMOLOGY,
):
    """Midpoint-integrated expected events in one unequal-width z slice."""
    if z_final <= z_initial:
        raise ValueError("z_final must be greater than z_initial")

    mass_edges = np.asarray(log_mass_edges, dtype=np.float64)
    if mass_edges.ndim != 1 or mass_edges.size < 2:
        raise ValueError("log_mass_edges must be a 1D array with at least 2 edges")
    if np.any(np.diff(mass_edges) <= 0.0):
        raise ValueError("log_mass_edges must be strictly increasing")

    z_center = 0.5 * (z_initial + z_final)
    dz = z_final - z_initial
    mass_centers = 0.5 * (mass_edges[:-1] + mass_edges[1:])
    mass_widths = np.diff(mass_edges)
    rates = np.asarray(rate_function(z_center, mass_centers))

    volume_time_factor = (
        differential_comoving_volume(z_center, cosmology)
        * solid_angle_sr(area_deg2)
        * dz
        * observing_years
        / (1.0 + z_center)
    )
    expected = np.sum(rates * mass_widths) * volume_time_factor
    if not np.isfinite(expected) or expected < 0.0:
        return 0.0
    return float(expected)


def expected_events_all_slices(
    zi,
    zf,
    log_mass_edges,
    area_deg2,
    observing_years,
    rate_functions=RATE_FUNCTIONS,
    cosmology=OUTER_RIM_COSMOLOGY,
):
    """Expected BBH/BNS/BHNS event counts for every redshift slice."""
    zi_array = np.atleast_1d(np.asarray(zi, dtype=np.float64))
    zf_array = np.atleast_1d(np.asarray(zf, dtype=np.float64))
    if zi_array.shape != zf_array.shape:
        raise ValueError("zi and zf must have equal shape")

    result = {}
    for name, function in rate_functions.items():
        result[name] = np.array(
            [
                expected_events_for_slice(
                    function,
                    z_initial,
                    z_final,
                    log_mass_edges,
                    area_deg2,
                    observing_years,
                    cosmology,
                )
                for z_initial, z_final in zip(zi_array, zf_array)
            ],
            dtype=np.float64,
        )
    return result


def draw_poisson(expected, rng=None):
    """Draw reproducible Poisson counts from scalar or array expectations."""
    expected_array = np.asarray(expected, dtype=np.float64)
    if np.any(~np.isfinite(expected_array)) or np.any(expected_array < 0.0):
        raise ValueError("expected must contain only finite non-negative values")
    if rng is None:
        rng = np.random.default_rng()
    draw = rng.poisson(expected_array)
    return _return_scalar_if_scalar(draw, expected)


__all__ = [
    "A20A_BBH_Z1_1D",
    "A20A_BBH_Z1_3D",
    "BBH_RATE_MODELS",
    "BBH_A20A_CAPPED_INTEGRAL_Z1",
    "BBH_A20A_CAPPED_MATCH_FACTOR",
    "BBH_A20A_CAPPED_MEAN_LOG_MASS",
    "BBH_A20A_CAPPED_SIGMA_LOG_MASS",
    "BBH_REFERENCE_INTEGRAL_Z1",
    "OUTER_RIM_COSMOLOGY",
    "P_BBH",
    "P_BHNS",
    "P_BNS",
    "RATE_FUNCTIONS",
    "a20a_bbh_n_gw_1d",
    "a20a_bbh_n_gw_3d",
    "bbh_model_metadata",
    "differential_comoving_volume",
    "draw_poisson",
    "expected_events_all_slices",
    "expected_events_for_slice",
    "expected_events_in_bin",
    "gaussian_term",
    "heaviside_term",
    "madau_dickinson_sfrd",
    "psi",
    "rate_bbh",
    "rate_bbh_a20a_1d_fixed_z",
    "rate_bbh_a20a_3d_fixed_z",
    "rate_bbh_a20a_3d_madau_sfrd",
    "rate_bbh_bhns",
    "rate_bbh_bimodal_fixed_z",
    "rate_bbh_bimodal_low10_fixed_z",
    "rate_bbh_flat_fixed_z",
    "rate_bbh_gaussian_fixed_z",
    "rate_bbh_menote_heaviside_scaled",
    "rate_bbh_menote_a20a1d_capped",
    "rate_bbh_menote_a20a1d_capped_matched_total",
    "rate_bhns",
    "rate_bns",
    "rate_functions_for_model",
    "solid_angle_sr",
]
