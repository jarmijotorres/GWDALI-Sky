"""Prepare SkySim5000 GW host catalogues for a later GWDALI run.

1. read sources in a true-redshift interval;
2. sample binary masses, spins, and extrinsic angles;
3. compute luminosity distance in the Outer Rim cosmology;
4. add detector-frame masses and derived mass combinations;
5. save one row-column NPZ catalogue per binary class.

Masses named ``*_source_msun`` are source-frame masses.  GWDALI/LAL
waveforms normally consume the corresponding ``*_detector_msun`` values.
Angles are saved in radians; the original RA/Dec values in degrees are also
retained.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

from skysim_gw.settings.population import (
    BASE_DIR,
    SOURCE_FILE,
    OUTPUT_DIR,
    SOURCE_TYPES,
    ZI,
    ZF,
    RANDOM_SEED,
    BBH_MASS_MODEL,
    BBH_MASS_PARAMETERS,
)


# Outer Rim: h=0.71, Omega_cdm=0.22, Omega_b=0.0448.
OUTERRIM_COSMOLOGY = FlatLambdaCDM(
    H0=71.0,
    Ob0=0.0448,
    Om0=0.22 + 0.0448,
    Tcmb0=2.7255,
    name="OuterRim",
)


@dataclass(frozen=True)
class PopulationConfig:
    """Hyperparameters used by the population samplers.

    Menote et al. specify the BBH functional family in their equation (17)
    but refer elsewhere for its numerical hyperparameters.  These defaults
    are therefore deliberately collected here so they can be replaced
    without changing any sampling code.
    """

    # BBH Power Law + Peak parameters (representative GWTC-3 medians).
    bbh_alpha: float = 3.40
    bbh_beta_q: float = 1.10
    bbh_mmin: float = 5.10
    bbh_mmax: float = 86.90
    bbh_delta_m: float = 4.80
    bbh_lambda_peak: float = 0.039
    bbh_mu_peak: float = 33.70
    bbh_sigma_peak: float = 3.60

    # Choice of the intrinsic BBH component-mass sampler.  ``bimodal_pairing``
    # is the production default. It is a deliberately phenomenological model
    # inspired by the bimodal-remnant scenario of Schneider, Podsiadlowski,
    # Laplace, and Willcox et al.: components are drawn from low/high mass
    # populations after first choosing an LL, LH, or HH binary pairing.
    # It is not a substitute for a population-synthesis calculation.
    bbh_mass_model: str = "bimodal_pairing"

    # Bimodal-pairing component distributions.  Each component is a
    # truncated normal in source-frame mass.  Keeping both means, widths, and
    # bounds explicit makes these parameters suitable for forecast scans.
    bbh_low_mean_msun: float = 10.0
    bbh_low_sigma_msun: float = 1.0
    bbh_low_min_msun: float = 5.0
    bbh_low_max_msun: float = 15.0
    bbh_high_mean_msun: float = 30.0
    bbh_high_sigma_msun: float = 10.0
    bbh_high_min_msun: float = 20.0
    bbh_high_max_msun: float = 80.0

    # Unnormalised pairing weights.  The default suppresses the mixed LH
    # channel relative to LL and HH, as suggested by isolated-binary models.
    # They are intentionally not presented as a fit to LVK data.
    bbh_pairing_weight_ll: float = 1.0
    bbh_pairing_weight_lh: float = 0.20
    bbh_pairing_weight_hh: float = 1.0

    # Menote equations (18)--(20).
    ns_mmin: float = 1.20
    ns_mmax: float = 2.00
    ns_alpha: float = -2.10
    ns_edge_b: float = 15.0
    bhns_bh_mmin: float = 3.0
    bhns_bh_mmax: float = 60.0

    # Menote equations (23)--(25).
    bh_spin_alpha: float = 1.60
    bh_spin_beta: float = 4.12
    ns_spin_sigma: float = 0.10
    aligned_spin_fraction: float = 0.66
    aligned_tilt_sigma: float = 1.50


CONFIG = PopulationConfig(
    bbh_mass_model=BBH_MASS_MODEL,
    **BBH_MASS_PARAMETERS,
)
VALID_SOURCE_TYPES = ("BBH", "BHNS", "BNS")


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------

def read_sources_in_true_redshift_slice(path, zi, zf, source_types):
    """Read the portable seed-host HDF5; no GCR or SkySim5000 access is needed.

    Expect /sources/{BBH,BHNS,BNS} groups with equally sized 1-D ra and dec
    datasets in degrees and dimensionless redshift_true. Read configured
    source types and retain finite coordinates with zi <= redshift_true < zf.
    Missing source-type groups are skipped; extra host properties are ignored.
    """
    if not np.isfinite(zi) or not np.isfinite(zf) or zi >= zf:
        raise ValueError(f"Require finite zi < zf; received zi={zi}, zf={zf}")

    selected_sources = {}
    with h5py.File(path, "r") as data:
        if "sources" not in data:
            raise KeyError(f"Missing /sources group in {path}")

        for source_type in source_types:
            source_type = source_type.upper()
            if source_type not in VALID_SOURCE_TYPES:
                raise ValueError(
                    f"Unknown source type {source_type!r}; choose from "
                    f"{VALID_SOURCE_TYPES}"
                )

            group_path = f"sources/{source_type}"
            if group_path not in data:
                print(f"Warning: missing {group_path}; skipping")
                continue

            group = data[group_path]
            for field in ("ra", "dec", "redshift_true"):
                if field not in group:
                    raise KeyError(f"Missing {group_path}/{field}")

            # Redshift is read first so the usually-small selection can be
            # applied immediately to the other one-dimensional datasets.
            redshift_true = np.asarray(group["redshift_true"][:], dtype=float)
            selected = (
                np.isfinite(redshift_true)
                & (redshift_true >= zi)
                & (redshift_true < zf)
            )
            ra = np.asarray(group["ra"][:], dtype=float)
            dec = np.asarray(group["dec"][:], dtype=float)
            selected &= np.isfinite(ra) & np.isfinite(dec)

            selected_sources[source_type] = {
                "ra_deg": ra[selected],
                "dec_deg": dec[selected],
                "redshift_true": redshift_true[selected],
            }

    return selected_sources


# -----------------------------------------------------------------------------
# Generic probability helpers
# -----------------------------------------------------------------------------

def _sample_tabulated_pdf(rng, size, xmin, xmax, pdf, grid_size=32768):
    """Sample a one-dimensional PDF by a tabulated inverse CDF."""
    if size == 0:
        return np.empty(0, dtype=float)
    x = np.linspace(xmin, xmax, grid_size)
    density = np.asarray(pdf(x), dtype=float)
    density[~np.isfinite(density)] = 0.0
    density = np.clip(density, 0.0, None)
    if not np.any(density > 0.0):
        raise ValueError("The requested PDF is zero over its entire interval")

    dx = np.diff(x)
    cdf = np.empty_like(x)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * dx)
    cdf /= cdf[-1]
    return np.interp(rng.random(size), cdf, x)


def _low_mass_smoothing(mass, mmin, delta_m):
    """LVK low-mass taper S(m | mmin, delta_m)."""
    mass = np.asarray(mass, dtype=float)
    if delta_m <= 0.0:
        return (mass >= mmin).astype(float)

    result = np.zeros_like(mass)
    result[mass >= mmin + delta_m] = 1.0
    transition = (mass > mmin) & (mass < mmin + delta_m)
    x = mass[transition] - mmin
    exponent = delta_m / x + delta_m / (x - delta_m)
    result[transition] = 1.0 / (1.0 + np.exp(np.clip(exponent, -700, 700)))
    return result


def _sample_truncated_normal(rng, size, mean, sigma, low, high):
    """Simple rejection sampler for a scalar truncated normal."""
    output = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        candidates = rng.normal(mean, sigma, max(32, 2 * (size - filled)))
        candidates = candidates[(candidates >= low) & (candidates <= high)]
        take = min(candidates.size, size - filled)
        output[filled:filled + take] = candidates[:take]
        filled += take
    return output


# -----------------------------------------------------------------------------
# Mass distributions
# -----------------------------------------------------------------------------

def sample_ns_masses(rng, size, config=CONFIG):
    """Sample Menote's smoothed NS power law, equations (18)--(19)."""
    def pdf(mass):
        lower = 1.0 / (1.0 + np.exp(-config.ns_edge_b * (mass - config.ns_mmin)))
        upper = 1.0 / (1.0 + np.exp(config.ns_edge_b * (mass - config.ns_mmax)))
        return mass**config.ns_alpha * lower * upper

    # The logistic edges have nonzero tails. Table 2 bounds NS masses to
    # [1, 2] Msun, while the text sets the lower smoothing location to 1.2.
    return _sample_tabulated_pdf(rng, size, 1.0, config.ns_mmax, pdf)


def _sample_bbh_masses_powerlaw_peak(rng, size, config):
    """Sample the existing source-frame Power Law + Peak BBH model."""
    normalization_grid = np.linspace(config.bbh_mmin, config.bbh_mmax, 32768)
    taper_grid = _low_mass_smoothing(
        normalization_grid, config.bbh_mmin, config.bbh_delta_m
    )
    power_grid = normalization_grid ** (-config.bbh_alpha) * taper_grid
    gaussian_grid = (
        np.exp(
            -0.5
            * ((normalization_grid - config.bbh_mu_peak) / config.bbh_sigma_peak) ** 2
        )
        / (np.sqrt(2.0 * np.pi) * config.bbh_sigma_peak)
        * taper_grid
    )
    # ``trapezoid`` was introduced after NumPy versions still used by some
    # NERSC notebooks; ``trapz`` is mathematically identical here.
    integrate = getattr(np, "trapezoid", np.trapz)
    power_norm = integrate(power_grid, normalization_grid)
    gaussian_norm = integrate(gaussian_grid, normalization_grid)

    def primary_pdf(mass):
        taper = _low_mass_smoothing(mass, config.bbh_mmin, config.bbh_delta_m)
        power_law = mass ** (-config.bbh_alpha) * taper / power_norm
        gaussian = np.exp(
            -0.5 * ((mass - config.bbh_mu_peak) / config.bbh_sigma_peak) ** 2
        ) * taper / (
            np.sqrt(2.0 * np.pi) * config.bbh_sigma_peak * gaussian_norm
        )
        return (
            (1.0 - config.bbh_lambda_peak) * power_law
            + config.bbh_lambda_peak * gaussian
        )

    m1 = _sample_tabulated_pdf(
        rng, size, config.bbh_mmin, config.bbh_mmax, primary_pdf
    )

    # q | m1 is proportional to q**beta_q times the same low-mass taper
    # evaluated at m2=q*m1. Rejection sampling preserves that correlation.
    q = np.empty(size, dtype=float)
    pending = np.arange(size)
    while pending.size:
        qmin = config.bbh_mmin / m1[pending]
        beta_plus_one = config.bbh_beta_q + 1.0
        u = rng.random(pending.size)
        if np.isclose(beta_plus_one, 0.0):
            proposal = qmin * (1.0 / qmin) ** u
        else:
            proposal = (
                u * (1.0 - qmin**beta_plus_one) + qmin**beta_plus_one
            ) ** (1.0 / beta_plus_one)
        accept_probability = _low_mass_smoothing(
            proposal * m1[pending], config.bbh_mmin, config.bbh_delta_m
        )
        accepted = rng.random(pending.size) < accept_probability
        q[pending[accepted]] = proposal[accepted]
        pending = pending[~accepted]

    return m1, q * m1


def _sample_bbh_masses_bimodal_pairing(rng, size, config):
    """Sample a phenomenological bimodal BH model with binary pairing.

    First draw the latent pairing class (LL, LH, or HH), then draw each
    component from its low- or high-mass truncated-normal population.  The
    resulting source-frame masses are ordered before being returned.  This
    construction can generate three chirp-mass features without incorrectly
    imposing a trimodal distribution on individual black-hole masses.
    """
    if size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    component_parameters = (
        (config.bbh_low_mean_msun, config.bbh_low_sigma_msun,
         config.bbh_low_min_msun, config.bbh_low_max_msun),
        (config.bbh_high_mean_msun, config.bbh_high_sigma_msun,
         config.bbh_high_min_msun, config.bbh_high_max_msun),
    )
    for label, (mean, sigma, lower, upper) in zip(
        ("low", "high"), component_parameters
    ):
        if not (sigma > 0.0 and lower < upper and lower <= mean <= upper):
            raise ValueError(
                f"Invalid {label}-mass component: require sigma > 0 and "
                "minimum <= mean <= maximum."
            )

    weights = np.asarray(
        (
            config.bbh_pairing_weight_ll,
            config.bbh_pairing_weight_lh,
            config.bbh_pairing_weight_hh,
        ),
        dtype=float,
    )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or not weights.sum() > 0.0:
        raise ValueError("BBH pairing weights must be finite, non-negative, and sum to > 0.")
    pair_class = rng.choice(3, size=size, p=weights / weights.sum())

    low = lambda count: _sample_truncated_normal(
        rng, count, config.bbh_low_mean_msun, config.bbh_low_sigma_msun,
        config.bbh_low_min_msun, config.bbh_low_max_msun,
    )
    high = lambda count: _sample_truncated_normal(
        rng, count, config.bbh_high_mean_msun, config.bbh_high_sigma_msun,
        config.bbh_high_min_msun, config.bbh_high_max_msun,
    )

    first = np.empty(size, dtype=float)
    second = np.empty(size, dtype=float)
    ll = pair_class == 0
    lh = pair_class == 1
    hh = pair_class == 2
    first[ll], second[ll] = low(ll.sum()), low(ll.sum())
    first[lh], second[lh] = low(lh.sum()), high(lh.sum())
    first[hh], second[hh] = high(hh.sum()), high(hh.sum())
    return np.maximum(first, second), np.minimum(first, second)


def sample_bbh_masses(rng, size, config=CONFIG):
    """Sample source-frame BBH component masses with ``m1 >= m2``.

    ``bimodal_pairing`` is the default flexible phenomenological model with
    explicit LL/LH/HH pairing weights. ``powerlaw_peak`` remains available
    for comparison with the historical prescription.
    """
    if config.bbh_mass_model == "powerlaw_peak":
        return _sample_bbh_masses_powerlaw_peak(rng, size, config)
    if config.bbh_mass_model == "bimodal_pairing":
        return _sample_bbh_masses_bimodal_pairing(rng, size, config)
    raise ValueError(
        f"Unknown BBH mass model {config.bbh_mass_model!r}; choose "
        "'powerlaw_peak' or 'bimodal_pairing'."
    )


def sample_component_masses(source_type, rng, size, config=CONFIG):
    """Return ordered source-frame component masses for one binary class."""
    source_type = source_type.upper()
    if source_type == "BBH":
        return sample_bbh_masses(rng, size, config)
    if source_type == "BHNS":
        m1 = rng.uniform(config.bhns_bh_mmin, config.bhns_bh_mmax, size)
        m2 = sample_ns_masses(rng, size, config)
        return m1, m2
    if source_type == "BNS":
        first = sample_ns_masses(rng, size, config)
        second = sample_ns_masses(rng, size, config)
        return np.maximum(first, second), np.minimum(first, second)
    raise ValueError(f"Unknown source type {source_type!r}")


# -----------------------------------------------------------------------------
# Spin and extrinsic distributions
# -----------------------------------------------------------------------------

def sample_spin_tilts(rng, size, config=CONFIG):
    """Sample the joint isotropic/aligned tilt mixture in equation (25)."""
    aligned = rng.random(size) < config.aligned_spin_fraction
    cos_theta_1 = rng.uniform(-1.0, 1.0, size)
    cos_theta_2 = rng.uniform(-1.0, 1.0, size)
    n_aligned = int(aligned.sum())
    cos_theta_1[aligned] = _sample_truncated_normal(
        rng, n_aligned, 1.0, config.aligned_tilt_sigma, -1.0, 1.0
    )
    cos_theta_2[aligned] = _sample_truncated_normal(
        rng, n_aligned, 1.0, config.aligned_tilt_sigma, -1.0, 1.0
    )
    return np.arccos(cos_theta_1), np.arccos(cos_theta_2)


def sample_spins(source_type, rng, size, config=CONFIG):
    """Sample dimensionless spin magnitudes and directions."""
    source_type = source_type.upper()
    if source_type == "BBH":
        chi_1 = rng.beta(config.bh_spin_alpha, config.bh_spin_beta, size)
        chi_2 = rng.beta(config.bh_spin_alpha, config.bh_spin_beta, size)
    elif source_type == "BHNS":
        chi_1 = rng.beta(config.bh_spin_alpha, config.bh_spin_beta, size)
        chi_2 = _sample_truncated_normal(
            rng, size, 0.0, config.ns_spin_sigma, 0.0, 1.0
        )
    elif source_type == "BNS":
        chi_1 = _sample_truncated_normal(
            rng, size, 0.0, config.ns_spin_sigma, 0.0, 1.0
        )
        chi_2 = _sample_truncated_normal(
            rng, size, 0.0, config.ns_spin_sigma, 0.0, 1.0
        )
    else:
        raise ValueError(f"Unknown source type {source_type!r}")

    theta_1, theta_2 = sample_spin_tilts(rng, size, config)
    return {
        "chi_1": chi_1,
        "chi_2": chi_2,
        "theta_1_rad": theta_1,
        "theta_2_rad": theta_2,
        "phi_1_rad": rng.uniform(0.0, 2.0 * np.pi, size),
        "phi_2_rad": rng.uniform(0.0, 2.0 * np.pi, size),
    }


def prepare_population(source_type, sources, rng, cosmology=OUTERRIM_COSMOLOGY,
                       config=CONFIG):
    """Attach all sampled and derived GW parameters to selected hosts."""
    redshift = np.asarray(sources["redshift_true"], dtype=float)
    size = redshift.size
    m1_source, m2_source = sample_component_masses(
        source_type, rng, size, config
    )
    one_plus_z = 1.0 + redshift
    m1_detector = one_plus_z * m1_source
    m2_detector = one_plus_z * m2_source
    chirp_source = (
        (m1_source * m2_source) ** (3.0 / 5.0)
        / (m1_source + m2_source) ** (1.0 / 5.0)
    )
    eta = m1_source * m2_source / (m1_source + m2_source) ** 2
    cos_iota = rng.uniform(-1.0, 1.0, size)

    result = {
        "ra_deg": np.asarray(sources["ra_deg"], dtype=float),
        "dec_deg": np.asarray(sources["dec_deg"], dtype=float),
        "ra_rad": np.deg2rad(sources["ra_deg"]),
        "dec_rad": np.deg2rad(sources["dec_deg"]),
        "redshift_true": redshift,
        "luminosity_distance_mpc": cosmology.luminosity_distance(redshift).value,
        "m1_source_msun": m1_source,
        "m2_source_msun": m2_source,
        "m1_detector_msun": m1_detector,
        "m2_detector_msun": m2_detector,
        "chirp_mass_source_msun": chirp_source,
        "chirp_mass_detector_msun": one_plus_z * chirp_source,
        "symmetric_mass_ratio": eta,
        "mass_ratio": m2_source / m1_source,
        "cos_iota": cos_iota,
        "iota_rad": np.arccos(cos_iota),
        "psi_rad": rng.uniform(0.0, np.pi, size),
        "phi_coal_rad": rng.uniform(0.0, 2.0 * np.pi, size),
        "t_coal_s": np.zeros(size, dtype=float),
    }
    result.update(sample_spins(source_type, rng, size, config))
    # Cartesian dimensionless spins required by GWDALI/LAL.  The z axis is
    # aligned with the binary orbital angular momentum.
    for component in (1, 2):
        chi = result[f"chi_{component}"]
        theta = result[f"theta_{component}_rad"]
        phi = result[f"phi_{component}_rad"]
        result[f"sx{component}"] = chi * np.sin(theta) * np.cos(phi)
        result[f"sy{component}"] = chi * np.sin(theta) * np.sin(phi)
        result[f"sz{component}"] = chi * np.cos(theta)
    return result


# -----------------------------------------------------------------------------
# Output and runnable example
# -----------------------------------------------------------------------------

GWDALI_COLUMNS = (
    "m1", "m2",                 # detector-frame component masses [Msun]
    "RA", "Dec",               # sky position [rad]
    "psi", "t_coal", "phi_coal",
    "dL", "iota",              # luminosity distance [Gpc], angle [rad]
    "sx1", "sy1", "sz1",
    "sx2", "sy2", "sz2",
)


def population_to_gwdali_table(population):
    """Return a named, row-column array ready to turn into GWDALI dicts.

    The two masses are detector-frame solar masses, dL is in Gpc, and every
    angle is in radians. Extra host/source-frame fields are retained for
    validation and future catalogue matching.
    """
    values = {
        "m1": population["m1_detector_msun"],
        "m2": population["m2_detector_msun"],
        "RA": population["ra_rad"],
        "Dec": population["dec_rad"],
        "psi": population["psi_rad"],
        "t_coal": population["t_coal_s"],
        "phi_coal": population["phi_coal_rad"],
        "dL": population["luminosity_distance_mpc"] / 1000.0,
        "iota": population["iota_rad"],
        "sx1": population["sx1"],
        "sy1": population["sy1"],
        "sz1": population["sz1"],
        "sx2": population["sx2"],
        "sy2": population["sy2"],
        "sz2": population["sz2"],
        "redshift_true": population["redshift_true"],
        "ra_deg": population["ra_deg"],
        "dec_deg": population["dec_deg"],
        "m1_source_msun": population["m1_source_msun"],
        "m2_source_msun": population["m2_source_msun"],
    }
    size = len(population["redshift_true"])
    table = np.empty(size, dtype=[(name, "f8") for name in values])
    for name, column in values.items():
        table[name] = column
    return table


def gwdali_dict_from_row(row):
    """Convert one structured-array row into a GWDALI source dictionary."""
    return {name: float(row[name]) for name in GWDALI_COLUMNS}


def write_npz_catalogs(output_dir, populations, zi, zf, seed,
                       cosmology=OUTERRIM_COSMOLOGY, config=CONFIG):
    """Write one compressed NPZ row-column catalogue per source type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for source_type, population in populations.items():
        table = population_to_gwdali_table(population)
        path = output_dir / (
            f"skysim5000_{source_type}_gwdali_z{zi:.4f}_{zf:.4f}.npz"
        )
        np.savez_compressed(
            path,
            catalog=table,
            gwdali_columns=np.asarray(GWDALI_COLUMNS),
            source_type=np.asarray(source_type),
            zi=np.asarray(zi),
            zf=np.asarray(zf),
            random_seed=np.asarray(seed),
            cosmology=np.asarray(cosmology.name),
            H0_km_s_Mpc=np.asarray(cosmology.H0.value),
            Omega_m=np.asarray(cosmology.Om0),
            Omega_b=np.asarray(cosmology.Ob0),
            distance_unit=np.asarray("Gpc"),
            mass_unit=np.asarray("Msun; m1 and m2 are detector-frame"),
            angle_unit=np.asarray("radian"),
            population_config=np.asarray(repr(config)),
        )
        paths[source_type] = path
    return paths


def main():
    selected = read_sources_in_true_redshift_slice(
        SOURCE_FILE, ZI, ZF, SOURCE_TYPES
    )
    seed_sequence = np.random.SeedSequence(RANDOM_SEED)
    child_seeds = seed_sequence.spawn(len(selected))
    prepared = {}
    for (source_type, sources), child_seed in zip(selected.items(), child_seeds):
        prepared[source_type] = prepare_population(
            source_type, sources, np.random.default_rng(child_seed)
        )
        print(
            f"{source_type}: prepared {len(sources['redshift_true']):,} "
            f"sources in {ZI:.4f} <= z < {ZF:.4f}"
        )

    output_paths = write_npz_catalogs(
        OUTPUT_DIR, prepared, ZI, ZF, RANDOM_SEED
    )
    for source_type, path in output_paths.items():
        print(f"Saved {source_type}: {path}")


if __name__ == "__main__":
    main()
