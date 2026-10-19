#!/usr/bin/env python3
"""Step zero: calculate expected and drawn GW source counts before host IO."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from skysim_gw.rates import (
    OUTER_RIM_COSMOLOGY,
    P_BBH,
    P_BHNS,
    P_BNS,
    bbh_model_metadata,
    draw_poisson,
    expected_events_in_bin,
    rate_functions_for_model,
)


DEFAULT_CONFIG = Path("gw_rate_config.json")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate GW event targets on a redshift/stellar-mass grid."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"JSON configuration file (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    required = {
        "model_name",
        "redshift_slices_file",
        "output_file",
        "area_deg2",
        "observing_years",
        "log_stellar_mass_min",
        "log_stellar_mass_max",
        "dlog_stellar_mass",
        "poisson_seed",
        "draw_mode",
        "overwrite",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise KeyError(f"Missing configuration entries: {missing}")
    if config["draw_mode"] != "independent_by_type":
        raise ValueError("Only draw_mode='independent_by_type' is implemented")
    parameters = config.get("bbh_model_parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("bbh_model_parameters must be a JSON object when supplied")
    rate_functions_for_model(config["model_name"], parameters)
    if config["area_deg2"] <= 0.0:
        raise ValueError("area_deg2 must be positive")
    if config["observing_years"] < 0.0:
        raise ValueError("observing_years must be non-negative")
    return config


def load_redshift_bins(path):
    zi, zf = np.loadtxt(
        path, usecols=(0, 1), unpack=True, dtype=np.float64
    )
    zi = np.atleast_1d(zi)
    zf = np.atleast_1d(zf)
    if zi.shape != zf.shape or zi.size == 0:
        raise ValueError("The zi and zf columns must be non-empty and equal length")
    if np.any(~np.isfinite(zi)) or np.any(~np.isfinite(zf)):
        raise ValueError("Redshift-bin edges must be finite")
    if np.any(zf <= zi):
        raise ValueError("Every redshift bin must satisfy zf > zi")
    if np.any(np.diff(zi) < 0.0):
        raise ValueError("Redshift bins must be ordered by increasing zi")
    if zi.size > 1 and np.any(zi[1:] < zf[:-1]):
        raise ValueError("Redshift bins must not overlap")
    return zi, zf


def make_mass_edges(config):
    lower = float(config["log_stellar_mass_min"])
    upper = float(config["log_stellar_mass_max"])
    width = float(config["dlog_stellar_mass"])
    if upper <= lower or width <= 0.0:
        raise ValueError("Invalid stellar-mass limits or bin width")

    n_bins_float = (upper - lower) / width
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins_float, n_bins):
        raise ValueError(
            "The stellar-mass range must be an integer multiple of the bin width"
        )
    return np.linspace(lower, upper, n_bins + 1, dtype=np.float64)


def calculate_grids(zi, zf, mass_edges, config):
    z_center = 0.5 * (zi + zf)
    dz = zf - zi
    mass_center = 0.5 * (mass_edges[:-1] + mass_edges[1:])
    dlog_mass = np.diff(mass_edges)

    # Shapes broadcast to (n_redshift_bins, n_mass_bins).
    z_grid = z_center[:, None]
    mass_grid = mass_center[None, :]
    dz_grid = dz[:, None]
    mass_width_grid = dlog_mass[None, :]

    expected = {}
    for source_type, rate_function in rate_functions_for_model(
        config["model_name"], config.get("bbh_model_parameters", {})
    ).items():
        rate_density = rate_function(z_grid, mass_grid)
        expected[source_type] = np.asarray(
            expected_events_in_bin(
                rate_density=rate_density,
                z_center=z_grid,
                dz=dz_grid,
                dlog_mass=mass_width_grid,
                area_deg2=float(config["area_deg2"]),
                observing_years=float(config["observing_years"]),
            ),
            dtype=np.float64,
        )
    return z_center, mass_center, expected


def write_output(path, config, config_path, zi, zf, mass_edges,
                 z_center, mass_center, expected, drawn):
    output_path = Path(path)
    if output_path.exists() and not config["overwrite"]:
        raise FileExistsError(
            f"Output already exists: {output_path}; set overwrite=true or change it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    parameter_snapshot = {
        "BBH": bbh_model_metadata(
            config["model_name"], config.get("bbh_model_parameters", {})
        ),
        "BNS": P_BNS,
        "BHNS": P_BHNS,
    }

    try:
        with h5py.File(temporary_path, "w") as output:
            output.attrs["description"] = (
                "Expected and independently Poisson-drawn GW source counts "
                "before SkySim5000 host assignment"
            )
            output.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            output.attrs["config_path"] = str(config_path.resolve())
            output.attrs["config_json"] = json.dumps(config, sort_keys=True)
            output.attrs["rate_parameters_json"] = json.dumps(
                parameter_snapshot, sort_keys=True
            )
            output.attrs["cosmology"] = (
                f"FlatLambdaCDM(H0={OUTER_RIM_COSMOLOGY.H0.value}, "
                f"Om0={OUTER_RIM_COSMOLOGY.Om0})"
            )
            output.attrs["rate_units"] = "Gpc^-3 source-frame yr^-1 dex^-1"
            output.attrs["expected_count_axes"] = "redshift_bin,mass_bin"

            bins = output.create_group("bins")
            bins.create_dataset("zi", data=zi)
            bins.create_dataset("zf", data=zf)
            bins.create_dataset("z_center", data=z_center)
            bins.create_dataset("log_stellar_mass_edges", data=mass_edges)
            bins.create_dataset("log_stellar_mass_center", data=mass_center)
            bins.attrs["redshift_interval"] = "zi <= redshift_true < zf"
            bins.attrs["mass_variable"] = "log10(stellar_mass / Msun)"

            expected_group = output.create_group("expected")
            drawn_group = output.create_group("drawn")
            summary = output.create_group("summary")

            for source_type in ("BBH", "BNS", "BHNS"):
                expected_dataset = expected_group.create_dataset(
                    source_type, data=expected[source_type]
                )
                expected_dataset.attrs["units"] = "expected events per bin"

                drawn_dataset = drawn_group.create_dataset(
                    source_type, data=drawn[source_type], dtype=np.int64
                )
                drawn_dataset.attrs["units"] = "Poisson-drawn events per bin"

                summary.create_dataset(
                    f"expected_by_redshift_{source_type}",
                    data=expected[source_type].sum(axis=1),
                )
                summary.create_dataset(
                    f"drawn_by_redshift_{source_type}",
                    data=drawn[source_type].sum(axis=1),
                )
                summary.attrs[f"expected_total_{source_type}"] = float(
                    expected[source_type].sum()
                )
                summary.attrs[f"drawn_total_{source_type}"] = int(
                    drawn[source_type].sum(dtype=np.int64)
                )

        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def main():
    arguments = parse_arguments()
    config = load_config(arguments.config)
    zi, zf = load_redshift_bins(config["redshift_slices_file"])
    mass_edges = make_mass_edges(config)
    z_center, mass_center, expected = calculate_grids(
        zi, zf, mass_edges, config
    )

    rng = np.random.default_rng(int(config["poisson_seed"]))
    drawn = {
        source_type: np.asarray(draw_poisson(values, rng), dtype=np.int64)
        for source_type, values in expected.items()
    }

    print(f"Redshift bins: {len(zi)}")
    print(f"Stellar-mass bins: {len(mass_center)}")
    print(f"Area: {config['area_deg2']:.6f} deg^2")
    print(f"Observing time: {config['observing_years']} yr")
    print(f"Poisson seed: {config['poisson_seed']}")
    for source_type in ("BBH", "BNS", "BHNS"):
        print(
            f"{source_type}: expected={expected[source_type].sum():,.3f}, "
            f"drawn={drawn[source_type].sum(dtype=np.int64):,}"
        )

    write_output(
        config["output_file"],
        config,
        arguments.config,
        zi,
        zf,
        mass_edges,
        z_center,
        mass_center,
        expected,
        drawn,
    )
    print(f"Saved: {config['output_file']}")


if __name__ == "__main__":
    main()
