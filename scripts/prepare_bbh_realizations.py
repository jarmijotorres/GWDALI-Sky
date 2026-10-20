#!/usr/bin/env python3
"""Prepare BBH injection catalogues for every fh0/fh1 GWseeds realization.

Each realization receives an independent, deterministic population RNG stream.
Outputs are grouped by model (``OUTPUT_ROOT/fh0`` and ``OUTPUT_ROOT/fh1``),
with the realization tag embedded in the NPZ filename to prevent overwrites.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skysim_gw.population import (
    CONFIG,
    GWDALI_COLUMNS,
    OUTERRIM_COSMOLOGY,
    population_to_gwdali_table,
    prepare_population,
    read_sources_in_true_redshift_slice,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zi", type=float, default=0.0)
    parser.add_argument("--zf", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def realization_info(path: Path):
    match = re.fullmatch(r"gw_hosts_(fh[01])_r(\d{4})\.hdf5", path.name)
    if match is None:
        raise ValueError(f"Unexpected realization filename: {path.name}")
    return match.group(1), int(match.group(2))


def write_catalogue(path: Path, population, zi: float, zf: float, seed: int):
    table = population_to_gwdali_table(population)
    np.savez_compressed(
        path,
        catalog=table,
        gwdali_columns=np.asarray(GWDALI_COLUMNS),
        source_type=np.asarray("BBH"),
        zi=np.asarray(zi),
        zf=np.asarray(zf),
        random_seed=np.asarray(seed),
        cosmology=np.asarray(OUTERRIM_COSMOLOGY.name),
        H0_km_s_Mpc=np.asarray(OUTERRIM_COSMOLOGY.H0.value),
        Omega_m=np.asarray(OUTERRIM_COSMOLOGY.Om0),
        Omega_b=np.asarray(OUTERRIM_COSMOLOGY.Ob0),
        distance_unit=np.asarray("Gpc"),
        mass_unit=np.asarray("Msun; m1 and m2 are detector-frame"),
        angle_unit=np.asarray("radian"),
        population_config=np.asarray(repr(CONFIG)),
    )


def main():
    args = parse_args()
    files = sorted(args.input_dir.glob("gw_hosts_fh[01]_r????.hdf5"))
    expected = {f"fh{f_h}_r{run:04d}" for f_h in (0, 1) for run in range(10)}
    found = {path.stem.removeprefix("gw_hosts_") for path in files}
    if found != expected:
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        raise ValueError(f"Expected exactly 20 realizations; missing={missing}, unexpected={unexpected}")

    for source_path in files:
        model_tag, realization = realization_info(source_path)
        # Separating model and realization identifiers gives every catalogue an
        # independent but reproducible binary-population draw.
        model_index = int(model_tag[-1])
        seed_sequence = np.random.SeedSequence([args.seed, model_index, realization])
        rng = np.random.default_rng(seed_sequence)
        stored_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
        output_dir = args.output_root / model_tag
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"skysim5000_BBH_gwdali_{model_tag}_r{realization:04d}_"
            f"z{args.zi:.4f}_{args.zf:.4f}.npz"
        )
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}")

        sources = read_sources_in_true_redshift_slice(
            source_path, args.zi, args.zf, ("BBH",)
        )["BBH"]
        population = prepare_population("BBH", sources, rng, config=CONFIG)
        write_catalogue(output_path, population, args.zi, args.zf, stored_seed)
        print(f"{model_tag}_r{realization:04d}: {len(sources['redshift_true']):,} BBH -> {output_path}")


if __name__ == "__main__":
    main()
