#!/usr/bin/env python3
"""Prepare independent BBH population draws for each catalogue in an S22 HDF5.

The default production plan is 100 BBH-parameter draws for each of the 100
stored host catalogues (10,000 output catalogues). This script only prepares
injection catalogues; detection and localisation are later pipeline stages.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skysim_gw.population import (  # noqa: E402
    CONFIG,
    OUTERRIM_COSMOLOGY,
    population_to_gwdali_table,
    prepare_population,
    read_sources_in_true_redshift_slice,
)
from skysim_gw.population import GWDALI_COLUMNS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zi", type=float, default=0.0)
    parser.add_argument("--zf", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--catalogues", type=int, default=100)
    parser.add_argument("--draws-per-catalogue", type=int, default=100)
    parser.add_argument(
        "--catalogue-start", type=int, default=0,
        help="First catalogue ID to process (inclusive; for HPC array chunks)",
    )
    parser.add_argument(
        "--catalogue-stop", type=int,
        help="Catalogue ID stop (exclusive; defaults to --catalogues)",
    )
    parser.add_argument(
        "--draw-start", type=int, default=0,
        help="First draw ID to process (inclusive; for HPC array chunks)",
    )
    parser.add_argument(
        "--draw-stop", type=int,
        help="Draw ID stop (exclusive; defaults to --draws-per-catalogue)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already completed output files (safe to rerun after interruption)",
    )
    return parser.parse_args()


def validate_input(path: Path, expected_catalogues: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as data:
        if "sources/BBH" not in data:
            raise KeyError(f"Missing /sources/BBH in {path}")
        group = data["sources/BBH"]
        required = {"ra", "dec", "redshift_true", "catalogue_id"}
        missing = sorted(required - set(group.keys()))
        if missing:
            raise ValueError(f"/sources/BBH is missing fields: {missing}")
        declared = int(group.attrs.get("n_catalogues", -1))
        if declared != expected_catalogues:
            raise ValueError(
                f"Expected {expected_catalogues} catalogues, file declares {declared}"
            )
        factor = int(group.attrs.get("event_oversample_factor", -1))
        if factor != 8:
            raise ValueError(
                f"Expected event_oversample_factor=8, found {factor}; check input provenance"
            )
        if int(data.attrs.get("n_catalogues", -1)) != expected_catalogues:
            raise ValueError("Root n_catalogues metadata does not match requested count")
        ids = np.unique(group["catalogue_id"][:])
        expected_ids = np.arange(expected_catalogues, dtype=ids.dtype)
        if not np.array_equal(ids, expected_ids):
            raise ValueError(
                "Catalogue IDs must be consecutive integers from 0 through "
                f"{expected_catalogues - 1}; found {ids[:10]}"
            )


def write_output(path, population, zi, zf, stored_seed, catalogue_id, draw):
    table = population_to_gwdali_table(population)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}_", suffix=".npz"
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary_path,
            catalog=table,
            gwdali_columns=np.asarray(GWDALI_COLUMNS),
            source_type=np.asarray("BBH"),
            zi=np.asarray(zi),
            zf=np.asarray(zf),
            random_seed=np.asarray(stored_seed, dtype=np.uint64),
            host_realization=np.asarray(-1),
            population_draw=np.asarray(draw),
            catalogue_id=np.asarray(catalogue_id),
            event_oversample_factor=np.asarray(8),
            event_weight=np.asarray(1.0 / 8.0),
            cosmology=np.asarray(OUTERRIM_COSMOLOGY.name),
            H0_km_s_Mpc=np.asarray(OUTERRIM_COSMOLOGY.H0.value),
            Omega_m=np.asarray(OUTERRIM_COSMOLOGY.Om0),
            Omega_b=np.asarray(OUTERRIM_COSMOLOGY.Ob0),
            distance_unit=np.asarray("Gpc"),
            mass_unit=np.asarray("Msun; m1 and m2 are detector-frame"),
            angle_unit=np.asarray("radian"),
            population_config=np.asarray(repr(CONFIG)),
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main():
    args = parse_args()
    stop_catalogue = args.catalogues if args.catalogue_stop is None else args.catalogue_stop
    stop_draw = args.draws_per_catalogue if args.draw_stop is None else args.draw_stop
    if args.catalogues <= 0 or args.draws_per_catalogue <= 0:
        raise ValueError("Catalogue and draw counts must be positive")
    if args.overwrite and args.resume:
        raise ValueError("Choose either --overwrite or --resume")
    if not (0 <= args.catalogue_start < stop_catalogue <= args.catalogues):
        raise ValueError("Require 0 <= catalogue-start < catalogue-stop <= catalogues")
    if not (0 <= args.draw_start < stop_draw <= args.draws_per_catalogue):
        raise ValueError("Require 0 <= draw-start < draw-stop <= draws-per-catalogue")
    if not np.isfinite(args.zi) or not np.isfinite(args.zf) or args.zi >= args.zf:
        raise ValueError("Require finite zi < zf")

    validate_input(args.input_file, args.catalogues)
    args.output_root.mkdir(parents=True, exist_ok=True)
    planned = (stop_catalogue - args.catalogue_start) * (stop_draw - args.draw_start)
    print(
        f"Preparing {planned:,} BBH catalogues from {args.input_file}; "
        f"catalogue IDs [{args.catalogue_start}, {stop_catalogue}), "
        f"draw IDs [{args.draw_start}, {stop_draw})"
    )

    for catalogue_id in range(args.catalogue_start, stop_catalogue):
        sources = read_sources_in_true_redshift_slice(
            args.input_file, args.zi, args.zf, ("BBH",), catalogue_id
        )["BBH"]
        n_sources = len(sources["redshift_true"])
        if n_sources == 0:
            raise ValueError(
                f"Catalogue {catalogue_id} has no BBHs in [{args.zi}, {args.zf})"
            )
        output_dir = args.output_root / f"c{catalogue_id:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for draw in range(args.draw_start, stop_draw):
            # Key streams on (master seed, catalogue ID, draw ID), so runs can
            # be split across jobs and resumed without changing any samples.
            sequence = np.random.SeedSequence([args.seed, catalogue_id, draw])
            rng = np.random.default_rng(sequence)
            stored_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
            output = output_dir / (
                f"skysim5000_BBH_gwdali_s22_c{catalogue_id:04d}"
                f"_draw{draw:04d}_z{args.zi:.4f}_{args.zf:.4f}.npz"
            )
            if output.exists() and not args.overwrite:
                if args.resume:
                    print(f"Skipping existing sample: {output}")
                    continue
                raise FileExistsError(f"Refusing to overwrite {output}")
            population = prepare_population("BBH", sources, rng, config=CONFIG)
            write_output(
                output, population, args.zi, args.zf, stored_seed,
                catalogue_id, draw,
            )
            print(
                f"c{catalogue_id:04d} draw={draw:04d}: "
                f"{n_sources:,} BBHs -> {output}"
            )


if __name__ == "__main__":
    main()
