#!/usr/bin/env python3
"""Save Fisher-Gaussian event parameters in one HDF5 per S22 seed catalogue."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np


# Ensure spawned Fisher workers can import the project package when this
# script is launched directly from the repository checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True,
                        help="Root containing cXXXX prepared population catalogues.")
    parser.add_argument("--snr-root", type=Path, required=True,
                        help="Matched SNR output root containing cXXXX directories.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Destination directory for cXXXX.h5 files.")
    parser.add_argument("--catalogue-start", type=int, default=0)
    parser.add_argument("--catalogue-stop", type=int, default=100,
                        help="Exclusive seed-catalogue ID stop.")
    parser.add_argument("--draw-start", type=int, default=0)
    parser.add_argument("--draw-stop", type=int, default=100,
                        help="Exclusive population-draw ID stop.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def population_file(root, catalogue_id, draw):
    folder = root / f"c{catalogue_id:04d}"
    matches = sorted(folder.glob(
        f"skysim5000_BBH_gwdali_s22_c{catalogue_id:04d}_"
        f"draw{draw:04d}_z*.npz"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one prepared catalogue for c{catalogue_id:04d} "
            f"draw{draw:04d}; found {len(matches)} in {folder}"
        )
    return matches[0]


def snr_file_for(population_path, snr_root, catalogue_id):
    name = population_path.name.replace(
        "_gwdali_", "_lvk_snr_radec_deg_", 1
    )
    path = snr_root / f"c{catalogue_id:04d}" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing matched SNR file: {path}")
    return path


def detected_indices(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"results", "completed"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing SNR arrays {sorted(missing)}")
        results = data["results"]
        completed = data["completed"].astype(bool)
    if not {"success", "detected"}.issubset(results.dtype.names or ()):
        raise ValueError(f"{path} results have no success/detected fields")
    return np.flatnonzero(
        completed & results["success"] & results["detected"]
    ).astype(np.int64)


def main():
    args = parse_args()
    if not (0 <= args.catalogue_start < args.catalogue_stop <= 100):
        raise ValueError("Require 0 <= catalogue-start < catalogue-stop <= 100")
    if not (0 <= args.draw_start < args.draw_stop <= 100):
        raise ValueError("Require 0 <= draw-start < draw-stop <= 100")
    if args.workers <= 0 or args.native_threads <= 0:
        raise ValueError("workers and native-threads must be positive")

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        from skysim_gw.stages.doublet_parallel import main as fisher_main

    for catalogue_id in range(args.catalogue_start, args.catalogue_stop):
        hdf5_path = args.output_root / f"c{catalogue_id:04d}.h5"
        for draw in range(args.draw_start, args.draw_stop):
            population = population_file(
                args.population_root, catalogue_id, draw
            )
            snr = snr_file_for(population, args.snr_root, catalogue_id)
            indices = detected_indices(snr)
            print(
                f"c{catalogue_id:04d} population_draw={draw:04d}: "
                f"{len(indices):,} detected events -> {hdf5_path}",
                flush=True,
            )
            if args.dry_run:
                continue

            with tempfile.TemporaryDirectory(prefix="fisher_indices_") as temp_dir:
                index_file = Path(temp_dir) / "indices.txt"
                np.savetxt(index_file, indices, fmt="%d")
                fisher_main([
                    "--pipeline-module", "skysim_gw.doublet_adapter",
                    "--output-hdf5", str(hdf5_path),
                    "--catalogue-id", str(catalogue_id),
                    "--population-draw", str(draw),
                    "--indices", str(index_file),
                    "--input-file", str(population),
                    "--snr-file", str(snr),
                    "--workers", str(args.workers),
                    "--native-threads", str(args.native_threads),
                ])


if __name__ == "__main__":
    main()
