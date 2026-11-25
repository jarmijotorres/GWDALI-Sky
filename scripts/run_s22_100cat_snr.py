#!/usr/bin/env python3
"""Run the existing checkpointed GWDALI SNR stage over prepared S22 files."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import os
from pathlib import Path
import re
import sys


# Also covers spawned ProcessPool workers when this script is run by path
# without installing the project as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CATALOGUE_PATTERN = re.compile(
    r"skysim5000_BBH_gwdali_s22_c(?P<catalogue>\d{4})_"
    r"draw(?P<draw>\d{4})_z[0-9.]+_[0-9.]+\.npz"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Population root containing cXXXX/*.npz")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Root for matching checkpointed SNR outputs")
    parser.add_argument("--catalogue-start", type=int, default=0)
    parser.add_argument("--catalogue-stop", type=int, default=100,
                        help="Exclusive catalogue ID stop")
    parser.add_argument("--draw-start", type=int, default=0)
    parser.add_argument("--draw-stop", type=int, default=100,
                        help="Exclusive draw ID stop")
    parser.add_argument("--workers", type=int, default=1,
                        help="Catalogue files processed concurrently")
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true",
                        help="List input/output pairs without computing SNR")
    return parser.parse_args()


def run_one_snr(input_path: str, output_path: str) -> tuple[str, int, int, int]:
    """Run one SNR catalogue quietly and return a compact result summary."""
    from skysim_gw.stages.snr import main as snr_main
    import numpy as np

    # The stage reports source-level progress every few rows. Across thousands
    # of catalogues that would flood the terminal/log; the batch runner reports
    # compact file-level progress in the parent instead.
    with open(os.devnull, "w") as quiet:
        with redirect_stdout(quiet):
            snr_main([
                "--input-file", input_path,
                "--output-file", output_path,
                "--workers", "1",
                "--native-threads", "1",
            ])
    with np.load(output_path, allow_pickle=False) as data:
        results = data["results"]
        completed = data["completed"].astype(bool)
    return (
        input_path,
        int(completed.sum()),
        int((completed & results["success"]).sum()),
        int((completed & results["detected"]).sum()),
    )


def main():
    args = parse_args()
    if not (0 <= args.catalogue_start < args.catalogue_stop <= 100):
        raise ValueError("Require 0 <= catalogue-start < catalogue-stop <= 100")
    if not (0 <= args.draw_start < args.draw_stop <= 100):
        raise ValueError("Require 0 <= draw-start < draw-stop <= 100")
    if args.workers < 1 or args.native_threads < 1:
        raise ValueError("workers and native-threads must be positive")

    pairs = []
    for input_path in sorted(args.input_root.glob("c????/*.npz")):
        match = CATALOGUE_PATTERN.fullmatch(input_path.name)
        if match is None:
            continue
        catalogue_id = int(match.group("catalogue"))
        draw = int(match.group("draw"))
        if not (args.catalogue_start <= catalogue_id < args.catalogue_stop):
            continue
        if not (args.draw_start <= draw < args.draw_stop):
            continue
        output_path = args.output_root / f"c{catalogue_id:04d}" / (
            input_path.name.replace(
                "_gwdali_", "_lvk_snr_radec_deg_", 1
            )
        )
        pairs.append((input_path, output_path))

    expected = (
        (args.catalogue_stop - args.catalogue_start)
        * (args.draw_stop - args.draw_start)
    )
    if len(pairs) != expected:
        raise FileNotFoundError(
            f"Found {len(pairs)} prepared catalogues; expected {expected} "
            "for the selected catalogue/draw ranges"
        )

    print(f"Selected {len(pairs):,} prepared catalogues")
    if args.dry_run:
        for input_path, output_path in pairs:
            print(f"{input_path} -> {output_path}")
        return

    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = str(args.native_threads)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one_snr, str(input_path), str(output_path)): input_path
            for input_path, output_path in pairs
        }
        totals = [0, 0, 0]
        for index, future in enumerate(as_completed(futures), start=1):
            input_path, completed, successful, detected = future.result()
            totals[0] += completed
            totals[1] += successful
            totals[2] += detected
            if index % 100 == 0 or index == len(pairs):
                print(
                    f"files={index:,}/{len(pairs):,} "
                    f"source_rows_completed={totals[0]:,} "
                    f"successful={totals[1]:,} detected={totals[2]:,}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
