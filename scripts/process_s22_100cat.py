#!/usr/bin/env python3
"""Prepare S22 BBH populations and run their GWdali SNR calculations."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True,
                        help="Multi-catalogue S22 host HDF5")
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--snr-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "prepare", "snr"), default="all")
    parser.add_argument("--zi", type=float, default=0.0)
    parser.add_argument("--zf", type=float, default=0.5,
                        help="Exclusive upper redshift cut; default z < 0.5")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--catalogues", type=int, default=100)
    parser.add_argument("--draws-per-catalogue", type=int, default=100)
    parser.add_argument("--catalogue-start", type=int, default=0)
    parser.add_argument("--catalogue-stop", type=int)
    parser.add_argument("--draw-start", type=int, default=0)
    parser.add_argument("--draw-stop", type=int)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of catalogue SNR runs to execute concurrently")
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--overwrite-samples", action="store_true",
                        help="Regenerate sample files instead of resuming/skipping them")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the stage commands without running them")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    command_base = [sys.executable]
    cat_stop = args.catalogues if args.catalogue_stop is None else args.catalogue_stop
    draw_stop = (
        args.draws_per_catalogue if args.draw_stop is None else args.draw_stop
    )

    prepare = [
        *command_base, str(script_dir / "prepare_s22_100cat.py"),
        "--input-file", str(args.input_file),
        "--output-root", str(args.population_root),
        "--zi", str(args.zi), "--zf", str(args.zf), "--seed", str(args.seed),
        "--catalogues", str(args.catalogues),
        "--draws-per-catalogue", str(args.draws_per_catalogue),
        "--catalogue-start", str(args.catalogue_start),
        "--catalogue-stop", str(cat_stop),
        "--draw-start", str(args.draw_start), "--draw-stop", str(draw_stop),
    ]
    prepare.append("--overwrite" if args.overwrite_samples else "--resume")

    snr = [
        *command_base, str(script_dir / "run_s22_100cat_snr.py"),
        "--input-root", str(args.population_root),
        "--output-root", str(args.snr_root),
        "--catalogue-start", str(args.catalogue_start),
        "--catalogue-stop", str(cat_stop),
        "--draw-start", str(args.draw_start), "--draw-stop", str(draw_stop),
        "--workers", str(args.workers), "--native-threads", str(args.native_threads),
    ]

    commands = []
    if args.stage in ("all", "prepare"):
        commands.append(prepare)
    if args.stage in ("all", "snr"):
        commands.append(snr)
    if args.dry_run:
        print("\n".join(shlex.join(command) for command in commands))
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
