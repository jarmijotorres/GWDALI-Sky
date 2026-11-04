#!/usr/bin/env python3
"""Build a compact S22 Mstar/SFR/logZ/redshift histogram without NumPy.

The output is a standard compressed NPZ file readable with ``numpy.load``.
Keeping this builder standard-library-only allows it to run on a lightweight
login machine; the production model itself still uses NumPy.
"""

from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import struct
import sys
import zipfile


SNAPSHOTS = {
    28: 0.0, 27: 0.10063853859901428, 26: 0.18270987272262573,
    25: 0.27090108394622803, 24: 0.36566856503486633,
    23: 0.5031073093414307, 22: 0.615189790725708,
    21: 0.7356296181678772, 20: 0.8650505542755127,
    19: 1.0041216611862183, 18: 1.259, 17: 1.487,
    16: 1.737, 15: 2.012,
}


def regular_edges(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [start + step * index for index in range(count + 1)]


def bin_index(value: float, edges: list[float]) -> int:
    step = edges[1] - edges[0]
    index = math.floor((value - edges[0]) / step)
    return index if 0 <= index < len(edges) - 1 else -1


def npy_payload(typecode: str, descr: str, shape: tuple[int, ...], values) -> bytes:
    data = array(typecode, values)
    if sys.byteorder != "little" and data.itemsize > 1:
        data.byteswap()
    header = repr({"descr": descr, "fortran_order": False, "shape": shape})
    header_length = len(header.encode("latin1")) + 1
    padding = (-10 - header_length) % 16
    encoded_header = (header + " " * padding + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded_header)) + encoded_header + data.tobytes()


def bytes_npy_payload(values: bytes) -> bytes:
    header = repr({"descr": "|u1", "fortran_order": False, "shape": (len(values),)})
    header_length = len(header.encode("latin1")) + 1
    padding = (-10 - header_length) % 16
    encoded_header = (header + " " * padding + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded_header)) + encoded_header + values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metallicity-relation", default="M57")
    parser.add_argument("--common-envelope", default="A3")
    args = parser.parse_args()

    mass_edges = regular_edges(7.0, 12.0, 0.1)
    sfr_edges = regular_edges(-4.0, 2.5, 0.1)
    logz_edges = regular_edges(-5.0, 0.0, 0.1)
    ordered = sorted(SNAPSHOTS.items(), key=lambda item: item[1])
    shape = (len(ordered), len(mass_edges) - 1, len(sfr_edges) - 1, len(logz_edges) - 1)
    cells_per_snapshot = math.prod(shape[1:])
    rate_sum = array("d", [0.0]) * math.prod(shape)
    occupancy = array("I", [0]) * math.prod(shape)
    input_rows = []
    accepted_rows = []
    input_rate_sums = []

    for redshift_index, (snapshot, redshift) in enumerate(ordered):
        path = args.data_dir / (
            f"HG_{args.metallicity_relation}_{args.common_envelope}_"
            f"GSSI_v2_BBHs_{snapshot}_12Z_50_No_0.14_2.txt"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        seen = kept = 0
        summed_rate = 0.0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line or line.startswith("#"):
                    continue
                seen += 1
                fields = line.split()
                if len(fields) < 4:
                    continue
                mass, sfr, logz, rate = map(float, fields[:4])
                if not (mass > 0.0 and sfr > 0.0 and rate >= 0.0):
                    continue
                log_mass, log_sfr = math.log10(mass), math.log10(sfr)
                indices = (
                    bin_index(log_mass, mass_edges),
                    bin_index(log_sfr, sfr_edges),
                    bin_index(logz, logz_edges),
                )
                if min(indices) < 0 or not all(math.isfinite(v) for v in (log_mass, log_sfr, logz, rate)):
                    continue
                mass_index, sfr_index, logz_index = indices
                flat = (
                    redshift_index * cells_per_snapshot
                    + mass_index * shape[2] * shape[3]
                    + sfr_index * shape[3]
                    + logz_index
                )
                rate_sum[flat] += rate
                occupancy[flat] += 1
                summed_rate += rate
                kept += 1
        input_rows.append(seen)
        accepted_rows.append(kept)
        input_rate_sums.append(summed_rate)
        print(f"snapshot={snapshot} z={redshift:.6f} accepted={kept:,}/{seen:,}")

    metadata = {
        "format": "s22_bbh_histogram_v1",
        "metallicity_relation": args.metallicity_relation,
        "common_envelope": args.common_envelope,
        "rate_input_units": "Gyr^-1 per catalogue galaxy",
        "rate_sum_semantics": "sum nGW in each Mstar/SFR/logZ bin",
        "metallicity_coordinate": "raw S22 logZ column; not 12+log10(O/H)",
        "source_volume_cmpc3": 100.0**3,
        "input_rows": input_rows,
        "accepted_rows": accepted_rows,
        "input_rate_sums_gyr_inverse": input_rate_sums,
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        output.writestr("snapshot_numbers.npy", npy_payload("q", "<i8", (len(ordered),), [item[0] for item in ordered]))
        output.writestr("redshifts.npy", npy_payload("d", "<f8", (len(ordered),), [item[1] for item in ordered]))
        output.writestr("mass_edges.npy", npy_payload("d", "<f8", (len(mass_edges),), mass_edges))
        output.writestr("sfr_edges.npy", npy_payload("d", "<f8", (len(sfr_edges),), sfr_edges))
        output.writestr("log_metallicity_edges.npy", npy_payload("d", "<f8", (len(logz_edges),), logz_edges))
        output.writestr("rate_sum.npy", npy_payload("d", "<f8", shape, rate_sum))
        output.writestr("occupancy.npy", npy_payload("I", "<u4", shape, occupancy))
        output.writestr("metadata_json.npy", bytes_npy_payload(metadata_bytes))
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
