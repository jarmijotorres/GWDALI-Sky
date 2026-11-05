#!/usr/bin/env python3
"""NERSC-only preparation: assign SkySim5000 galaxies to GW targets with MPI.

This optional upstream script requires the original heavy SkySim5000 catalogue
hosted at NERSC, GCRCatalogs, MPI, and membership in the NERSC `lsst` group.
Run it there only when rebuilding the seed-host HDF5. The portable pipeline
starts from that exported HDF5 with `python -m skysim_gw prepare` and needs
neither GCRCatalogs nor access to the original SkySim5000 catalogue.

Native SkySim5000 HEALPixels are used only to partition file IO among ranks.
Host sampling is uniform without replacement inside each true-redshift and
stellar-mass bin, using deterministic 64-bit priorities based on galaxy ID.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import GCRCatalogs
import h5py
import numpy as np
from mpi4py import MPI
from skysim_gw.host_properties import derive_host_properties


assert "lsst" in subprocess.check_output(["groups"]).decode().split(), (
    "NERSC-only seed preparation requires the `lsst` group and SkySim5000 access; use `prepare` with an existing seed HDF5 elsewhere"
)

from skysim_gw.settings.assign_hosts import (
    SOURCE_TYPES,
)
DEFAULT_CONFIG = Path("gw_host_assignment_config.json")

CANDIDATE_DTYPE = np.dtype(
    [
        ("priority", "<u8"),
        ("galaxy_id", "<u8"),
        ("ra", "<f8"),
        ("dec", "<f8"),
        ("redshift_true", "<f8"),
        ("redshift", "<f8"),
        ("stellar_mass", "<f8"),
        ("halo_mass", "<f8"),
        ("is_central", "u1"),
        ("sfr_uv_corrected", "<f8"),
        ("log_sfr_uv_corrected", "<f8"),
        ("metallicity_fmr_12logoh", "<f8"),
        ("fmr_mu", "<f8"),
        ("fmr_valid", "u1"),
        ("fmr_extrapolated", "u1"),
    ]
)

TYPE_HASH_KEYS = {
    "BBH": np.uint64(0x243F6A8885A308D3),
    "BNS": np.uint64(0x13198A2E03707344),
    "BHNS": np.uint64(0xA4093822299F31D0),
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Assign SkySim5000 host galaxies to precomputed GW targets."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "catalog_name",
        "targets_file",
        "output_file",
        "galaxy_id_column",
        "selection_seed",
        "centrals_only",
        "progress_every_native_pixels",
        "compression_level",
        "overwrite",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise KeyError(f"Missing configuration entries: {missing}")
    source_types = config.get("source_types", list(SOURCE_TYPES))
    if (
        not isinstance(source_types, list)
        or not source_types
        or any(not isinstance(value, str) for value in source_types)
    ):
        raise ValueError("source_types must be a non-empty JSON list of strings")
    source_types = [value.upper() for value in source_types]
    unknown = sorted(set(source_types) - set(SOURCE_TYPES))
    if unknown:
        raise ValueError(
            f"Unsupported source_types {unknown}; choose from {SOURCE_TYPES}"
        )
    if len(set(source_types)) != len(source_types):
        raise ValueError("source_types must not contain duplicates")
    config["source_types"] = source_types
    factor = config.get("event_oversample_factor", 1)
    if isinstance(factor, bool) or int(factor) != factor or int(factor) < 1:
        raise ValueError("event_oversample_factor must be a positive integer")
    config["event_oversample_factor"] = int(factor)
    return config


def load_targets(path, source_types=SOURCE_TYPES):
    with h5py.File(path, "r") as data:
        zi = data["bins/zi"][:]
        zf = data["bins/zf"][:]
        mass_edges = data["bins/log_stellar_mass_edges"][:]
        targets = {
            source_type: data[f"drawn/{source_type}"][:].astype(np.int64)
            for source_type in source_types
        }
        expected = {
            source_type: data[f"expected/{source_type}"][:].astype(np.float64)
            for source_type in source_types
        }
        target_metadata = {
            key: value for key, value in data.attrs.items()
        }

    shape = (len(zi), len(mass_edges) - 1)
    if zi.shape != zf.shape or np.any(zf <= zi):
        raise ValueError("Invalid redshift bins in target file")
    if np.any(np.diff(mass_edges) <= 0.0):
        raise ValueError("Invalid stellar-mass edges in target file")
    for source_type in source_types:
        if targets[source_type].shape != shape:
            raise ValueError(
                f"drawn/{source_type} has shape {targets[source_type].shape}, "
                f"expected {shape}"
            )
        if np.any(targets[source_type] < 0):
            raise ValueError(f"Negative target count for {source_type}")
    return zi, zf, mass_edges, targets, expected, target_metadata


def oversample_targets(targets, factor):
    """Scale already Poisson-drawn event targets for catalogue reuse.

    This deliberately does *not* alter the physical rate density or create
    additional independent SkySim volume.  It creates more draws from the
    same lightcone, which are tagged with a reciprocal event weight.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be a positive integer")
    maximum = np.iinfo(np.int64).max
    expanded = {}
    for source_type, values in targets.items():
        values = np.asarray(values, dtype=np.int64)
        if np.any(values > maximum // factor):
            raise OverflowError(
                f"Oversampled {source_type} target count exceeds int64"
            )
        expanded[source_type] = values * factor
    return expanded


def splitmix64(values, key):
    """Fast deterministic 64-bit hash used as a sampling priority."""
    with np.errstate(over="ignore"):
        x = np.asarray(values, dtype=np.uint64) ^ np.uint64(key)
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def assign_bins(redshift_true, stellar_mass, zi, zf, mass_edges):
    """Return flattened (redshift, mass) bin IDs, with -1 for invalid rows."""
    n_mass_bins = len(mass_edges) - 1
    z_index = np.searchsorted(zi, redshift_true, side="right") - 1
    z_candidate = (z_index >= 0) & (z_index < len(zi))
    valid_z = np.zeros(redshift_true.size, dtype=bool)
    valid_z[z_candidate] = redshift_true[z_candidate] < zf[z_index[z_candidate]]

    log_mass = np.full(stellar_mass.size, np.nan, dtype=np.float64)
    positive_mass = np.isfinite(stellar_mass) & (stellar_mass > 0.0)
    np.log10(stellar_mass, out=log_mass, where=positive_mass)
    mass_index = np.searchsorted(mass_edges, log_mass, side="right") - 1
    valid_mass = (mass_index >= 0) & (mass_index < n_mass_bins)

    valid = valid_z & valid_mass
    flat_bin = np.full(redshift_true.size, -1, dtype=np.int32)
    flat_bin[valid] = (
        z_index[valid] * n_mass_bins + mass_index[valid]
    ).astype(np.int32)
    return flat_bin


def make_candidate_records(data, derived, indices, priorities, id_column):
    records = np.empty(len(indices), dtype=CANDIDATE_DTYPE)
    records["priority"] = priorities
    records["galaxy_id"] = np.asarray(data[id_column])[indices].astype(np.uint64)
    records["ra"] = np.asarray(data["ra"])[indices]
    records["dec"] = np.asarray(data["dec"])[indices]
    records["redshift_true"] = np.asarray(data["redshift_true"])[indices]
    records["redshift"] = np.asarray(data["redshift"])[indices]
    records["stellar_mass"] = np.asarray(data["stellar_mass"])[indices]
    records["halo_mass"] = np.asarray(data["halo_mass"])[indices]
    records["is_central"] = np.asarray(data["is_central"])[indices].astype(np.uint8)
    records["sfr_uv_corrected"] = derived["sfr_uv_corrected"][indices]
    records["log_sfr_uv_corrected"] = derived["log_sfr_uv_corrected"][indices]
    records["metallicity_fmr_12logoh"] = derived["metallicity_fmr_12logoh"][indices]
    records["fmr_mu"] = derived["fmr_mu"][indices]
    records["fmr_valid"] = derived["fmr_valid"][indices].astype(np.uint8)
    records["fmr_extrapolated"] = derived["fmr_extrapolated"][indices].astype(np.uint8)
    return records


def keep_smallest(existing, new_records, limit):
    """Keep at most ``limit`` records with the smallest hash priorities."""
    if limit <= 0:
        return np.empty(0, dtype=CANDIDATE_DTYPE)
    if existing is None or existing.size == 0:
        combined = new_records
    elif new_records.size == 0:
        return existing
    else:
        combined = np.concatenate((existing, new_records))
    if combined.size <= limit:
        return combined
    chosen = np.argpartition(combined["priority"], limit - 1)[:limit]
    return combined[chosen]


def process_chunk(data, zi, zf, mass_edges, targets, availability,
                  reservoirs, config):
    id_column = config["galaxy_id_column"]
    galaxy_id = np.asarray(data[id_column])
    redshift_true = np.asarray(data["redshift_true"])
    stellar_mass = np.asarray(data["stellar_mass"])
    property_config = config.get("host_property_config", {})
    derived = derive_host_properties(
        data,
        sed_column=property_config.get(
            "sed_column", "sed_1246_306_disk_no_host_extinction"
        ),
        av_column=property_config.get("av_column", "A_v_disk"),
        stellar_mass_column="stellar_mass",
        sed_bin_width_angstrom=float(property_config.get("sed_bin_width_angstrom", 306.0)),
        sfr_calibration=property_config.get("sfr_calibration", "kennicutt2012"),
        fmr_alpha=float(property_config.get("fmr_alpha", 0.32)),
    )

    valid = (
        np.isfinite(redshift_true)
        & np.isfinite(stellar_mass)
        & (stellar_mass > 0.0)
        & np.isfinite(np.asarray(data["ra"]))
        & np.isfinite(np.asarray(data["dec"]))
    )
    if config["centrals_only"]:
        valid &= np.asarray(data["is_central"]).astype(bool)

    flat_bin = assign_bins(redshift_true, stellar_mass, zi, zf, mass_edges)
    valid &= flat_bin >= 0
    if not np.any(valid):
        return 0, 0

    valid_indices = np.flatnonzero(valid)
    valid_bins = flat_bin[valid_indices]
    availability.ravel()[:] += np.bincount(
        valid_bins, minlength=availability.size
    ).astype(np.int64, copy=False)

    seed_key = np.uint64(int(config["selection_seed"]))
    for bin_number in np.unique(valid_bins):
        in_bin = valid_indices[valid_bins == bin_number]
        if in_bin.size == 0:
            continue
        for source_type in targets:
            limit = int(targets[source_type].ravel()[bin_number])
            if limit == 0:
                continue
            priorities = splitmix64(
                galaxy_id[in_bin], seed_key ^ TYPE_HASH_KEYS[source_type]
            )
            if in_bin.size > limit:
                selected_local = np.argpartition(priorities, limit - 1)[:limit]
                indices = in_bin[selected_local]
                priorities = priorities[selected_local]
            else:
                indices = in_bin
            new_records = make_candidate_records(
                data, derived, indices, priorities, id_column
            )
            key = (source_type, int(bin_number))
            reservoirs[key] = keep_smallest(
                reservoirs.get(key), new_records, limit
            )
    return len(galaxy_id), len(valid_indices)


def merge_distributed_reservoirs(comm, reservoirs, targets):
    """Merge rank-local top-k reservoirs one target bin at a time.

    A single pickled gather of all populations can exceed mpi4py/MPI message
    limits for oversampled catalogues. Bin-wise collectives keep each message
    bounded by one target cell and let rank 0 discard non-winning candidates
    immediately.
    """
    merged = {} if comm.Get_rank() == 0 else None
    empty = np.empty(0, dtype=CANDIDATE_DTYPE)
    for source_type in targets:
        for bin_number, limit_value in enumerate(targets[source_type].ravel()):
            limit = int(limit_value)
            if limit == 0:
                continue
            local = reservoirs.get((source_type, bin_number), empty)
            arrays = comm.gather(local, root=0)
            if comm.Get_rank() == 0:
                nonempty = [array for array in arrays if array.size]
                if nonempty:
                    combined = np.concatenate(nonempty)
                    merged[(source_type, bin_number)] = keep_smallest(
                        None, combined, limit
                    )
                else:
                    merged[(source_type, bin_number)] = empty.copy()
    return merged


def finalize_sources(merged, targets, availability, selection_seed,
                     event_oversample_factor=1):
    """Create final per-type event records and handle deficient bins."""
    rng = np.random.default_rng(int(selection_seed))
    final = {}
    assigned_grid = {}
    replacement_grid = {}

    output_dtype = np.dtype(
        CANDIDATE_DTYPE.descr
        + [
            ("z_bin", "<i4"),
            ("mass_bin", "<i4"),
            ("sampled_with_replacement", "u1"),
            ("event_weight", "<f8"),
        ]
    )
    n_mass_bins = availability.shape[1]

    for source_type in targets:
        pieces = []
        assigned = np.zeros_like(targets[source_type], dtype=np.int64)
        replacements = np.zeros_like(targets[source_type], dtype=np.int64)

        for bin_number, requested_value in enumerate(targets[source_type].ravel()):
            requested = int(requested_value)
            if requested == 0:
                continue
            candidates = merged.get(
                (source_type, bin_number), np.empty(0, dtype=CANDIDATE_DTYPE)
            )
            if candidates.size == 0:
                continue

            order = np.lexsort((candidates["galaxy_id"], candidates["priority"]))
            candidates = candidates[order]
            n_unique = min(requested, candidates.size)
            chosen = candidates[:n_unique]
            replacement_flags = np.zeros(n_unique, dtype=np.uint8)

            if n_unique < requested:
                extra_indices = rng.choice(
                    candidates.size, size=requested - n_unique, replace=True
                )
                chosen = np.concatenate((chosen, candidates[extra_indices]))
                replacement_flags = np.concatenate(
                    (
                        replacement_flags,
                        np.ones(requested - n_unique, dtype=np.uint8),
                    )
                )
                replacements.ravel()[bin_number] = requested - n_unique

            records = np.empty(chosen.size, dtype=output_dtype)
            for name in CANDIDATE_DTYPE.names:
                records[name] = chosen[name]
            records["z_bin"] = bin_number // n_mass_bins
            records["mass_bin"] = bin_number % n_mass_bins
            records["sampled_with_replacement"] = replacement_flags
            records["event_weight"] = 1.0 / float(event_oversample_factor)
            pieces.append(records)
            assigned.ravel()[bin_number] = chosen.size

        final[source_type] = (
            np.concatenate(pieces) if pieces else np.empty(0, dtype=output_dtype)
        )
        assigned_grid[source_type] = assigned
        replacement_grid[source_type] = replacements
    return final, assigned_grid, replacement_grid


def write_output(path, config, config_path, targets_path, zi, zf, mass_edges,
                 targets, expected, availability, final_sources,
                 assigned_grid, replacement_grid, run_summary):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    compression = int(config["compression_level"])

    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["description"] = (
                "SkySim5000 galaxies assigned as hosts of precomputed GW sources"
            )
            output.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            output.attrs["config_path"] = str(config_path.resolve())
            output.attrs["config_json"] = json.dumps(config, sort_keys=True)
            output.attrs["targets_file"] = str(Path(targets_path).resolve())
            for key, value in run_summary.items():
                output.attrs[key] = value

            bins = output.create_group("bins")
            bins.create_dataset("zi", data=zi)
            bins.create_dataset("zf", data=zf)
            bins.create_dataset("log_stellar_mass_edges", data=mass_edges)

            available_group = output.create_group("available_hosts")
            available_group.create_dataset("counts", data=availability)

            expected_group = output.create_group("expected")
            target_group = output.create_group("target")
            assigned_group = output.create_group("assigned")
            replacement_group = output.create_group("replacement")
            sources_group = output.create_group("sources")

            for source_type in targets:
                expected_group.create_dataset(source_type, data=expected[source_type])
                target_group.create_dataset(source_type, data=targets[source_type])
                assigned_group.create_dataset(source_type, data=assigned_grid[source_type])
                replacement_group.create_dataset(
                    source_type, data=replacement_grid[source_type]
                )

                group = sources_group.create_group(source_type)
                records = final_sources[source_type]
                group.attrs["n_sources"] = len(records)
                group.attrs["n_requested"] = int(targets[source_type].sum())
                group.attrs["n_poisson_drawn_before_oversampling"] = int(
                    targets[source_type].sum()
                    // config["event_oversample_factor"]
                )
                group.attrs["n_replacement"] = int(
                    replacement_grid[source_type].sum()
                )
                group.attrs["event_oversample_factor"] = int(
                    config["event_oversample_factor"]
                )
                group.attrs["event_weight"] = 1.0 / float(
                    config["event_oversample_factor"]
                )
                for field in records.dtype.names:
                    group.create_dataset(
                        field,
                        data=records[field],
                        compression="gzip" if len(records) else None,
                        compression_opts=compression if len(records) else None,
                        shuffle=bool(len(records)),
                    )

        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def main():
    arguments = parse_arguments()
    config = load_config(arguments.config)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        output_exists = Path(config["output_file"]).exists()
        preflight_error = (
            f"Output exists: {config['output_file']}"
            if output_exists and not config["overwrite"]
            else None
        )
        target_payload = load_targets(
            config["targets_file"], config["source_types"]
        )
    else:
        preflight_error = None
        target_payload = None
    preflight_error = comm.bcast(preflight_error, root=0)
    if preflight_error:
        raise FileExistsError(preflight_error)
    zi, zf, mass_edges, base_targets, expected, _ = comm.bcast(
        target_payload, root=0
    )
    targets = oversample_targets(
        base_targets, config["event_oversample_factor"]
    )

    GCRCatalogs.ConfigSource.set_config_source(dr=False)
    catalog = GCRCatalogs.load_catalog(config["catalog_name"])
    id_column = config["galaxy_id_column"]
    required_quantities = [
        id_column,
        "ra",
        "dec",
        "redshift_true",
        "redshift",
        "stellar_mass",
        "halo_mass",
        "is_central",
        config.get("host_property_config", {}).get(
            "sed_column", "sed_1246_306_disk_no_host_extinction"
        ),
        config.get("host_property_config", {}).get("av_column", "A_v_disk"),
    ]
    missing = [q for q in required_quantities if not catalog.has_quantity(q)]
    all_missing = comm.gather(missing, root=0)
    if rank == 0:
        missing_union = sorted({item for items in all_missing for item in items})
    else:
        missing_union = None
    missing_union = comm.bcast(missing_union, root=0)
    if missing_union:
        raise KeyError(f"Catalog is missing required quantities: {missing_union}")

    all_native_pixels = np.asarray(
        catalog.available_healpix_pixels, dtype=np.int64
    )
    my_native_pixels = all_native_pixels[rank::size]
    availability = np.zeros(
        next(iter(targets.values())).shape, dtype=np.int64
    )
    reservoirs = {}
    filters = [
        "stellar_mass > 0",
        f"redshift_true >= {np.min(zi):.17g}",
        f"redshift_true < {np.max(zf):.17g}",
    ]

    comm.Barrier()
    start = MPI.Wtime()
    rows_read = 0
    valid_rows = 0
    chunks = 0
    progress_every = int(config["progress_every_native_pixels"])

    print(
        f"rank={rank}/{size} assigned_native_pixels={len(my_native_pixels)}",
        flush=True,
    )
    for pixel_number, native_pixel in enumerate(my_native_pixels, start=1):
        iterator = catalog.get_quantities(
            required_quantities,
            filters=filters,
            native_filters=[f"healpix_pixel == {int(native_pixel)}"],
            return_iterator=True,
        )
        for data in iterator:
            chunks += 1
            n_read, n_valid = process_chunk(
                data,
                zi,
                zf,
                mass_edges,
                targets,
                availability,
                reservoirs,
                config,
            )
            rows_read += n_read
            valid_rows += n_valid

        if pixel_number % progress_every == 0 or pixel_number == len(my_native_pixels):
            print(
                f"rank={rank} pixels={pixel_number}/{len(my_native_pixels)} "
                f"chunks={chunks:,} rows={rows_read:,} valid={valid_rows:,} "
                f"elapsed={(MPI.Wtime() - start) / 60:.1f} min",
                flush=True,
            )

    global_availability = (
        np.empty_like(availability) if rank == 0 else None
    )
    comm.Reduce(availability, global_availability, op=MPI.SUM, root=0)
    total_rows = comm.reduce(rows_read, op=MPI.SUM, root=0)
    total_valid = comm.reduce(valid_rows, op=MPI.SUM, root=0)
    total_chunks = comm.reduce(chunks, op=MPI.SUM, root=0)
    merged = merge_distributed_reservoirs(comm, reservoirs, targets)

    if rank == 0:
        final_sources, assigned_grid, replacement_grid = finalize_sources(
            merged,
            targets,
            global_availability,
            config["selection_seed"],
            config["event_oversample_factor"],
        )
        elapsed = MPI.Wtime() - start
        property_config = config.get("host_property_config", {})
        sed_column = property_config.get(
            "sed_column", "sed_1246_306_disk_no_host_extinction"
        )
        av_column = property_config.get("av_column", "A_v_disk")
        run_summary = {
            "catalog": config["catalog_name"],
            "mpi_ranks": size,
            "native_pixels": len(all_native_pixels),
            "native_chunks": total_chunks,
            "rows_read": total_rows,
            "valid_host_rows": total_valid,
            "elapsed_seconds": elapsed,
            "selection_method": "deterministic distributed top-k hash priority",
            "mass_variable": "log10(stellar_mass / Msun)",
            "redshift_variable": "redshift_true",
            "source_types": ",".join(config["source_types"]),
            "event_oversample_factor": config["event_oversample_factor"],
            "event_weight": 1.0 / config["event_oversample_factor"],
            "oversampling_interpretation": (
                "Repeated sampling of one SkySim lightcone; not independent "
                "sky area or reduced cosmic variance"
            ),
            "host_properties": (
                "dust-corrected UV SFR from "
                f"{sed_column} and {av_column}; "
                "Mannucci2010 FMR gas-phase 12+log10(O/H)"
            ),
        }
        write_output(
            config["output_file"],
            config,
            arguments.config,
            config["targets_file"],
            zi,
            zf,
            mass_edges,
            targets,
            expected,
            global_availability,
            final_sources,
            assigned_grid,
            replacement_grid,
            run_summary,
        )
        print(
            f"[{datetime.now():%H:%M:%S}] Saved {config['output_file']} "
            f"in {elapsed / 60:.1f} min",
            flush=True,
        )
        for source_type in targets:
            print(
                f"{source_type}: requested={targets[source_type].sum():,} "
                f"assigned={len(final_sources[source_type]):,} "
                f"replacement={replacement_grid[source_type].sum():,}",
                flush=True,
            )


if __name__ == "__main__":
    main()
