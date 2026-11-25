#!/usr/bin/env python3
"""Run independent GWDALI Fisher events on one multi-core node.

The project-specific numerical code is deliberately kept behind ``--pipeline-module``.
That module must expose ``run_event(source_index) -> dict`` and return at least
``fisher`` and ``covariance`` matrices in the FREE_PARAMS order below. It may
also return scalar localisation products. Legacy mode writes per-source NPZ
files; HDF5 mode saves the Fisher-Gaussian parameters and appends each
event result from the parent process into one population-draw group.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys
import tempfile
import time
from multiprocessing import get_context


FREE_PARAMS = ("RA", "Dec", "inv_dL", "Mc", "eta", "iota", "psi", "phi_coal")
_RUN_EVENT = None
_OUTPUT_DIR = None


def _limit_native_threads(n: int) -> None:
    # Must happen before importing numpy/JAX/native waveform libraries in workers.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(n)
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")


def _init_worker(
    module_name: str,
    native_threads: int,
    output_dir: str,
    input_file: str | None,
    snr_file: str | None,
) -> None:
    global _RUN_EVENT, _OUTPUT_DIR
    _limit_native_threads(native_threads)
    _OUTPUT_DIR = output_dir
    if input_file is not None:
        os.environ["GWDALI_INPUT_FILE"] = input_file
        os.environ["GWDALI_SNR_FILE"] = snr_file
    else:
        os.environ.pop("GWDALI_INPUT_FILE", None)
        os.environ.pop("GWDALI_SNR_FILE", None)
    module = importlib.import_module(module_name)
    _RUN_EVENT = module.run_event


def _atomic_save(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp = pathlib.Path(tmp_name)
    try:
        import numpy as np
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _one(index: int) -> tuple[int, str]:
    import numpy as np
    out = pathlib.Path(_OUTPUT_DIR) / f"source_{index:06d}.npz"
    if out.exists():
        return index, "skipped"
    started = time.time()
    try:
        result = dict(_RUN_EVENT(index))
        for name in ("fisher", "covariance"):
            if name not in result:
                raise KeyError(f"run_event() did not return {name!r}")
            matrix = np.asarray(result[name], dtype=float)
            if matrix.shape != (len(FREE_PARAMS), len(FREE_PARAMS)):
                raise ValueError(
                    f"{name} must have shape ({len(FREE_PARAMS)}, "
                    f"{len(FREE_PARAMS)}), got {matrix.shape}"
                )
            result[name] = matrix
        result["source_index"] = np.int64(index)
        result["free_params"] = np.asarray(FREE_PARAMS)
        result["runtime_s_worker"] = np.float64(time.time() - started)
        result["success"] = np.bool_(True)
        _atomic_save(out, result)
        return index, "completed"
    except Exception as exc:  # keep other events running; failed events are retryable
        error_path = out.with_suffix(".error.json")
        error_path.write_text(json.dumps({"source_index": index, "error": repr(exc)}, indent=2))
        return index, f"failed: {exc!r}"


def _one_result(index: int):
    """Calculate one Fisher result and return it for parent-owned HDF5 I/O."""
    import numpy as np
    started = time.time()
    try:
        result = dict(_RUN_EVENT(index))
        for name in ("fisher", "covariance"):
            if name not in result:
                raise KeyError(f"run_event() did not return {name!r}")
            matrix = np.asarray(result[name], dtype=float)
            if matrix.shape != (len(FREE_PARAMS), len(FREE_PARAMS)):
                raise ValueError(
                    f"{name} must have shape ({len(FREE_PARAMS)}, "
                    f"{len(FREE_PARAMS)}), got {matrix.shape}"
                )
            result[name] = matrix
        result["runtime_s_worker"] = time.time() - started
        return index, result, ""
    except Exception as exc:
        return index, None, f"{type(exc).__name__}: {exc}"


def _initialize_hdf5_group(
    store, catalogue_id, population_draw, indices, input_file, snr_file,
):
    """Create or resume one draw group; only the parent process touches HDF5."""
    import h5py
    import numpy as np
    path = f"draws/draw{population_draw:04d}"
    if (
        "format" in store.attrs
        and store.attrs["format"] != "fisher_gaussian_parameters_v1"
    ):
        raise ValueError("Output HDF5 has an unsupported format")
    if (
        "seed_catalogue_id" in store.attrs
        and int(store.attrs["seed_catalogue_id"]) != int(catalogue_id)
    ):
        raise ValueError("Output HDF5 belongs to a different seed catalogue")
    if path not in store:
        store.attrs["format"] = "fisher_gaussian_parameters_v1"
        store.attrs["seed_catalogue_id"] = int(catalogue_id)
        store.attrs["free_params"] = np.asarray(FREE_PARAMS, dtype="S")
        store.attrs["parameter_units"] = (
            "RA,Dec: degree; inv_dL: Gpc^-1; Mc: detector-frame Msun; "
            "eta: dimensionless; iota,psi,phi_coal: radian"
        )
        group = store.require_group(path)
        group.attrs["catalogue_id"] = int(catalogue_id)
        group.attrs["population_draw"] = int(population_draw)
        group.attrs["population_file"] = str(input_file)
        group.attrs["snr_file"] = str(snr_file)
        group.attrs["sampling"] = "Gaussian centered at injected parameters; covariance is inverse Fisher"
        group.create_dataset("source_index", data=np.asarray(indices, dtype=np.int64))
        for name, shape in (("mean", (len(indices), len(FREE_PARAMS))),
                            ("fisher", (len(indices), len(FREE_PARAMS), len(FREE_PARAMS))),
                            ("covariance", (len(indices), len(FREE_PARAMS), len(FREE_PARAMS)))):
            group.create_dataset(name, shape=shape, dtype="f8", fillvalue=np.nan)
        group.create_dataset("completed", data=np.zeros(len(indices), dtype=bool))
        group.create_dataset("success", data=np.zeros(len(indices), dtype=bool))
        group.create_dataset(
            "error_message", shape=(len(indices),),
            dtype=h5py.string_dtype("utf-8"),
        )
    else:
        group = store[path]
        if (
            int(group.attrs["catalogue_id"]) != int(catalogue_id)
            or int(group.attrs["population_draw"]) != int(population_draw)
            or group.attrs["population_file"] != str(input_file)
            or group.attrs["snr_file"] != str(snr_file)
            or not np.array_equal(group["source_index"][:], indices)
        ):
            raise ValueError(f"Existing HDF5 group {path} has incompatible metadata")
    return group


def _save_hdf5_event(
    group, row, index, result, error, catalogue, catalogue_id,
):
    """Save one event Gaussian and its Fisher matrix in the parent process."""
    import numpy as np
    from skysim_gw.catalogue import get_gwdali_source
    if error:
        group["success"][row] = False
        group["error_message"][row] = error
        group["completed"][row] = True
        return
    try:
        source = get_gwdali_source(
            catalogue, index, distance_parameter="inv_dL",
            mass_parameterization="mc_eta",
        )
        mean = np.asarray([source[name] for name in FREE_PARAMS], dtype=float)
        group["mean"][row] = mean
        group["fisher"][row] = result["fisher"]
        group["covariance"][row] = result["covariance"]
        group["success"][row] = True
        group["error_message"][row] = ""
    except Exception as exc:
        group["success"][row] = False
        group["error_message"][row] = f"{type(exc).__name__}: {exc}"
    group["completed"][row] = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-module", required=True,
                    help="importable module containing run_event(source_index)")
    ap.add_argument("--output-dir",
                    help="Legacy output directory for one NPZ per event.")
    ap.add_argument("--output-hdf5",
                    help="Append Fisher-Gaussian samples to a grouped HDF5 file.")
    ap.add_argument("--catalogue-id", type=int,
                    help="Host seed-catalogue ID used in deterministic sample seeds.")
    ap.add_argument("--population-draw", type=int,
                    help="Population draw ID stored under /draws/drawXXXX.")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="Flush HDF5 progress after this many events.")
    ap.add_argument("--indices", required=True, help="text file: one detected catalogue index per line")
    ap.add_argument("--input-file",
                    help="Prepared population NPZ; passed to the standard adapter.")
    ap.add_argument("--snr-file",
                    help="Matched completed SNR NPZ; passed to the standard adapter.")
    ap.add_argument("--workers", type=int, default=32,
                    help="processes; conservative default for a 128-core node")
    ap.add_argument("--native-threads", type=int, default=1)
    args = ap.parse_args(argv)
    if args.workers < 1 or args.native_threads < 1:
        ap.error("--workers and --native-threads must be positive")
    if args.workers * args.native_threads > 128:
        ap.error("workers * native-threads must not exceed 128")
    if (args.input_file is None) != (args.snr_file is None):
        ap.error("--input-file and --snr-file must be supplied together")
    if (args.output_dir is None) == (args.output_hdf5 is None):
        ap.error("supply exactly one of --output-dir or --output-hdf5")
    hdf5_mode = args.output_hdf5 is not None
    if hdf5_mode and (
        args.input_file is None or args.catalogue_id is None
        or args.population_draw is None
    ):
        ap.error(
            "HDF5 mode requires --input-file, --snr-file, --catalogue-id, "
            "and --population-draw"
        )
    if args.checkpoint_every <= 0:
        ap.error("--checkpoint-every must be positive")

    # Set limits in the parent before spawn/imports as well as in each child.
    _limit_native_threads(args.native_threads)
    indices = [int(line) for line in pathlib.Path(args.indices).read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        ap.error("indices must be unique non-negative catalogue row numbers")
    output = pathlib.Path(args.output_dir) if args.output_dir else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    group = None
    catalogue = None
    store = None
    if hdf5_mode:
        import h5py
        import numpy as np
        from skysim_gw.catalogue import load_gwdali_catalogue
        catalogue, _ = load_gwdali_catalogue(args.input_file)
        if any(index >= len(catalogue) for index in indices):
            ap.error("an index lies outside the input population catalogue")
        hdf5_path = pathlib.Path(args.output_hdf5)
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        store = h5py.File(hdf5_path, "a")
        try:
            group = _initialize_hdf5_group(
                store, args.catalogue_id, args.population_draw, indices,
                args.input_file, args.snr_file,
            )
        except Exception:
            store.close()
            raise
        row_for_index = {index: row for row, index in enumerate(indices)}
        completed = group["completed"][:].astype(bool)
        success = group["success"][:].astype(bool)
        todo = [
            index for index in indices
            if not completed[row_for_index[index]] or not success[row_for_index[index]]
        ]
    else:
        todo = [i for i in indices if not (output / f"source_{i:06d}.npz").exists()]
    print(f"indices={len(indices)} existing={len(indices)-len(todo)} todo={len(todo)} "
          f"workers={args.workers} native_threads={args.native_threads}", flush=True)
    if not todo:
        if store is not None:
            store.close()
        return 0

    ctx = get_context("spawn")
    try:
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(args.pipeline_module, args.native_threads,
                                "" if output is None else str(output),
                                args.input_file, args.snr_file)) as pool:
            if hdf5_mode:
                row_for_index = {index: row for row, index in enumerate(indices)}
                for count, (index, result, error) in enumerate(
                    pool.imap_unordered(_one_result, todo, chunksize=1), start=1
                ):
                    _save_hdf5_event(
                        group, row_for_index[index], index, result, error,
                        catalogue, args.catalogue_id,
                    )
                    if count % args.checkpoint_every == 0:
                        store.flush()
                    print(
                        f"{index}: {'failed: ' + error if error else 'saved'}",
                        flush=True,
                    )
                store.flush()
            else:
                for index, status in pool.imap_unordered(_one, todo, chunksize=1):
                    print(f"{index}: {status}", flush=True)
    finally:
        if store is not None:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
