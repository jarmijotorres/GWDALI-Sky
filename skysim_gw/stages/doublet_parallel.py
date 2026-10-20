#!/usr/bin/env python3
"""Run independent GWDALI Doublet events on one multi-core node.

The project-specific numerical code is deliberately kept behind ``--pipeline-module``.
That module must expose ``run_event(source_index) -> dict`` and return at least
``samples`` (N x 6), with columns ordered as FREE_PARAMS below.  It may also return
any scalar products from the serial pipeline; those are persisted alongside samples.
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


FREE_PARAMS = ("RA", "Dec", "inv_dL", "iota", "psi", "phi_coal")
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
        if "samples" not in result:
            raise KeyError("run_event() did not return 'samples'")
        samples = np.asarray(result["samples"])
        if samples.ndim != 2 or samples.shape[1] != len(FREE_PARAMS):
            raise ValueError(f"samples must have shape (N, 6), got {samples.shape}")
        result["samples"] = samples
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-module", required=True,
                    help="importable module containing run_event(source_index)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--indices", required=True, help="text file: one detected catalogue index per line")
    ap.add_argument("--input-file",
                    help="Prepared population NPZ; passed to the standard adapter.")
    ap.add_argument("--snr-file",
                    help="Matched completed SNR NPZ; passed to the standard adapter.")
    ap.add_argument("--workers", type=int, default=32,
                    help="processes; conservative default for a 128-core node")
    ap.add_argument("--native-threads", type=int, default=1)
    args = ap.parse_args()
    if args.workers < 1 or args.native_threads < 1:
        ap.error("--workers and --native-threads must be positive")
    if args.workers * args.native_threads > 128:
        ap.error("workers * native-threads must not exceed 128")
    if (args.input_file is None) != (args.snr_file is None):
        ap.error("--input-file and --snr-file must be supplied together")

    # Set limits in the parent before spawn/imports as well as in each child.
    _limit_native_threads(args.native_threads)
    indices = [int(line) for line in pathlib.Path(args.indices).read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    todo = [i for i in indices if not (output / f"source_{i:06d}.npz").exists()]
    print(f"indices={len(indices)} existing={len(indices)-len(todo)} todo={len(todo)} "
          f"workers={args.workers} native_threads={args.native_threads}", flush=True)
    if not todo:
        return 0

    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.pipeline_module, args.native_threads, str(output),
                            args.input_file, args.snr_file)) as pool:
        for index, status in pool.imap_unordered(_one, todo, chunksize=1):
            print(f"{index}: {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
