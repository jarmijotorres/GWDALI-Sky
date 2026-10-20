"""Step 7: compute serial LVK SNRs for a prepared GW source catalogue.

The calculation is checkpointed into a compressed NPZ file. Re-running the
same command resumes unfinished rows. Each result row maps to the original
injection catalogue through ``source_index``.
"""

import argparse
from pathlib import Path
import os
import tempfile
import time

import GWDALI as gw
import numpy as np

from skysim_gw.catalogue import (
    catalog_path,
    get_gwdali_source,
    load_gwdali_catalogue,
)
from skysim_gw.detectors import LVK_LABELS, get_lvk_detectors


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

from skysim_gw.settings.snr import (
    BASE_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    SOURCE_TYPE,
    ZI,
    ZF,
    INCLUDE_KAGRA,
    APPROXIMANTS,
    FMIN_HZ,
    FMAX_HZ,
    FSIZE,
    NETWORK_SNR_THRESHOLD,
    SINGLE_DETECTOR_SNR_THRESHOLD,
    MIN_DETECTORS_ABOVE_THRESHOLD,
    START_INDEX,
    STOP_INDEX,
    CHECKPOINT_EVERY,
    PRINT_EVERY,
    RESUME,
)


# Set STOP_INDEX to an integer for a small test. None processes to the end.


RESULT_DTYPE = np.dtype([
    ("source_index", "i8"),
    ("snr_hanford", "f8"),
    ("snr_livingston", "f8"),
    ("snr_virgo", "f8"),
    ("snr_kagra", "f8"),
    ("snr_network", "f8"),
    ("n_detectors_above_threshold", "i2"),
    ("detected", "?"),
    ("success", "?"),
    ("error_message", "U512"),
])


from skysim_gw.products import snr_output_path as output_path


def parse_arguments(argv=None):
    """Parse optional explicit paths for tagged realization catalogues."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=Path,
        help="Prepared GWDALI NPZ catalogue; overrides settings INPUT_DIR.",
    )
    parser.add_argument(
        "--output-file", type=Path,
        help="Checkpointed SNR NPZ output; overrides settings OUTPUT_DIR.",
    )
    arguments = parser.parse_args(argv)
    if (arguments.input_file is None) != (arguments.output_file is None):
        parser.error("--input-file and --output-file must be supplied together")
    return arguments


def empty_results(number_of_sources):
    results = np.zeros(number_of_sources, dtype=RESULT_DTYPE)
    results["source_index"] = np.arange(number_of_sources)
    for field in (
        "snr_hanford", "snr_livingston", "snr_virgo",
        "snr_kagra", "snr_network",
    ):
        results[field] = np.nan
    return results


def save_checkpoint(path, results, completed, metadata):
    """Atomically replace a checkpoint so interruption cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}_", suffix=".npz"
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary_path,
            results=results,
            completed=completed,
            **{name: np.asarray(value) for name, value in metadata.items()},
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(path, expected_size, metadata):
    with np.load(path, allow_pickle=False) as saved:
        results = saved["results"]
        completed = saved["completed"].astype(bool)
        if len(results) != expected_size or len(completed) != expected_size:
            raise ValueError("Checkpoint size does not match the input catalogue")
        for name, expected in metadata.items():
            if name not in saved.files:
                raise KeyError(f"Checkpoint is missing metadata {name!r}")
            actual = saved[name].item()
            if isinstance(expected, float):
                matches = np.isclose(actual, expected, rtol=0.0, atol=1e-12)
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f"Checkpoint setting {name!r}={actual!r} does not match "
                    f"current value {expected!r}"
                )
    return results, completed


def parse_snr_result(raw_result, number_of_detectors):
    """Parse the verified GWDALI return: (detector SNR list, network SNR)."""
    if not isinstance(raw_result, tuple) or len(raw_result) != 2:
        raise TypeError(
            "Unexpected get_SNR return; expected (detector_snrs, network_snr)"
        )
    detector_snrs = np.asarray(raw_result[0], dtype=float)
    network_snr = float(raw_result[1])
    if detector_snrs.shape != (number_of_detectors,):
        raise ValueError(
            f"Expected {number_of_detectors} detector SNRs; "
            f"received shape {detector_snrs.shape}"
        )
    if not np.all(np.isfinite(detector_snrs)) or not np.isfinite(network_snr):
        raise ValueError("GWDALI returned a non-finite SNR")

    quadrature_snr = float(np.sqrt(np.sum(detector_snrs**2)))
    if not np.isclose(network_snr, quadrature_snr, rtol=1e-8, atol=1e-10):
        raise ValueError(
            f"Network SNR {network_snr} disagrees with quadrature sum "
            f"{quadrature_snr}"
        )
    return detector_snrs, network_snr


def main(argv=None):
    arguments = parse_arguments(argv)
    source_type = SOURCE_TYPE.upper()
    if source_type not in APPROXIMANTS:
        raise ValueError(f"Unknown source type {source_type!r}")
    if CHECKPOINT_EVERY <= 0 or PRINT_EVERY <= 0:
        raise ValueError("CHECKPOINT_EVERY and PRINT_EVERY must be positive")

    input_path = (
        arguments.input_file
        if arguments.input_file is not None
        else catalog_path(INPUT_DIR, source_type, ZI, ZF)
    )
    save_path = (
        arguments.output_file
        if arguments.output_file is not None
        else output_path(OUTPUT_DIR, source_type, ZI, ZF)
    )
    catalogue, _ = load_gwdali_catalogue(input_path)
    detectors = get_lvk_detectors(include_kagra=INCLUDE_KAGRA)
    detector_labels = LVK_LABELS[:len(detectors)]
    approximant = APPROXIMANTS[source_type]

    stop = (
        len(catalogue)
        if STOP_INDEX is None
        else min(int(STOP_INDEX), len(catalogue))
    )
    start = max(0, int(START_INDEX))
    if start >= stop:
        raise ValueError(f"Empty index interval [{start}, {stop})")

    metadata = {
        "source_type": source_type,
        "zi": float(ZI),
        "zf": float(ZF),
        "approximant": approximant,
        "include_kagra": bool(INCLUDE_KAGRA),
        "fmin_hz": float(FMIN_HZ),
        "fmax_hz": float(FMAX_HZ),
        "fsize": int(FSIZE),
        "network_snr_threshold": float(NETWORK_SNR_THRESHOLD),
        "single_detector_snr_threshold": float(SINGLE_DETECTOR_SNR_THRESHOLD),
        "minimum_detectors": int(MIN_DETECTORS_ABOVE_THRESHOLD),
        "input_catalog": str(input_path),
        "stored_ra_dec_unit": "radian",
        "gwdali_ra_dec_unit": "degree",
        "coordinate_convention": "gwdali_ra_dec_degrees_v1",
    }

    if RESUME and save_path.is_file():
        results, completed = load_checkpoint(save_path, len(catalogue), metadata)
        print(f"Resuming checkpoint: {save_path}")
    else:
        results = empty_results(len(catalogue))
        completed = np.zeros(len(catalogue), dtype=bool)

    pending = [index for index in range(start, stop) if not completed[index]]
    print(f"Input: {input_path}")
    print(f"Output: {save_path}")
    print(f"Sources in catalogue: {len(catalogue):,}")
    print("Stored RA/Dec: radians; GWDALI RA/Dec: degrees")
    print(f"Requested interval: [{start:,}, {stop:,})")
    print(f"Pending sources: {len(pending):,}")
    print(f"Approximant: {approximant}")
    print(f"Detectors: {detector_labels}")

    run_start = time.perf_counter()
    since_checkpoint = 0
    for count, index in enumerate(pending, start=1):
        try:
            GwPrms = get_gwdali_source(catalogue, index)
            raw_result = gw.get_SNR(
                detectors,
                GwPrms,
                approximant,
                enable_jax_waveforms=False,
                fmin=FMIN_HZ,
                fmax=FMAX_HZ,
                fsize=FSIZE,
            )
            detector_snrs, network_snr = parse_snr_result(
                raw_result, len(detectors)
            )

            results["snr_hanford"][index] = detector_snrs[0]
            results["snr_livingston"][index] = detector_snrs[1]
            results["snr_virgo"][index] = detector_snrs[2]
            if INCLUDE_KAGRA:
                results["snr_kagra"][index] = detector_snrs[3]

            number_above = int(np.count_nonzero(
                detector_snrs >= SINGLE_DETECTOR_SNR_THRESHOLD
            ))
            results["snr_network"][index] = network_snr
            results["n_detectors_above_threshold"][index] = number_above
            results["detected"][index] = (
                network_snr >= NETWORK_SNR_THRESHOLD
                and number_above >= MIN_DETECTORS_ABOVE_THRESHOLD
            )
            results["success"][index] = True
            results["error_message"][index] = ""
        except Exception as error:
            results["success"][index] = False
            results["detected"][index] = False
            results["error_message"][index] = (
                f"{type(error).__name__}: {error}"[:512]
            )

        completed[index] = True
        since_checkpoint += 1

        if count % PRINT_EVERY == 0 or count == len(pending):
            elapsed = time.perf_counter() - run_start
            rate = count / elapsed if elapsed > 0 else np.nan
            successes = int(results["success"][start:stop].sum())
            detections = int(results["detected"][start:stop].sum())
            print(
                f"processed={count:,}/{len(pending):,} index={index:,} "
                f"success={successes:,} detected={detections:,} "
                f"rate={rate:.3f} source/s elapsed={elapsed/60:.2f} min",
                flush=True,
            )

        if since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(save_path, results, completed, metadata)
            since_checkpoint = 0

    save_checkpoint(save_path, results, completed, metadata)

    selected = completed[start:stop]
    success = results["success"][start:stop] & selected
    detected = results["detected"][start:stop] & selected
    elapsed = time.perf_counter() - run_start
    print("\nFinished")
    print(f"Completed: {int(selected.sum()):,}")
    print(f"Successful: {int(success.sum()):,}")
    print(f"Failed: {int(selected.sum() - success.sum()):,}")
    print(f"Detected: {int(detected.sum()):,}")
    print(f"Elapsed: {elapsed/60:.2f} min")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
