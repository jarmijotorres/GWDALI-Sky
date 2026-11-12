"""Compute Fisher sky localisations for all detected GW sources, serially.

Inputs are the prepared injection NPZ and the completed SNR checkpoint. Only
rows with ``success & detected & completed`` enter GWDALI.

The physical injection catalogue stores luminosity distance ``dL`` in Gpc.
At the GWDALI localisation interface this is transformed to
``inv_dL = 1 / dL`` in Gpc^-1.

For each source, GWDALI computes a Fisher matrix in the detector-frame chirp
mass and symmetric-mass-ratio parameterisation. Its inverse is the local
Gaussian covariance used for all stored sky and distance summaries.

Results retain the original ``source_index`` and are checkpointed atomically
for safe resumption.
"""

from pathlib import Path
import argparse
import os
import tempfile
import time

import GWDALI as gw
import numpy as np

from skysim_gw.products import snr_output_path
from skysim_gw.catalogue import (
    catalog_path,
    get_gwdali_source,
    load_gwdali_catalogue,
)
from skysim_gw.detectors import get_lvk_detectors
from skysim_gw.results import (
    find_matrix,
    unwrap_gwdali_result,
)


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

from skysim_gw.settings.localize import (
    BASE_DIR,
    INPUT_DIR,
    SNR_DIR,
    OUTPUT_DIR,
    SOURCE_TYPE,
    ZI,
    ZF,
    APPROXIMANTS,
    METHOD,
    FMIN_HZ,
    FMAX_HZ,
    FSIZE,
    FISHER_RCOND,
    MAX_NEW_SOURCES,
    CHECKPOINT_EVERY,
    PRINT_EVERY,
    RESUME,
)


FREE_PARAMS = (
    "RA",
    "Dec",
    "inv_dL",
    "Mc",
    "eta",
    "iota",
    "psi",
    "phi_coal",
)


# Production:
# MAX_NEW_SOURCES = None
# CHECKPOINT_EVERY = 5
# PRINT_EVERY = 1
# RESUME = True


# -----------------------------------------------------------------------------
# Output structure
# -----------------------------------------------------------------------------

RESULT_DTYPE = np.dtype([
    ("source_index", "i8"),
    ("snr_network", "f8"),
    ("sky_area_90_deg2", "f8"),
    ("sigma_ra_deg", "f8"),
    ("sigma_dec_deg", "f8"),
    ("correlation_ra_dec", "f8"),
    ("fisher_condition_number", "f8"),
    ("fisher_used_pseudoinverse", "?"),
    ("runtime_s", "f8"),
    ("success", "?"),
    ("error_message", "U512"),
])


from skysim_gw.products import localization_output_path


def empty_results(number_of_sources, snr_results):

    results = np.zeros(
        number_of_sources,
        dtype=RESULT_DTYPE,
    )

    results["source_index"] = np.arange(number_of_sources)
    results["snr_network"] = snr_results["snr_network"]

    for name in (
        "sky_area_90_deg2",
        "sigma_ra_deg",
        "sigma_dec_deg",
        "correlation_ra_dec",
        "fisher_condition_number",
        "runtime_s",
    ):
        results[name] = np.nan

    fisher = np.full(
        (
            number_of_sources,
            len(FREE_PARAMS),
            len(FREE_PARAMS),
        ),
        np.nan,
    )

    covariance = np.full_like(
        fisher,
        np.nan,
    )

    completed = np.zeros(
        number_of_sources,
        dtype=bool,
    )

    return results, fisher, covariance, completed


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

def save_checkpoint(
    path,
    results,
    fisher,
    covariance,
    completed,
    metadata,
):

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}_",
        suffix=".npz",
    )

    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:

        np.savez_compressed(
            temporary_path,
            results=results,
            fisher=fisher,
            covariance=covariance,
            completed=completed,
            free_params=np.asarray(FREE_PARAMS),
            **{
                name: np.asarray(value)
                for name, value in metadata.items()
            },
        )

        temporary_path.replace(path)

    finally:

        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    path,
    expected_size,
    metadata,
):

    with np.load(
        path,
        allow_pickle=False,
    ) as saved:

        results = saved["results"]
        fisher = saved["fisher"]
        covariance = saved["covariance"]
        completed = saved["completed"].astype(bool)

        expected_matrix_shape = (
            expected_size,
            len(FREE_PARAMS),
            len(FREE_PARAMS),
        )

        if (
            len(results) != expected_size
            or len(completed) != expected_size
        ):
            raise ValueError(
                "Localization checkpoint has the wrong row count"
            )

        if (
            fisher.shape != expected_matrix_shape
            or covariance.shape != expected_matrix_shape
        ):
            raise ValueError(
                "Localization checkpoint has the wrong matrix shape"
            )

        if tuple(saved["free_params"].tolist()) != FREE_PARAMS:
            raise ValueError(
                "Checkpoint FreeParams differ from current FreeParams"
            )

        for name, expected in metadata.items():

            if name not in saved.files:
                raise KeyError(
                    f"Checkpoint is missing metadata {name!r}"
                )

            actual = saved[name].item()

            if isinstance(expected, float):
                matches = np.isclose(
                    actual,
                    expected,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            else:
                matches = actual == expected

            if not matches:
                raise ValueError(
                    f"Checkpoint {name}={actual!r} "
                    f"differs from {expected!r}"
                )

    return (
        results,
        fisher,
        covariance,
        completed,
    )


# -----------------------------------------------------------------------------
# Fisher diagnostics
# -----------------------------------------------------------------------------

def invert_fisher(fisher):
    """Invert Fisher matrix for conditioning diagnostics only."""

    fisher = 0.5 * (
        fisher + fisher.T
    )

    used_pseudoinverse = False

    try:
        covariance = np.linalg.inv(fisher)

    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(
            fisher,
            rcond=FISHER_RCOND,
        )
        used_pseudoinverse = True

    covariance = 0.5 * (
        covariance + covariance.T
    )

    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "Fisher inversion produced a non-finite covariance"
        )

    return covariance, used_pseudoinverse


# -----------------------------------------------------------------------------
# Localization summary
# -----------------------------------------------------------------------------

def localisation_summary(
    covariance,
    declination_deg,
):
    """Return Gaussianised sky errors from the Fisher covariance.

    RA and Dec are in degrees. Therefore their covariance block is in
    degree^2. The spherical Jacobian contributes cos(declination).

    The resulting area is the 90% credible ellipse of the local Gaussian
    Fisher approximation.
    """

    i_ra = FREE_PARAMS.index("RA")
    i_dec = FREE_PARAMS.index("Dec")

    sky_covariance = covariance[
        np.ix_(
            [i_ra, i_dec],
            [i_ra, i_dec],
        )
    ]

    determinant = float(
        np.linalg.det(sky_covariance)
    )

    if (
        determinant <= 0.0
        or not np.isfinite(determinant)
    ):
        raise ValueError(
            f"Invalid sky covariance determinant "
            f"{determinant}"
        )

    variance_ra = float(
        sky_covariance[0, 0]
    )

    variance_dec = float(
        sky_covariance[1, 1]
    )

    if (
        variance_ra <= 0.0
        or variance_dec <= 0.0
    ):
        raise ValueError(
            "RA or Dec marginalized variance "
            "is not positive"
        )

    sigma_ra = np.sqrt(
        variance_ra
    )

    sigma_dec = np.sqrt(
        variance_dec
    )

    correlation = float(
        sky_covariance[0, 1]
        / (sigma_ra * sigma_dec)
    )

    area_90 = (
        np.pi
        * (-2.0 * np.log(0.1))
        * abs(
            np.cos(
                np.radians(declination_deg)
            )
        )
        * np.sqrt(determinant)
    )

    return (
        area_90,
        sigma_ra,
        sigma_dec,
        correlation,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_arguments(argv=None):
    """Parse optional explicit product paths and a bounded smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--snr-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument(
        "--max-new-sources", type=int,
        help="Maximum new detected sources to localize in this run.",
    )
    limit.add_argument(
        "--all", action="store_true",
        help="Process every remaining detected source, overriding MAX_NEW_SOURCES.",
    )
    args = parser.parse_args(argv)
    explicit_paths = (args.input_file, args.snr_file, args.output_file)
    if any(path is not None for path in explicit_paths) and not all(
        path is not None for path in explicit_paths
    ):
        parser.error(
            "--input-file, --snr-file, and --output-file must be supplied together"
        )
    if args.max_new_sources is not None and args.max_new_sources <= 0:
        parser.error("--max-new-sources must be positive")
    return args


def main(argv=None):

    args = parse_arguments(argv)

    source_type = SOURCE_TYPE.upper()
    approximant = APPROXIMANTS[source_type]

    injection_path = catalog_path(
        INPUT_DIR,
        source_type,
        ZI,
        ZF,
    ) if args.input_file is None else args.input_file

    snr_path = snr_output_path(
        SNR_DIR,
        source_type,
        ZI,
        ZF,
    ) if args.snr_file is None else args.snr_file

    output_path = localization_output_path(
        OUTPUT_DIR,
        source_type,
        ZI,
        ZF,
    ) if args.output_file is None else args.output_file

    catalogue, _ = load_gwdali_catalogue(
        injection_path
    )

    with np.load(
        snr_path,
        allow_pickle=False,
    ) as snr_file:

        snr_results = snr_file["results"]
        snr_completed = snr_file[
            "completed"
        ].astype(bool)

    if len(catalogue) != len(snr_results):
        raise ValueError(
            "Injection and SNR catalogues "
            "have different lengths"
        )

    eligible = (
        snr_completed
        & snr_results["success"]
        & snr_results["detected"]
    )

    detectors = get_lvk_detectors(
        include_kagra=True
    )

    if METHOD != "Fisher":
        raise ValueError(
            "This stage is Fisher-only; set METHOD='Fisher' in "
            "skysim_gw.settings.localize"
        )

    metadata = {
        "source_type": source_type,
        "zi": float(ZI),
        "zf": float(ZF),
        "approximant": approximant,
        "method": METHOD,
        "run_sampler": False,
        "diff_method": "numdiff",
        "mass_parameterization": "Mc_eta_detector_frame",
        "distance_parameter": "inv_dL",
        "distance_parameter_unit": "Gpc^-1",
        "t_coal_fixed_s": 0.0,
        "fmin_hz": float(FMIN_HZ),
        "fmax_hz": float(FMAX_HZ),
        "fsize": int(FSIZE),
        "fisher_rcond": float(FISHER_RCOND),
        "input_catalog": str(injection_path),
        "snr_catalog": str(snr_path),
        "stored_ra_dec_unit": "radian",
        "gwdali_ra_dec_unit": "degree",
        "sky_covariance_unit": "degree_squared",
        "coordinate_convention":
            "gwdali_ra_dec_degrees_v1",
        "saved_fisher":
            "Fisher information matrix",
        "saved_covariance":
            "Inverse Fisher matrix (Gaussian covariance)",
    }

    if RESUME and output_path.is_file():

        (
            results,
            fisher_all,
            covariance_all,
            completed,
        ) = load_checkpoint(
            output_path,
            len(catalogue),
            metadata,
        )

        print(
            f"Resuming: {output_path}"
        )

    else:

        (
            results,
            fisher_all,
            covariance_all,
            completed,
        ) = empty_results(
            len(catalogue),
            snr_results,
        )

    pending = np.flatnonzero(
        eligible & ~completed
    )

    maximum_new_sources = (
        None if args.all
        else args.max_new_sources
        if args.max_new_sources is not None
        else MAX_NEW_SOURCES
    )
    if maximum_new_sources is not None:
        pending = pending[
            :int(maximum_new_sources)
        ]

    print(
        f"Injection catalogue: {injection_path}"
    )
    print(
        f"SNR catalogue: {snr_path}"
    )
    print(
        f"Output: {output_path}"
    )
    print(
        f"Catalogue sources: {len(catalogue):,}"
    )
    print(
        f"Eligible detections: {int(eligible.sum()):,}"
    )
    print(
        f"Already localized: "
        f"{int((eligible & completed).sum()):,}"
    )
    print(
        f"Processing this run: {len(pending):,}"
    )
    print(
        f"FreeParams: {FREE_PARAMS}"
    )
    print(
        f"Method: {METHOD}; Gaussian covariance = inverse Fisher"
    )
    print(
        "Distance parameter: inv_dL [Gpc^-1]"
    )
    print(
        "Stored RA/Dec: radians; "
        "GWDALI RA/Dec: degrees"
    )

    run_start = time.perf_counter()
    since_checkpoint = 0

    for count, index in enumerate(
        pending,
        start=1,
    ):

        event_start = time.perf_counter()

        try:

            GwPrms = get_gwdali_source(
                catalogue,
                index,
                distance_parameter="inv_dL",
                mass_parameterization="mc_eta",
            )

            if not np.isclose(
                GwPrms["t_coal"],
                0.0,
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise ValueError(
                    "Expected fixed t_coal=0; "
                    f"received "
                    f"{GwPrms['t_coal']}"
                )

            raw_result = gw.GWDALI(
                GwPrms,
                detectors,
                list(FREE_PARAMS),
                approx=approximant,
                method=METHOD,
                diff_method="numdiff",
                run_sampler=False,

                # Keep False while validating.
                # This also avoids GWDALI redirecting
                # sys.stdout to /dev/null.
                hide_info=False,

                output_name=None,
                save_bilby_path=False,
                enable_jax_waveforms=False,
                fmin=FMIN_HZ,
                fmax=FMAX_HZ,
                fsize=FSIZE,
            )

            # Fisher covariance used as the Gaussian localization posterior.

            tensors = unwrap_gwdali_result(
                raw_result
            )

            _, fisher = find_matrix(
                tensors,
                (
                    "Fisher",
                    "fisher",
                    "FisherMatrix",
                    "fisher_matrix",
                ),
            )

            expected_shape = (
                len(FREE_PARAMS),
                len(FREE_PARAMS),
            )

            if (
                fisher is None
                or fisher.shape != expected_shape
            ):
                raise ValueError(
                    "GWDALI did not return the "
                    "expected Fisher matrix"
                )

            covariance, fisher_used_pseudoinverse = invert_fisher(fisher)

            (
                area_90,
                sigma_ra,
                sigma_dec,
                correlation,
            ) = localisation_summary(
                covariance,
                GwPrms["Dec"],
            )

            # -------------------------------------------------------------
            # Save results
            # -------------------------------------------------------------

            fisher_all[index] = fisher
            covariance_all[index] = covariance

            results[
                "sky_area_90_deg2"
            ][index] = area_90

            results[
                "sigma_ra_deg"
            ][index] = sigma_ra

            results[
                "sigma_dec_deg"
            ][index] = sigma_dec

            results[
                "correlation_ra_dec"
            ][index] = correlation

            results[
                "fisher_condition_number"
            ][index] = np.linalg.cond(
                fisher
            )

            results[
                "fisher_used_pseudoinverse"
            ][index] = (
                fisher_used_pseudoinverse
            )

            results[
                "success"
            ][index] = True

            results[
                "error_message"
            ][index] = ""

        except Exception as error:

            results[
                "success"
            ][index] = False

            results[
                "error_message"
            ][index] = (
                f"{type(error).__name__}: "
                f"{error}"
            )[:512]

        results[
            "runtime_s"
        ][index] = (
            time.perf_counter()
            - event_start
        )

        completed[index] = True
        since_checkpoint += 1

        # -------------------------------------------------------------
        # Compact progress output
        # -------------------------------------------------------------

        if (
            count % PRINT_EVERY == 0
            or count == len(pending)
        ):

            elapsed = (
                time.perf_counter()
                - run_start
            )

            status = (
                "OK"
                if results["success"][index]
                else "FAILED"
            )

            area = results[
                "sky_area_90_deg2"
            ][index]

            print(
                f"processed={count:,}/{len(pending):,} "
                f"index={index:,} "
                f"status={status} "
                f"area90={area:.6g} deg2 "
                f"event="
                f"{results['runtime_s'][index]:.2f}s "
                f"elapsed={elapsed/60:.2f}min",
                flush=True,
            )

            if not results["success"][index]:

                print(
                    "  error: "
                    f"{results['error_message'][index]}",
                    flush=True,
                )

        # -------------------------------------------------------------
        # Checkpoint
        # -------------------------------------------------------------

        if since_checkpoint >= CHECKPOINT_EVERY:

            save_checkpoint(
                output_path,
                results,
                fisher_all,
                covariance_all,
                completed,
                metadata,
            )

            since_checkpoint = 0

    # Final checkpoint
    save_checkpoint(
        output_path,
        results,
        fisher_all,
        covariance_all,
        completed,
        metadata,
    )

    localized = (
        eligible & completed
    )

    successful = (
        localized
        & results["success"]
    )

    print(
        "\nCheckpoint summary"
    )

    print(
        f"Localized attempts: "
        f"{int(localized.sum()):,}"
    )

    print(
        f"Successful: "
        f"{int(successful.sum()):,}"
    )

    print(
        f"Failed: "
        f"{int((localized & ~results['success']).sum()):,}"
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
