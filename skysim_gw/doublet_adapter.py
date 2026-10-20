"""Adapter that reuses compute_gwdali_localization_catalogue.py."""
import importlib
import os
from pathlib import Path
import re
import numpy as np

_serial = None
_catalogue = None
_detectors = None
_snr_results = None
_input_catalogue_path = None
_snr_catalogue_path = None
_realization_tag = None


def realization_tag(path) -> str:
    """Extract the optional fh-model/realization tag from a population name."""
    path = Path(path)
    match = re.search(r"_gwdali_(fh[01]_r\d{4})_z", path.stem)
    return match.group(1) if match is not None else path.stem

def _load():
    global _serial, _catalogue, _detectors, _snr_results
    global _input_catalogue_path, _snr_catalogue_path, _realization_tag
    if _serial is not None:
        return
    _serial = importlib.import_module(os.environ.get(
        "GWDALI_SERIAL_MODULE", "skysim_gw.stages.localize"))
    _serial.FSIZE = int(os.environ.get("GWDALI_FSIZE", "3000"))
    input_override = os.environ.get("GWDALI_INPUT_FILE")
    _input_catalogue_path = (
        Path(input_override)
        if input_override is not None
        else _serial.catalog_path(
            _serial.INPUT_DIR, _serial.SOURCE_TYPE, _serial.ZI, _serial.ZF,
        )
    )
    _catalogue, _ = _serial.load_gwdali_catalogue(_input_catalogue_path)
    _detectors = _serial.get_lvk_detectors(include_kagra=True)
    snr_override = os.environ.get("GWDALI_SNR_FILE")
    _snr_catalogue_path = (
        Path(snr_override)
        if snr_override is not None
        else _serial.snr_output_path(
            _serial.SNR_DIR, _serial.SOURCE_TYPE, _serial.ZI, _serial.ZF,
        )
    )
    with np.load(_snr_catalogue_path, allow_pickle=False) as f:
        _snr_results = f["results"]
        snr_input_path = (
            Path(str(f["input_catalog"].item()))
            if "input_catalog" in f.files else None
        )
    if len(_catalogue) != len(_snr_results):
        raise ValueError("Population and SNR catalogues have different lengths")
    if (
        snr_input_path is not None
        and snr_input_path.resolve() != _input_catalogue_path.resolve()
    ):
        raise ValueError(
            "SNR metadata input catalogue does not match the supplied "
            "population catalogue"
        )
    _realization_tag = realization_tag(_input_catalogue_path)

def run_event(source_index: int) -> dict:
    """Run the existing Doublet/Nestle calculation for one catalogue row."""
    _load()
    s = _serial
    p = s.get_gwdali_source(_catalogue, int(source_index), distance_parameter="inv_dL")
    if not np.isclose(p["t_coal"], 0.0, rtol=0.0, atol=1.0e-15):
        raise ValueError(f"Expected fixed t_coal=0; got {p['t_coal']}")
    raw = s.gw.GWDALI(
        p, _detectors, list(s.FREE_PARAMS),
        approx=s.APPROXIMANTS[s.SOURCE_TYPE], method="Doublet",
        sampler="nestle",
        npoints=int(os.environ.get("GWDALI_NPOINTS", "3000")),
        diff_method="numdiff", run_sampler=True,
        hide_info=False, output_name=None, save_bilby_path=False,
        enable_jax_waveforms=False, fmin=s.FMIN_HZ, fmax=s.FMAX_HZ, fsize=s.FSIZE)
    tensors = s.unwrap_gwdali_result(raw)
    _, fisher = s.find_matrix(tensors, ("Fisher", "fisher", "FisherMatrix", "fisher_matrix"))
    if fisher is None or fisher.shape != (len(s.FREE_PARAMS),) * 2:
        raise ValueError("GWDALI did not return the expected Fisher matrix")
    _, pinv = s.invert_fisher(fisher)
    samples = s.get_doublet_samples(raw)  # raw[0][0]: actual Nestle posterior samples
    covariance = np.cov(samples, rowvar=False)
    covariance = 0.5 * (covariance + covariance.T)
    area, sr, sd, corr = s.localisation_summary(covariance, p["Dec"])
    return {"samples": samples, "fisher": fisher, "covariance": covariance,
            "input_catalog": str(_input_catalogue_path),
            "snr_catalog": str(_snr_catalogue_path),
            "realization_tag": _realization_tag,
            "snr_network": _snr_results[int(source_index)]["snr_network"],
            "sky_area_90_deg2": area,
            "sigma_ra_deg": sr, "sigma_dec_deg": sd,
            "correlation_ra_dec": corr,
            "fisher_condition_number": np.linalg.cond(fisher),
            "fisher_used_pseudoinverse": pinv}
