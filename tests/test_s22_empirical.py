import numpy as np

from skysim_gw.s22 import (
    S22BBHEmpiricalModel,
    S22BBHParameters,
    S22_LOW_REDSHIFT_SNAPSHOTS,
)
from skysim_gw.rates import rate_functions_for_model


def _write_catalogue(directory, snapshot, rate):
    rows = []
    for log_mass in (9.5, 10.5):
        for log_sfr in (-1.5, -0.5):
            rows.append((10.0**log_mass, 10.0**log_sfr, -2.0, rate, 1.0))
    path = directory / f"HG_M57_A3_GSSI_v2_BBHs_{snapshot}_12Z_50_No_0.14_2.txt"
    np.savetxt(path, rows, header="Mstar SFR logZ nGW Type")


def _model(tmp_path):
    _write_catalogue(tmp_path, 28, rate=10.0)
    _write_catalogue(tmp_path, 19, rate=30.0)
    return S22BBHEmpiricalModel(
        tmp_path,
        snapshots={28: 0.0, 19: 1.0},
        mass_edges=np.array([9.0, 10.0, 11.0]),
        sfr_edges=np.array([-2.0, -1.0, 0.0]),
    )


def test_rate_per_galaxy_interpolates_s22_templates(tmp_path):
    model = _model(tmp_path)
    kwargs = dict(stellar_mass=10.0**9.5, sfr=10.0**-1.5)
    assert model.rate_per_galaxy(0.0, **kwargs) == 10.0
    assert model.rate_per_galaxy(1.0, **kwargs) == 30.0
    assert model.rate_per_galaxy(0.5, **kwargs) == 20.0


def test_rate_per_galaxy_applies_low_dimensional_tilts(tmp_path):
    model = _model(tmp_path)
    rate = model.rate_per_galaxy(
        0.5,
        stellar_mass=10.0**10.5,
        sfr=10.0**-0.5,
        parameters=S22BBHParameters(rate_scale=2.0, mass_tilt=1.0),
    )
    np.testing.assert_allclose(rate, 20.0 * 2.0 * 10.0**0.5)


def test_sample_hosts_stays_inside_template_support(tmp_path):
    model = _model(tmp_path)
    sample = model.sample_hosts(0.5, size=100, rng=np.random.default_rng(1234))
    assert set(sample) == {"stellar_mass", "sfr"}
    assert np.all((sample["stellar_mass"] >= 1.0e9) & (sample["stellar_mass"] <= 1.0e11))
    assert np.all((sample["sfr"] >= 1.0e-2) & (sample["sfr"] <= 1.0))


def test_s22_model_is_available_to_rate_target_pipeline(tmp_path):
    for snapshot in S22_LOW_REDSHIFT_SNAPSHOTS:
        _write_catalogue(tmp_path, snapshot, rate=10.0)
    functions = rate_functions_for_model(
        "s22_bbh_empirical", {"data_dir": str(tmp_path)}
    )
    base = functions["BBH"](0.5, 9.5)
    assert np.isfinite(base) and base > 0.0
    doubled = rate_functions_for_model(
        "s22_bbh_empirical", {"data_dir": str(tmp_path), "rate_scale": 2.0}
    )["BBH"](0.5, 9.5)
    np.testing.assert_allclose(doubled, 2.0 * base)
