import numpy as np

from skysim_gw.host_properties import derive_host_properties


def test_zero_extinction_unit_sfr_and_fmr_pivot():
    sed = 10.0**43.35 / (306.0 * 3.846e33)
    result = derive_host_properties({
        "sed_1246_306_disk_no_host_extinction": np.array([sed]),
        "A_v_disk": np.array([0.0]),
        "stellar_mass": np.array([1.0e10]),
    })
    np.testing.assert_allclose(result["log_sfr_uv_corrected"], [0.0])
    np.testing.assert_allclose(result["sfr_uv_corrected"], [1.0])
    np.testing.assert_allclose(result["fmr_mu"], [10.0])
    np.testing.assert_allclose(result["metallicity_fmr_12logoh"], [8.90])
    assert result["fmr_valid"].tolist() == [True]
    assert result["fmr_extrapolated"].tolist() == [False]


def test_invalid_sed_does_not_receive_an_sfr_floor():
    result = derive_host_properties({
        "sed_1246_306_disk_no_host_extinction": np.array([0.0, np.nan]),
        "A_v_disk": np.array([0.0, 0.0]),
        "stellar_mass": np.array([1.0e10, 1.0e10]),
    })
    assert np.all(np.isnan(result["sfr_uv_corrected"]))
    assert np.all(np.isnan(result["metallicity_fmr_12logoh"]))
    assert result["fmr_valid"].tolist() == [False, False]
