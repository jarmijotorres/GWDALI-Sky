import numpy as np

from skysim_gw.host_properties import derive_host_properties


def sed_for_log_sfr(log_sfr):
    central_wavelength = 1246.0 + 0.5 * 306.0
    frequency_hz = 2.99792458e18 / central_wavelength
    return 10.0 ** (43.35 + log_sfr) / (4.4659e20 * frequency_hz)


def test_zero_extinction_unit_sfr_and_fmr_pivot():
    sed = sed_for_log_sfr(0.0)
    result = derive_host_properties({
        "sed_1246_306_disk_no_host_extinction": np.array([sed]),
        "A_v_disk": np.array([0.0]),
        "stellar_mass": np.array([1.0e10]),
    })
    np.testing.assert_allclose(result["log_sfr_uv_corrected"], [0.0])
    np.testing.assert_allclose(result["sfr_uv_corrected"], [1.0])
    np.testing.assert_allclose(result["fmr_mu"], [10.0])
    np.testing.assert_allclose(result["metallicity_fmr_12logoh"], [8.90])
    np.testing.assert_allclose(
        result["metallicity_fmr_log10_z"],
        [np.log10(0.0153) + 8.90 - 8.76],
    )
    assert result["fmr_sfr_clipped"].tolist() == [False]
    assert result["fmr_valid"].tolist() == [True]
    assert result["fmr_extrapolated"].tolist() == [False]


def test_no_host_extinction_sed_is_not_corrected_twice():
    sed = sed_for_log_sfr(0.0)
    data = {
        "sed_1246_306_disk_no_host_extinction": np.array([sed]),
        "A_v_disk": np.array([2.0]),
        "stellar_mass": np.array([1.0e10]),
    }
    intrinsic = derive_host_properties(data)
    corrected = derive_host_properties(
        data, apply_host_extinction_correction=True
    )
    np.testing.assert_allclose(intrinsic["log_sfr_uv_corrected"], [0.0])
    assert corrected["log_sfr_uv_corrected"][0] > 2.0


def test_mannucci2011_piecewise_fmr():
    result = derive_host_properties({
        "sed_1246_306_disk_no_host_extinction": np.array([
            sed_for_log_sfr(1.0), sed_for_log_sfr(0.0)
        ]),
        "stellar_mass": np.array([1.0e10, 1.0e9]),
    })
    # High-mu two-dimensional surface: m=0, s=1.
    np.testing.assert_allclose(
        result["metallicity_fmr_12logoh"][0], 8.90 - 0.14 - 0.054
    )
    # Low-mu extension: mu=9, so 8.93 + 0.51*(9-10).
    np.testing.assert_allclose(
        result["metallicity_fmr_12logoh"][1], 8.42
    )


def test_fmr_uses_s22_passive_sfr_floor_and_flags_it():
    result = derive_host_properties({
        "sed_1246_306_disk_no_host_extinction": np.array([
            sed_for_log_sfr(-6.0), sed_for_log_sfr(-3.0)
        ]),
        "stellar_mass": np.array([1.0e10, 1.0e10]),
    })
    np.testing.assert_allclose(result["log_sfr_uv_corrected"], [-6.0, -3.0])
    np.testing.assert_allclose(result["log_sfr_fmr"], [-4.0, -3.0])
    assert result["fmr_sfr_clipped"].tolist() == [True, False]
    assert result["fmr_extrapolated"].tolist() == [False, False]


def test_invalid_sed_does_not_receive_an_sfr_floor():
    result = derive_host_properties({
        "sed_1246_306_disk_no_host_extinction": np.array([0.0, np.nan]),
        "A_v_disk": np.array([0.0, 0.0]),
        "stellar_mass": np.array([1.0e10, 1.0e10]),
    })
    assert np.all(np.isnan(result["sfr_uv_corrected"]))
    assert np.all(np.isnan(result["metallicity_fmr_12logoh"]))
    assert np.all(np.isnan(result["metallicity_fmr_log10_z"]))
    assert result["fmr_valid"].tolist() == [False, False]
