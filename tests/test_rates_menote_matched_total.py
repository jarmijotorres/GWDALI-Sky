import numpy as np

from skysim_gw.rates import (
    BBH_MODEL_LOG_MASS_MAX,
    BBH_MODEL_LOG_MASS_MIN,
    rate_bbh,
    rate_bbh_menote_heaviside_scaled_matched_total,
)


def test_matched_heaviside_models_preserve_menote_mass_integral():
    masses = np.linspace(BBH_MODEL_LOG_MASS_MIN, BBH_MODEL_LOG_MASS_MAX, 8193)
    redshifts = np.array([0.0, 0.5, 1.0, 2.0])
    reference = np.trapz(rate_bbh(redshifts[:, None], masses), masses, axis=1)

    for scale in (0.0, 0.1, 1.0, 2.0):
        matched = rate_bbh_menote_heaviside_scaled_matched_total(
            redshifts[:, None], masses, heaviside_scale=scale
        )
        integral = np.trapz(matched, masses, axis=1)
        np.testing.assert_allclose(integral, reference, rtol=2e-6)


def test_matched_heaviside_scale_one_is_menote():
    redshifts = np.array([0.0, 0.5, 1.0, 2.0])
    masses = np.linspace(7.0, 12.5, 101)
    np.testing.assert_allclose(
        rate_bbh_menote_heaviside_scaled_matched_total(
            redshifts[:, None], masses, heaviside_scale=1.0
        ),
        rate_bbh(redshifts[:, None], masses),
        rtol=2e-12,
    )
