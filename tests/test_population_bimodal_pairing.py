import numpy as np

from skysim_gw.population import PopulationConfig, sample_bbh_masses


def test_bimodal_pairing_returns_ordered_component_masses():
    config = PopulationConfig(
        bbh_mass_model="bimodal_pairing",
        bbh_pairing_weight_ll=1.0,
        bbh_pairing_weight_lh=0.2,
        bbh_pairing_weight_hh=1.0,
    )
    m1, m2 = sample_bbh_masses(np.random.default_rng(1234), 2_000, config)

    assert m1.shape == (2_000,)
    assert np.all(m1 >= m2)
    assert np.all(m2 >= config.bbh_low_min_msun)
    assert np.all(m1 <= config.bbh_high_max_msun)


def test_bimodal_pairing_rejects_invalid_weights():
    config = PopulationConfig(
        bbh_mass_model="bimodal_pairing",
        bbh_pairing_weight_ll=0.0,
        bbh_pairing_weight_lh=0.0,
        bbh_pairing_weight_hh=0.0,
    )

    try:
        sample_bbh_masses(np.random.default_rng(1234), 1, config)
    except ValueError as error:
        assert "pairing weights" in str(error)
    else:
        raise AssertionError("Expected invalid pairing weights to raise ValueError")
