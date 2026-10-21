# SkySim5000 GW catalogue: review guide

This guide describes the full scientific data flow, including the optional
upstream construction of the seed-host catalogue. The normal portable run
begins at step 2, with an existing seed-host HDF5 file.

## The pipeline at a glance

```text
0. Choose merger-rate density R_type(z, log10 Mstar)
   -> expected and Poisson-drawn counts in (true-z, log10 Mstar) bins
1. Assign the drawn targets to SkySim5000 galaxies
   -> possible-host HDF5 catalogue
2. Sample binary and extrinsic parameters for each host
   -> GWDALI injection NPZ catalogue
3. Calculate network SNR and apply the detection rule
   -> SNR catalogue and detected original-row indices
4. Localise detected sources (Fisher and/or Doublet)
   -> covariance summaries and/or per-source posterior samples
5. Project localisations to HEALPix and compare with galaxy slices
   -> maps, areas, and overlays
```

Steps 0--1 build the source population; they do not simulate detection.
Steps 2--5 turn that possible-host population into synthetic injections,
detections, and localisation products.

## Step 0: merger-rate density and target counts

The current implementation is `skysim_gw.rates`, run through:

```bash
python -m skysim_gw rate-targets --config /path/gw_rate_config.json
```

It evaluates a source-frame merger-rate density in units of
`Gpc^-3 yr^-1 dex^-1` at the midpoint of every true-redshift and stellar-mass
bin. For a bin centred at `z` and `log10(Mstar/Msun)`, it calculates

```text
lambda = R(z, logM) [dVc/(dz dOmega)](z) Omega dz dlogM Tobs / (1 + z).
N ~ Poisson(lambda).
```

Here `Omega` is the SkySim5000 footprint in steradians, `Tobs` is the
observer-frame duration, and `1/(1+z)` converts the source-frame rate into an
observer-frame count. This is midpoint integration, not an integral evaluated
with a finer quadrature grid. The output HDF5 retains both `expected/<TYPE>`
and `drawn/<TYPE>` arrays, as well as bin edges, the configuration JSON, rate
parameters, cosmology, and random-seed provenance.

### Built-in models

`rates.py` always calculates all three source classes:

- `BBH` and `BHNS`: a redshift factor multiplied by a smooth stellar-mass
  window plus a Gaussian stellar-mass component.
- `BNS`: a distinct stellar-mass-dependent expression.
- Parameters are the supplied Menote et al. (2025) Table 1 values stored as
  `P_BBH`, `P_BHNS`, and `P_BNS`.
- Cosmology is `FlatLambdaCDM(H0=71 km s^-1 Mpc^-1, Om0=0.265)`.

`s22_bbh_empirical` is also available for BBHs. It reads the S22 released
catalogues, marginalizes their rate-weighted host grid over SFR, and supplies
an empirical `R(z, logMstar)` to the same target grid. BHNS and BNS continue
to use their Menote templates. The S22 catalogue directory and its M54/M57,
A1/A3/A5 selection are explicit JSON model parameters and are retained in the
target-file provenance.

The required config key `model_name` selects the BBH rate function. `draw_mode`
presently accepts only `independent_by_type`, meaning independent
Poisson draws for BBH, BNS, and BHNS using one reproducible NumPy generator.

### Decisions to record before a production run

1. **Rate model and parameters.** Use the supplied fixed Menote parameter set,
   or provide an explicitly named alternate parameter set. Record a citation,
   parameter values, units, and the stellar-mass definition expected by the
   model.
2. **Population mix.** Retain all BBH/BHNS/BNS classes, or limit the target
   stage and downstream `SOURCE_TYPES` consistently. The current target code
   always writes all three, while later stages may select a subset.
3. **Count realisation.** The current supported choice is an independent
   Poisson realisation. For deterministic sensitivity tests, add an explicit
   `expected`/Asimov mode rather than rounding counts silently.
4. **Survey and binning.** Fix footprint area, observer-frame duration,
   true-redshift slices, stellar-mass edges, and the adopted cosmology. These
   jointly control the count normalisation.
5. **Random provenance.** Preserve `poisson_seed`; use a new output filename
   or directory for any changed scientific setting.

The rate-target JSON must include:

```json
{
  "model_name": "menote_2025_table1",
  "redshift_slices_file": "/path/redshift_slices.info",
  "output_file": "/path/gw_rate_targets.hdf5",
  "area_deg2": 1.0,
  "observing_years": 1.0,
  "log_stellar_mass_min": 7.0,
  "log_stellar_mass_max": 12.0,
  "dlog_stellar_mass": 0.1,
  "poisson_seed": 12345,
  "draw_mode": "independent_by_type",
  "overwrite": false
}
```

The numerical values above are a schema example only; the validated area,
duration, mass range, and bin width were not supplied with this repository.

## Step 1: assign targets to SkySim5000 hosts

`assign-hosts` consumes `drawn/<TYPE>` from step 0 and must run where the full
SkySim5000 catalogue, GCRCatalogs, MPI, and NERSC `lsst` access are available.
Within each `(true-z, logMstar)` bin, it selects candidates by deterministic
64-bit hash priority of galaxy ID. The selection is uniform without
replacement while enough candidates exist. If a target bin is undersupplied,
it fills the shortfall by sampling those candidates with replacement and marks
the affected records and `replacement/<TYPE>` counts in the output HDF5.

Review this replacement fraction before treating the output as a host sample:
it measures whether the requested rate targets are supportable by the simulated
galaxy population at the chosen binning.

## Steps 2--5: portable GW pipeline

| Step | Code/stage | Input | Output and review point |
|---|---|---|---|
| 2 | `prepare` / `population.py` | `/sources/<TYPE>` host groups | Per-type GWDALI NPZ. Samples masses, spins and extrinsic angles; uses host true redshift for detector-frame mass and luminosity distance. |
| 3a | `snr` | Injection NPZ | Checkpointed detector and network SNR catalogue. |
| 3b | `detect` | SNR catalogue | Original injection-row indices passing completion, success, network-SNR, and detector-multiplicity cuts. |
| 4 | `doublet-serial` or `doublet-parallel` | Detected rows, injections, SNR | Fisher/Doublet summaries or actual posterior samples. Match products by `source_index`, never detected-list position. |
| 5 | `fisher-maps` or `doublet-maps` | Localisation products plus injections | HEALPix maps. Assign events to redshift slices from injected `redshift_true`, not inferred distance. |

The prepared injection NPZ preserves `redshift_true`, sky coordinates, and
source-frame masses. Its GWDALI angles are stored in radians; the SNR and
localisation interfaces convert RA/Dec to the degree convention used by
GWDALI. This unit boundary is worth retaining in any methods description.

### BBH intrinsic-mass options during preparation

`PopulationConfig` uses the phenomenological `bimodal_pairing` sampler by
default. Its `BBH_MASS_PARAMETERS` in `skysim_gw/settings/population.py` can override
the low/high truncated-normal component parameters and the unnormalised
`bbh_pairing_weight_ll`, `bbh_pairing_weight_lh`, and
`bbh_pairing_weight_hh` weights.  This model draws the LL, LH, or HH pairing
class before drawing component masses, so a bimodal component distribution can
produce LL, LH, and HH chirp-mass features.  The defaults suppress LH but are
not an LVK fit or a replacement for population synthesis; record every
overridden parameter in forecast products.

## Recommended next implementation choices

If multiple rate options are needed, make them explicit in the JSON and in
the output provenance rather than overloading `model_name`. A minimal,
auditable design is:

```text
rate_model: "menote_table1" | "menote_custom"
source_types: ["BBH", "BHNS", "BNS"]
rate_parameters: { ... }        # required for menote_custom
draw_mode: "independent_poisson" | "expected_counts"
```

The first option preserves the present science. The second supplies an
alternate parameter set with the same functional forms. A genuinely different
functional form should be a separately named model and receive its own tests
and reference, rather than being squeezed into the existing parameter keys.

Before implementation, decide whether an alternate model must change only the
rate normalisation/evolution or also the stellar-mass dependence; the latter
changes the host population, not merely the total event count.

## BBH comparison models

The rate-target stage now accepts a `model_name` which selects the BBH density
while retaining the existing Menote BNS and BHNS density models.  The available
BBH choices are `menote_bbh`, `menote_heaviside_scaled_bbh`,
`menote_heaviside_scaled_matched_total_bbh`,
`flat_bbh_fixed_z`, `gaussian_bbh_fixed_z`, `bimodal_bbh_fixed_z`, and
`bimodal_bbh_low10_fixed_z`, and `s22_bbh_empirical`.

`s22_bbh_empirical` requires `bbh_model_parameters.data_dir`, pointing at the
S22 BBH text catalogues. It uses the SFR-marginalized rate for the existing
two-dimensional Seed workflow; optional `rate_scale`, `mass_tilt`, and
`sfr_tilt` parameters modify its normalization and empirical host shape.

The flat curve is normalized to the integral of the Menote BBH density at
`z=1` over `7 <= log10(Mstar/Msun) <= 12.5`. The Gaussian and bimodal curves
instead use direct peak amplitudes, so they can be compared without forcing an
identical total event abundance.

The A20a 1D and 3D helpers remain available for future per-galaxy host
weighting. The narrow Gaussian is centred on the Menote Gaussian component and
has default width 0.38 dex. The bimodal model retains that high-mass Gaussian and adds a
low-mass Gaussian centred at log10(Mstar/Msun)=8.7. By default its peak is
half the Menote z=1 peak. Supply `bbh_model_parameters` in the target JSON to
change `peak_amplitude`/`sigma_dex` for the Gaussian or
`low_peak_amplitude`, `low_sigma_dex`, `high_peak_amplitude`, and
`high_sigma_dex` for the bimodal model.

`bimodal_bbh_low10_fixed_z` has the same form but defaults to a low-mass peak
of 10% of the Menote z=1 peak; its amplitude can be overridden with the same
`low_peak_amplitude` parameter.

`menote_heaviside_scaled_bbh` exposes the low/mid-mass Menote contribution
directly: supply `bbh_model_parameters: {"heaviside_scale": f_H}`. At
`f_H=0` it is the Menote Gaussian component alone; at `f_H=1` it exactly
reproduces `menote_bbh`. The standard Menote `psi(z)` factor applies to both
terms. The discontinuous A20a-low-mass splice is retained only as a deprecated
internal helper and is not selectable.

`menote_heaviside_scaled_matched_total_bbh` uses the same `heaviside_scale`
parameter but applies a redshift-dependent factor `C(z)`. It preserves the
full Menote BBH rate integrated over `7 <= log10(Mstar/Msun) <= 12.5` at every
redshift, so `f_H=0` and `f_H=0.1` alter the host-mass profile without changing
the expected total BBH abundance.
