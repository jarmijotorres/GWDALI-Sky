# Methodology plan: BBH host demographics, GW bias, and peak-siren cosmology

Status: working methodology and implementation record, updated 2026-11-05.
This document separates the agreed scientific design, the components already
implemented, and the choices that remain open. It does not report completed
measurements.

## 0. Overview for collaborators

### 0.1 Research question

We want to determine how the galaxy populations hosting binary black hole
(BBH) mergers set the large-scale clustering bias of gravitational-wave (GW)
sources, and how uncertainty in that host connection propagates into
cross-correlation or “peak-siren” constraints on the distance–redshift
relation, especially H0.

The analysis connects four layers:

```text
SkySim5000 galaxy lightcone
        |
        | Mstar, SED-derived SFR, FMR metallicity, position, redshift
        v
Santoliquido et al. (2022) BBH host-rate model
        |
        | per-galaxy merger weights and Poisson event counts
        v
intrinsic and O5-detected GW populations
        |
        | GWDALI selection and sky/distance localization
        v
GW–galaxy clustering, BBH bias, and peak-siren cosmology
```

The primary scientific contribution is the galaxy/LSS part of this chain:
how stellar mass, star-formation activity, and metallicity determine the
mixture of BBH hosts and therefore the effective GW clustering bias.

### 0.2 Baseline model choices

The current baseline uses:

- the SkySim5000 lightcone, covering approximately 5000 deg2;
- the Santoliquido et al. (2022, S22) BBH host catalogues with the M57
  metallicity relation and common-envelope model alpha_CE=3;
- direct linear interpolation between the S22 fixed-redshift snapshots,
  without forcing their redshift evolution to follow the Menote functional
  template;
- a compact four-dimensional S22 histogram in
  `(redshift, logMstar, logSFR, raw S22 logZ)` rather than repeatedly reading
  the released galaxy catalogues;
- UV SFR inferred galaxy by galaxy from the integrated rest-frame AB
  luminosity in the SkySim `sed_1246_306_disk_no_host_extinction` chunk. This
  field is already free of host attenuation, so no second `A_v_disk`
  correction is applied;
- deterministic Mannucci et al. (2011) equation-2 FMR oxygen abundance
  calculated from each galaxy’s stellar mass and inferred SFR, converted to
  the S22 absolute `log10(Z)` convention with `Z_sun=0.0153` and solar
  `12+log10(O/H)=8.76`;
- twelve demographic classes: 2 stellar-mass classes x 3 activity classes x
  2 FMR-metallicity classes;
- an LVK O5-like GWDALI detection/localization forecast.

### 0.3 Current implementation state

The following pieces are implemented:

1. The raw M57/alpha3 S22 catalogues can be reduced to a compact NPZ
   histogram containing summed BBH rates and galaxy occupancy.
2. `rate-targets` reads that histogram, marginalizes over SFR and S22 logZ,
   evaluates `R(z, logMstar)`, and Poisson-samples target event counts.
3. `assign-hosts` reads SkySim5000 once with MPI and calculates SFR and FMR
   metallicity for each candidate galaxy in each catalogue chunk.
4. The selected seed HDF5 retains galaxy ID, position, redshift, stellar and
   halo mass, inferred SFR, FMR abundance, validity flags, and provenance.
5. `event_oversample_factor=N` can generate a denser Monte Carlo realization
   of the same fixed SkySim lightcone. Each event receives weight `1/N` for
   physical-yield calculations.
6. Host properties and event weights propagate into the prepared GWDALI NPZ.

One important step is not yet implemented: within each `(z, logMstar)` target
cell, host assignment is still uniform through deterministic hash priorities.
The intended next change is to sample candidates proportionally to the S22
per-galaxy rate `lambda(z, Mstar, SFR[, logZ])`. Therefore, current catalogues
contain galaxy-level SFR/FMR information but are not yet SFR- or
metallicity-weighted host realizations.

### 0.4 Interpretation of oversampling

SkySim5000 contains far more galaxies than expected GW events. We exploit
`N_gal >> N_GW` by drawing an enlarged event sample from the same galaxy
field. This reduces GW-event shot noise when measuring clustering bias on the
one available simulation. It does not create independent cosmic volume,
reduce cosmic variance, or turn the 5000-deg2 lightcone into a full-sky
realization. Physical rates and yields use the stored event weight `1/N`;
high-density clustering diagnostics may use all oversampled tracers while
retaining the original footprint/cosmic-variance interpretation.

### 0.5 Metallicity convention

Two metallicity variables are retained explicitly:

- the S22 histogram coordinate is the raw simulation `logZ` column;
- the SkySim-derived FMR value is gas-phase `12 + log10(O/H)`, with a second
  stored field converted to absolute `log10(Z)` using the S22 solar reference.

The adopted mapping is `log10(Z) = log10(0.0153) + [12+log10(O/H)-8.76]`.
The inferred UV SFR remains unchanged in the catalogue, but the FMR evaluation
uses the S22 passive-galaxy floor `log10(SFR)=-4` and records whether this floor
was applied. A metallicity-resolved merger-rate weight remains a planned
sensitivity extension rather than the current baseline assignment.

## 1. Scientific objective

Determine how binary black hole (BBH) host demographics set the large-scale clustering bias of intrinsic and detected gravitational-wave (GW) sources, and how uncertainty in that connection affects peak-siren inference of H0 and Omega_m.

The framework combines the SkySim5000 synthetic galaxy lightcone (approximately 5000 deg^2, LSST-like photometry and galaxy physical properties, based on the cosmoDC2 framework), astrophysically motivated merger prescriptions, and GWDALI GW signal/parameter-estimation forecasts for an LVK O5 scenario.

The primary contribution is the galaxy/LSS connection: host demographics -> per-galaxy merger weights -> intrinsic and detected GW clustering -> cosmological precision and robustness. H0 and GW bias are primary targets; joint H0–Omega_m inference should be tested without promising a useful independent Omega_m constraint in advance.

## 2. Inputs and provenance

- SkySim5000: galaxy positions, true and photometric redshifts, stellar mass,
  photometry/SED chunks, and available halo/environment information. SFR is
  not taken as a ready-made catalogue truth in this workflow; it is inferred
  from a specified intrinsic UV SED chunk. Completeness and
  property-dependent clustering still require validation.
- Baseline merger-rate prescription: Santoliquido et al. (2022), M57,
  alpha_CE=3. The compact histogram stores rate sums and occupancy in
  `(snapshot, logMstar, logSFR, raw logZ)` cells. Menote rate functions remain
  useful comparators, not the baseline redshift evolution for the present
  low-redshift analysis.
- Metallicity is not a native observable used directly in the user's
  SkySim5000 workflow. The implemented FMR combines stellar mass and inferred
  SFR and returns gas-phase `12 + log10(O/H)` using Mannucci et al. (2011)
  equation 2, together with the S22-compatible absolute `log10(Z)`. Invalid or
  zero SED values remain invalid. The raw SFR is not floored; only the FMR
  evaluation uses the documented S22 passive-galaxy floor.
- GWDALI: generate/forecast BBH signals and joint sky-position, luminosity-distance, and chirp-mass likelihood/posterior information for a specified O5 detector configuration. Do not silently equate Fisher/DALI forecasts with exact parameter estimation.

Distinguish galaxy stellar mass M_star, Schechter characteristic mass M_knee,
binary chirp mass Mc, raw S22 logZ, and FMR oxygen abundance in code and
figures.

## 3. Agreed demographic grid: 2 mass x 3 activity x 2 metallicity

There are TWO stellar-mass bins, not six. The six mass–activity combinations are each subdivided into two metallicity classes, giving TWELVE possible classes.

### 3.1 Stellar mass: split at the fitted Schechter knee

Fit a double-Schechter mass function to a reliable, completeness-controlled parent population. In a common-knee convention, per unit mass:

phi(M) = exp(-M/M_knee)/M_knee * [phi1*(M/M_knee)^alpha1 + phi2*(M/M_knee)^alpha2].

Baseline classes:

- Below knee: M_min <= M_star < M_knee.
- Above knee: M_star >= M_knee, within the reliable catalog range.

The knee is a characteristic mass-function scale, not proof of two physically distinct populations. The fitted components overlap. This replaces an earlier tentative proposal of three fixed mass intervals; those intervals are NOT the current baseline.

A distinct optional diagnostic is the component crossover:
M_cross = M_knee * (phi2/phi1)^[1/(alpha1-alpha2)].
Do not confuse M_cross with M_knee or substitute it without documenting the scientific reason. It can be outside the well-constrained fit range.

Open choice: use a fixed knee from a reference-redshift fit, or an evolving M_knee(z). The former preserves a fixed physical boundary; the latter classifies relative to the evolving mass function. State which is used. Account for the conversion between mass functions per dM and per dlog10(M) when fitting/integrating.

### 3.2 Activity: three bins relative to the star-forming main sequence

Define:
Delta_MS = log10(SFR) - log10[SFR_MS(M_star,z)].

Equivalently, this is the offset in log sSFR from the main-sequence sSFR at the same mass/redshift. It controls for both mass dependence and redshift evolution, unlike a single fixed sSFR threshold.

Agreed baseline:

- Star-forming: Delta_MS > -0.5 dex.
- Transition: -1.0 < Delta_MS <= -0.5 dex.
- Passive: Delta_MS <= -1.0 dex.

Estimate the star-forming ridge from the unweighted parent galaxy population in redshift slices, fit its mass dependence (allow curvature where supported), and smooth across redshift. Do not use the median of all galaxies where passive systems dominate. Compare the recovered sequence with catalog construction prescriptions and an observational reference.

Classify zero-SFR objects separately before taking logarithms; verify whether zeros represent physical quiescence, numerical floors, or missing data. Test reasonable boundary variations (approximately 0.3 dex) and a documented fixed-sSFR alternative. These thresholds are analysis conventions.

### 3.3 Metallicity: deterministic FMR assignment, two absolute classes

For the baseline, assign Z_FMR = F(M_star,SFR,z) using the user's adopted deterministic FMR.

- Metal-poor: Z_FMR < Z_cut.
- Metal-rich: Z_FMR >= Z_cut.

Z_cut is NOT yet numerically agreed. Choose it using the metallicity dependence of the BBH prescription and the actual FMR calibration/range. Do not silently adopt solar metallicity or a conditional median.

Important agreed interpretation: deterministic metallicity is still a useful classification. It defines a joint boundary in the mass–SFR plane that is not necessarily resolved by the broad mass/activity bins. Two galaxies in the same broad mass/activity cell may lie on opposite sides of Z_cut. However, it adds no independent information at exactly fixed M_star, SFR, and z. Report results as FMR-selected metallicity populations; do not claim a separately measured metallicity/environment effect at fixed inputs.

The earlier suggestion to make metallicity solely a conditional model ingredient, and the earlier conditional-median metallicity split, are NOT the final agreed primary grid. A residual-metallicity split can be an optional diagnostic with a different interpretation.

Some of the 12 combinations can be sparse or empty because the FMR is deterministic. Report occupancy and merger contributions; do not force equal populations or invent galaxies to fill cells.

Check whether the FMR returns 12+log10(O/H), a mass fraction Z, or Z/Z_sun. State the abundance calibration, adopted solar reference, and conversion assumptions. Gas-phase oxygen abundance, total stellar metallicity, and BBH progenitor-birth metallicity are distinct quantities.

FMR extrapolation for passive/zero-SFR galaxies requires an explicit, validated treatment. Do not insert a tiny SFR floor and treat its extrapolated metallicity as established physics.

### 3.4 Explicit grid

1. Below knee, star-forming, metal-poor/rich.
2. Below knee, transition, metal-poor/rich.
3. Below knee, passive, metal-poor/rich.
4. Above knee, star-forming, metal-poor/rich.
5. Above knee, transition, metal-poor/rich.
6. Above knee, passive, metal-poor/rich.

Each row contains two classes. Retain continuous physical properties and merger weights within every class.

## 4. Per-galaxy merger assignment

### 4.1 Implemented two-stage normalization

The current pipeline separates the total event yield from the conditional
choice of hosts:

1. `rate-targets` evaluates the S22 histogram marginalized over SFR and raw
   logZ to obtain a volumetric `R(z, logMstar)` in
   `Gpc^-3 source-frame yr^-1 dex^-1`.
2. It converts this density into expected lightcone counts using comoving
   volume, footprint area, observing time, bin widths, and `1/(1+z)`, then
   draws Poisson counts.
3. `assign-hosts` uses those counts as the required number of events in each
   `(true-z, logMstar)` cell.
4. The next implementation step is to choose galaxies within each cell with
   probability proportional to the S22 per-galaxy rate, rather than uniformly.

This design preserves the S22 volumetric normalization and prevents accidental
double counting of galaxy number density. Within-cell per-galaxy weights are
conditional probabilities; they must be normalized over the available
SkySim candidates in that cell.

For the baseline metallicity-marginalized assignment:

```text
p(g | z-cell, Mstar-cell) = lambda_S22(z_g, Mstar_g, SFR_g)
                            / sum_candidates lambda_S22.
```

After a validated conversion between FMR oxygen abundance and the S22 raw
logZ convention is adopted, a metallicity-resolved extension can replace this
with `lambda_S22(z_g, Mstar_g, SFR_g, logZ_g)`.

Distinguish volumetric merger-rate density R(z) from a per-galaxy source-frame rate lambda_g [mergers / source-frame time]. A global R(z) alone does not identify the host distribution. Construct/calibrate lambda(M_star,SFR,Z,z) consistently with the adopted astrophysical prescription and verify that summing galaxy rates over comoving volume recovers its intended R(z).

Do not multiply a volumetric rate contribution by galaxy number density again. If only relative galaxy weights are available, document their normalization to R(z).

Current host properties do not directly specify BBH formation conditions. Delayed mergers can occur in passive, metal-rich hosts after formation in metal-poor progenitors. Use a prescription already incorporating histories/delay times, or explicitly model the mapping. Do not convolve delays twice if the supplied rate already describes mergers at the epoch of interest.

Optional metallicity sensitivity extension:
mean_lambda_g = integral dZ p(Z | M_star,SFR,z) lambda(M_star,SFR,Z,z).

This reduces to the deterministic FMR case for a delta distribution. Scatter matters for nonlinear merger efficiencies. Independently assigned scatter does not create an independently calibrated environmental dependence at fixed mass/SFR. Vary FMR calibration, scatter, and a mass–metallicity alternative as model-sensitivity tests.

For observing time T_obs, expected intrinsic event counts on the lightcone use lambda_g*T_obs/(1+z_g) when lambda_g is in source-frame time. Document Poisson sampling, any fixed-count experiment, and angular/volume boundaries.

## 5. Two complementary bias strategies

### 5.1 Intrinsic seeds, before detector selection

Select/simulate host seeds with probabilities proportional to per-galaxy merger rates. Measure their large-scale cross-correlation with a galaxy tracer whose bias is independently calibrated and held fixed during the GW-bias fit.

In narrow true-redshift bins and the deterministic linear-bias regime:
P_GW,gal(k,z) approximately b_GW(z)*b_gal(z)*P_mm(k,z).

Calibrate b_gal for the actual tracer selection; do not assume the same bias for every demographic subset. Use consistent masks, scale cuts, redshift kernels, and shot-noise treatment. If GW hosts are contained in the galaxy tracer catalog, account for their shared-object cross shot noise, or validate with a disjoint tracer construction. Holding b_gal fixed is an idealized baseline; propagate its uncertainty in realistic inference.

### 5.2 Independent rate-weighted prediction

For a mass interval [M_a,M_b] at fixed redshift:

b_GW([M_a,M_b],z) =
[integral_Ma^Mb dM n(M,z) mean_lambda(M,z) b_rate(M,z)] /
[integral_Ma^Mb dM n(M,z) mean_lambda(M,z)].

Here b_rate must be the appropriate merger-weighted conditional galaxy bias. Replacing it by an unweighted b_gal(M,z) is an approximation that can fail if rate and environment correlate within a mass interval.

The user's original integral from zero to M_star is a CUMULATIVE b_GW(<M_star), with the same upper limit in numerator and denominator; replace zero by the reliable lower mass limit in practice. Show cumulative curves as well as class summaries.

General expression for class C, with x=(M_star,Delta_MS,Z):
b_GW,C(z) = [integral_C dx n(x,z) lambda(x,z) b_gal(x,z)] /
             [integral_C dx n(x,z) lambda(x,z)].

For deterministic FMR, n(x,z) is supported on the FMR relation; evaluate this through galaxy sums or the corresponding conditional distribution, not an independent product n(M)n(SFR)n(Z).

Class merger fraction:
f_C^R(z) = sum_(g in C) lambda_g / sum_all_g lambda_g.

For exhaustive, disjoint classes at fixed redshift:
b_GW,total(z) = sum_C f_C^R(z)*b_GW,C(z).

Compare the analytic/catalog-weighted prediction with clustering of repeated intrinsic seed realizations. If prediction and measurement use the same fitted clustering inputs, describe the check as internal consistency rather than completely independent physical validation.

## 6. Detected sources and recovered bias

Keep three quantities distinct:

1. Intrinsic merger-rate-field bias.
2. Bias of the population selected by the O5 detector network.
3. Bias recovered from noisy/localized GW data and the selected LSST tracer sample.

For binary parameters theta:
w_g^det = lambda_g * integral dtheta p(theta | g,z) P_det(theta,z,sky_g).

For expected observed counts multiply by T_obs/(1+z). Include observing duty cycle and time dependence consistently. Model inclination/orientation, source/detector-frame masses, detector sensitivity, detection criterion, and network configuration.

Host-dependent binary mass distributions can couple metallicity to detectability. A universal binary population is a useful control, but excludes this mechanism. GWDALI's redshifted chirp mass must not be confused with source-frame mass.

Generate/use joint sky-distance-mass information, preserving relevant correlations. Validate approximate likelihoods and their priors for representative detections. Localized GW maps introduce angular/radial smoothing, which must be modelled rather than interpreted as a change in intrinsic bias.

## 7. Experimental comparisons

Run both:

- Physical-yield forecasts: retain the rate normalization and detector selection, allowing event counts and n(z) to differ.
- Controlled comparisons: match counts and, where feasible, redshift distributions across host models to isolate clustering effects.

Within demographic comparisons, show natural populations and comparisons at matched stellar mass/redshift over shared support. The latter tests activity dependence beyond mass composition. Do not automatically match metallicity away when studying the full activity–metallicity connection.

## 8. Peak-siren cosmology and comparator

Construct galaxy maps in redshift shells and GW maps in luminosity-distance shells, including selection and localization kernels. Infer the distance–redshift relation from the cross-correlation structure and fit H0/Omega_m with an explicit cosmological model.

Host models may change amplitude and precision more strongly than peak location, but test rather than assume this. Finite bins, evolving bias/rates, photo-z errors, and selection can couple amplitudes to the recovered peak.

Truth-versus-analysis tests:

- Physical host weights with the correct bias/population model: baseline recovery.
- Same mock with constant or incorrect b_GW(z): mismodelling sensitivity.
- Alternative host/FMR prescriptions with flexible bias marginalization: robustness and precision cost.
- Identical mock data analysed with a galaxy-catalog dark-siren baseline: method comparison.

Use the same events, galaxy selection, priors where comparable, and consistent GW/galaxy completeness corrections. Do not add a spectral-siren comparison unless separately justified and implemented. Quantify posterior shifts, uncertainties, interval coverage, and correlation signal-to-noise across realizations. Show prior sensitivity for Omega_m and degeneracies with galaxy/GW bias.

## 9. LSST Y1/Y5 and finite footprint

Use distinct Y1 and Y5 photometric noise, depth/completeness, redshift PDFs, calibration shifts, and outlier models. These are galaxy-survey scenarios paired with a defined O5 exposure, not automatically simultaneous observing schedules.

SkySim5000 covers approximately 12% of the full sky. Label results as 5000-deg^2 overlap forecasts unless extrapolation is explicitly justified. Model detector angular selection, survey masks, and events/posterior support crossing footprint boundaries consistently.

Compare true-property classes with observationally inferred classifications, including correlated mass/SFR/photo-z errors. FMR-derived metallicity errors inherit these correlations. Treat property-selected samples as potential observables only to the extent the photometry supports their classification.

Repeated GW realizations on one sky quantify event sampling variance, not independent cosmic variance. Assess covariance limitations and finite-volume effects. Validate property-dependent galaxy clustering, including relevant halo occupation/environment behavior, rather than assuming realistic photometry guarantees it.

## 10. Implementation sequence and deliverables

1. **Implemented:** audit the existing repository workflow and establish data
   provenance for targets, hosts, injections, SNR, and localization products.
2. **Implemented:** compact the S22 M57/alpha3 catalogues into a portable 4D
   rate/occupancy histogram and retain direct redshift interpolation.
3. **Implemented:** calculate UV SFR and deterministic FMR abundance galaxy by
   galaxy during the MPI SkySim scan; propagate these properties into GW
   products.
4. **Implemented:** support event oversampling on the fixed lightcone with
   explicit event weights and provenance.
5. **Next:** replace uniform within-cell hash selection by S22 per-galaxy
   rate-weighted selection and verify recovery of the target volumetric rate.
6. Establish completeness ranges, redshift bins, the main-sequence fit, the
   double-Schechter fit, and the abundance conversion/calibration.
7. Set and record the M_knee convention and Z_cut; create the 12 class labels
   with occupancy and boundary tests.
8. Produce galaxy and merger fractions and distributions in
   Mstar/Delta_MS/Z/redshift, including cumulative merger contributions.
9. Compute intrinsic weighted-bias predictions and validate them against
   repeated intrinsic seed cross-correlations on the fixed SkySim lightcone.
10. Apply O5 detection and GWDALI forecasts; compare intrinsic, detected, and
    recovered biases.
11. Implement Y1/Y5 observation models and peak-siren inference; run
    population-model mismatch and controlled-comparison tests.
12. Analyse identical mocks with the galaxy-catalog dark-siren comparator.
13. Deliver figures/tables for b_GW(z), class contributions, detection yields,
    correlation kernels, H0–Omega_m posteriors, systematic shifts, and
    recovery coverage. Release configurations and seeds where permitted.

The 12 demographic classes are a population decomposition, not a commitment to fit 12 independent GW-bias parameters. Choose the inference dimension based on the actual information content.

## 11. Anticipated scientific outcomes

If the methodology is validated, the central result is not simply a statement
that BBHs prefer a particular type of galaxy. It is a physical prediction for
the clustering bias of the GW population and an assessment of the resulting
systematics in peak-siren cosmology.

### 11.1 BBH clustering bias as an emergent host mixture

The merger-weighted contribution of each demographic class defines the
effective GW bias:

\[
b_{\rm GW}(z) = \sum_{k=1}^{12} f_k(z)\,b_k(z),
\]

where \(f_k(z)\) is the merger-weighted fraction of class \(k\) and
\(b_k(z)\) is its galaxy bias. This replaces an unconstrained GW-bias
nuisance parameter with a prediction based on host demographics.

An especially relevant outcome is that delayed BBH mergers may occur in
massive, passive, highly biased galaxies even when their progenitors formed in
low-metallicity, star-forming systems. The distinction between formation and
merger environments is therefore central to interpreting the inferred GW
bias.

### 11.2 Which demographic variables matter beyond stellar mass?

The twelve classes allow the relative importance of stellar mass, activity,
and FMR-selected metallicity to be quantified. At fixed redshift, compare the
change in \(b_{\rm GW}\) induced by successively resolving:

- stellar-mass class;
- main-sequence offset or activity class;
- FMR-metallicity class.

Stellar mass is expected to provide the dominant contribution because of its
strong connection to halo mass and large-scale bias. A null result after
conditioning on mass and redshift—namely, little remaining sensitivity to
activity and FMR-metallicity class—would be valuable: it establishes that a
simple, robust host model captures the relevant clustering signal. A residual
activity or metallicity trend would be more informative, but requires careful
interpretation because deterministic FMR metallicity is constructed from mass
and SFR rather than independently measured at fixed values of those variables.

### 11.3 A test of formation and delay-time physics

Compare the S22-weighted catalogue to deliberately simpler host prescriptions,
such as instantaneous SFR tracing and mass-only weighting. This tests the
physical pathway

\[
\text{low-mass, star-forming formation environment}
\;\longrightarrow\;
\text{more massive merger host}.
\]

Differences in host fractions, halo occupation, and \(b_{\rm GW}(z)\) isolate
which aspects of the observable merger-host population arise from merger-delay
and galaxy-evolution physics.

### 11.4 Peak-siren cosmology bias from host-model mismatch

Generate GW mocks from the S22-weighted host population and analyse them with
simplified galaxy priors, for example uniform-galaxy or mass-only assignment.
The central diagnostic is the induced cosmological shift relative to the
statistical error:

\[
\Delta H_0 / \sigma(H_0).
\]

This establishes whether host-demographic uncertainty is negligible or a
limiting systematic for peak/dark-siren inference. The same exercise can test
the effects of assuming the wrong activity or metallicity mixture, or of
mis-specifying the redshift evolution of the host population.

### 11.5 Requirements for robust future analyses

The sensitivity study can be converted into practical requirements:

- the precision needed for stellar mass, activity, and redshift estimates;
- whether broad mass bins are sufficient or additional demographic modes are
  needed;
- the GW-event count at which host modeling becomes important;
- the extent to which improved sky and distance localization mitigates host
  model systematics;
- the redshift and host-property ranges contributing most of the cosmological
  information.

The intended paper narrative is therefore: astrophysical BBH formation and
delay-time physics select a non-trivial mixture of host galaxies; that mixture
sets the effective GW clustering bias; and an overly simple host prior can bias
peak-siren cosmology. Conversely, demonstrating that a small number of
host-demographic modes captures the effect is itself a route to a practical,
robust analysis model.

## 12. Open choices: do not silently invent defaults

- Exact Menote reference/provenance and its role as a comparator rather than
  the baseline low-redshift rate.
- FMR redshift evolution and sensitivity to the adopted passive-galaxy SFR
  floor; invalid-SFR galaxies remain explicitly invalid.
- Mapping gas metallicity to progenitor-birth metallicity in the merger prescription.
- Numerical Z_cut and optional scatter model.
- Fixed versus evolving Schechter knee; fit range and redshift binning.
- Catalog completeness, reliable lower stellar mass, and MS fitting procedure.
- O5 network, sensitivities, exposure, duty cycles, detection threshold, and binary populations.
- Y1/Y5 selection/photo-z/property-error models and galaxy-bias calibration.
- Linear scale cuts, covariance strategy, inference priors, and useful bias parameterization.

## 13. Starting references

- Santoliquido et al. (2022), Modelling the host galaxies of binary compact object mergers with observational scaling relations: https://arxiv.org/abs/2205.05099
- Mannucci et al. (2010), A fundamental relation between mass, SFR and metallicity in local and high redshift galaxies: https://arxiv.org/abs/1005.0006
- Mannucci et al. (2011), The metallicity of long-GRB hosts and the low-mass FMR extension: https://arxiv.org/abs/1011.4506
- Speagle et al. (2014), A Highly Consistent Framework for the Evolution of the Star-Forming Main Sequence: https://arxiv.org/abs/1405.2041
- Weigel et al. (2016), Stellar mass functions: methods, systematics and results for the local Universe: https://academic.oup.com/mnras/article/459/2/2150/2595150
- Ferri et al., A robust cosmic standard ruler from the cross-correlations of galaxies and dark sirens: https://arxiv.org/abs/2412.00202
- Matos et al., First measurement of the Hubble constant from gravitational wave-galaxy cross-correlations: https://arxiv.org/abs/2512.15380
- Gravitational-wave dark siren cosmology systematics from galaxy weighting: https://arxiv.org/abs/2405.14818
- GWDALI methodology: https://arxiv.org/abs/2307.10154
- GWDALI documentation: https://gwdali.readthedocs.io/en/latest/

Verify current paper versions and exact prescriptions before implementation. These links provide provenance; the methodological decisions above are the current conversation agreement, not claims that every cited work uses this exact pipeline.
