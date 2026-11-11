# SkySim5000 GW localisation methodology

This document preserves the supplied scientific description. Original script names and old command examples below are historical; their compatibility wrappers now live in `legacy/`. Use the current pipeline guide for runnable launcher commands. See [the current pipeline guide](PIPELINE.md) for package locations, configuration, and missing components in the supplied collection.

This repository contains the scripts used to construct SkySim5000 compact-binary source catalogues, select detected GW events, run GWDALI/LVK localisation, save Doublet posterior samples, and project the resulting localisations onto galaxy redshift slices.

The central provenance rule is:

```text
source_XXXXXX.npz
        -> XXXXXX = original row index in the prepared GWDALI catalogue
        -> catalogue[XXXXXX]["redshift_true"]
        -> galaxy redshift slice
```

The index is preserved through SNR selection and localisation. The prepared catalogue also retains `redshift_true`, so downstream plotting does not need to infer redshift from the Doublet `inv_dL` posterior.

## Practical input boundary

The full scientific chain below includes the origin of the seeds. In normal use, the portable pipeline starts from the exported seed-host HDF5 at injection preparation; the heavy SkySim5000 catalogue is not required. The GCR/MPI host-assignment script is an optional NERSC-only seed-building tool. See [the pipeline guide](PIPELINE.md) for the HDF5 schema and current execution order.

## Scientific methodology

The analysis begins with a galaxy catalogue, not with an already detected GW catalogue. SkySim5000 supplies the simulated galaxy population and its galaxy properties, including position, true redshift, photometric redshift, stellar mass, halo information, and galaxy identifiers. These galaxies are the possible host pool. They are not automatically GW detections.

The full conceptual chain is:

```text
SkySim5000 galaxy simulation
        |
        | galaxy positions, redshifts, stellar/halo properties
        v
merger-rate model R(z, M_star)
        |
        | expected event counts in redshift/mass bins
        v
GW event targets
        |
        | deterministic distributed host assignment
        v
possible GW host/source catalogue
        |
        | sample binary and extrinsic parameters
        v
GWDALI injection catalogue
        |
        | O5 detector network response and waveform model
        v
detector SNR catalogue
        |
        | detection thresholds and detector multiplicity cut
        v
detected GW source indices
        |
        +--------------------------+
        |                          |
        v                          v
Fisher localisation             Doublet localisation
Gaussian covariance             DALI tensors + nested sampling
        |                          |
        v                          v
Fisher HEALPix map               saved RA/Dec posterior samples
                                   |
                                   v
                              Doublet HEALPix map
        +--------------------------+
        |
        v
galaxy redshift-slice overlays and localisation comparisons
```

### 1. Galaxy seeds and provenance

The starting data are SkySim5000 simulated galaxies. The galaxy catalogue provides the spatial and population context; it does not itself specify which objects are GW events. The host-assignment stage selects a controlled set of possible hosts according to the merger-rate targets. The selected host rows retain their true redshift and galaxy identity in the assignment output, while the prepared GWDALI NPZ currently preserves the filtered row order and `redshift_true`.

### 2. Merger-rate density and event targets

The merger-rate model provides a source-frame rate density as a function of redshift and stellar mass, schematically `R(z, M_star)`. The expected number in each bin additionally uses survey solid angle, cosmological differential comoving volume, redshift-bin width, stellar-mass-bin width, observing time, and the source-to-observer factor `1/(1+z)`. Poisson draws convert expected counts into concrete GW event targets.

This is a target-generation step: it says how many events should be assigned to each population bin. It is distinct from assigning those targets to individual SkySim5000 galaxies.

### 3. Host assignment

`mpi_assign_gw_hosts.py` scans the galaxy catalogue in parallel, restricts candidates by true-redshift and stellar-mass bin, and uses deterministic 64-bit hash priorities to select hosts reproducibly. The result is the possible GW source/host catalogue used by the later waveform calculations. If a bin has fewer eligible galaxies than requested, the output records replacement sampling; that should be treated explicitly in downstream interpretation.

### 4. Menote-inspired GW parameter sampling

For each selected host, `sampling_GW_sirens_properties.py` samples the binary population and extrinsic parameters following the configured Menote-inspired distributions. It samples source-frame masses, converts them to detector-frame masses using `1+z`, samples spins and orientations, assigns coalescence angles, computes luminosity distance in the Outer Rim cosmology, and writes the GWDALI input table. The catalogue is therefore a synthetic injection catalogue attached to a simulated host population, not a direct list of measured events.

### 5. O5 detector framework

The GWDALI stage uses the configured O5-era detector network response: Hanford, Livingston, Virgo, and optionally KAGRA, with the detector geometries and built-in sensitivity names defined in `gwdali_lvk_network.py`. The waveform approximant and frequency range are configured separately. The validated production path uses `IMRPhenomXPHM`, 10–2048 Hz, numerical derivatives, and the installed LAL/GWDALI implementation.

### 6. SNR and detection

The SNR catalogue evaluates each prepared source in the detector network and records detector-level SNRs, network SNR, success, and completion. The detection catalogue is then defined by the configured selection rule, currently network SNR ≥ 12, at least two detectors above single-detector SNR 4, and successful completion. This stage is where the possible source catalogue becomes the detected GW sample.

### 7. Localisation

The production localisation uses a Fisher approximation with the free-parameter order `(RA, Dec, inv_dL, Mc, eta, iota, psi, phi_coal)`. `Mc` is detector-frame chirp mass and `eta` is symmetric mass ratio; the inverse Fisher matrix is the local Gaussian covariance, so sky and distance errors are marginalized over these mass parameters and the extrinsic angles.

The Fisher and Doublet catalogues must be matched by the original prepared-catalogue `source_index`; they must not be matched by the ordinal position inside a filtered detected list.

### 8. Map making and galaxy overlays

The Fisher map path turns each covariance into a local tangent-plane Gaussian and projects it onto HEALPix. The Doublet path assigns the actual posterior RA/Dec samples to HEALPix pixels and computes a discrete sampled HPD area. The galaxy overlay stage uses the known injected `redshift_true` of each detected source to place it into the galaxy redshift bin. It then overlays either the HEALPix localisation support or a reproducible subsample of posterior points on the corresponding galaxy count map.

The galaxy map's slice coordinate is the galaxy/photo-z analysis bin, while GW event membership is assigned using the injected true redshift. This distinction should remain explicit in figures and captions.

## Repository contents

### Population and host preparation

- `merger_rate_density_models.py` — merger-rate-density models, cosmology, expected event counts, and Poisson draws.
- `calculate_merger_rate_density_targets.py` — evaluates the redshift/stellar-mass target grid and writes the target NPZ/HDF5 products used by host assignment.
- `mpi_assign_gw_hosts.py` — MPI/GCRCatalogs host assignment. It partitions native HEALPixels across MPI ranks and selects hosts deterministically using 64-bit hash priorities inside true-redshift and stellar-mass bins.
- `sampling_GW_sirens_properties.py` — reads the selected GW-host HDF5 catalogue, samples binary masses/spins/extrinsic parameters, computes luminosity distance with the Outer Rim cosmology, and writes row-column GWDALI NPZ catalogues for BBH, BHNS, and BNS.

The prepared catalogue stores GWDALI parameters plus useful provenance fields including `redshift_true`, `ra_deg`, `dec_deg`, and source-frame masses. GWDALI angles are stored in radians in the catalogue; the GWDALI adapter converts RA/Dec to degrees and can replace `dL` with `inv_dL = 1/dL`.

### Shared GWDALI configuration

- `gwdali_load_sources.py` — catalogue naming, schema validation, row access, unit conversion, and `dL`/`inv_dL` conversion.
- `gwdali_lvk_network.py` — LVK detector geometry and sensitivity configuration for Hanford, Livingston, Virgo, and optional KAGRA.

The production Doublet parameter order is:

```python
("RA", "Dec", "inv_dL", "iota", "psi", "phi_coal")
```

The production waveform choices are `IMRPhenomXPHM` for BBH/BHNS and `IMRPhenomXHM` for BNS. The current validated path uses numerical derivatives (`numdiff`), LAL waveforms (`enable_jax_waveforms=False`), fixed `t_coal`, and nested sampling with `nestle`.

### SNR and detection

- `test_gwdali_snr_one_source.py` — one-source SNR diagnostic; useful for validating the GWDALI return structure, detector configuration, and thresholds before a catalogue run.
- `compute_gwdali_snr_catalogue.py` — serial SNR catalogue with atomic checkpointing and resume support. It records per-detector/network SNRs, success, detection status, and completion state.
- `make_detected_indices.py` — writes the original catalogue row indices satisfying `completed & success & detected`.

The current detection configuration requires network SNR ≥ 12, at least two detectors above single-detector SNR 4, and includes KAGRA when enabled.

### Fisher and Doublet localisation

- `test_gwdali_localization_one_source.py` — one-source Fisher/localisation validation, matrix extraction, and covariance-area checks.
- `compute_gwdali_localization_catalogue.py` — serial Fisher catalogue localisation with atomic checkpointing. It stores Fisher matrices, inverse-Fisher covariances, summary sky areas, uncertainties, correlations, runtime, and status fields.
- `gwdali_doublet_parallel.py` — generic one-node multiprocessing driver. It uses `spawn`, limits native numerical-library threads, skips existing `source_XXXXXX.npz` files, and atomically writes one compressed NPZ per event.
- `gwdali_doublet_serial_adapter.py` — adapter exposing the serial GWDALI calculation as `run_event(source_index)` for the parallel driver.
- `compare_fisher_localization.py` — matches Fisher and Doublet results by original `source_index`, reports area/error ratios, and writes comparison plots and NPZ output.
- `diagnose_gwdal_fisher_catalogue.py` — catalogue-level Fisher quality-control plots covering area, conditioning, SNR, redshift, and tangent-plane validity.
- `diagnose_gwdali_localisation_issue.py` — targeted one-source diagnosis of broad or unstable localisations, including parameter-set and finite-difference-step comparisons.

The parallel driver expects an importable module exposing:

```python
run_event(source_index: int) -> dict
```

The returned dictionary must contain `samples`, with columns in the production `FREE_PARAMS` order. The recommended one-node starting point is 32 workers and one native numerical thread per worker, subject to the node's memory and library behaviour.

Example:

```bash
python gwdali_doublet_parallel.py \
  --pipeline-module gwdali_doublet_serial_adapter \
  --output-dir /pscratch/.../GWInputCatalogs \
  --indices /pscratch/.../detected_indices.txt \
  --workers 32 \
  --native-threads 1
```

Each successful event is saved as:

```text
source_XXXXXX.npz
```

The files contain the full posterior samples, parameter labels, Fisher/covariance diagnostics, SNR, and runtime metadata. Existing files are skipped, so interrupted production runs can be resumed safely.

### HEALPix products

- `make_gw_healpix_probability_maps.py` — converts Fisher covariance ellipses into sparse per-event HEALPix probability maps, sums them, and optionally writes redshift-slice maps.
- `make_gw_doublet_healpix_maps.py` — converts the actual saved Doublet RA/Dec posterior samples directly into HEALPix maps. It computes discrete sampled 90% HPD areas and can write per-redshift-slice Doublet products.

For Doublet maps, posterior samples are assigned to pixels and each event is normalized to unit probability. The summed map therefore integrates to the number of mapped events. The Doublet HPD area is based on the sorted discrete pixel probabilities, not a Gaussian covariance approximation.

### Visualisation

The current lightweight visualisation is notebook-oriented: it reads the galaxy count map, assigns each production source to a slice using `catalogue["redshift_true"]`, and overlays a reproducible random subset of Doublet RA/Dec posterior points. It deliberately avoids KDE and contour grids because production posterior files can contain approximately 100,000 samples per source.

The full-slice view is:

```text
galaxy HEALPix density slice
    + posterior sample points for every source in that slice
```

For source-level inspection, use the same map and samples in a fixed 30° × 30° `healpy.gnomview` window centred on the source posterior sky position.

## Recommended execution order

1. Compute rate targets with `calculate_merger_rate_density_targets.py`.
2. Assign SkySim5000 hosts with `mpi_assign_gw_hosts.py`.
3. Prepare GWDALI source catalogues with `sampling_GW_sirens_properties.py`.
4. Validate one source with `test_gwdali_snr_one_source.py`.
5. Run the serial SNR catalogue with `compute_gwdali_snr_catalogue.py`.
6. Extract detected original row indices with `make_detected_indices.py`.
7. Validate one localisation with `test_gwdali_localization_one_source.py`.
8. Run a small Doublet test through the serial adapter.
9. Run production Doublet events with `gwdali_doublet_parallel.py`.
10. Compare Fisher and Doublet results with `compare_fisher_localization.py`.
11. Run catalogue diagnostics with the two `diagnose_*.py` scripts.
12. Build Fisher or Doublet HEALPix products with the corresponding map script.
13. Assign events to galaxy redshift slices from the prepared catalogue's `redshift_true` field and make the full-slice/sample-point plots.

## Data layout used by the scripts

The configured production layout is:

```text
/pscratch/sd/j/jatorres/data/lsst/SkySim5000/NumberCountMaps/
├── SkySim5000_photoz_number_count_maps_nside512.hdf5
├── SkySim5000_gw_host_catalogue1.hdf5
└── GWInputCatalogs/
    ├── skysim5000_BBH_gwdali_z0.0000_0.8000.npz
    ├── SNR/
    ├── Localization/
    ├── LocalizationDoublet/
    ├── Healpix/
    └── source_XXXXXX.npz
```

Shared paths and waveform defaults now live in `skysim_gw/settings/common.py`; individual settings modules preserve stage-specific options. Existing JSON and command-line interfaces remain available. See the pipeline guide for configuration details.

## Important caveats and cleanup items

- The original host catalogue row number is not saved in the prepared GWDALI NPZ. The prepared row index is stable relative to the filtered preparation array, but recovering the original HDF5 row requires reapplying the exact preparation mask.
- `source_XXXXXX.npz` filenames should be treated as prepared-catalogue row indices. Production plotting and HEALPix scripts validate the index against the prepared catalogue.
- The original target-generator import mismatch has been corrected: it now imports `skysim_gw.rates`.
- `merge_gwdali_doublet_samples.py` is currently empty and should be implemented if a compact merged Doublet catalogue is required. Per-source files remain the authoritative posterior-sample products.
- GWDALI return structures can vary by installed version. Keep the result-unwrapping and sample extraction checks in the validation scripts when changing environments.
- Do not use posterior `inv_dL` samples to assign events to galaxy redshift slices when the prepared catalogue contains `redshift_true`; use the known injected true redshift for deterministic membership.

## Dependencies

The scripts require the project environment containing at least NumPy, SciPy, Astropy, h5py, healpy, matplotlib, GWDALI, nestle, and the relevant SkySim5000/GCRCatalogs/MPI stack. The MPI host-assignment stage additionally requires `mpi4py` and `GCRCatalogs`; the localisation stages require the installed GWDALI and waveform dependencies.
