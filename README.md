# SkySim5000 GW pipeline

The portable pipeline starts from an existing HDF5 catalogue of possible seed hosts, then samples GW injections, computes O5-network detection and GWDALI localisation, and builds HEALPix maps. It does not need the original heavy SkySim5000 catalogue, GCRCatalogs, or MPI. Optional upstream tools retain the merger-rate calculation and NERSC-only SkySim5000 seed-host assignment. The code is now separated into reusable modules, production stages, diagnostics, and settings. Existing NERSC paths and scientific defaults are preserved.

## Start here

```bash
# Run from this directory; scientific dependencies are unnecessary for these commands.
python3 -m skysim_gw --list
python3 -m skysim_gw --dry-run snr
```

Read [the pipeline guide](docs/PIPELINE.md) for execution order, configuration, portability, and known missing inputs. The [scientific methodology](docs/METHODOLOGY.md) explains the analysis and product conventions.
For a review-oriented account of the complete catalogue construction, including
the merger-rate-density target stage and its present configuration limits, see
[the pipeline review guide](docs/PIPELINE_REVIEW.md).

Set `SOURCE_FILE` in `skysim_gw/settings/population.py` to the exported seed-host HDF5, then start with:

```bash
python -m skysim_gw prepare
```

The existing path remains a placeholder. See the pipeline guide for the required HDF5 schema. `assign-hosts` is only for rebuilding that file at NERSC; skip it and `rate-targets` when reusing existing seeds.

## Tagged BBH realization workflow

For the GWseeds realizations, keep the host selections and the prepared
populations separate by model:

```text
GWseeds/
├── Seeds/
│   ├── gw_hosts_fh0_r0000.hdf5 ... gw_hosts_fh0_r0009.hdf5
│   └── gw_hosts_fh1_r0000.hdf5 ... gw_hosts_fh1_r0009.hdf5
└── Populations/
    ├── fh0/
    └── fh1/
```

The local batch helper prepares the 20 tagged host files as BBH catalogues,
using the configured bimodal mass distribution. It assigns an independent,
deterministic random stream to each model/realization pair:

```bash
python scripts/prepare_bbh_realizations.py \
  --input-dir /path/to/GWseeds/Seeds \
  --output-root /path/to/GWseeds/Populations \
  --zi 0.0 --zf 1.0 --seed 12345
```

This produces files such as
`Populations/fh0/skysim5000_BBH_gwdali_fh0_r0000_z0.0000_1.0000.npz`.
Each catalogue contains the sampled BBH parameters (including source-frame
and detector-frame masses, luminosity distance, sky position, orientation,
spins, and `redshift_true`) plus provenance metadata. The full host-galaxy
properties, such as galaxy ID and stellar mass, remain in the corresponding
file under `Seeds/`.

Run SNR estimation per prepared realization by passing its explicit input and
output paths. This preserves the realization tag rather than using the
standard redshift-only filename:

```bash
python -m skysim_gw snr -- \
  --input-file /path/to/GWseeds/Populations/fh0/skysim5000_BBH_gwdali_fh0_r0000_z0.0000_1.0000.npz \
  --output-file /path/to/GWseeds/Populations/fh0/skysim5000_BBH_lvk_snr_fh0_r0000_z0.0000_1.0000.npz
```

The usual checkpoint/resume behavior still applies; a completed SNR output
must match the input catalogue and detector-configuration metadata before it
is reused. Detector/network settings are configured in `settings/snr.py`.

### NERSC SkySim5000 host catalogue

The fixed GCR catalogue name is `skysim5000_v1.2`. Its verified host-assignment
ID column is `galaxy_id`. The catalogue provides every required quantity:
`galaxy_id`, `ra`, `dec`, `redshift_true`, `redshift`, `stellar_mass`,
`halo_mass`, and `is_central`.

## Repository layout

```text
skysim_gw/
├── rates.py                # Exportable merger-rate models and event counts
├── population.py           # Exportable sampling, cosmology, catalogue preparation
├── catalogue.py            # Catalogue loading, validation, units, row access
├── detectors.py            # LVK geometry and sensitivity definitions
├── results.py              # GWDALI result/matrix extraction helpers
├── products.py             # Shared product filenames
├── doublet_adapter.py      # Importable run_event(source_index) worker interface
├── stages/                 # Production calculations and map making
├── diagnostics/            # All original diagnosers and one-source checks
├── settings/
│   ├── common.py           # Shared paths, redshift range, source and waveform defaults
│   └── *.py                # Explicit stage-specific settings
└── cli.py                  # Unified, dependency-lazy stage launcher
docs/                      # Workflow guide and scientific methodology
tests/                     # Dependency-free structural checks
legacy/                    # Archived compatibility wrappers and empty merge placeholder
```

The obsolete root-level entry points now live in `legacy/`; use the package launcher for calculations. Edit implementations inside `skysim_gw/` and configuration inside `skysim_gw/settings/`. All diagnosers remain active in `skysim_gw/diagnostics/`. The empty `legacy/merge_gwdali_doublet_samples.py` remains an unimplemented placeholder. See [the archive notes](legacy/README.md) for compatibility usage.

## Import reusable code

In the scientific environment:

```python
from skysim_gw.rates import expected_events_in_bin
from skysim_gw.population import PopulationConfig, prepare_population
from skysim_gw.catalogue import load_gwdali_catalogue, get_gwdali_source
from skysim_gw.detectors import get_lvk_detectors
from skysim_gw.results import find_matrix, unwrap_gwdali_result
```

Importing these modules does not launch catalogue calculations. Individual modules still require their scientific dependencies; importing the top-level package and listing stages does not.

## Provenance and validation

`source_XXXXXX.npz` identifies original **prepared-catalogue row XXXXXX**, even after detection filtering. Match catalogues using `source_index`; assign GW redshift slices using injected `redshift_true`. Preserve the distinction between Fisher covariance maps and maps of actual Doublet posterior samples.

The original scripts were tested end-to-end on NERSC by the author. This refactor has passed syntax and eight structural checks locally, plus a comparison confirming preserved extracted settings and 115 unchanged function/class syntax trees. It has **not** been numerically revalidated: this machine lacks NumPy, Astropy, h5py, healpy, and GWDALI.

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q skysim_gw tests
```
