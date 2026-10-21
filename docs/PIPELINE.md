# Pipeline guide

## Portable starting point: seed-host HDF5

Start at `prepare` using an existing exported seed-host catalogue. No original SkySim5000 files, GCRCatalogs, mpi4py, or MPI runtime are needed for this workflow. Set `SOURCE_FILE` in `skysim_gw/settings/population.py`; the existing NERSC path is retained as a placeholder.

The reader expects this schema for each configured population (`BBH`, `BHNS`, `BNS`):

```text
/sources/BBH/ra              # 1-D numeric array, degrees
/sources/BBH/dec             # 1-D numeric array, degrees
/sources/BBH/redshift_true   # 1-D numeric array, dimensionless
```

Arrays within a population must have equal length. Additional host properties are allowed but not used by preparation. Missing population groups are skipped. The reader selects finite coordinates and true redshifts within the configured `[ZI, ZF)` interval. These rows are already selected possible event hosts; this stage does not reapply the merger-rate model to an arbitrary unassigned galaxy pool.

`rate-targets` and `assign-hosts` are optional upstream steps for rebuilding seeds. Only `skysim_gw/stages/assign_hosts.py` imports GCRCatalogs and mpi4py. It is **NERSC-only as configured**, where the original SkySim5000 catalogue lives, and requires `lsst` group access. Run seed building there and transfer the resulting HDF5 to the analysis machine. Rate calculation itself does not require GCR or NERSC.

### S22 empirical BBH Seeds

`model_name: "s22_bbh_empirical"` adds the metallicity-marginalized
Santoliquido et al. (2022) BBH model to the existing Seed workflow. It returns
the S22 \(R(z,\log M_\star)\) obtained by summing the empirical host grid over
SFR. This is intentionally the shared projection with the existing Menote
fh0/fh1 models, so `assign-hosts` still selects uniformly within its existing
`(true-z, logMstar)` cells.

The released S22 text catalogues must be readable on the system running
`rate-targets`. A minimal BBH configuration addition is:

```json
"model_name": "s22_bbh_empirical",
"bbh_model_parameters": {
  "data_dir": "/path/to/MergerRateDensityModels",
  "metallicity_relation": "M57",
  "common_envelope": "A3",
  "rate_scale": 1.0,
  "mass_tilt": 0.0,
  "sfr_tilt": 0.0
}
```

`data_dir` is required. `M57`/`M54` and `A1`/`A3`/`A5` select the supplied
S22 catalogue family. The direct S22 rate is used when `rate_scale=1`; choose
and record an explicit scale if matching a Menote total for a shape-only
comparison. The separate `S22BBHEmpiricalModel.rate_per_galaxy(z, Mstar, sfr)`
API retains SFR and is the appropriate path for a future fully galaxy-weighted
lightcone assignment. It is not used by the current 2-D Seed assignment.

To prepare S22 Seed files after the NERSC host-assignment step, use the normal
tagged-realization helper with `--models s22` (or include `s22` alongside
`fh0 fh1`). It recognizes `gw_hosts_s22_r0000.hdf5` through
`gw_hosts_s22_r0009.hdf5`, writes `Populations/s22/`, and assigns a distinct
deterministic random stream to that model tag. The Doublet provenance tagger
also recognizes `s22_rXXXX`.

## Configuration and moving to another computer

Edit `skysim_gw/settings/common.py` for the shared NERSC base directory, BBH/BHNS/BNS selection, redshift bounds, waveform approximants, frequency range, and frequency-grid size. The `/pscratch/...` and `/global/...` paths are deliberately unchanged placeholders. Stage settings derive output directories from the shared base directory.

Inspect the other files in `skysim_gw/settings/` for stage-specific options: sampling seed and input host file (`population.py`), SNR thresholds and checkpointing (`snr.py`), Doublet settings (`localize.py`), map resolution and support (`fisher_maps.py`), and diagnostic paths/settings. Values are imported when Python starts; restart the process after editing settings. `PopulationConfig` in `population.py` retains the configured distribution parameters and can be passed explicitly to sampling functions.

The standalone catalogue demonstration retains its original different defaults: `/pscratch/sd/j/jatorres/data/lsst/SkySim5000/GWInputCatalogs` and `zf=3.5`. Production stages explicitly pass their configured catalogue directory and redshift limits, so they use the shared `NumberCountMaps/GWInputCatalogs` path and `zf=0.8`.

Existing JSON interfaces and command-line interfaces are preserved. Rate-target and host-assignment JSON files were **not supplied**, but are unnecessary when starting from existing seeds. Their default filenames resolve in the working directory, or provide absolute paths with `--config`. Recover the validated NERSC versions; no scientific survey area, observing duration, mass limits, or host-selection defaults have been invented here. Required JSON keys are listed by `load_config()` in the respective stage modules.

Parallel Doublet retains `GWDALI_FSIZE` (default 3000), `GWDALI_NPOINTS` (default 3000), and `GWDALI_SERIAL_MODULE` (now defaulting to `skysim_gw.stages.localize`). Its adapter continues to include KAGRA. Serial Doublet retains `NPOINTS=300` and `MAX_NEW_SOURCES=9`; set the latter to `None` for an unrestricted serial run. These differing defaults have deliberately not been harmonized without numerical validation. The parallel parameter-order contract remains `(RA, Dec, inv_dL, iota, psi, phi_coal)`.

Use the Python environment that passed validation on NERSC, including NumPy, SciPy, Astropy, h5py, healpy, matplotlib, GWDALI, nestle, LAL/waveform dependencies, and, only for optional NERSC seed building, MPI/GCRCatalogs. Export its actual package versions before migration. `pyproject.toml` packages the code but does not attempt to reconstruct an unknown scientific environment or install GWDALI. From the validated environment, optionally install with `python -m pip install --no-deps -e .`; this enables `skysim-gw` as an equivalent launcher.

## Execution order and data contracts

Each command runs one stage, allowing MPI and one-node multiprocessing to retain their existing execution models. The launcher does not automatically submit jobs or infer that outputs are complete. `--dry-run` prints the corresponding Python module command without importing scientific libraries or reading data.

| Stage | Launcher command | Input → output |
|---|---|---|
| Optional seed-building rate targets | `rate-targets --config /path/gw_rate_config.json` | Redshift slices + rate configuration → target HDF5 |
| Optional NERSC-only seed assignment | `assign-hosts --config /path/gw_host_assignment_config.json` | Target HDF5 + GCR SkySim5000 → host HDF5 |
| Injection preparation | `prepare` | Selected host HDF5 → per-population GWDALI NPZ |
| One-source SNR check | `check-snr` | Injection row → printed diagnostic |
| SNR catalogue | `snr` | Injection NPZ → checkpointed SNR NPZ |
| Detection selection | `detect SNR_FILE OUTPUT.txt` | SNR completion/success/detection masks → original row indices |
| One-source Fisher check | `check-localisation` | Detected injection + SNR → diagnostic |
| Serial Doublet | `doublet-serial` | Injection + SNR → covariance/diagnostic summary NPZ |
| Parallel Doublet | `doublet-parallel ...` | Detected indices + injection/SNR → actual `source_XXXXXX.npz` samples |
| Comparison | `compare` | Existing Fisher + serial Doublet summaries → matched comparisons |
| Fisher diagnostics | `diagnose-fisher` | Configured localisation catalogue → quality-control plots |
| Localisation diagnostics | `diagnose-localisation` | One detected source → parameter/step-size comparisons |
| Fisher maps | `fisher-maps` | Existing Fisher covariance + injections → HEALPix products |
| Doublet maps | `doublet-maps ...` | Actual posterior files + injections + slices → HEALPix products |

Prefix commands with `python -m skysim_gw`. The normal workflow starts from the seed-host HDF5:

```bash
python -m skysim_gw prepare
python -m skysim_gw check-snr
python -m skysim_gw snr
python -m skysim_gw snr -- \
  --input-file /path/to/skysim5000_BBH_gwdali_fh0_r0000_z0.0000_1.0000.npz \
  --output-file /path/to/SNR/skysim5000_BBH_lvk_snr_fh0_r0000_z0.0000_1.0000.npz \
  --workers 32 --native-threads 1
python -m skysim_gw detect /path/to/snr.npz /path/to/detected_indices.txt
python -m skysim_gw check-localisation
python -m skysim_gw doublet-serial
python -m skysim_gw doublet-parallel \
  --pipeline-module skysim_gw.doublet_adapter \
  --input-file /path/to/skysim5000_BBH_gwdali_fh0_r0000_z0.0000_1.0000.npz \
  --snr-file /path/to/SNR/skysim5000_BBH_lvk_snr_fh0_r0000_z0.0000_1.0000.npz \
  --indices /path/to/detected_indices_fh0_r0000.txt \
  --output-dir /path/to/doublet/fh0/r0000 \
  --workers 32 --native-threads 1
python -m skysim_gw doublet-maps \
  --samples-dir /pscratch/.../GWInputCatalogs \
  --injection-file /pscratch/.../skysim5000_BBH_gwdali_z0.0000_0.8000.npz \
  --slice-file /global/.../redshift_slices.info \
  --output /pscratch/.../Healpix/doublet_maps.npz \
  --slice-output-dir /pscratch/.../Healpix/DoubletRedshiftSlices
```

The abbreviated `/path/...` arguments above are examples to replace; the original full paths remain in settings. Choose MPI resources for the target machine. For native CLI options, run `python -m skysim_gw doublet-parallel -- --help` (also supported for rate targets, host assignment, detection, and Doublet maps). Other stages use settings files and reject extra arguments at the unified launcher.

## Preserved diagnostics

No diagnoser was removed. Active implementations live below; the obsolete wrappers with the original filenames have been archived in `legacy/`. Use the diagnostic launcher commands listed above.

| Original filename | Implementation |
|---|---|
| `test_gwdali_snr_one_source.py` | `skysim_gw/diagnostics/one_snr.py` |
| `test_gwdali_localization_one_source.py` | `skysim_gw/diagnostics/one_localisation.py` |
| `diagnose_gwdal_fisher_catalogue.py` | `skysim_gw/diagnostics/fisher.py` |
| `diagnose_gwdali_localisation_issue.py` | `skysim_gw/diagnostics/localisation.py` |
| `compare_fisher_localization.py` | `skysim_gw/diagnostics/compare.py` |

The `test_*` scripts are scientific one-source diagnostics, not automated unit tests. Production result-extraction helpers now live in `skysim_gw/results.py`, so production code no longer imports a diagnostic script. Product-naming helpers are independent of GWDALI. Importing the comparison module no longer creates an output directory; running it still does.

## Boundaries in the supplied collection

- The supplied catalogue localisation implementation is Doublet-specific: it always extracts posterior samples. Changing `METHOD` to `Fisher` alone will not create a valid Fisher catalogue pipeline. The full Fisher catalogue generator described in the old workflow is absent; retain/recover that validated generator from NERSC. The supplied one-source Fisher check, Fisher diagnostics, Fisher map maker, and comparison tools are preserved.
- `diagnose-fisher` retains its original product-name helper, which currently constructs a **Doublet** summary filename under its configured localisation directory. Check that input path against the actual Fisher product before using this diagnostic; the refactor does not silently change the scientific input.
- `legacy/merge_gwdali_doublet_samples.py` was empty and remains an archived placeholder. Parallel posterior files do not automatically become the summary catalogue expected by `compare`; the serial Doublet stage produces that summary separately.
- Galaxy-slice overlays were described as notebook code, but no notebook/overlay script or galaxy-map schema was supplied. Map generation is available; recover that plotting notebook to finish the exact previously validated visualisation.
- Prepared-catalogue row IDs, output schemas, random sampling, thresholds, waveform calls, coordinate conversions, and posterior-map algorithms were preserved. Original HDF5 host-row indices are still not saved in the prepared NPZ.

The local structural tests verify dispatch, argument forwarding, module compatibility, package import boundaries, preserved key settings, and stage availability. A one-time syntax-tree comparison against the supplied originals found 115 unchanged function/class bodies. The three adjusted functions only rename the shared SNR filename helper, change the adapter's default import path, and move comparison directory creation into `main()`. Population execution was wrapped in `main()` without changing its body. End-to-end scientific revalidation is still required in the scientific environment.
