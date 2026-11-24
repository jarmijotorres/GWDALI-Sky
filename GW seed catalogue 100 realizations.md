# GW seed catalogue: 100 realizations

## Purpose

This note defines the catalogue and output structure for the S22 BBH
workflow. The goal is to start from 100 independent host-seed catalogues,
populate each host catalogue 100 times with independent BBH population draws,
run the SNR and Fisher calculations, and store Fisher-Gaussian parameter
realizations without creating one small file per event.

The intended totals are:

- 100 host-seed catalogues;
- 100 BBH population draws per host-seed catalogue;
- 10,000 population catalogues and 10,000 matched SNR catalogues;
- one Fisher-sample HDF5 file per host-seed catalogue, with 100 population
  draw groups in each file.

The Fisher-Gaussian realization count per detected event is a separate
setting. It currently defaults to 100 and can be changed independently.

## Seed input

Input file:

~~~text
Documents/Data/GWseeds/gw_hosts_s22_100cat.hdf5
~~~

The inspected file contains /sources/BBH, 21,624,648 BBH host rows across
catalogue IDs 0 through 99, and metadata declaring:

- n_catalogues = 100;
- event_oversample_factor = 8;
- event_weight = 1/8.

The eight-fold oversampling applies to GW-event sampling. It does not mean
there are eight independent copies of the galaxy sky, and an additional
factor of eight must not be applied to the event count. The input keeps the
S22 catalogue identifier, octant-copy identifier, host properties,
redshift, position, and event weight on each assigned source.

The sampler selects one catalogue_id at a time and applies the half-open
true-redshift cut:

~~~text
zi <= redshift_true < zf
~~~

The planned selection is zi=0.0, zf=0.5, so objects at exactly z=0.5
are excluded.

## Workflow and file contracts

### 1. Prepare BBH population catalogues

New entry point: scripts/prepare_s22_100cat.py.

It reads the seed HDF5, selects rows for one host catalogue and the requested
redshift interval, then samples BBH component masses, spins, and extrinsic
parameters for each row. It produces one compressed NPZ per population draw:

~~~text
Populations/s22_100cat/
  c0000/
    skysim5000_BBH_gwdali_s22_c0000_draw0000_z0.0000_0.5000.npz
    ...
    skysim5000_BBH_gwdali_s22_c0000_draw0099_z0.0000_0.5000.npz
  ...
  c0099/
    ...
~~~

Each NPZ is one complete population catalogue, not one file per BBH. Its
catalog array is a structured row table. It contains the GWDALI parameters
(m1, m2, RA, Dec, psi, t_coal, phi_coal, dL, iota, and Cartesian spin
components) plus the available host-provenance columns, including
redshift_true, galaxy_id, catalogue_id, and event_weight. The NPZ also
contains file-level metadata such as zi, zf, the random seed, population draw
ID, units, cosmology, and population configuration.

The row number in catalog is the source index used by downstream stages.
The population draw ID is file-level metadata and is also in the filename.
The row-level catalogue_id is the host-seed catalogue ID.

Run the preparation stage alone:

~~~bash
python scripts/prepare_s22_100cat.py \
  --input-file /path/to/GWseeds/gw_hosts_s22_100cat.hdf5 \
  --output-root /path/to/GWseeds/Populations/s22_100cat \
  --zi 0.0 --zf 0.5 --seed 12345
~~~

Random streams are deterministic for (master seed, catalogue ID, population
draw ID). Catalogue and draw ranges can be supplied to divide work. With
--resume, existing population files are skipped; use a new output root when
changing the seed or population configuration.

### 2. Calculate SNR

Existing calculation: skysim_gw/stages/snr.py.

New batch entry point: scripts/run_s22_100cat_snr.py.

For each input population NPZ, the SNR stage loads the structured table and
passes its source rows to GWDALI as individual source parameter dictionaries.
It writes one matched SNR NPZ per population NPZ:

~~~text
Populations/s22_100cat/SNR/
  c0000/
    skysim5000_BBH_lvk_snr_radec_deg_s22_c0000_draw0000_z0.0000_0.5000.npz
    ...
  ...
~~~

Each SNR results array is aligned to the original population catalogue row
order. Its source_index refers to that row; completed, success, and detected
identify which calculations completed successfully and which sources pass
the configured detector thresholds. SNR checkpoints resume unfinished rows
and validate their input and detector metadata.

The batch runner processes all matching input/output pairs in one command.
The workers option is the number of population files processed concurrently;
native-threads limits native math-library threads in each worker. For a
machine with about 20 CPU cores, start with 16 workers and one native thread.

~~~bash
python scripts/run_s22_100cat_snr.py \
  --input-root /path/to/GWseeds/Populations/s22_100cat \
  --output-root /path/to/GWseeds/Populations/s22_100cat/SNR \
  --workers 16 --native-threads 1
~~~

Combined preparation and SNR entry point: scripts/process_s22_100cat.py.
Its default stage is all, so one command can prepare the population files
and then run SNR. It also supports prepare-only or SNR-only operation,
scheduler ranges, and dry-run.

~~~bash
python scripts/process_s22_100cat.py \
  --input-file /path/to/GWseeds/gw_hosts_s22_100cat.hdf5 \
  --population-root /path/to/GWseeds/Populations/s22_100cat \
  --snr-root /path/to/GWseeds/Populations/s22_100cat/SNR \
  --workers 16 --native-threads 1
~~~

### 3. Fisher calculation and Gaussian sampling

Existing calculation adapter: skysim_gw/doublet_adapter.py.

Modified parallel driver: skysim_gw/stages/doublet_parallel.py.

New S22 batch entry point: scripts/run_s22_fisher_hdf5.py.

The Fisher runner reads each population NPZ and its matched SNR NPZ. It
selects only rows marked completed & success & detected, then calls the
GWDALI Fisher calculation for each detected event. GWDALI receives one source
parameter dictionary per call; it does not read the population file itself.
The Fisher covariance describes a local multivariate Gaussian in the
parameter order:

~~~text
RA, Dec, inv_dL, Mc, eta, iota, psi, phi_coal
~~~

The Gaussian mean is the injected parameter vector from the population row,
transformed to the GWDALI parameterization. The covariance is the inverse
Fisher matrix. The code generates Fisher-Gaussian parameter vectors
deterministically from (master seed, host catalogue ID, population draw ID,
source index).

The result is one HDF5 per host-seed catalogue:

~~~text
FisherSamples/
  c0000.h5
  ...
  c0099.h5
~~~

Each file contains one group per population draw:

~~~text
/draws/draw0000/
  source_index                 (number of detected events,)
  fisher_gaussian_samples      (events, gaussian draws per event, 8)
  completed                    (events,)
  success                      (events,)
  error_message                (events,)
...
/draws/draw0099/
~~~

The sample array is chunked and gzip-compressed. Its last dimension follows
the parameter order above. Units are RA and Dec in degrees, inv_dL in Gpc^-1,
Mc in detector-frame solar masses, eta dimensionless, and iota, psi, and
phi_coal in radians. The HDF5 stores samples, source indices, progress status,
and enough metadata to identify the population and SNR inputs. It does not
duplicate the event-property or SNR tables; join those from the matched NPZ
files using host catalogue ID, population draw ID, and source_index.

The HDF5 file is opened in append mode and receives one population-draw group
at a time. Only one process writes a given HDF5 file; Fisher events run in
parallel and the parent process writes results. Do not run concurrent jobs
against the same cXXXX.h5. Existing completed event rows are skipped on
rerun; incomplete or failed rows are retried.

The number of Fisher-Gaussian vectors per detected event is independent of
the 100 population draws. It currently defaults to 100 and is set with
--gaussian-draws. Use --gaussian-draws 1 for one vector per detected event.

~~~bash
python scripts/run_s22_fisher_hdf5.py \
  --population-root /path/to/GWseeds/Populations/s22_100cat \
  --snr-root /path/to/GWseeds/Populations/s22_100cat/SNR \
  --output-root /path/to/GWseeds/Populations/s22_100cat/FisherSamples \
  --workers 16 --native-threads 1 --gaussian-draws 100
~~~

This is a Fisher-Gaussian approximation around the injected point, not a
GWDALI posterior sampler run. The existing skysim_gw/stages/localize.py
serial stage separately writes catalogue-wide Fisher and covariance arrays
to NPZ; its MAX_NEW_SOURCES=9 setting is a smoke-run limit. The grouped
HDF5 workflow uses the parallel adapter and is not limited by that setting.

## Structure summary

| Product | One file represents | Used by |
|---|---|---|
| Seed HDF5 | 100 host catalogues, selected host rows, properties | Population preparation |
| Population NPZ | One host catalogue × one population draw | SNR and Fisher stages |
| SNR NPZ | SNR results aligned with one population NPZ | Detection selection and Fisher stage |
| Fisher-sample HDF5 | One host catalogue, containing its population-draw groups | Downstream analysis of Fisher-Gaussian vectors |

The population and SNR stages therefore use 10,000 matched NPZ pairs. The
Fisher-Gaussian output uses 100 HDF5 files containing 100 groups each. The
HDF5 output is not an input format for the current GWDALI SNR stage.

## Implementation status

The current working tree contains the preparation, batch SNR, combined
preparation/SNR, and grouped Fisher-HDF5 code described above. Production
catalogue generation and Fisher calculations have not been run here. Syntax
compilation, command help, and whitespace checks were run; no scientific
production validation has been performed.
