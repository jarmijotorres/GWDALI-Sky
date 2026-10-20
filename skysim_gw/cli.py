"""Lazy stage launcher: inspection requires only Python's standard library."""
import argparse
import importlib
import shlex
import sys

STAGES = {
    'rate-targets': ('skysim_gw.stages.rate_targets', 'Optional seed-building prerequisite: rate targets from --config; no GCR required.'),
    'assign-hosts': ('skysim_gw.stages.assign_hosts', 'NERSC-only seed building: GCR/SkySim5000 + MPI; requires --config.'),
    'prepare': ('skysim_gw.population', 'Portable pipeline start: read seed-host HDF5 and sample injections.'),
    'check-snr': ('skysim_gw.diagnostics.one_snr', 'Validate SNR for one source.'),
    'snr': ('skysim_gw.stages.snr', 'Compute checkpointed SNR catalogue.'),
    'detect': ('skysim_gw.stages.detect', 'Write detected original row indices: SNR_FILE OUTPUT.'),
    'check-localisation': ('skysim_gw.diagnostics.one_localisation', 'Validate Fisher localisation for one detected source.'),
    'doublet-serial': ('skysim_gw.stages.localize', 'Compute Doublet summary catalogue; default limit is 9 new sources.'),
    'doublet-parallel': ('skysim_gw.stages.doublet_parallel', 'Write actual per-source Doublet posterior samples; see native --help.'),
    'compare': ('skysim_gw.diagnostics.compare', 'Compare existing Fisher and serial Doublet summary catalogues.'),
    'diagnose-fisher': ('skysim_gw.diagnostics.fisher', 'Inspect catalogue quality; verify input filename before running.'),
    'diagnose-localisation': ('skysim_gw.diagnostics.localisation', 'Investigate one-source numerical behaviour.'),
    'fisher-maps': ('skysim_gw.stages.fisher_maps', 'Map an existing Fisher covariance catalogue.'),
    'doublet-maps': ('skysim_gw.stages.doublet_maps', 'Map actual saved posterior samples using injected redshift.'),
}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Pass stage arguments after the stage name; use -- --help for native CLI help.')
    parser.add_argument('--list', action='store_true', help='Show stages in workflow order without importing scientific dependencies.')
    parser.add_argument('--dry-run', action='store_true', help='Print the module command without executing it.')
    parser.add_argument('stage', nargs='?', choices=STAGES)
    parser.add_argument('stage_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.list:
        for name, (_, description) in STAGES.items():
            print(f'{name:24s} {description}')
        return 0
    if args.stage is None:
        parser.error('choose a stage or use --list')
    module, _ = STAGES[args.stage]
    forwarded = args.stage_args
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    cli_stages = {'rate-targets', 'assign-hosts', 'snr', 'detect', 'doublet-parallel', 'doublet-maps'}
    if forwarded and args.stage not in cli_stages:
        parser.error(f'{args.stage} uses skysim_gw/settings/; it accepts no command-line options')
    if args.dry_run:
        print(shlex.join([sys.executable, '-m', module, *forwarded]))
        return 0
    previous = sys.argv
    try:
        sys.argv = [module, *forwarded]
        result = importlib.import_module(module).main()
        return 0 if result is None else result
    finally:
        sys.argv = previous
    return 0
