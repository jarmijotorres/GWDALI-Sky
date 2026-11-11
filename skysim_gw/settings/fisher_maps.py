"""Preserved settings for skysim_gw.stages.fisher_maps. Edit before a new production run."""
from pathlib import Path
from .common import (BASE_DIR, REDSHIFT_SLICE_FILE, SOURCE_TYPE, ZI, ZF)

INPUT_DIR = BASE_DIR / "GWInputCatalogs"

LOCALISATION_DIR = INPUT_DIR / "LocalizationFisher"

OUTPUT_DIR = INPUT_DIR / "Healpix"

SLICE_OUTPUT_DIR = OUTPUT_DIR / "RedshiftSlices"

NSIDE = 512

NEST = False

NSIGMA = 6.0

MAX_EVENTS = None

PRINT_EVERY = 10
