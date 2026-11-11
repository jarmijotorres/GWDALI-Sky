"""Preserved settings for skysim_gw.stages.localize. Edit before a new production run."""
from pathlib import Path
from .common import (BASE_DIR, SOURCE_TYPE, ZI, ZF, APPROXIMANTS, FMIN_HZ, FMAX_HZ, FSIZE)

INPUT_DIR = BASE_DIR / "GWInputCatalogs"

SNR_DIR = INPUT_DIR / "SNR"

OUTPUT_DIR = INPUT_DIR / "LocalizationFisher"

METHOD = "Fisher"

FISHER_RCOND = 1.0e-12

MAX_NEW_SOURCES = 9

CHECKPOINT_EVERY = 5

PRINT_EVERY = 5

RESUME = True
