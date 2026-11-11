"""Preserved settings for skysim_gw.diagnostics.fisher. Edit before a new production run."""
from pathlib import Path
from .common import (BASE_DIR, SOURCE_TYPE, ZI, ZF)

INPUT_DIR = BASE_DIR / "GWInputCatalogs"

LOCALIZATION_DIR = INPUT_DIR / "LocalizationFisher"

DIAGNOSTIC_DIR = LOCALIZATION_DIR / "Diagnostics"

CONDITION_WARNING = 1.0e10
