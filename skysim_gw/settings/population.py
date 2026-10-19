"""Preserved settings for skysim_gw.population. Edit before a new production run."""
from pathlib import Path
from .common import (BASE_DIR, ZI, ZF)

# Portable pipeline input: exported seed-host HDF5, not the full SkySim5000 catalogue.
SOURCE_FILE = BASE_DIR / "SkySim5000_gw_host_catalogue1.hdf5"

OUTPUT_DIR = BASE_DIR / "GWInputCatalogs"

SOURCE_TYPES = ("BBH", "BHNS", "BNS")

RANDOM_SEED = 12345

# BBH intrinsic component-mass prescription.  The phenomenological
# ``bimodal_pairing`` model is the production default; ``powerlaw_peak``
# remains available for comparison. Override any fields in BBH_MASS_PARAMETERS
# while retaining the remaining PopulationConfig defaults.
BBH_MASS_MODEL = "bimodal_pairing"
BBH_MASS_PARAMETERS = {}
