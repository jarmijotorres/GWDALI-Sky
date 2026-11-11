"""Load and validate SkySim5000 source catalogues for GWDALI.

The NPZ files are produced by ``prepare_gw_injections.py``.  Each row can be
converted to a scalar ``GwPrms`` dictionary accepted by ``GWDALI.get_SNR`` and
``GWDALI.GWDALI``.
"""

from pathlib import Path

import numpy as np


from skysim_gw.settings.catalogue import (
    BASE_DIR,
    INPUT_DIR,
    SOURCE_TYPE,
    ZI,
    ZF,
)


GWDALI_COLUMNS = (
    "m1", "m2",
    "RA", "Dec",
    "psi", "t_coal", "phi_coal",
    "dL", "iota",
    "sx1", "sy1", "sz1",
    "sx2", "sy2", "sz2",
)


def catalog_path(input_dir, source_type, zi, zf):
    """Construct the filename written by prepare_gw_injections.py."""
    source_type = source_type.upper()
    if source_type not in ("BBH", "BHNS", "BNS"):
        raise ValueError("source_type must be BBH, BHNS, or BNS")
    return Path(input_dir) / (
        f"skysim5000_{source_type}_gwdali_z{zi:.4f}_{zf:.4f}.npz"
    )


def validate_catalogue(catalog):
    """Validate schema and basic physical bounds of a GWDALI table."""
    if catalog.dtype.names is None:
        raise TypeError("Expected a structured row-column array")

    missing = [name for name in GWDALI_COLUMNS if name not in catalog.dtype.names]
    if missing:
        raise KeyError(f"Catalogue is missing GWDALI columns: {missing}")

    for name in GWDALI_COLUMNS:
        if not np.issubdtype(catalog.dtype[name], np.number):
            raise TypeError(f"Column {name!r} is not numeric")

    if len(catalog) == 0:
        return

    finite = np.ones(len(catalog), dtype=bool)
    for name in GWDALI_COLUMNS:
        finite &= np.isfinite(catalog[name])
    if not np.all(finite):
        bad = np.flatnonzero(~finite)
        raise ValueError(
            f"Found {bad.size} rows with non-finite GWDALI values; "
            f"first bad index is {bad[0]}"
        )

    if np.any(catalog["m1"] < catalog["m2"]):
        raise ValueError("Found rows with m1 < m2")
    if np.any(catalog["m2"] <= 0.0):
        raise ValueError("Component masses must be positive")
    if np.any(catalog["dL"] <= 0.0):
        raise ValueError("Luminosity distance dL must be positive")
    if np.any((catalog["RA"] < 0.0) | (catalog["RA"] >= 2.0 * np.pi)):
        raise ValueError("RA must satisfy 0 <= RA < 2*pi radians")
    if np.any((catalog["Dec"] < -0.5 * np.pi) | (catalog["Dec"] > 0.5 * np.pi)):
        raise ValueError("Dec must satisfy -pi/2 <= Dec <= pi/2 radians")
    if np.any((catalog["iota"] < 0.0) | (catalog["iota"] > np.pi)):
        raise ValueError("iota must satisfy 0 <= iota <= pi radians")

    for component in (1, 2):
        spin_squared = sum(
            catalog[f"s{axis}{component}"] ** 2 for axis in ("x", "y", "z")
        )
        if np.any(spin_squared > 1.0 + 1.0e-12):
            raise ValueError(f"Found dimensionless spin-{component} magnitude > 1")


def load_gwdali_catalogue(path, validate=True):
    """Load the table and metadata from one compressed NPZ file.

    NPZ compression requires the array to be decompressed into memory.  The
    returned table is independent of the closed ``np.load`` context.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        if "catalog" not in data.files:
            raise KeyError(f"Missing 'catalog' array in {path}")
        catalog = data["catalog"]
        metadata = {
            name: data[name].item() if data[name].ndim == 0 else data[name].copy()
            for name in data.files
            if name != "catalog"
        }

    if validate:
        validate_catalogue(catalog)
    return catalog, metadata


def catalogue_source_from_row(row):
    """Return a stored source without changing its catalogue units."""
    if row.dtype.names is None:
        raise TypeError("Expected one row from a structured catalogue")
    missing = [name for name in GWDALI_COLUMNS if name not in row.dtype.names]
    if missing:
        raise KeyError(f"Row is missing GWDALI parameters: {missing}")
    return {name: float(row[name]) for name in GWDALI_COLUMNS}


def gwdali_source_from_row(
    row, distance_parameter="dL", mass_parameterization="m1_m2"
):
    """Convert a stored row to the parameterisation required by GWDALI.

    The catalogue stores detector-frame component masses and dL in Gpc.  For
    Fisher localisation it can instead pass `(Mc, eta)` and `inv_dL = 1/dL`,
    while leaving the stored catalogue unchanged.
    """
    if distance_parameter not in ("dL", "inv_dL"):
        raise ValueError(
            "distance_parameter must be either 'dL' or 'inv_dL'"
        )
    if mass_parameterization not in ("m1_m2", "mc_eta"):
        raise ValueError(
            "mass_parameterization must be either 'm1_m2' or 'mc_eta'"
        )

    source = catalogue_source_from_row(row)

    # GWDALI sky-coordinate convention
    source["RA"] = float(np.degrees(source["RA"]))
    source["Dec"] = float(np.degrees(source["Dec"]))

    if mass_parameterization == "mc_eta":
        m1 = source.pop("m1")
        m2 = source.pop("m2")
        total_mass = m1 + m2
        eta = m1 * m2 / total_mass**2
        source["Mc"] = (m1 * m2) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0)
        source["eta"] = eta

    if distance_parameter == "inv_dL":
        dL = source.pop("dL")

        if dL <= 0.0:
            raise ValueError("Luminosity distance dL must be positive")

        source["inv_dL"] = 1.0 / dL

    return source


def get_catalogue_source(catalog, index):
    """Return one checked source in the units stored in the catalogue."""
    index = int(index)
    if index < 0 or index >= len(catalog):
        raise IndexError(f"Source index {index} outside [0, {len(catalog)})")
    return catalogue_source_from_row(catalog[index])


def get_gwdali_source(
    catalog, index, distance_parameter="dL", mass_parameterization="m1_m2"
):
    """Return one checked source in the units expected by GWDALI."""
    index = int(index)

    if index < 0 or index >= len(catalog):
        raise IndexError(
            f"Source index {index} outside [0, {len(catalog)})"
        )

    return gwdali_source_from_row(
        catalog[index],
        distance_parameter=distance_parameter,
        mass_parameterization=mass_parameterization,
    )

def iter_gwdali_sources(catalog, start=0, stop=None, step=1):
    """Yield ``(index, GwPrms)`` without constructing a list of dictionaries."""
    if stop is None:
        stop = len(catalog)
    source_slice = slice(start, stop, step)
    begin, end, stride = source_slice.indices(len(catalog))
    for index in range(begin, end, stride):
        yield index, gwdali_source_from_row(catalog[index])


if __name__ == "__main__":
    path = catalog_path(INPUT_DIR, SOURCE_TYPE, ZI, ZF)
    catalog, metadata = load_gwdali_catalogue(path)

    print(f"Loaded: {path}")
    print(f"Sources: {len(catalog):,}")
    print(f"Columns: {catalog.dtype.names}")
    print(f"Metadata keys: {tuple(metadata)}")

    if len(catalog):
        source = get_gwdali_source(catalog, 0)
        print("\nFirst GwPrms source:")
        for name, value in source.items():
            print(f"  {name:8s} = {value:.10g}")


# Backwards-compatible aliases for older scripts. New code uses British
# spelling, but existing injection files retain their internal ``catalog`` key.
validate_catalog = validate_catalogue
load_gwdali_catalog = load_gwdali_catalogue
