"""Empirical Santoliquido et al. (2022) BBH host-rate templates.

The S22 catalogues are reduced to a continuous, rate-weighted function of
merger redshift, stellar mass, and SFR. Metallicity is marginalized out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


S22_LOW_REDSHIFT_SNAPSHOTS = {
    28: 0.0, 27: 0.10063853859901428, 26: 0.18270987272262573,
    25: 0.27090108394622803, 24: 0.36566856503486633,
    23: 0.5031073093414307, 22: 0.615189790725708,
    21: 0.7356296181678772, 20: 0.8650505542755127,
    19: 1.0041216611862183, 18: 1.259, 17: 1.487,
    16: 1.737, 15: 2.012,
}
S22_VOLUME_CMPC3 = 100.0**3


@dataclass(frozen=True)
class S22BBHParameters:
    """Global normalization and mass/SFR tilts relative to the S22 tables."""

    rate_scale: float = 1.0
    mass_tilt: float = 0.0
    sfr_tilt: float = 0.0

    def validate(self) -> None:
        values = np.asarray((self.rate_scale, self.mass_tilt, self.sfr_tilt))
        if not np.all(np.isfinite(values)) or self.rate_scale < 0.0:
            raise ValueError("S22 parameters must be finite and rate_scale non-negative")


def _centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


class S22BBHEmpiricalModel:
    """Continuous, metallicity-marginalized S22 BBH host-rate model.

    ``rate_per_galaxy`` weights galaxies in a lightcone. M57/M54 selects the
    S22 catalogue calibration but does not require a metallicity per galaxy.
    """

    def __init__(
        self, data_dir: str | Path, metallicity_relation="M57", common_envelope="A3",
        snapshots=None, mass_edges=None, sfr_edges=None,
    ):
        if metallicity_relation not in {"M54", "M57"}:
            raise ValueError("metallicity_relation must be M54 or M57")
        if common_envelope not in {"A1", "A3", "A5"}:
            raise ValueError("common_envelope must be A1, A3, or A5")
        self.data_dir = Path(data_dir)
        self.metallicity_relation = metallicity_relation
        self.common_envelope = common_envelope
        source_snapshots = S22_LOW_REDSHIFT_SNAPSHOTS if snapshots is None else snapshots
        ordered = sorted(source_snapshots.items(), key=lambda item: item[1])
        self.snapshot_numbers = np.asarray([item[0] for item in ordered], dtype=int)
        self.redshifts = np.asarray([item[1] for item in ordered], dtype=float)
        if self.redshifts.size < 2 or np.any(np.diff(self.redshifts) <= 0):
            raise ValueError("snapshots must contain at least two increasing redshifts")
        self.mass_edges = np.asarray(np.arange(7.0, 12.05, 0.1) if mass_edges is None else mass_edges, dtype=float)
        self.sfr_edges = np.asarray(np.arange(-4.0, 2.55, 0.1) if sfr_edges is None else sfr_edges, dtype=float)
        for name, edges in (("mass_edges", self.mass_edges), ("sfr_edges", self.sfr_edges)):
            if edges.ndim != 1 or edges.size < 3 or np.any(np.diff(edges) <= 0):
                raise ValueError(f"{name} must have at least two increasing bins")
        self._rate_sum, self._occupancy = self._load_templates()
        self._mean_rate = np.divide(
            self._rate_sum, self._occupancy, out=np.full_like(self._rate_sum, np.nan),
            where=self._occupancy > 0,
        )

    def _path(self, snapshot):
        return self.data_dir / (
            f"HG_{self.metallicity_relation}_{self.common_envelope}_"
            f"GSSI_v2_BBHs_{snapshot}_12Z_50_No_0.14_2.txt"
        )

    def _load_templates(self):
        shape = (self.snapshot_numbers.size, self.mass_edges.size - 1, self.sfr_edges.size - 1)
        rate_sum, occupancy = np.zeros(shape), np.zeros(shape)
        for index, snapshot in enumerate(self.snapshot_numbers):
            path = self._path(int(snapshot))
            if not path.is_file():
                raise FileNotFoundError(f"Missing S22 BBH catalogue: {path}")
            values = np.loadtxt(path)
            if values.ndim != 2 or values.shape[1] < 4:
                raise ValueError(f"Invalid S22 BBH catalogue: {path}")
            mass, sfr, rate = values[:, 0], values[:, 1], values[:, 3]
            valid = np.isfinite(mass) & (mass > 0) & np.isfinite(sfr) & (sfr > 0) & np.isfinite(rate) & (rate >= 0)
            points = np.column_stack((np.log10(mass[valid]), np.log10(sfr[valid])))
            rate_sum[index] = np.histogramdd(points, bins=(self.mass_edges, self.sfr_edges), weights=rate[valid])[0]
            occupancy[index] = np.histogramdd(points, bins=(self.mass_edges, self.sfr_edges))[0]
        return rate_sum, occupancy

    def _bracket(self, redshift):
        z = np.asarray(redshift, dtype=float)
        if np.any(~np.isfinite(z)) or np.any(z < self.redshifts[0]) or np.any(z > self.redshifts[-1]):
            raise ValueError(f"redshift must be in [{self.redshifts[0]}, {self.redshifts[-1]}]")
        upper = np.clip(np.searchsorted(self.redshifts, z, side="right"), 1, len(self.redshifts) - 1)
        lower = upper - 1
        return lower, upper, (z - self.redshifts[lower]) / (self.redshifts[upper] - self.redshifts[lower])

    @staticmethod
    def _bilinear(grid, x, y, x_edges, y_edges):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        result = np.zeros(x.shape)
        xc, yc = _centers(x_edges), _centers(y_edges)
        valid = np.isfinite(x) & np.isfinite(y) & (x >= xc[0]) & (x <= xc[-1]) & (y >= yc[0]) & (y <= yc[-1])
        if not np.any(valid):
            return result
        ix1 = np.clip(np.searchsorted(xc, x, side="right"), 1, len(xc) - 1)
        iy1 = np.clip(np.searchsorted(yc, y, side="right"), 1, len(yc) - 1)
        ix0, iy0 = ix1 - 1, iy1 - 1
        fx = (x - xc[ix0]) / (xc[ix1] - xc[ix0])
        fy = (y - yc[iy0]) / (yc[iy1] - yc[iy0])
        numerator, denominator = np.zeros(x.shape), np.zeros(x.shape)
        for dx in (0, 1):
            for dy in (0, 1):
                weight = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                corner = grid[ix0 + dx, iy0 + dy]
                usable = valid & np.isfinite(corner)
                numerator[usable] += weight[usable] * corner[usable]
                denominator[usable] += weight[usable]
        return np.divide(numerator, denominator, out=result, where=denominator > 0)

    def _tilt_grid(self, grid, parameters):
        mass, sfr = np.meshgrid(_centers(self.mass_edges), _centers(self.sfr_edges), indexing="ij")
        return grid * parameters.rate_scale * 10.0 ** (parameters.mass_tilt * (mass - 10) + parameters.sfr_tilt * sfr)

    def host_rate_grid(self, redshift: float, parameters: S22BBHParameters | None = None):
        """Rate-weighted host grid in Gpc^-3 yr^-1 per (Mstar,SFR) bin."""
        parameters = S22BBHParameters() if parameters is None else parameters
        parameters.validate()
        low, high, fraction = self._bracket(float(redshift))
        return self._tilt_grid(((1 - fraction) * self._rate_sum[low] + fraction * self._rate_sum[high]) / S22_VOLUME_CMPC3, parameters)

    def rate_per_galaxy(self, redshift, stellar_mass, sfr, parameters: S22BBHParameters | None = None):
        """Interpolated BBH rate per galaxy in Gyr^-1."""
        parameters = S22BBHParameters() if parameters is None else parameters
        parameters.validate()
        redshift, stellar_mass, sfr = np.broadcast_arrays(np.asarray(redshift, dtype=float), np.asarray(stellar_mass, dtype=float), np.asarray(sfr, dtype=float))
        log_mass, log_sfr = np.full(redshift.shape, np.nan), np.full(redshift.shape, np.nan)
        np.log10(stellar_mass, out=log_mass, where=stellar_mass > 0)
        np.log10(sfr, out=log_sfr, where=sfr > 0)
        lower, _, fraction = self._bracket(redshift)
        result = np.zeros(redshift.shape)
        for low in np.unique(lower):
            choose = lower == low
            first = self._bilinear(self._mean_rate[low], log_mass[choose], log_sfr[choose], self.mass_edges, self.sfr_edges)
            second = self._bilinear(self._mean_rate[low + 1], log_mass[choose], log_sfr[choose], self.mass_edges, self.sfr_edges)
            result[choose] = (1 - fraction[choose]) * first + fraction[choose] * second
        tilt = parameters.mass_tilt * (log_mass - 10) + parameters.sfr_tilt * log_sfr
        result *= parameters.rate_scale * np.where(np.isfinite(tilt), 10.0 ** tilt, 0)
        result[~np.isfinite(result)] = 0
        return float(result) if result.ndim == 0 else result

    def sample_hosts(self, redshift: float, size: int, rng=None, parameters: S22BBHParameters | None = None):
        """Sample rate-weighted S22 stellar mass and SFR at one redshift."""
        if np.ndim(redshift) != 0 or size < 0:
            raise ValueError("redshift must be scalar and size non-negative")
        grid = self.host_rate_grid(float(redshift), parameters)
        if not np.any(grid > 0):
            raise ValueError("No positive S22 host weights at this redshift")
        rng = np.random.default_rng() if rng is None else rng
        chosen = rng.choice(grid.size, size=size, p=grid.ravel() / grid.sum())
        i_mass, i_sfr = np.unravel_index(chosen, grid.shape)
        return {
            "stellar_mass": 10.0 ** rng.uniform(self.mass_edges[i_mass], self.mass_edges[i_mass + 1]),
            "sfr": 10.0 ** rng.uniform(self.sfr_edges[i_sfr], self.sfr_edges[i_sfr + 1]),
        }


__all__ = ["S22BBHEmpiricalModel", "S22BBHParameters", "S22_LOW_REDSHIFT_SNAPSHOTS", "S22_VOLUME_CMPC3"]
