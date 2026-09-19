"""SGP4 propagation of catalog mean elements to a common epoch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from astropy.time import Time, TimeDelta
from sgp4.api import SGP4_ERRORS, WGS72, Satrec, SatrecArray

from ._core import CartesianState, Epoch, ReferenceFrame
from .catalog import Catalog, CatalogAcquisition, OmmRecord
from .coordinates import elapsed_time_s, epoch_to_iso

_minutes_per_day = 1_440.0
_sgp4_epoch_julian_date = 2_433_281.5
_maximum_sgp4_satellite_number = 339_999
_teme = ReferenceFrame("TEME")


@dataclass(frozen=True, slots=True)
class SynchronizedCatalogObject:
    """An original mean-element record and its propagated TEME state."""

    record: OmmRecord
    state: CartesianState
    propagation_offset_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.record, OmmRecord):
            raise TypeError("record must be an OmmRecord")
        if not isinstance(self.state, CartesianState):
            raise TypeError("state must be a CartesianState")
        if not math.isfinite(self.propagation_offset_s):
            raise ValueError("propagation_offset_s must be finite")
        if self.state.frame != _teme:
            raise ValueError("SGP4 state frame must be TEME")


@dataclass(frozen=True, slots=True)
class SynchronizedCatalog:
    """Catalog objects propagated to one shared absolute epoch."""

    epoch: Epoch
    objects: tuple[SynchronizedCatalogObject, ...]
    acquisitions: tuple[CatalogAcquisition, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, Epoch):
            raise TypeError("epoch must be an Epoch")
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "acquisitions", tuple(self.acquisitions))
        if any(not isinstance(item, SynchronizedCatalogObject) for item in self.objects):
            raise TypeError("objects must contain only SynchronizedCatalogObject values")
        if any(not isinstance(item, CatalogAcquisition) for item in self.acquisitions):
            raise TypeError("acquisitions must contain only CatalogAcquisition values")
        if any(item.state.epoch != self.epoch for item in self.objects):
            raise ValueError("all states must use the catalog synchronization epoch")
        catalog_ids = tuple(item.record.catalog_id for item in self.objects)
        if catalog_ids != tuple(sorted(catalog_ids)) or len(catalog_ids) != len(set(catalog_ids)):
            raise ValueError("objects must have unique catalog IDs in ascending order")

    def __len__(self) -> int:
        return len(self.objects)


def _utc_julian_date(epoch: Epoch) -> tuple[float, float]:
    if not isinstance(epoch, Epoch):
        raise TypeError("epoch must be an Epoch")
    time = Time(epoch_to_iso(epoch, "UTC"), format="isot", scale="utc")
    return float(time.jd1), float(time.jd2)


def _satrec(record: OmmRecord, epoch_days: float | None = None) -> Satrec:
    if not isinstance(record, OmmRecord):
        raise TypeError("record must be an OmmRecord")
    if epoch_days is None:
        epoch_jd, epoch_fraction = _utc_julian_date(record.epoch)
        epoch_days = epoch_jd - _sgp4_epoch_julian_date + epoch_fraction
    satellite = Satrec()
    # The backend stores legacy Alpha-5 metadata even for OMM input. This value does not enter the
    # equations of motion; ODSS retains the complete nine-digit catalog ID on the original record.
    satellite_number = (
        record.catalog_id if record.catalog_id <= _maximum_sgp4_satellite_number else 0
    )
    satellite.sgp4init(
        WGS72,
        "i",
        satellite_number,
        epoch_days,
        record.bstar_1_earth_radii,
        record.mean_motion_dot_rev_day2
        * 2.0
        * math.pi
        / _minutes_per_day**2,
        record.mean_motion_ddot_rev_day3
        * 2.0
        * math.pi
        / _minutes_per_day**3,
        record.eccentricity,
        math.radians(record.argument_of_pericenter_deg),
        math.radians(record.inclination_deg),
        math.radians(record.mean_anomaly_deg),
        record.mean_motion_rev_day * 2.0 * math.pi / _minutes_per_day,
        math.radians(record.raan_deg),
    )
    return satellite


def _state(
    record: OmmRecord,
    epoch: Epoch,
    error_code: int,
    position_km: np.ndarray | tuple[float, float, float],
    velocity_km_s: np.ndarray | tuple[float, float, float],
    propagation_offset_s: float | None = None,
) -> SynchronizedCatalogObject:
    if error_code:
        detail = SGP4_ERRORS.get(error_code, "unknown SGP4 error")
        raise RuntimeError(f"SGP4 failed for catalog ID {record.catalog_id}: {detail}")
    state = CartesianState(
        position_m=tuple(float(component) * 1_000.0 for component in position_km),
        velocity_m_s=tuple(float(component) * 1_000.0 for component in velocity_km_s),
        epoch=epoch,
        frame=_teme,
    )
    return SynchronizedCatalogObject(
        record=record,
        state=state,
        propagation_offset_s=(
            elapsed_time_s(record.epoch, epoch)
            if propagation_offset_s is None
            else propagation_offset_s
        ),
    )


def propagate_omm_sgp4(record: OmmRecord, epoch: Epoch) -> SynchronizedCatalogObject:
    """Propagate one OMM record with the reference WGS-72 SGP4 implementation."""
    julian_date, fraction = _utc_julian_date(epoch)
    error_code, position_km, velocity_km_s = _satrec(record).sgp4(julian_date, fraction)
    return _state(record, epoch, error_code, position_km, velocity_km_s)


def synchronize_catalog_sgp4(catalog: Catalog, epoch: Epoch) -> SynchronizedCatalog:
    """Batch-propagate every catalog record to one epoch with WGS-72 SGP4."""
    if not isinstance(catalog, Catalog):
        raise TypeError("catalog must be a Catalog")
    julian_date, fraction = _utc_julian_date(epoch)
    if not catalog.records:
        return SynchronizedCatalog(epoch=epoch, objects=(), acquisitions=catalog.acquisitions)

    record_times = Time(
        [record.epoch.reference_epoch for record in catalog.records],
        format="isot",
        scale="utc",
    ) + TimeDelta(
        np.array([record.epoch.offset_s for record in catalog.records]),
        format="sec",
    )
    epoch_days = (
        np.asarray(record_times.jd1) - _sgp4_epoch_julian_date + np.asarray(record_times.jd2)
    )
    target_time = Time(epoch_to_iso(epoch, "UTC"), format="isot", scale="utc")
    propagation_offsets_s = np.asarray((target_time.tai - record_times.tai).to_value("s"))

    satellites = SatrecArray(
        [_satrec(record, float(epoch_days[index])) for index, record in enumerate(catalog.records)]
    )
    errors, positions_km, velocities_km_s = satellites.sgp4(
        np.array([julian_date]),
        np.array([fraction]),
    )
    objects = tuple(
        _state(
            record,
            epoch,
            int(errors[index, 0]),
            positions_km[index, 0],
            velocities_km_s[index, 0],
            float(propagation_offsets_s[index]),
        )
        for index, record in enumerate(catalog.records)
    )
    return SynchronizedCatalog(epoch=epoch, objects=objects, acquisitions=catalog.acquisitions)


__all__ = [
    "SynchronizedCatalog",
    "SynchronizedCatalogObject",
    "propagate_omm_sgp4",
    "synchronize_catalog_sgp4",
]
