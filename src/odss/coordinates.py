"""Explicit astronomical time and geocentric frame conversions."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import astropy.units as u
from astropy.coordinates import (
    GCRS,
    ITRS,
    TEME,
    BaseCoordinateFrame,
    CartesianDifferential,
    CartesianRepresentation,
)
from astropy.time import Time, TimeDelta
from astropy.utils import iers

from ._core import CartesianState, Epoch, ReferenceFrame
from .provenance import InputAssetMetadata

_supported_time_scales = frozenset({"TAI", "TT", "UTC"})
_supported_frames = frozenset({"GCRS", "ITRS", "TEME"})


def _normalize_time_scale(time_scale: str) -> str:
    if not isinstance(time_scale, str):
        raise ValueError("time_scale must be UTC, TAI, or TT")
    normalized = time_scale.upper()
    if normalized not in _supported_time_scales:
        raise ValueError("time_scale must be UTC, TAI, or TT")
    return normalized


def _time_from_epoch(epoch: Epoch) -> Time:
    if not isinstance(epoch, Epoch):
        raise TypeError("epoch must be an Epoch")
    time_scale = _normalize_time_scale(epoch.time_scale)
    reference = Time(epoch.reference_epoch, format="isot", scale=time_scale.lower())
    return reference + TimeDelta(epoch.offset_s, format="sec")


def _iso_value(time: Time) -> str:
    time.precision = 9
    return str(time.to_value("isot"))


def epoch_from_iso(value: str, time_scale: str) -> Epoch:
    """Create an epoch from an ISO-8601 timestamp in an explicit time scale."""
    normalized_scale = _normalize_time_scale(time_scale)
    with iers.conf.set_temp("auto_download", False):
        time = Time(value, format="isot", scale=normalized_scale.lower())
        return Epoch(0.0, _iso_value(time), normalized_scale)


def epoch_to_iso(epoch: Epoch, time_scale: str | None = None) -> str:
    """Represent an epoch as ISO-8601 in its own or an explicitly requested scale."""
    if not isinstance(epoch, Epoch):
        raise TypeError("epoch must be an Epoch")
    target_scale = epoch.time_scale if time_scale is None else time_scale
    normalized_scale = _normalize_time_scale(target_scale)
    with iers.conf.set_temp("auto_download", False):
        time = getattr(_time_from_epoch(epoch), normalized_scale.lower())
        return _iso_value(time)


def convert_epoch(epoch: Epoch, time_scale: str) -> Epoch:
    """Represent the same instant in another supported time scale."""
    normalized_scale = _normalize_time_scale(time_scale)
    return Epoch(0.0, epoch_to_iso(epoch, normalized_scale), normalized_scale)


def elapsed_time_s(start: Epoch, end: Epoch) -> float:
    """Return SI seconds elapsed from start to end, including UTC leap seconds."""
    with iers.conf.set_temp("auto_download", False):
        return float((_time_from_epoch(end) - _time_from_epoch(start)).to_value(u.s))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class EarthOrientationData:
    """Loaded IERS Earth-orientation data with content identity for provenance."""

    bulletin: Literal["A", "B"]
    metadata: InputAssetMetadata
    _table: iers.IERS = field(repr=False, compare=False)

    def __init__(self, path: str | os.PathLike[str], bulletin: Literal["A", "B"]) -> None:
        if bulletin not in ("A", "B"):
            raise ValueError("bulletin must be A or B")
        source = Path(path)
        table_type = iers.IERS_A if bulletin == "A" else iers.IERS_B
        table = table_type.open(str(source))
        metadata = InputAssetMetadata(
            logical_name=source.name,
            content_sha256=_sha256_file(source),
            size_bytes=source.stat().st_size,
        )
        object.__setattr__(self, "bulletin", bulletin)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "_table", table)


def bundled_iers_a() -> EarthOrientationData:
    """Load Astropy's bundled predictive IERS Bulletin A table."""
    return EarthOrientationData(iers.IERS_A_FILE, "A")


def bundled_iers_b() -> EarthOrientationData:
    """Load Astropy's bundled definitive IERS Bulletin B table."""
    return EarthOrientationData(iers.IERS_B_FILE, "B")


def _frame(
    identifier: str, representation: CartesianRepresentation, obstime: Time
) -> BaseCoordinateFrame:
    if identifier == "TEME":
        return TEME(representation, obstime=obstime)
    if identifier == "GCRS":
        return GCRS(representation, obstime=obstime)
    if identifier == "ITRS":
        return ITRS(representation, obstime=obstime)
    raise ValueError(f"unsupported reference frame: {identifier}")


def _empty_frame(identifier: str, obstime: Time) -> BaseCoordinateFrame:
    if identifier == "TEME":
        return TEME(obstime=obstime)
    if identifier == "GCRS":
        return GCRS(obstime=obstime)
    if identifier == "ITRS":
        return ITRS(obstime=obstime)
    raise ValueError(f"unsupported reference frame: {identifier}")


def transform_state(
    state: CartesianState,
    target_frame: ReferenceFrame,
    earth_orientation: EarthOrientationData,
) -> CartesianState:
    """Explicitly transform a Cartesian state between TEME, GCRS, and ITRS."""
    if not isinstance(state, CartesianState):
        raise TypeError("state must be a CartesianState")
    if not isinstance(target_frame, ReferenceFrame):
        raise TypeError("target_frame must be a ReferenceFrame")
    if not isinstance(earth_orientation, EarthOrientationData):
        raise TypeError("earth_orientation must be EarthOrientationData")
    if state.frame.identifier not in _supported_frames:
        raise ValueError(f"unsupported reference frame: {state.frame.identifier}")
    if target_frame.identifier not in _supported_frames:
        raise ValueError(f"unsupported reference frame: {target_frame.identifier}")
    if state.frame == target_frame:
        return state

    position = CartesianRepresentation(
        state.position_m[0] * u.m,
        state.position_m[1] * u.m,
        state.position_m[2] * u.m,
    )
    velocity = CartesianDifferential(
        state.velocity_m_s[0] * u.m / u.s,
        state.velocity_m_s[1] * u.m / u.s,
        state.velocity_m_s[2] * u.m / u.s,
    )

    with iers.conf.set_temp("auto_download", False):
        obstime = _time_from_epoch(state.epoch)
        source = _frame(state.frame.identifier, position.with_differentials(velocity), obstime)
        with iers.earth_orientation_table.set(earth_orientation._table):
            transformed = source.transform_to(_empty_frame(target_frame.identifier, obstime))

    cartesian = transformed.cartesian
    differential = cartesian.differentials["s"]
    return CartesianState(
        position_m=tuple(float(value) for value in cartesian.xyz.to_value(u.m)),
        velocity_m_s=tuple(float(value) for value in differential.d_xyz.to_value(u.m / u.s)),
        epoch=state.epoch,
        frame=target_frame,
    )


__all__ = [
    "EarthOrientationData",
    "bundled_iers_a",
    "bundled_iers_b",
    "convert_epoch",
    "elapsed_time_s",
    "epoch_from_iso",
    "epoch_to_iso",
    "transform_state",
]
