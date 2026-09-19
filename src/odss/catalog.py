"""Immutable OMM/GP catalog ingestion and Sun-synchronous filtering."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ._core import Epoch
from .coordinates import elapsed_time_s, epoch_from_iso, epoch_to_iso
from .provenance import InputAssetMetadata

_seconds_per_day = 86_400.0
_tropical_year_days = 365.242_189_7
_sun_mean_rate_deg_day = 360.0 / _tropical_year_days
_wgs72_mu_m3_s2 = 398_600.8e9
_wgs72_earth_radius_m = 6_378_135.0
_wgs72_j2 = 0.001_082_616


def _require_text(value: object, field_name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _require_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_range(
    value: float,
    field_name: str,
    minimum: float,
    maximum: float,
    *,
    include_maximum: bool = True,
) -> None:
    valid = minimum <= value <= maximum if include_maximum else minimum <= value < maximum
    if not valid:
        boundary = "inclusive" if include_maximum else "exclusive maximum"
        raise ValueError(f"{field_name} must be in [{minimum}, {maximum}] ({boundary})")


@dataclass(frozen=True, slots=True)
class OmmRecord:
    """One Earth-centered TEME/UTC SGP4 mean-element record."""

    catalog_id: int
    object_name: str | None
    international_designator: str | None
    epoch: Epoch
    mean_motion_rev_day: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    argument_of_pericenter_deg: float
    mean_anomaly_deg: float
    ephemeris_type: int
    classification: str
    element_set_number: int
    revolution_number: int
    bstar_1_earth_radii: float
    mean_motion_dot_rev_day2: float
    mean_motion_ddot_rev_day3: float

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.catalog_id, "catalog_id"),
            (self.ephemeris_type, "ephemeris_type"),
            (self.element_set_number, "element_set_number"),
            (self.revolution_number, "revolution_number"),
        ):
            _require_integer(value, field_name)
        if not 1 <= self.catalog_id <= 999_999_999:
            raise ValueError("catalog_id must be in [1, 999999999]")
        _require_text(self.object_name, "object_name", optional=True)
        _require_text(self.international_designator, "international_designator", optional=True)
        if not isinstance(self.epoch, Epoch):
            raise TypeError("epoch must be an Epoch")
        if self.epoch.time_scale != "UTC":
            raise ValueError("OMM epoch must use UTC")
        for value, field_name in (
            (self.mean_motion_rev_day, "mean_motion_rev_day"),
            (self.eccentricity, "eccentricity"),
            (self.inclination_deg, "inclination_deg"),
            (self.raan_deg, "raan_deg"),
            (self.argument_of_pericenter_deg, "argument_of_pericenter_deg"),
            (self.mean_anomaly_deg, "mean_anomaly_deg"),
            (self.bstar_1_earth_radii, "bstar_1_earth_radii"),
            (self.mean_motion_dot_rev_day2, "mean_motion_dot_rev_day2"),
            (self.mean_motion_ddot_rev_day3, "mean_motion_ddot_rev_day3"),
        ):
            _require_float(value, field_name)
        if self.mean_motion_rev_day <= 0.0:
            raise ValueError("mean_motion_rev_day must be positive and finite")
        _require_range(self.eccentricity, "eccentricity", 0.0, 1.0, include_maximum=False)
        _require_range(self.inclination_deg, "inclination_deg", 0.0, 180.0)
        for value, field_name in (
            (self.raan_deg, "raan_deg"),
            (self.argument_of_pericenter_deg, "argument_of_pericenter_deg"),
            (self.mean_anomaly_deg, "mean_anomaly_deg"),
        ):
            _require_range(value, field_name, 0.0, 360.0, include_maximum=False)
        if self.ephemeris_type < 0:
            raise ValueError("ephemeris_type must be non-negative")
        if self.classification not in {"C", "S", "U"}:
            raise ValueError("classification must be C, S, or U")
        if self.element_set_number < 0:
            raise ValueError("element_set_number must be non-negative")
        if self.revolution_number < 0:
            raise ValueError("revolution_number must be non-negative")
        for value, field_name in (
            (self.bstar_1_earth_radii, "bstar_1_earth_radii"),
            (self.mean_motion_dot_rev_day2, "mean_motion_dot_rev_day2"),
            (self.mean_motion_ddot_rev_day3, "mean_motion_ddot_rev_day3"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")


@dataclass(frozen=True, slots=True)
class CatalogAcquisition:
    """Explicit source, acquisition epoch, and content identity for catalog input."""

    source_uri: str
    acquired_at: Epoch
    asset: InputAssetMetadata

    def __post_init__(self) -> None:
        _require_text(self.source_uri, "source_uri")
        if not isinstance(self.acquired_at, Epoch):
            raise TypeError("acquired_at must be an Epoch")
        epoch_to_iso(self.acquired_at, "TAI")
        if not isinstance(self.asset, InputAssetMetadata):
            raise TypeError("asset must be InputAssetMetadata")


@dataclass(frozen=True, slots=True)
class Catalog:
    """Immutable catalog with unique, catalog-ID-sorted records."""

    records: tuple[OmmRecord, ...]
    acquisitions: tuple[CatalogAcquisition, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "acquisitions", tuple(self.acquisitions))
        if any(not isinstance(record, OmmRecord) for record in self.records):
            raise TypeError("records must contain only OmmRecord values")
        if any(not isinstance(item, CatalogAcquisition) for item in self.acquisitions):
            raise TypeError("acquisitions must contain only CatalogAcquisition values")
        catalog_ids = tuple(record.catalog_id for record in self.records)
        if catalog_ids != tuple(sorted(catalog_ids)) or len(catalog_ids) != len(set(catalog_ids)):
            raise ValueError("records must have unique catalog IDs in ascending order")

    def __len__(self) -> int:
        return len(self.records)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _unique_json_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _field(item: Mapping[str, object], name: str) -> object:
    if name not in item:
        raise ValueError(f"missing OMM field: {name}")
    return item[name]


def _validate_fixed_metadata(item: Mapping[str, object], name: str, expected: str) -> None:
    if name in item and item[name] != expected:
        raise ValueError(f"{name} must be {expected}")


def _parse_omm_record(item: object) -> OmmRecord:
    if not isinstance(item, Mapping):
        raise ValueError("each OMM record must be a JSON object")
    _validate_fixed_metadata(item, "CENTER_NAME", "EARTH")
    _validate_fixed_metadata(item, "REF_FRAME", "TEME")
    _validate_fixed_metadata(item, "TIME_SYSTEM", "UTC")
    _validate_fixed_metadata(item, "MEAN_ELEMENT_THEORY", "SGP4")

    epoch_text = _require_text(_field(item, "EPOCH"), "EPOCH")
    assert epoch_text is not None
    if epoch_text.endswith("Z"):
        epoch_text = epoch_text[:-1]

    catalog_id = _require_integer(_field(item, "NORAD_CAT_ID"), "NORAD_CAT_ID")
    classification = _require_text(_field(item, "CLASSIFICATION_TYPE"), "CLASSIFICATION_TYPE")
    assert classification is not None
    return OmmRecord(
        catalog_id=catalog_id,
        object_name=_require_text(item.get("OBJECT_NAME"), "OBJECT_NAME", optional=True),
        international_designator=_require_text(
            item.get("OBJECT_ID"), "OBJECT_ID", optional=True
        ),
        epoch=epoch_from_iso(epoch_text, "UTC"),
        mean_motion_rev_day=_require_float(_field(item, "MEAN_MOTION"), "MEAN_MOTION"),
        eccentricity=_require_float(_field(item, "ECCENTRICITY"), "ECCENTRICITY"),
        inclination_deg=_require_float(_field(item, "INCLINATION"), "INCLINATION"),
        raan_deg=_require_float(_field(item, "RA_OF_ASC_NODE"), "RA_OF_ASC_NODE"),
        argument_of_pericenter_deg=_require_float(
            _field(item, "ARG_OF_PERICENTER"), "ARG_OF_PERICENTER"
        ),
        mean_anomaly_deg=_require_float(_field(item, "MEAN_ANOMALY"), "MEAN_ANOMALY"),
        ephemeris_type=_require_integer(_field(item, "EPHEMERIS_TYPE"), "EPHEMERIS_TYPE"),
        classification=classification,
        element_set_number=_require_integer(_field(item, "ELEMENT_SET_NO"), "ELEMENT_SET_NO"),
        revolution_number=_require_integer(_field(item, "REV_AT_EPOCH"), "REV_AT_EPOCH"),
        bstar_1_earth_radii=_require_float(_field(item, "BSTAR"), "BSTAR"),
        mean_motion_dot_rev_day2=_require_float(
            _field(item, "MEAN_MOTION_DOT"), "MEAN_MOTION_DOT"
        ),
        mean_motion_ddot_rev_day3=_require_float(
            _field(item, "MEAN_MOTION_DDOT"), "MEAN_MOTION_DDOT"
        ),
    )


def _preferred_record(first: OmmRecord, second: OmmRecord) -> OmmRecord:
    epoch_difference_s = elapsed_time_s(first.epoch, second.epoch)
    if epoch_difference_s > 0.0:
        return second
    if epoch_difference_s < 0.0:
        return first
    if first.element_set_number != second.element_set_number:
        return first if first.element_set_number > second.element_set_number else second
    if first != second:
        raise ValueError(
            f"conflicting OMM records for catalog ID {first.catalog_id} at the same epoch"
        )
    return first


def _deduplicate(records: Sequence[OmmRecord]) -> tuple[OmmRecord, ...]:
    selected: dict[int, OmmRecord] = {}
    for record in records:
        existing = selected.get(record.catalog_id)
        selected[record.catalog_id] = (
            record if existing is None else _preferred_record(existing, record)
        )
    return tuple(selected[catalog_id] for catalog_id in sorted(selected))


def parse_omm_json(
    data: str | bytes,
    *,
    logical_name: str,
    source_uri: str,
    acquired_at: Epoch,
) -> Catalog:
    """Parse CelesTrak-compatible OMM/JSON and retain exact input provenance."""
    if not isinstance(data, (str, bytes)):
        raise TypeError("data must be str or bytes")
    raw_bytes = data.encode("utf-8") if isinstance(data, str) else data
    parsed = json.loads(
        raw_bytes,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(parsed, list):
        raise ValueError("OMM JSON root must be an array")
    records = _deduplicate(tuple(_parse_omm_record(item) for item in parsed))
    asset = InputAssetMetadata(
        logical_name=logical_name,
        content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        size_bytes=len(raw_bytes),
    )
    acquisition = CatalogAcquisition(
        source_uri=source_uri,
        acquired_at=acquired_at,
        asset=asset,
    )
    return Catalog(records=records, acquisitions=(acquisition,))


def _acquisition_sort_key(item: CatalogAcquisition) -> tuple[str, str, str, str]:
    return (
        item.asset.logical_name,
        item.asset.content_sha256,
        item.source_uri,
        epoch_to_iso(item.acquired_at, "TAI"),
    )


def union_catalogs(*catalogs: Catalog) -> Catalog:
    """Return a deterministic union selecting the newest element set for each catalog ID."""
    if any(not isinstance(catalog, Catalog) for catalog in catalogs):
        raise TypeError("catalogs must contain only Catalog values")
    records = _deduplicate(tuple(record for catalog in catalogs for record in catalog.records))
    unique_acquisitions: list[CatalogAcquisition] = []
    for catalog in catalogs:
        for item in catalog.acquisitions:
            if item not in unique_acquisitions:
                unique_acquisitions.append(item)
    acquisitions = tuple(sorted(unique_acquisitions, key=_acquisition_sort_key))
    return Catalog(records=records, acquisitions=acquisitions)


def nodal_precession_rate_deg_day(record: OmmRecord) -> float:
    """Approximate the WGS-72 J2 secular ascending-node rate in degrees per day."""
    if not isinstance(record, OmmRecord):
        raise TypeError("record must be an OmmRecord")
    mean_motion_rad_s = record.mean_motion_rev_day * 2.0 * math.pi / _seconds_per_day
    semimajor_axis_m = (_wgs72_mu_m3_s2 / mean_motion_rad_s**2) ** (1.0 / 3.0)
    semilatus_rectum_m = semimajor_axis_m * (1.0 - record.eccentricity**2)
    rate_rad_s = (
        -1.5
        * _wgs72_j2
        * mean_motion_rad_s
        * (_wgs72_earth_radius_m / semilatus_rectum_m) ** 2
        * math.cos(math.radians(record.inclination_deg))
    )
    return math.degrees(rate_rad_s) * _seconds_per_day


def filter_sso(catalog: Catalog, *, max_precession_error_deg_day: float) -> Catalog:
    """Filter by consistency with the Sun's mean nodal rate, not by an orbit box."""
    if not isinstance(catalog, Catalog):
        raise TypeError("catalog must be a Catalog")
    if (
        not math.isfinite(max_precession_error_deg_day)
        or max_precession_error_deg_day < 0.0
    ):
        raise ValueError("max_precession_error_deg_day must be finite and non-negative")
    records = tuple(
        record
        for record in catalog.records
        if abs(nodal_precession_rate_deg_day(record) - _sun_mean_rate_deg_day)
        <= max_precession_error_deg_day
    )
    return Catalog(records=records, acquisitions=catalog.acquisitions)


__all__ = [
    "Catalog",
    "CatalogAcquisition",
    "OmmRecord",
    "filter_sso",
    "nodal_precession_rate_deg_day",
    "parse_omm_json",
    "union_catalogs",
]
