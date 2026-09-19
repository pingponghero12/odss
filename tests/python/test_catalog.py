import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import odss

_fixtures = Path(__file__).parents[1] / "fixtures"


def acquisition_epoch() -> odss.Epoch:
    return odss.epoch_from_iso("2026-06-19T12:30:00", "UTC")


def load_catalog(name: str) -> odss.Catalog:
    path = _fixtures / name
    return odss.parse_omm_json(
        path.read_bytes(),
        logical_name=path.name,
        source_uri=f"https://celestrak.org/test/{path.name}",
        acquired_at=acquisition_epoch(),
    )


def test_parse_celestrak_omm_json_with_six_digit_catalog_id() -> None:
    catalog = load_catalog("omm_catalog_a.json")

    assert len(catalog) == 3
    assert tuple(record.catalog_id for record in catalog.records) == (25544, 33591, 100001)
    six_digit = catalog.records[2]
    assert six_digit.object_name is None
    assert six_digit.international_designator is None
    assert six_digit.epoch == odss.epoch_from_iso("2026-06-19T09:30:00", "UTC")
    assert six_digit.mean_motion_rev_day == 14.2
    assert six_digit.inclination_deg == 98.7
    with pytest.raises(FrozenInstanceError):
        six_digit.catalog_id = 100002


def test_parse_records_exact_input_and_acquisition_provenance() -> None:
    catalog = load_catalog("omm_catalog_a.json")
    acquisition = catalog.acquisitions[0]

    assert acquisition.source_uri == "https://celestrak.org/test/omm_catalog_a.json"
    assert acquisition.acquired_at == acquisition_epoch()
    assert acquisition.asset == odss.InputAssetMetadata(
        logical_name="omm_catalog_a.json",
        content_sha256="62ae8cee1aa268b1fcf060d90c2db7dc16fb3f80d192aea695642c89e428975a",
        size_bytes=1658,
    )


def test_catalog_union_is_order_independent_and_selects_newest_epoch() -> None:
    first = load_catalog("omm_catalog_a.json")
    second = load_catalog("omm_catalog_b.json")

    combined = odss.union_catalogs(first, second)
    reversed_order = odss.union_catalogs(second, first)

    assert combined == reversed_order
    assert tuple(record.catalog_id for record in combined.records) == (
        25544,
        33591,
        43013,
        100001,
    )
    noaa = next(record for record in combined.records if record.catalog_id == 33591)
    assert noaa.epoch == odss.epoch_from_iso("2026-06-20T08:00:00", "UTC")
    assert noaa.element_set_number == 902
    assert len(combined.acquisitions) == 2
    assert odss.union_catalogs(first, first) == first


def test_catalog_union_uses_element_number_for_equal_epochs() -> None:
    original = load_catalog("omm_catalog_a.json").records[0]
    newer_element = replace(original, element_set_number=1000)

    combined = odss.union_catalogs(
        odss.Catalog((original,), ()),
        odss.Catalog((newer_element,), ()),
    )

    assert combined.records == (newer_element,)


def test_catalog_union_rejects_unresolved_conflicts() -> None:
    original = load_catalog("omm_catalog_a.json").records[0]
    conflicting = replace(original, mean_anomaly_deg=original.mean_anomaly_deg + 1.0)

    with pytest.raises(ValueError, match="conflicting OMM records"):
        odss.union_catalogs(
            odss.Catalog((original,), ()),
            odss.Catalog((conflicting,), ()),
        )


def test_sso_filter_uses_j2_nodal_precession_consistency() -> None:
    catalog = load_catalog("omm_catalog_a.json")
    rates = {
        record.catalog_id: odss.nodal_precession_rate_deg_day(record)
        for record in catalog.records
    }

    assert rates[25544] == pytest.approx(-4.9505754877, abs=1e-10)
    assert rates[33591] == pytest.approx(1.0270042635, abs=1e-10)
    assert rates[100001] == pytest.approx(0.9844306558, abs=1e-10)
    filtered = odss.filter_sso(catalog, max_precession_error_deg_day=0.05)
    assert tuple(record.catalog_id for record in filtered.records) == (33591, 100001)
    assert filtered.acquisitions == catalog.acquisitions


def test_sso_filter_validates_inputs() -> None:
    catalog = load_catalog("omm_catalog_a.json")

    with pytest.raises(ValueError, match="finite and non-negative"):
        odss.filter_sso(catalog, max_precession_error_deg_day=-0.1)
    with pytest.raises(TypeError, match="OmmRecord"):
        odss.nodal_precession_rate_deg_day("not a record")


def test_parser_rejects_wrong_root_duplicate_fields_and_nonfinite_numbers() -> None:
    arguments = {
        "logical_name": "invalid.json",
        "source_uri": "https://example.test/invalid.json",
        "acquired_at": acquisition_epoch(),
    }

    with pytest.raises(ValueError, match="root must be an array"):
        odss.parse_omm_json("{}", **arguments)
    with pytest.raises(ValueError, match="duplicate JSON field"):
        odss.parse_omm_json('[{"EPOCH":"a","EPOCH":"b"}]', **arguments)
    with pytest.raises(ValueError, match="invalid JSON numeric constant"):
        odss.parse_omm_json("[NaN]", **arguments)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("REF_FRAME", "GCRS", "REF_FRAME must be TEME"),
        ("TIME_SYSTEM", "TAI", "TIME_SYSTEM must be UTC"),
        ("CENTER_NAME", "MARS", "CENTER_NAME must be EARTH"),
        ("MEAN_ELEMENT_THEORY", "OTHER", "MEAN_ELEMENT_THEORY must be SGP4"),
        ("ECCENTRICITY", 1.0, "eccentricity"),
        ("INCLINATION", 181.0, "inclination_deg"),
        ("NORAD_CAT_ID", 1_000_000_000, "catalog_id"),
    ],
)
def test_parser_validates_omm_scientific_metadata(
    field_name: str,
    value: object,
    message: str,
) -> None:
    item = json.loads((_fixtures / "omm_catalog_a.json").read_text())[0]
    item[field_name] = value

    with pytest.raises(ValueError, match=message):
        odss.parse_omm_json(
            json.dumps([item]),
            logical_name="invalid.json",
            source_uri="https://example.test/invalid.json",
            acquired_at=acquisition_epoch(),
        )


def test_parser_requires_complete_numeric_omm_fields() -> None:
    item = json.loads((_fixtures / "omm_catalog_a.json").read_text())[0]
    del item["MEAN_MOTION"]

    with pytest.raises(ValueError, match="missing OMM field: MEAN_MOTION"):
        odss.parse_omm_json(
            json.dumps([item]),
            logical_name="invalid.json",
            source_uri="https://example.test/invalid.json",
            acquired_at=acquisition_epoch(),
        )
