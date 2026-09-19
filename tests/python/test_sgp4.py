import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import odss

_fixtures = Path(__file__).parents[1] / "fixtures"


def load_catalog() -> odss.Catalog:
    path = _fixtures / "omm_catalog_a.json"
    return odss.parse_omm_json(
        path.read_bytes(),
        logical_name=path.name,
        source_uri="https://celestrak.org/test/omm_catalog_a.json",
        acquired_at=odss.epoch_from_iso("2026-06-19T12:30:00", "UTC"),
    )


def test_reference_sgp4_known_state_in_si_units() -> None:
    record = load_catalog().records[0]
    result = odss.propagate_omm_sgp4(record, record.epoch)

    assert result.record is record
    assert result.propagation_offset_s == pytest.approx(0.0, abs=1e-9)
    assert result.state.frame == odss.ReferenceFrame("TEME")
    assert result.state.position_m == pytest.approx(
        (2_167_810.4309, -6_445_660.7017, -4.7725), abs=0.1
    )
    assert result.state.velocity_m_s == pytest.approx(
        (4_501.4572680, 1_519.3295486, 6_005.6301219), abs=0.0001
    )


def test_batch_and_scalar_sgp4_agree() -> None:
    catalog = load_catalog()
    epoch = odss.epoch_from_iso("2026-06-20T00:00:00", "UTC")
    batch = odss.synchronize_catalog_sgp4(catalog, epoch)

    for batch_object, record in zip(batch.objects, catalog.records, strict=True):
        scalar = odss.propagate_omm_sgp4(record, epoch)
        assert batch_object.record == scalar.record
        assert batch_object.propagation_offset_s == pytest.approx(
            scalar.propagation_offset_s, abs=1e-9
        )
        assert batch_object.state.position_m == pytest.approx(
            scalar.state.position_m, abs=1e-6
        )
        assert batch_object.state.velocity_m_s == pytest.approx(
            scalar.state.velocity_m_s, abs=1e-9
        )


def test_catalog_is_synchronized_without_losing_original_epochs() -> None:
    catalog = load_catalog()
    original_epochs = tuple(record.epoch for record in catalog.records)
    epoch = odss.epoch_from_iso("2026-06-20T00:00:00", "UTC")

    synchronized = odss.synchronize_catalog_sgp4(catalog, epoch)

    assert synchronized.epoch is epoch
    assert len(synchronized) == len(catalog)
    assert synchronized.acquisitions == catalog.acquisitions
    assert tuple(item.record.epoch for item in synchronized.objects) == original_epochs
    assert all(item.state.epoch == epoch for item in synchronized.objects)
    assert all(item.state.frame == odss.ReferenceFrame("TEME") for item in synchronized.objects)
    assert tuple(item.propagation_offset_s for item in synchronized.objects) == pytest.approx(
        tuple(odss.elapsed_time_s(record.epoch, epoch) for record in catalog.records),
        abs=1e-9,
    )
    with pytest.raises(FrozenInstanceError):
        synchronized.epoch = catalog.records[0].epoch


def test_synchronization_supports_empty_catalog() -> None:
    epoch = odss.epoch_from_iso("2026-06-20T00:00:00", "TAI")
    synchronized = odss.synchronize_catalog_sgp4(odss.Catalog((), ()), epoch)

    assert synchronized == odss.SynchronizedCatalog(epoch, (), ())


def test_equivalent_absolute_epochs_in_different_time_scales_agree() -> None:
    record = load_catalog().records[0]
    utc_epoch = odss.epoch_from_iso("2026-06-20T00:00:00", "UTC")
    tai_epoch = odss.epoch_from_iso("2026-06-20T00:00:37", "TAI")

    utc_result = odss.propagate_omm_sgp4(record, utc_epoch)
    tai_result = odss.propagate_omm_sgp4(record, tai_epoch)

    assert utc_result.state.position_m == pytest.approx(tai_result.state.position_m, abs=1e-6)
    assert utc_result.state.velocity_m_s == pytest.approx(
        tai_result.state.velocity_m_s, abs=1e-9
    )
    assert utc_result.propagation_offset_s == pytest.approx(
        tai_result.propagation_offset_s, abs=1e-9
    )


def test_reference_backend_supports_deep_space_catalog_objects() -> None:
    record = replace(
        load_catalog().records[0],
        mean_motion_rev_day=1.0027,
        eccentricity=0.0001,
        inclination_deg=0.05,
        raan_deg=20.0,
        argument_of_pericenter_deg=10.0,
        mean_anomaly_deg=30.0,
        bstar_1_earth_radii=0.0,
        mean_motion_dot_rev_day2=0.0,
    )

    result = odss.propagate_omm_sgp4(
        record,
        odss.epoch_from_iso("2026-06-20T12:16:41.638656", "UTC"),
    )

    radius_m = math.sqrt(sum(component**2 for component in result.state.position_m))
    assert radius_m == pytest.approx(42_164_000.0, rel=0.001)


def test_reference_backend_preserves_nine_digit_omm_catalog_id() -> None:
    original = load_catalog().records[0]
    record = replace(original, catalog_id=999_999_999)

    result = odss.propagate_omm_sgp4(record, record.epoch)
    original_result = odss.propagate_omm_sgp4(original, original.epoch)

    assert result.record.catalog_id == 999_999_999
    assert result.state == original_result.state


def test_reference_backend_reports_sgp4_errors() -> None:
    record = replace(load_catalog().records[0], eccentricity=0.9999)

    with pytest.raises(RuntimeError, match=r"catalog ID 25544: semilatus rectum"):
        odss.propagate_omm_sgp4(record, record.epoch)


def test_synchronization_validates_inputs() -> None:
    catalog = load_catalog()
    epoch = odss.epoch_from_iso("2026-06-20T00:00:00", "UTC")

    with pytest.raises(TypeError, match="Catalog"):
        odss.synchronize_catalog_sgp4("not a catalog", epoch)
    with pytest.raises(TypeError, match="Epoch"):
        odss.synchronize_catalog_sgp4(catalog, "not an epoch")
    with pytest.raises(TypeError, match="OmmRecord"):
        odss.propagate_omm_sgp4("not a record", epoch)
