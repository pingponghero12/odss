from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from astropy.utils import iers

import odss


@pytest.fixture(scope="module")
def earth_orientation() -> odss.EarthOrientationData:
    return odss.bundled_iers_b()


def satellite_teme_state() -> odss.CartesianState:
    return odss.CartesianState(
        position_m=(-6102443.276428913, -986332.0160861297, -2820313.0707199224),
        velocity_m_s=(-1455.2527284474308, -5527.413835655969, 5101.042029427083),
        epoch=odss.epoch_from_iso("2019-12-09T20:42:09.072", "UTC"),
        frame=odss.ReferenceFrame("TEME"),
    )


def test_epoch_conversion_handles_the_2016_leap_second() -> None:
    before = odss.epoch_from_iso("2016-12-31T23:59:59", "UTC")
    leap_second = odss.Epoch(1.0, before.reference_epoch, before.time_scale)
    after = odss.Epoch(2.0, before.reference_epoch, before.time_scale)

    assert odss.epoch_to_iso(leap_second) == "2016-12-31T23:59:60.000000000"
    assert odss.epoch_to_iso(after) == "2017-01-01T00:00:00.000000000"
    assert odss.elapsed_time_s(before, after) == pytest.approx(2.0, abs=2e-12)
    assert odss.epoch_to_iso(before, "TAI") == "2017-01-01T00:00:35.000000000"
    assert odss.epoch_to_iso(before, "TT") == "2017-01-01T00:01:07.184000000"


def test_epoch_conversion_preserves_the_instant_and_is_explicit() -> None:
    utc = odss.epoch_from_iso("2020-01-02T03:04:05.123456789", "utc")
    tai = odss.convert_epoch(utc, "TAI")
    round_trip = odss.convert_epoch(tai, "UTC")

    assert tai.time_scale == "TAI"
    assert tai.offset_s == 0.0
    assert odss.elapsed_time_s(utc, tai) == pytest.approx(0.0, abs=1e-12)
    assert round_trip == utc


def test_epoch_conversion_rejects_implicit_or_unsupported_metadata() -> None:
    with pytest.raises(ValueError, match="UTC, TAI, or TT"):
        odss.epoch_from_iso("2020-01-01T00:00:00", "UT1")
    with pytest.raises(ValueError):
        odss.epoch_from_iso("not-a-time", "UTC")
    with pytest.raises(ValueError, match="UTC, TAI, or TT"):
        odss.epoch_to_iso(odss.Epoch(0.0, "2020-01-01T00:00:00", "GPS"))


def test_bundled_earth_orientation_data_has_provenance() -> None:
    data = odss.bundled_iers_b()
    source = Path(iers.IERS_B_FILE)

    assert data.bulletin == "B"
    assert data.metadata.logical_name == source.name
    assert data.metadata.size_bytes == source.stat().st_size
    assert len(data.metadata.content_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        data.metadata = data.metadata

    predictive = odss.bundled_iers_a()
    assert predictive.bulletin == "A"
    assert predictive.metadata.logical_name == Path(iers.IERS_A_FILE).name
    with pytest.raises(ValueError, match="bulletin must be A or B"):
        odss.EarthOrientationData(source, "C")


def test_teme_to_itrs_matches_the_documented_satellite_reference(
    earth_orientation: odss.EarthOrientationData,
) -> None:
    transformed = odss.transform_state(
        satellite_teme_state(),
        odss.ReferenceFrame("ITRS"),
        earth_orientation,
    )

    assert transformed.position_m == pytest.approx(
        (-5821359.671553324, 2079535.682393819, -2820307.345827059), abs=0.1
    )
    assert transformed.velocity_m_s == pytest.approx(
        (-3789.2909822110087, -3715.445128681138, 5101.039096770343), abs=1e-4
    )
    assert transformed.frame == odss.ReferenceFrame("ITRS")
    assert transformed.epoch == satellite_teme_state().epoch


def test_teme_gcrs_round_trip_preserves_state(
    earth_orientation: odss.EarthOrientationData,
) -> None:
    original = satellite_teme_state()
    gcrs = odss.transform_state(original, odss.ReferenceFrame("GCRS"), earth_orientation)
    round_trip = odss.transform_state(gcrs, odss.ReferenceFrame("TEME"), earth_orientation)

    assert gcrs.frame == odss.ReferenceFrame("GCRS")
    assert gcrs.epoch == original.epoch
    assert round_trip.position_m == pytest.approx(original.position_m, abs=1e-3)
    assert round_trip.velocity_m_s == pytest.approx(original.velocity_m_s, abs=1e-6)


def test_teme_eme2000_round_trip_preserves_state(
    earth_orientation: odss.EarthOrientationData,
) -> None:
    original = satellite_teme_state()
    eme2000 = odss.transform_state(
        original,
        odss.ReferenceFrame("EME2000"),
        earth_orientation,
    )
    round_trip = odss.transform_state(
        eme2000,
        odss.ReferenceFrame("TEME"),
        earth_orientation,
    )

    assert eme2000.frame == odss.ReferenceFrame("EME2000")
    assert eme2000.epoch == original.epoch
    assert round_trip.position_m == pytest.approx(original.position_m, abs=1e-3)
    assert round_trip.velocity_m_s == pytest.approx(original.velocity_m_s, abs=1e-6)


def test_frame_transform_requires_supported_explicit_frames(
    earth_orientation: odss.EarthOrientationData,
) -> None:
    state = satellite_teme_state()

    assert odss.transform_state(state, state.frame, earth_orientation) is state
    with pytest.raises(ValueError, match="unsupported reference frame"):
        odss.transform_state(state, odss.ReferenceFrame("GCRF"), earth_orientation)
    unsupported_source = odss.CartesianState(
        state.position_m,
        state.velocity_m_s,
        state.epoch,
        odss.ReferenceFrame("UNKNOWN"),
    )
    with pytest.raises(ValueError, match="unsupported reference frame"):
        odss.transform_state(unsupported_source, odss.ReferenceFrame("GCRS"), earth_orientation)


def test_frame_transform_requires_earth_orientation_data() -> None:
    with pytest.raises(TypeError, match="EarthOrientationData"):
        odss.transform_state(
            satellite_teme_state(),
            odss.ReferenceFrame("GCRS"),
            None,
        )
