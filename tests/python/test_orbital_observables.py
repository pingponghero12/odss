import math

import pytest

import odss

_mu_m3_s2 = 3.986_004_418e14
_earth_radius_m = 6_378_136.3


def population(
    radii_m: list[float],
    speed_factors: list[float],
    *,
    epoch_offset_s: float,
) -> odss.ParticlePopulation:
    speeds_m_s = [
        factor * math.sqrt(_mu_m3_s2 / radius_m)
        for radius_m, factor in zip(radii_m, speed_factors, strict=True)
    ]
    count = len(radii_m)
    return odss.ParticlePopulation(
        epoch=odss.Epoch(epoch_offset_s, "2026-06-20T12:00:00", "TAI"),
        frame=odss.ReferenceFrame("GCRS"),
        position_x_m=radii_m,
        position_y_m=[0.0] * count,
        position_z_m=[0.0] * count,
        velocity_x_m_s=[0.0] * count,
        velocity_y_m_s=speeds_m_s,
        velocity_z_m_s=[0.0] * count,
        mass_kg=[1.0] * count,
        area_m2=[1.0] * count,
    )


def test_circular_orbit_decay_has_analytical_elements_and_change() -> None:
    initial_radius_m = _earth_radius_m + 700_000.0
    final_radius_m = _earth_radius_m + 650_000.0

    result = odss.evaluate_decay(
        population([initial_radius_m], [1.0], epoch_offset_s=0.0),
        population([final_radius_m], [1.0], epoch_offset_s=60.0),
    )

    value = result.diagnostics[0]
    assert value.initial_semimajor_axis_m == pytest.approx(initial_radius_m)
    assert value.final_semimajor_axis_m == pytest.approx(final_radius_m)
    assert value.semimajor_axis_change_m == pytest.approx(-50_000.0)
    assert value.initial_perigee_altitude_m == pytest.approx(700_000.0)
    assert value.final_apogee_altitude_m == pytest.approx(650_000.0)
    assert value.classification == "retained"
    assert result.retained_fraction == 1.0


def test_reentry_removal_and_escape_classifications() -> None:
    initial = population([_earth_radius_m + 700_000.0] * 3, [1.0] * 3, epoch_offset_s=0.0)
    final = population(
        [
            _earth_radius_m + 80_000.0,
            _earth_radius_m + 150_000.0,
            _earth_radius_m + 700_000.0,
        ],
        [1.0, 1.0, 1.5],
        epoch_offset_s=60.0,
    )

    result = odss.evaluate_decay(initial, final)
    escape = odss.evaluate_escape(final)

    assert [value.classification for value in result.diagnostics] == [
        "reentry",
        "removal",
        "escape",
    ]
    assert result.reentry_fraction == pytest.approx(1.0 / 3.0)
    assert result.removal_fraction == pytest.approx(1.0 / 3.0)
    assert result.escape_fraction == pytest.approx(1.0 / 3.0)
    assert escape.escaped == (False, False, True)
    assert escape.escaped_fraction == pytest.approx(1.0 / 3.0)
    assert result.diagnostics[2].final_semimajor_axis_m is None


def test_impact_probability_matches_poisson_reference_equation() -> None:
    flux = odss.FluxResult(
        spec=odss.FluxSpec(1.0, (0.0, 10.0, 30.0), (0.0, math.inf)),
        target_count=2,
        bins=(
            odss.FluxBin(0, 0.0, 10.0, 0.0, math.inf, 1, 2.0e-6, 0.0, 0.0),
            odss.FluxBin(0, 10.0, 30.0, 0.0, math.inf, 1, 3.0e-6, 0.0, 0.0),
            odss.FluxBin(1, 0.0, 10.0, 0.0, math.inf, 0, 0.0, 0.0, 0.0),
            odss.FluxBin(1, 10.0, 30.0, 0.0, math.inf, 0, 0.0, 0.0, 0.0),
        ),
    )

    result = odss.evaluate_impact_risk(flux, (1.0, 10.0))

    by_area_target = {
        (value.reference_area_m2, value.target_index): value for value in result.values
    }
    integrated_flux_m2 = 2.0e-6 * 10.0 + 3.0e-6 * 20.0
    one = by_area_target[(1.0, 0)]
    assert one.expected_impacts == pytest.approx(integrated_flux_m2)
    assert one.model_impact_probability == pytest.approx(1.0 - math.exp(-integrated_flux_m2))
    assert by_area_target[(10.0, 0)].expected_impacts == pytest.approx(
        10.0 * integrated_flux_m2
    )
    assert by_area_target[(1.0, 1)].model_impact_probability == 0.0


def test_empty_populations_have_zero_fractions() -> None:
    empty = population([], [], epoch_offset_s=0.0)
    later = population([], [], epoch_offset_s=1.0)

    decay = odss.evaluate_decay(empty, later)
    escape = odss.evaluate_escape(empty)

    assert decay.diagnostics == ()
    assert decay.retained_fraction == 0.0
    assert escape.escaped == ()
    assert escape.escaped_fraction == 0.0


def test_orbital_observable_inputs_are_validated() -> None:
    valid = population([_earth_radius_m + 700_000.0], [1.0], epoch_offset_s=0.0)
    earlier = population([_earth_radius_m + 700_000.0], [1.0], epoch_offset_s=-1.0)

    with pytest.raises(ValueError, match="must not precede"):
        odss.evaluate_decay(valid, earlier)
    with pytest.raises(ValueError, match="reentry_altitude"):
        odss.OrbitalDecaySpec(reentry_altitude_m=300_000.0, removal_altitude_m=200_000.0)
    with pytest.raises(ValueError, match="reference_areas"):
        odss.evaluate_impact_risk(
            odss.FluxResult(odss.FluxSpec(1.0, (0.0, 1.0), (0.0, math.inf)), 0, ()),
            (),
        )
