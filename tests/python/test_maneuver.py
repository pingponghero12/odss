import math

import pytest

import odss


def population(count: int) -> odss.ParticlePopulation:
    return odss.ParticlePopulation(
        epoch=odss.Epoch(0.0, "2026-06-20T12:00:00", "TAI"),
        frame=odss.ReferenceFrame("GCRS"),
        position_x_m=[7_000_000.0] * count,
        position_y_m=[0.0] * count,
        position_z_m=[0.0] * count,
        velocity_x_m_s=[0.0] * count,
        velocity_y_m_s=[7_500.0] * count,
        velocity_z_m_s=[0.0] * count,
        mass_kg=[1.0] * count,
        area_m2=[1.0] * count,
    )


def test_trackability_distance_filter_and_physical_tca_deduplication() -> None:
    events = (
        odss.ConjunctionEvent(0, 0, 10.0, 100.0, 1_000.0),
        odss.ConjunctionEvent(1, 0, 20.0, 900.0, 2_000.0),
        odss.ConjunctionEvent(1, 0, 20.0005, 800.0, 2_000.0),
        odss.ConjunctionEvent(2, 1, 30.0, 1_100.0, 3_000.0),
        odss.ConjunctionEvent(2, 1, 40.0, 1_000.0, 3_000.0),
    )
    spec = odss.ManeuverDemandSpec(
        trackability_size_threshold_m=0.1,
        miss_distance_threshold_m=1_000.0,
        duration_s=100.0,
        deduplication_tolerance_s=0.001,
    )

    result = odss.evaluate_maneuver_demand(
        events,
        population(3),
        population(2),
        (0.05, 0.1, 0.2),
        spec,
    )

    assert result.spec == spec
    assert [(event.debris_index, event.target_index) for event in result.actionable_encounters] == [
        (1, 0),
        (2, 1),
    ]
    assert result.actionable_encounters[0].miss_distance_m == 800.0
    assert result.affected_target_indices == (0, 1)
    assert result.events_per_target == (1, 1)
    assert result.event_rate_s == pytest.approx(0.02)
    assert result.probability_at_least_one_actionable_encounter == pytest.approx(
        1.0 - math.exp(-2.0)
    )


def test_distinct_encounters_for_one_pair_are_not_deduplicated() -> None:
    events = (
        odss.ConjunctionEvent(0, 0, 10.0, 100.0, 1_000.0),
        odss.ConjunctionEvent(0, 0, 12.0, 100.0, 1_000.0),
    )

    result = odss.evaluate_maneuver_demand(
        events,
        population(1),
        population(1),
        (0.1,),
        odss.ManeuverDemandSpec(0.1, 1_000.0, 100.0, 0.1),
    )

    assert len(result.actionable_encounters) == 2
    assert result.events_per_target == (2,)


def test_no_actionable_events_has_zero_rate_and_probability() -> None:
    result = odss.evaluate_maneuver_demand(
        (odss.ConjunctionEvent(0, 0, 10.0, 2_000.0, 1_000.0),),
        population(1),
        population(2),
        (0.05,),
        odss.ManeuverDemandSpec(0.1, 1_000.0, 100.0),
    )

    assert result.actionable_encounters == ()
    assert result.affected_target_indices == ()
    assert result.events_per_target == (0, 0)
    assert result.event_rate_s == 0.0
    assert result.probability_at_least_one_actionable_encounter == 0.0


def test_events_after_policy_horizon_are_excluded() -> None:
    result = odss.evaluate_maneuver_demand(
        (odss.ConjunctionEvent(0, 0, 101.0, 100.0, 1_000.0),),
        population(1),
        population(1),
        (0.2,),
        odss.ManeuverDemandSpec(0.1, 1_000.0, 100.0),
    )

    assert result.actionable_encounters == ()


def test_maneuver_proxy_validates_thresholds_and_population_data() -> None:
    with pytest.raises(ValueError, match="trackability"):
        odss.ManeuverDemandSpec(0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="deduplication"):
        odss.ManeuverDemandSpec(0.1, 1.0, 1.0, -1.0)
    with pytest.raises(ValueError, match="match debris"):
        odss.evaluate_maneuver_demand(
            (),
            population(1),
            population(1),
            (),
            odss.ManeuverDemandSpec(0.1, 1.0, 1.0),
        )
    with pytest.raises(ValueError, match="outside"):
        odss.evaluate_maneuver_demand(
            (odss.ConjunctionEvent(1, 0, 0.5, 0.5, 1.0),),
            population(1),
            population(1),
            (0.1,),
            odss.ManeuverDemandSpec(0.1, 1.0, 1.0),
        )
