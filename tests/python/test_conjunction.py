import math
from dataclasses import FrozenInstanceError

import pytest

import odss


def population(
    positions_m: list[tuple[float, float, float]],
    velocities_m_s: list[tuple[float, float, float]],
    *,
    epoch: odss.Epoch | None = None,
    frame: str = "GCRS",
) -> odss.ParticlePopulation:
    count = len(positions_m)
    assert count == len(velocities_m_s)
    return odss.ParticlePopulation(
        epoch=epoch or odss.Epoch(0.0, "2026-06-20T12:00:00", "TAI"),
        frame=odss.ReferenceFrame(frame),
        position_x_m=[value[0] for value in positions_m],
        position_y_m=[value[1] for value in positions_m],
        position_z_m=[value[2] for value in positions_m],
        velocity_x_m_s=[value[0] for value in velocities_m_s],
        velocity_y_m_s=[value[1] for value in velocities_m_s],
        velocity_z_m_s=[value[2] for value in velocities_m_s],
        mass_kg=[1.0] * count,
        area_m2=[1.0] * count,
    )


def screen(
    debris_position_m: tuple[float, float, float],
    debris_velocity_m_s: tuple[float, float, float],
    target_position_m: tuple[float, float, float],
    target_velocity_m_s: tuple[float, float, float],
    *,
    duration_s: float = 10.0,
    threshold_m: float = 1.0,
) -> tuple[odss.ConjunctionEvent, ...]:
    return odss.screen_conjunctions_reference(
        population([debris_position_m], [debris_velocity_m_s]),
        population([target_position_m], [target_velocity_m_s]),
        duration_s=duration_s,
        threshold_m=threshold_m,
    )


def test_head_on_encounter_has_exact_continuous_tca() -> None:
    events = screen(
        (-10.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
    )

    assert events == (odss.ConjunctionEvent(0, 0, 5.0, 0.0, 4.0),)
    with pytest.raises(FrozenInstanceError):
        events[0].tca_s = 0.0


def test_grazing_encounter_at_threshold_is_included() -> None:
    events = screen(
        (-10.0, 3.0, 0.0),
        (2.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
        threshold_m=3.0,
    )

    assert len(events) == 1
    assert events[0].tca_s == pytest.approx(5.0)
    assert events[0].miss_distance_m == pytest.approx(3.0)
    assert events[0].relative_velocity_m_s == pytest.approx(4.0)


def test_parallel_motion_without_a_unique_local_minimum_has_no_event() -> None:
    diverging = screen(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        threshold_m=2.0,
    )
    co_moving = screen(
        (0.0, 0.5, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        threshold_m=1.0,
    )

    assert diverging == ()
    assert co_moving == ()


def test_no_encounter_returns_no_event() -> None:
    events = screen(
        (-10.0, 2.0, 0.0),
        (2.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
        threshold_m=1.0,
    )

    assert events == ()


def test_fast_crossing_between_coarse_sample_times_is_detected() -> None:
    events = screen(
        (-1000.0, 0.0, 0.0),
        (10_000.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        duration_s=1.0,
        threshold_m=0.01,
    )

    assert len(events) == 1
    assert events[0].tca_s == pytest.approx(0.1)
    assert events[0].miss_distance_m == pytest.approx(0.0)


def test_brute_force_checks_only_cross_population_pairs() -> None:
    debris = population(
        [(-2.0, 0.0, 0.0), (-2.0, 10.0, 0.0)],
        [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    )
    targets = population(
        [(2.0, 0.0, 0.0), (2.0, 10.0, 0.0)],
        [(-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)],
    )

    events = odss.screen_conjunctions_reference(
        debris,
        targets,
        duration_s=4.0,
        threshold_m=0.1,
    )

    assert [(event.debris_index, event.target_index) for event in events] == [(0, 0), (1, 1)]


def test_interval_boundaries_are_not_reported_as_conjunction_events() -> None:
    before = screen(
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        threshold_m=1.0,
    )
    after = screen(
        (-20.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        duration_s=10.0,
        threshold_m=10.0,
    )

    assert before == ()
    assert after == ()


def test_empty_population_and_input_validation() -> None:
    empty = population([], [])
    target = population([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)])

    assert (
        odss.screen_conjunctions_reference(
            empty,
            target,
            duration_s=1.0,
            threshold_m=1.0,
        )
        == ()
    )
    with pytest.raises(ValueError, match="same epoch"):
        odss.screen_conjunctions_reference(
            population([], [], epoch=odss.Epoch(1.0, "J2000", "TAI")),
            empty,
            duration_s=1.0,
            threshold_m=1.0,
        )
    with pytest.raises(ValueError, match="same reference frame"):
        odss.screen_conjunctions_reference(
            empty,
            population([], [], frame="TEME"),
            duration_s=1.0,
            threshold_m=1.0,
        )
    with pytest.raises(ValueError, match="duration_s"):
        odss.screen_conjunctions_reference(empty, empty, duration_s=0.0, threshold_m=1.0)
    with pytest.raises(ValueError, match="threshold_m"):
        odss.screen_conjunctions_reference(empty, empty, duration_s=1.0, threshold_m=math.nan)
