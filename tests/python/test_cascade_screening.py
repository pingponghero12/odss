import math
from types import SimpleNamespace

import numpy as np
import pytest

import odss
import odss.cascade_screening as cascade_screening


def population(
    positions_m: np.ndarray | list[tuple[float, float, float]],
    velocities_m_s: np.ndarray | list[tuple[float, float, float]],
) -> odss.ParticlePopulation:
    positions = np.asarray(positions_m, dtype=np.float64).reshape((-1, 3))
    velocities = np.asarray(velocities_m_s, dtype=np.float64).reshape((-1, 3))
    count = len(positions)
    return odss.ParticlePopulation(
        epoch=odss.Epoch(0.0, "2026-06-20T12:00:00", "TAI"),
        frame=odss.ReferenceFrame("GCRS"),
        position_x_m=positions[:, 0],
        position_y_m=positions[:, 1],
        position_z_m=positions[:, 2],
        velocity_x_m_s=velocities[:, 0],
        velocity_y_m_s=velocities[:, 1],
        velocity_z_m_s=velocities[:, 2],
        mass_kg=np.ones(count),
        area_m2=np.full(count, math.pi * 4.0),
    )


def fake_event(
    first: int,
    second: int,
    time_s: float,
    distance_m: float,
    first_velocity_m_s: tuple[float, float, float],
    second_velocity_m_s: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "i": first,
        "j": second,
        "time": time_s,
        "dist": distance_m,
        "state_i": np.array((0.0, 0.0, 0.0, *first_velocity_m_s)),
        "state_j": np.array((0.0, 0.0, 0.0, *second_velocity_m_s)),
    }


class FakeSimulation:
    last_call: tuple[np.ndarray, float, dict[str, object]] | None = None

    def __init__(self, state: np.ndarray, timestep_s: float, **arguments: object) -> None:
        FakeSimulation.last_call = (state, timestep_s, arguments)
        self.conjunctions = (
            fake_event(0, 1, 1.0, 0.0, (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            fake_event(2, 3, 2.0, 0.0, (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            fake_event(0, 2, 3.0, 4.0, (3.0, 4.0, 0.0), (0.0, 0.0, 0.0)),
        )

    def propagate_until(self, duration_s: float) -> str:
        assert duration_s == 10.0
        return "time_limit"


class FakeHeyoka:
    @staticmethod
    def make_vars(*names: str) -> tuple[str, ...]:
        return names

    @staticmethod
    def expression(value: float) -> tuple[str, float]:
        return ("expression", value)


def fake_backends() -> tuple[SimpleNamespace, type[FakeHeyoka]]:
    cascade = SimpleNamespace(
        sim=FakeSimulation,
        outcome=SimpleNamespace(time_limit="time_limit"),
    )
    return cascade, FakeHeyoka


def test_adapter_uses_whitelist_and_keeps_only_debris_target_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cascade_screening, "_load_screening_backends", fake_backends)
    debris = population(
        [(7_000_000.0, 0.0, 0.0), (7_000_010.0, 0.0, 0.0)],
        [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)],
    )
    targets = population(
        [(7_000_020.0, 0.0, 0.0), (7_000_030.0, 0.0, 0.0)],
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    )

    events = odss.screen_conjunctions_cascade(
        debris,
        targets,
        duration_s=10.0,
        threshold_m=5.0,
        collisional_timestep_s=2.0,
    )

    assert events == (odss.ConjunctionEvent(0, 0, 3.0, 4.0, 5.0),)
    assert FakeSimulation.last_call is not None
    state, timestep_s, arguments = FakeSimulation.last_call
    assert state.shape == (4, 7)
    assert state[:, 6] == pytest.approx((2.0, 2.0, 2.0, 2.0))
    assert timestep_s == 2.0
    assert arguments["conj_whitelist"] == {0, 1}
    assert arguments["min_coll_radius"] == math.inf
    assert arguments["conj_thresh"] > 5.0


def test_empty_cross_population_does_not_load_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cascade_screening,
        "_load_screening_backends",
        lambda: pytest.fail("empty screening must not load Cascade"),
    )
    empty = population([], [])
    one = population([(7_000_000.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)])

    assert (
        odss.screen_conjunctions_cascade(
            empty,
            one,
            duration_s=1.0,
            threshold_m=1.0,
            collisional_timestep_s=1.0,
        )
        == ()
    )


def test_smaller_target_role_is_whitelisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cascade_screening, "_load_screening_backends", fake_backends)
    debris = population(
        [(7_000_000.0 + index, 0.0, 0.0) for index in range(3)],
        [(0.0, 0.0, 0.0)] * 3,
    )
    targets = population([(7_000_010.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)])

    odss.screen_conjunctions_cascade(
        debris,
        targets,
        duration_s=10.0,
        threshold_m=5.0,
        collisional_timestep_s=2.0,
    )

    assert FakeSimulation.last_call is not None
    assert FakeSimulation.last_call[2]["conj_whitelist"] == {3}


def test_screening_configuration_is_validated() -> None:
    empty = population([], [])

    with pytest.raises(ValueError, match="collisional_timestep_s"):
        odss.screen_conjunctions_cascade(
            empty,
            empty,
            duration_s=1.0,
            threshold_m=1.0,
            collisional_timestep_s=0.0,
        )
    with pytest.raises(TypeError, match="collisional_timestep_s"):
        odss.screen_conjunctions_cascade(
            empty,
            empty,
            duration_s=1.0,
            threshold_m=1.0,
            collisional_timestep_s=True,
        )


def require_cascade() -> None:
    pytest.importorskip("cascade")
    pytest.importorskip("heyoka")


def test_real_cascade_reports_tca_distance_and_relative_velocity() -> None:
    require_cascade()
    debris = population([(7_000_000.0 - 10.0, 0.0, 0.0)], [(2.0, 0.0, 0.0)])
    targets = population([(7_000_000.0 + 10.0, 3.0, 0.0)], [(-2.0, 0.0, 0.0)])

    events = odss.screen_conjunctions_cascade(
        debris,
        targets,
        duration_s=10.0,
        threshold_m=3.0,
        collisional_timestep_s=1.0,
    )

    assert len(events) == 1
    assert events[0].tca_s == pytest.approx(5.0, abs=1.0e-10)
    assert events[0].miss_distance_m == pytest.approx(3.0, abs=1.0e-10)
    assert events[0].relative_velocity_m_s == pytest.approx(4.0, abs=1.0e-12)


@pytest.mark.parametrize("seed", range(10))
def test_real_cascade_has_no_false_negatives_against_reference(seed: int) -> None:
    require_cascade()
    generator = np.random.default_rng(seed)
    base_position_m = np.array((7_000_000.0, 0.0, 0.0))
    debris = population(
        base_position_m + generator.uniform(-50_000.0, 50_000.0, size=(8, 3)),
        generator.uniform(-5_000.0, 5_000.0, size=(8, 3)),
    )
    targets = population(
        base_position_m + generator.uniform(-50_000.0, 50_000.0, size=(7, 3)),
        generator.uniform(-5_000.0, 5_000.0, size=(7, 3)),
    )
    arguments = {"duration_s": 20.0, "threshold_m": 10_000.0}

    reference = odss.screen_conjunctions_reference(debris, targets, **arguments)
    production = odss.screen_conjunctions_cascade(
        debris,
        targets,
        collisional_timestep_s=2.0,
        **arguments,
    )

    production_by_pair = {
        (event.debris_index, event.target_index): event for event in production
    }
    assert {
        (event.debris_index, event.target_index) for event in reference
    } <= production_by_pair.keys()
    for expected in reference:
        actual = production_by_pair[(expected.debris_index, expected.target_index)]
        assert actual.tca_s == pytest.approx(expected.tca_s, abs=1.0e-8)
        assert actual.miss_distance_m == pytest.approx(expected.miss_distance_m, abs=1.0e-6)
        assert actual.relative_velocity_m_s == pytest.approx(
            expected.relative_velocity_m_s,
            abs=1.0e-10,
        )
