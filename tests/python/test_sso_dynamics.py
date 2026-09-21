import math
from types import SimpleNamespace

import numpy as np
import pytest

import odss
import odss.sso_dynamics as sso_dynamics


def population(count: int = 2, frame: str = "EME2000") -> odss.ParticlePopulation:
    return odss.ParticlePopulation(
        epoch=odss.epoch_from_iso("2026-06-20T12:00:00", "TT"),
        frame=odss.ReferenceFrame(frame),
        position_x_m=[7_078_000.0] * count,
        position_y_m=[float(index) for index in range(count)],
        position_z_m=[0.0] * count,
        velocity_x_m_s=[0.0] * count,
        velocity_y_m_s=[7_500.0] * count,
        velocity_z_m_s=[0.0] * count,
        mass_kg=[100.0, 200.0][:count],
        area_m2=[2.0, 8.0][:count],
    )


class FakeDynamics:
    arguments: dict[str, bool] | None = None

    @staticmethod
    def simple_earth(**arguments: bool) -> str:
        FakeDynamics.arguments = arguments
        return "earth_dynamics"


class FakeSimulation:
    last_call: tuple[np.ndarray, float, dict[str, object]] | None = None

    def __init__(self, state: np.ndarray, timestep_s: float, **arguments: object) -> None:
        self.state = np.array(state, copy=True)
        self.time = 0.0
        self.conjunctions: tuple[dict[str, object], ...] = ()
        FakeSimulation.last_call = (self.state, timestep_s, arguments)

    def propagate_until(self, final_time_s: float) -> str:
        duration_s = final_time_s - self.time
        self.state[:, :3] += self.state[:, 3:6] * duration_s
        self.time = final_time_s
        return "time_limit"


def fake_cascade() -> SimpleNamespace:
    return SimpleNamespace(
        dynamics=FakeDynamics,
        sim=FakeSimulation,
        outcome=SimpleNamespace(time_limit="time_limit"),
    )


def test_sso_adapter_configures_dynamics_epoch_and_particle_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sso_dynamics, "_load_cascade", fake_cascade)
    source = population()
    force_model = odss.SsoForceModelSpec(
        j2=True,
        j3=True,
        c22_s22=True,
        drag=True,
        sun=True,
        moon=True,
        srp=True,
        drag_coefficient=2.0,
        reflectivity_coefficient=1.5,
    )

    result = odss.propagate_sso(
        source,
        odss.SsoPropagationSpec(60.0, 10.0, force_model, tolerance=1.0e-12),
    )

    assert result.epoch.offset_s == source.epoch.offset_s + 60.0
    assert result.position_y_m == pytest.approx((450_000.0, 450_001.0))
    assert FakeDynamics.arguments == {
        "J2": True,
        "J3": True,
        "J4": False,
        "C22S22": True,
        "drag": True,
        "sun": True,
        "moon": True,
        "SRP": True,
    }
    assert FakeSimulation.last_call is not None
    _, timestep_s, arguments = FakeSimulation.last_call
    assert timestep_s == 10.0
    assert arguments["dyn"] == "earth_dynamics"
    assert arguments["min_coll_radius"] == math.inf
    assert arguments["n_par_ct"] == 1
    assert arguments["tol"] == 1.0e-12
    parameters = np.asarray(arguments["pars"])
    expected_area_to_mass = np.array((0.02, 0.04))
    expected_bstar = expected_area_to_mass * sso_dynamics._cascade_drag_reference_density_kg_m3
    assert parameters[:, 0] == pytest.approx(expected_bstar)
    assert parameters[:, 1] == pytest.approx(1.5 * expected_area_to_mass)


def test_sso_adapter_requires_explicit_eme2000_frame() -> None:
    with pytest.raises(ValueError, match="EME2000"):
        odss.propagate_sso(
            population(frame="GCRS"),
            odss.SsoPropagationSpec(1.0, 1.0, odss.SsoForceModelSpec()),
        )


def test_combined_sso_propagation_screens_only_debris_target_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_time_s = sso_dynamics.elapsed_time_s(
        sso_dynamics._j2000_tt,
        population(count=1).epoch,
    )

    class ScreeningSimulation(FakeSimulation):
        def __init__(self, state: np.ndarray, timestep_s: float, **arguments: object) -> None:
            super().__init__(state, timestep_s, **arguments)
            self.conjunctions = (
                {
                    "i": 0,
                    "j": 1,
                    "time": start_time_s + 30.0,
                    "dist": 250.0,
                    "state_i": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                    "state_j": (0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
                },
            )

    backend = fake_cascade()
    backend.sim = ScreeningSimulation
    monkeypatch.setattr(sso_dynamics, "_load_cascade", lambda: backend)
    debris = population(count=1)
    targets = population(count=2)

    result = odss.propagate_and_screen_sso(
        debris,
        targets,
        odss.SsoPropagationSpec(
            duration_s=60.0,
            collisional_timestep_s=10.0,
            force_model=odss.SsoForceModelSpec(j2=True, drag=True),
            collisional_steps_per_batch=120,
        ),
        threshold_m=1_000.0,
    )

    assert len(result.final_debris) == 1
    assert len(result.final_targets) == 2
    assert result.final_debris.epoch.offset_s == debris.epoch.offset_s + 60.0
    assert result.conjunctions == (odss.ConjunctionEvent(0, 0, 30.0, 250.0, 2.0),)
    assert ScreeningSimulation.last_call is not None
    _, _, arguments = ScreeningSimulation.last_call
    assert arguments["conj_whitelist"] == {0}
    assert arguments["conj_thresh"] > 1_000.0
    assert arguments["n_par_ct"] == 120
    assert np.asarray(arguments["pars"]).shape == (3, 1)


def test_force_model_and_propagation_specs_validate_values() -> None:
    with pytest.raises(TypeError, match="j2"):
        odss.SsoForceModelSpec(j2=1)
    with pytest.raises(ValueError, match="drag_coefficient"):
        odss.SsoForceModelSpec(drag_coefficient=0.0)
    with pytest.raises(ValueError, match="duration_s"):
        odss.SsoPropagationSpec(0.0, 1.0, odss.SsoForceModelSpec())
    with pytest.raises(TypeError, match="force_model"):
        odss.SsoPropagationSpec(1.0, 1.0, "j2")
    with pytest.raises(ValueError, match="collisional_steps_per_batch"):
        odss.SsoPropagationSpec(
            1.0,
            1.0,
            odss.SsoForceModelSpec(),
            collisional_steps_per_batch=0,
        )


def test_sensitivity_and_fastest_converged_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    source = population(count=1)
    simple = odss.SsoForceModelSpec(j2=False, drag=False)
    intermediate = odss.SsoForceModelSpec(j2=True, drag=False)
    reference = odss.SsoForceModelSpec(j2=True, j3=True, drag=False)
    offsets = {simple: 10.0, intermediate: 1.0, reference: 0.0}

    def fake_propagate(
        initial: odss.ParticlePopulation,
        spec: odss.SsoPropagationSpec,
    ) -> odss.ParticlePopulation:
        offset = offsets[spec.force_model]
        return odss.ParticlePopulation(
            initial.epoch,
            initial.frame,
            [initial.position_x_m[0] + offset],
            initial.position_y_m,
            initial.position_z_m,
            initial.velocity_x_m_s,
            [initial.velocity_y_m_s[0] + offset / 10.0],
            initial.velocity_z_m_s,
            initial.mass_kg,
            initial.area_m2,
        )

    monkeypatch.setattr(sso_dynamics, "propagate_sso", fake_propagate)
    results = odss.evaluate_force_model_sensitivity(
        source,
        duration_s=60.0,
        collisional_timestep_s=10.0,
        candidates=(simple, intermediate, reference),
        reference=reference,
    )

    assert [item.maximum_position_difference_m for item in results] == [10.0, 1.0, 0.0]
    assert [item.maximum_velocity_difference_m_s for item in results] == pytest.approx(
        [1.0, 0.1, 0.0]
    )
    selected = odss.fastest_converged_force_model(
        results,
        maximum_position_difference_m=1.0,
        maximum_velocity_difference_m_s=0.1,
    )
    assert selected in (intermediate, reference)
    with pytest.raises(ValueError, match="no force model"):
        odss.fastest_converged_force_model(
            results,
            maximum_position_difference_m=-1.0,
            maximum_velocity_difference_m_s=-1.0,
        )


def test_real_cascade_sso_dynamics_produces_finite_state() -> None:
    pytest.importorskip("cascade")
    result = odss.propagate_sso(
        population(count=1),
        odss.SsoPropagationSpec(
            60.0,
            10.0,
            odss.SsoForceModelSpec(
                j2=True,
                j3=True,
                c22_s22=True,
                drag=True,
                sun=True,
                moon=True,
                srp=True,
            ),
        ),
    )

    assert all(math.isfinite(value) for value in result.position_x_m)
    assert all(math.isfinite(value) for value in result.velocity_y_m_s)
