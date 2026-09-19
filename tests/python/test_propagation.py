import math
from types import SimpleNamespace

import numpy as np
import pytest

import odss
import odss.propagation as propagation


def population(*, frame: str = "GCRS", count: int = 2) -> odss.ParticlePopulation:
    return odss.ParticlePopulation(
        epoch=odss.Epoch(10.0, "2026-06-20T12:00:00", "TAI"),
        frame=odss.ReferenceFrame(frame),
        position_x_m=[7_000_000.0 + index for index in range(count)],
        position_y_m=[0.0] * count,
        position_z_m=[0.0] * count,
        velocity_x_m_s=[0.0] * count,
        velocity_y_m_s=[7_500.0] * count,
        velocity_z_m_s=[0.0] * count,
        mass_kg=[100.0] * count,
        area_m2=[math.pi * 4.0] * count,
    )


class FakeSimulation:
    last_call: tuple[np.ndarray, float, dict[str, object]] | None = None

    def __init__(self, state: np.ndarray, timestep_s: float, **arguments: object) -> None:
        self.state = np.array(state, copy=True)
        self.arguments = arguments
        self.timestep_s = timestep_s
        self.stopped_at_s: float | None = None
        FakeSimulation.last_call = (self.state.copy(), timestep_s, arguments)

    def propagate_until(self, duration_s: float) -> str:
        self.state[:, :3] += self.state[:, 3:6] * duration_s
        self.stopped_at_s = duration_s
        return "time_limit"


class FakeDynamics:
    @staticmethod
    def kepler(*, mu: float) -> tuple[str, float]:
        return ("kepler", mu)


def fake_cascade() -> SimpleNamespace:
    return SimpleNamespace(
        dynamics=FakeDynamics,
        sim=FakeSimulation,
        outcome=SimpleNamespace(time_limit="time_limit"),
    )


def test_cascade_adapter_preserves_metadata_and_uses_si_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(propagation, "_load_cascade", fake_cascade)
    source = population()
    spec = odss.CascadePropagationSpec(
        duration_s=60.0,
        collisional_timestep_s=10.0,
        gravitational_parameter_m3_s2=4.0e14,
        tolerance=1.0e-12,
        high_accuracy=True,
    )

    result = odss.propagate(source, spec)

    assert isinstance(result, odss.ParticlePopulation)
    assert result.epoch == odss.Epoch(70.0, source.epoch.reference_epoch, "TAI")
    assert result.frame == source.frame
    assert result.position_y_m == pytest.approx((450_000.0, 450_000.0))
    assert result.velocity_y_m_s == source.velocity_y_m_s
    assert result.mass_kg == source.mass_kg
    assert result.area_m2 == source.area_m2
    assert FakeSimulation.last_call is not None
    state, timestep_s, arguments = FakeSimulation.last_call
    assert state.shape == (2, 7)
    assert state[:, 6] == pytest.approx((2.0, 2.0))
    assert timestep_s == 10.0
    assert arguments == {
        "dyn": ("kepler", 4.0e14),
        "high_accuracy": True,
        "tol": 1.0e-12,
    }


def test_cascade_adapter_handles_an_empty_population_without_loading_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        propagation,
        "_load_cascade",
        lambda: pytest.fail("empty propagation must not load Cascade"),
    )

    result = odss.propagate(
        population(count=0),
        odss.CascadePropagationSpec(60.0, 10.0),
    )

    assert result.empty
    assert result.epoch.offset_s == 70.0


def test_cascade_adapter_rejects_ambiguous_non_inertial_input() -> None:
    with pytest.raises(ValueError, match="GCRS"):
        odss.propagate(
            population(frame="TEME"),
            odss.CascadePropagationSpec(60.0, 10.0),
        )


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ((0.0, 1.0), ValueError),
        ((1.0, math.inf), ValueError),
        ((1.0, 1.0, -1.0), ValueError),
        ((True, 1.0), TypeError),
    ],
)
def test_cascade_spec_rejects_invalid_numerical_configuration(
    arguments: tuple[object, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        odss.CascadePropagationSpec(*arguments)


def test_generic_contract_adapts_existing_sgp4_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = odss.Catalog((), ())
    epoch = odss.epoch_from_iso("2026-06-20T12:00:00", "UTC")

    result = odss.propagate(catalog, odss.Sgp4PropagationSpec(epoch))

    assert result == odss.SynchronizedCatalog(epoch, (), ())
    with pytest.raises(TypeError, match="Catalog"):
        odss.propagate(population(), odss.Sgp4PropagationSpec(epoch))
    with pytest.raises(TypeError, match="ParticlePopulation"):
        odss.propagate(catalog, odss.CascadePropagationSpec(1.0, 1.0))
    with pytest.raises(TypeError, match="spec"):
        odss.propagate(catalog, object())


def test_missing_optional_backend_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = propagation.importlib.import_module

    def fail_for_cascade(name: str) -> object:
        if name == "cascade":
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr(propagation.importlib, "import_module", fail_for_cascade)

    with pytest.raises(ImportError, match="conda-forge"):
        odss.propagate(population(), odss.CascadePropagationSpec(1.0, 1.0))


def test_real_cascade_kepler_orbit_returns_near_initial_state() -> None:
    pytest.importorskip("cascade")
    radius_m = 7_000_000.0
    mu_m3_s2 = 3.986_004_418e14
    period_s = 2.0 * math.pi * math.sqrt(radius_m**3 / mu_m3_s2)
    source = odss.ParticlePopulation(
        epoch=odss.epoch_from_iso("2026-06-20T12:00:00", "TAI"),
        frame=odss.ReferenceFrame("GCRS"),
        position_x_m=[radius_m],
        position_y_m=[0.0],
        position_z_m=[0.0],
        velocity_x_m_s=[0.0],
        velocity_y_m_s=[math.sqrt(mu_m3_s2 / radius_m)],
        velocity_z_m_s=[0.0],
        mass_kg=[100.0],
        area_m2=[1.0],
    )

    result = odss.propagate(
        source,
        odss.CascadePropagationSpec(period_s, 60.0, high_accuracy=True),
    )

    assert result.position_x_m == pytest.approx(source.position_x_m, abs=1.0e-5)
    assert result.position_y_m == pytest.approx(source.position_y_m, abs=1.0e-5)
    assert result.velocity_x_m_s == pytest.approx(source.velocity_x_m_s, abs=1.0e-8)
    assert result.velocity_y_m_s == pytest.approx(source.velocity_y_m_s, abs=1.0e-8)
