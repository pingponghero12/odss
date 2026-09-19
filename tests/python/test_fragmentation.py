import importlib
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import odss
import odss.fragmentation as fragmentation


class FakeVariable:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)


class FakeDataset:
    def __init__(self) -> None:
        self.fragment_size = FakeVariable([0.1, 0.2])
        self.fragment_mass = FakeVariable([1.0, 2.0])
        self.cross_sectional_area = FakeVariable([0.5, 0.4])
        self.area_to_mass_ratio = FakeVariable([0.5, 0.2])
        self.position = FakeVariable([[7000.0, 0.0, 0.0], [7000.0, 0.0, 0.0]])
        self.velocity = FakeVariable([[0.0, 7.51, 0.0], [0.01, 7.5, 0.0]])
        self.ejection_velocity = FakeVariable([[0.0, 0.01, 0.0], [0.01, 0.0, 0.0]])


class FakeBackend:
    def __init__(self) -> None:
        self.explosion_arguments: dict[str, object] | None = None
        self.collision_arguments: dict[str, object] | None = None

    def explosion(self, **arguments: object) -> FakeDataset:
        self.explosion_arguments = arguments
        return FakeDataset()

    def collision(self, **arguments: object) -> FakeDataset:
        self.collision_arguments = arguments
        return FakeDataset()


def epoch() -> odss.Epoch:
    return odss.epoch_from_iso("2026-06-20T12:00:00", "UTC")


def frame() -> odss.ReferenceFrame:
    return odss.ReferenceFrame("GCRS")


def state(velocity_m_s: tuple[float, float, float] = (0.0, 7_500.0, 0.0)) -> odss.CartesianState:
    return odss.CartesianState((7_000_000.0, 0.0, 0.0), velocity_m_s, epoch(), frame())


def random_key() -> odss.RandomKey:
    return odss.named_random_key(
        master_seed=2026,
        scenario_id=2,
        run_id=3,
        object_id=4,
        stream_name="nasa_sbm",
    )


def install_fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    backend = FakeBackend()
    monkeypatch.setattr(fragmentation, "_load_backend", lambda: backend)
    return backend


def test_explosion_converts_si_boundary_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = install_fake_backend(monkeypatch)
    parent_state = state()
    key = random_key()

    result = odss.generate_explosion_fragments(
        parent_state,
        odss.PhysicalProperties(20.0, 4.0),
        satellite_type="spacecraft",
        minimum_characteristic_length_m=0.1,
        random_key=key,
    )

    assert backend.explosion_arguments == {
        "mass": 20.0,
        "sat_type": "spacecraft",
        "cutoff": 0.1,
        "seed": odss.nasa_sbm_seed(key),
        "position": (7000.0, 0.0, 0.0),
        "velocity": (0.0, 7.5, 0.0),
    }
    assert result.event_kind == "explosion"
    assert result.random_key == key
    assert result.backend_seed == odss.random_u64(key, 0) & ((1 << 31) - 1)
    assert result.population.epoch == parent_state.epoch
    assert result.population.frame == parent_state.frame
    assert result.population.position_x_m == (7_000_000.0, 7_000_000.0)
    assert result.population.velocity_y_m_s == (7_510.0, 7_500.0)
    assert result.population.mass_kg == (1.0, 2.0)
    assert result.population.area_m2 == (0.5, 0.4)
    assert result.characteristic_length_m == (0.1, 0.2)
    assert result.area_to_mass_ratio_m2_kg == (0.5, 0.2)
    assert result.delta_velocity_y_m_s == (10.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        result.event_kind = "collision"


def test_collision_passes_relative_state_and_parent_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = install_fake_backend(monkeypatch)
    primary_state = state((0.0, 7_500.0, 0.0))
    secondary_state = state((0.0, -2_500.0, 0.0))

    result = odss.generate_collision_fragments(
        primary_state,
        odss.PhysicalProperties(20.0, 4.0),
        secondary_state,
        odss.PhysicalProperties(5.0, 1.0),
        primary_type="spacecraft",
        secondary_type="rocket_body",
        minimum_characteristic_length_m=0.1,
        random_key=random_key(),
    )

    assert backend.collision_arguments is not None
    assert backend.collision_arguments["mass1"] == 20.0
    assert backend.collision_arguments["mass2"] == 5.0
    assert backend.collision_arguments["velocity_relative"] == 10.0
    assert backend.collision_arguments["sat_type1"] == "spacecraft"
    assert backend.collision_arguments["sat_type2"] == "rocket_body"
    assert backend.collision_arguments["position"] == (7000.0, 0.0, 0.0)
    assert backend.collision_arguments["velocity1"] == (0.0, 7.5, 0.0)
    assert backend.collision_arguments["velocity2"] == (0.0, -2.5, 0.0)
    assert result.event_kind == "collision"


def test_larger_size_subsets_do_not_rerun_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = install_fake_backend(monkeypatch)
    generated = odss.generate_explosion_fragments(
        state(),
        odss.PhysicalProperties(20.0, 4.0),
        satellite_type="spacecraft",
        minimum_characteristic_length_m=0.1,
        random_key=random_key(),
    )

    selected = odss.select_fragments_by_size(generated, 0.15)

    assert backend.explosion_arguments is not None
    assert len(selected) == 1
    assert selected.characteristic_length_m == (0.2,)
    assert selected.population.mass_kg == (2.0,)
    assert selected.minimum_characteristic_length_m == 0.15
    assert selected.generated_minimum_characteristic_length_m == 0.1
    assert selected.random_key == generated.random_key
    assert selected.backend_seed == generated.backend_seed
    assert len(odss.select_fragments_by_size(generated, 1.0)) == 0
    with pytest.raises(ValueError, match="cannot be smaller"):
        odss.select_fragments_by_size(generated, 0.05)


def test_collision_requires_one_explicit_event_state(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_backend(monkeypatch)
    properties = odss.PhysicalProperties(20.0, 4.0)
    key = random_key()
    common = state()

    different_epoch = odss.CartesianState(
        common.position_m,
        common.velocity_m_s,
        odss.epoch_from_iso("2026-06-20T12:00:01", "UTC"),
        common.frame,
    )
    with pytest.raises(ValueError, match="same epoch"):
        odss.generate_collision_fragments(
            common,
            properties,
            different_epoch,
            properties,
            primary_type="spacecraft",
            secondary_type="spacecraft",
            minimum_characteristic_length_m=0.1,
            random_key=key,
        )

    different_position = odss.CartesianState(
        (7_000_001.0, 0.0, 0.0),
        common.velocity_m_s,
        common.epoch,
        common.frame,
    )
    with pytest.raises(ValueError, match="same event position"):
        odss.generate_collision_fragments(
            common,
            properties,
            different_position,
            properties,
            primary_type="spacecraft",
            secondary_type="spacecraft",
            minimum_characteristic_length_m=0.1,
            random_key=key,
        )

    with pytest.raises(ValueError, match="relative collision velocity"):
        odss.generate_collision_fragments(
            common,
            properties,
            common,
            properties,
            primary_type="spacecraft",
            secondary_type="spacecraft",
            minimum_characteristic_length_m=0.1,
            random_key=key,
        )


def test_fragmentation_validates_public_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_backend(monkeypatch)
    properties = odss.PhysicalProperties(20.0, 4.0)

    with pytest.raises(ValueError, match="satellite_type"):
        odss.generate_explosion_fragments(
            state(),
            properties,
            satellite_type="debris",
            minimum_characteristic_length_m=0.1,
            random_key=random_key(),
        )
    with pytest.raises(ValueError, match="positive and finite"):
        odss.generate_explosion_fragments(
            state(),
            properties,
            satellite_type="spacecraft",
            minimum_characteristic_length_m=math.nan,
            random_key=random_key(),
        )
    with pytest.raises(TypeError, match="RandomKey"):
        odss.nasa_sbm_seed("not a key")


def test_invalid_backend_output_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = install_fake_backend(monkeypatch)
    dataset = FakeDataset()
    dataset.fragment_mass = FakeVariable([-1.0, 2.0])
    monkeypatch.setattr(backend, "explosion", lambda **arguments: dataset)

    with pytest.raises(RuntimeError, match="non-physical"):
        odss.generate_explosion_fragments(
            state(),
            odss.PhysicalProperties(20.0, 4.0),
            satellite_type="spacecraft",
            minimum_characteristic_length_m=0.1,
            random_key=random_key(),
        )


def test_missing_optional_backend_has_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_backend(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(fragmentation.importlib, "import_module", missing_backend)

    with pytest.raises(ImportError, match="install the optional backend"):
        odss.generate_explosion_fragments(
            state(),
            odss.PhysicalProperties(20.0, 4.0),
            satellite_type="spacecraft",
            minimum_characteristic_length_m=0.1,
            random_key=random_key(),
        )


def test_real_nasa_sbm_explosion_and_collision_are_deterministic_and_physical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    pytest.importorskip("nasa_sbm")
    monkeypatch.setattr(fragmentation, "_load_backend", lambda: importlib.import_module("nasa_sbm"))
    parent = state()
    properties = odss.PhysicalProperties(20.0, 4.0)
    key = random_key()

    first = odss.generate_explosion_fragments(
        parent,
        properties,
        satellite_type="spacecraft",
        minimum_characteristic_length_m=0.2,
        random_key=key,
    )
    second = odss.generate_explosion_fragments(
        parent,
        properties,
        satellite_type="spacecraft",
        minimum_characteristic_length_m=0.2,
        random_key=key,
    )
    collision = odss.generate_collision_fragments(
        parent,
        properties,
        state((0.0, -2_500.0, 0.0)),
        odss.PhysicalProperties(5.0, 1.0),
        primary_type="spacecraft",
        secondary_type="rocket_body",
        minimum_characteristic_length_m=0.2,
        random_key=key,
    )

    assert first == second
    different_key = odss.named_random_key(
        master_seed=2027,
        scenario_id=2,
        run_id=3,
        object_id=4,
        stream_name="nasa_sbm",
    )
    different = odss.generate_explosion_fragments(
        parent,
        properties,
        satellite_type="spacecraft",
        minimum_characteristic_length_m=0.2,
        random_key=different_key,
    )
    assert different.backend_seed != first.backend_seed
    assert different.characteristic_length_m != first.characteristic_length_m
    assert len(first) > 0
    assert len(collision) > 0
    assert sum(first.population.mass_kg) <= properties.mass_kg
    assert sum(collision.population.mass_kg) <= 25.0
    for result in (first, collision):
        assert all(length >= 0.2 for length in result.characteristic_length_m)
        assert all(mass > 0.0 for mass in result.population.mass_kg)
        assert all(area > 0.0 for area in result.population.area_m2)
        assert tuple(
            area / mass
            for area, mass in zip(result.population.area_m2, result.population.mass_kg, strict=True)
        ) == pytest.approx(result.area_to_mass_ratio_m2_kg)
        assert all(
            math.isfinite(component)
            for components in (
                result.delta_velocity_x_m_s,
                result.delta_velocity_y_m_s,
                result.delta_velocity_z_m_s,
            )
            for component in components
        )
