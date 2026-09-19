"""NASA Standard Breakup Model integration."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import numpy as np

from ._core import (
    CartesianState,
    Epoch,
    ParticlePopulation,
    PhysicalProperties,
    RandomKey,
    ReferenceFrame,
    random_u64,
)

SatelliteType = Literal["spacecraft", "rocket_body"]

_maximum_backend_seed = (1 << 31) - 1


def _require_positive_finite(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be positive and finite")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be positive and finite")


def _require_satellite_type(value: str, field_name: str) -> None:
    if value not in ("spacecraft", "rocket_body"):
        raise ValueError(f"{field_name} must be spacecraft or rocket_body")


def nasa_sbm_seed(random_key: RandomKey) -> int:
    """Derive the non-negative seed accepted by nasa-sbm-py from an ODSS random key."""
    if not isinstance(random_key, RandomKey):
        raise TypeError("random_key must be a RandomKey")
    return random_u64(random_key, 0) & _maximum_backend_seed


@dataclass(frozen=True, slots=True)
class FragmentationResult:
    """An immutable fragment population with NASA SBM physical metadata."""

    population: ParticlePopulation
    characteristic_length_m: tuple[float, ...]
    area_to_mass_ratio_m2_kg: tuple[float, ...]
    delta_velocity_x_m_s: tuple[float, ...]
    delta_velocity_y_m_s: tuple[float, ...]
    delta_velocity_z_m_s: tuple[float, ...]
    minimum_characteristic_length_m: float
    generated_minimum_characteristic_length_m: float
    event_kind: Literal["explosion", "collision"]
    random_key: RandomKey
    backend_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.population, ParticlePopulation):
            raise TypeError("population must be a ParticlePopulation")
        for field_name in (
            "characteristic_length_m",
            "area_to_mass_ratio_m2_kg",
            "delta_velocity_x_m_s",
            "delta_velocity_y_m_s",
            "delta_velocity_z_m_s",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        expected_size = len(self.population)
        fields = (
            self.characteristic_length_m,
            self.area_to_mass_ratio_m2_kg,
            self.delta_velocity_x_m_s,
            self.delta_velocity_y_m_s,
            self.delta_velocity_z_m_s,
        )
        if any(len(values) != expected_size for values in fields):
            raise ValueError("fragment metadata fields must match the population size")
        _require_positive_finite(
            self.minimum_characteristic_length_m,
            "minimum_characteristic_length_m",
        )
        _require_positive_finite(
            self.generated_minimum_characteristic_length_m,
            "generated_minimum_characteristic_length_m",
        )
        if (
            self.minimum_characteristic_length_m
            < self.generated_minimum_characteristic_length_m
        ):
            raise ValueError("selection cutoff cannot be smaller than the generated cutoff")
        if any(
            not math.isfinite(value)
            or value < self.minimum_characteristic_length_m
            for value in self.characteristic_length_m
        ):
            raise ValueError("characteristic lengths must be finite and meet the generated cutoff")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.area_to_mass_ratio_m2_kg):
            raise ValueError("area-to-mass ratios must be positive and finite")
        if any(
            not math.isclose(ratio, area / mass, rel_tol=1e-12, abs_tol=0.0)
            for ratio, area, mass in zip(
                self.area_to_mass_ratio_m2_kg,
                self.population.area_m2,
                self.population.mass_kg,
                strict=True,
            )
        ):
            raise ValueError("area-to-mass ratios must match fragment areas and masses")
        if any(
            not math.isfinite(value)
            for values in (
                self.delta_velocity_x_m_s,
                self.delta_velocity_y_m_s,
                self.delta_velocity_z_m_s,
            )
            for value in values
        ):
            raise ValueError("delta velocities must be finite")
        if self.event_kind not in ("explosion", "collision"):
            raise ValueError("event_kind must be explosion or collision")
        if not isinstance(self.random_key, RandomKey):
            raise TypeError("random_key must be a RandomKey")
        if (
            isinstance(self.backend_seed, bool)
            or not isinstance(self.backend_seed, int)
            or not 0 <= self.backend_seed <= _maximum_backend_seed
        ):
            raise ValueError("backend_seed must be a non-negative 31-bit integer")

    def __len__(self) -> int:
        return len(self.population)


def _load_backend() -> ModuleType:
    try:
        return importlib.import_module("nasa_sbm")
    except ImportError as error:
        raise ImportError(
            "nasa-sbm-py is required for fragmentation; install the optional backend first"
        ) from error


def _values(dataset: object, field_name: str) -> np.ndarray:
    try:
        variable = getattr(dataset, field_name)
        return np.asarray(variable.values, dtype=float)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(f"nasa-sbm-py returned an invalid {field_name} field") from error


def _convert_result(
    dataset: object,
    *,
    epoch: Epoch,
    frame: ReferenceFrame,
    minimum_characteristic_length_m: float,
    event_kind: Literal["explosion", "collision"],
    random_key: RandomKey,
    backend_seed: int,
) -> FragmentationResult:
    length_m = _values(dataset, "fragment_size")
    mass_kg = _values(dataset, "fragment_mass")
    area_m2 = _values(dataset, "cross_sectional_area")
    area_to_mass_m2_kg = _values(dataset, "area_to_mass_ratio")
    position_km = _values(dataset, "position")
    velocity_km_s = _values(dataset, "velocity")
    delta_velocity_km_s = _values(dataset, "ejection_velocity")

    size = length_m.size
    scalar_fields = (length_m, mass_kg, area_m2, area_to_mass_m2_kg)
    vector_fields = (position_km, velocity_km_s, delta_velocity_km_s)
    if any(values.ndim != 1 or values.size != size for values in scalar_fields):
        raise RuntimeError("nasa-sbm-py returned inconsistent scalar fragment fields")
    if any(values.shape != (size, 3) for values in vector_fields):
        raise RuntimeError("nasa-sbm-py returned inconsistent Cartesian fragment fields")
    if any(not np.all(np.isfinite(values)) for values in (*scalar_fields, *vector_fields)):
        raise RuntimeError("nasa-sbm-py returned non-finite fragment values")
    if (
        np.any(length_m < minimum_characteristic_length_m)
        or np.any(mass_kg <= 0.0)
        or np.any(area_m2 <= 0.0)
        or np.any(area_to_mass_m2_kg <= 0.0)
    ):
        raise RuntimeError("nasa-sbm-py returned non-physical fragment values")

    position_m = position_km * 1_000.0
    velocity_m_s = velocity_km_s * 1_000.0
    delta_velocity_m_s = delta_velocity_km_s * 1_000.0
    population = ParticlePopulation(
        epoch=epoch,
        frame=frame,
        position_x_m=position_m[:, 0].tolist(),
        position_y_m=position_m[:, 1].tolist(),
        position_z_m=position_m[:, 2].tolist(),
        velocity_x_m_s=velocity_m_s[:, 0].tolist(),
        velocity_y_m_s=velocity_m_s[:, 1].tolist(),
        velocity_z_m_s=velocity_m_s[:, 2].tolist(),
        mass_kg=mass_kg.tolist(),
        area_m2=area_m2.tolist(),
    )
    return FragmentationResult(
        population=population,
        characteristic_length_m=tuple(float(value) for value in length_m),
        area_to_mass_ratio_m2_kg=tuple(float(value) for value in area_to_mass_m2_kg),
        delta_velocity_x_m_s=tuple(float(value) for value in delta_velocity_m_s[:, 0]),
        delta_velocity_y_m_s=tuple(float(value) for value in delta_velocity_m_s[:, 1]),
        delta_velocity_z_m_s=tuple(float(value) for value in delta_velocity_m_s[:, 2]),
        minimum_characteristic_length_m=minimum_characteristic_length_m,
        generated_minimum_characteristic_length_m=minimum_characteristic_length_m,
        event_kind=event_kind,
        random_key=random_key,
        backend_seed=backend_seed,
    )


def generate_explosion_fragments(
    parent_state: CartesianState,
    parent_properties: PhysicalProperties,
    *,
    satellite_type: SatelliteType,
    minimum_characteristic_length_m: float,
    random_key: RandomKey,
) -> FragmentationResult:
    """Generate one deterministic NASA SBM explosion at the parent state."""
    if not isinstance(parent_state, CartesianState):
        raise TypeError("parent_state must be a CartesianState")
    if not isinstance(parent_properties, PhysicalProperties):
        raise TypeError("parent_properties must be PhysicalProperties")
    _require_satellite_type(satellite_type, "satellite_type")
    _require_positive_finite(minimum_characteristic_length_m, "minimum_characteristic_length_m")
    backend_seed = nasa_sbm_seed(random_key)
    dataset = _load_backend().explosion(
        mass=parent_properties.mass_kg,
        sat_type=satellite_type,
        cutoff=minimum_characteristic_length_m,
        seed=backend_seed,
        position=tuple(value / 1_000.0 for value in parent_state.position_m),
        velocity=tuple(value / 1_000.0 for value in parent_state.velocity_m_s),
    )
    return _convert_result(
        dataset,
        epoch=parent_state.epoch,
        frame=parent_state.frame,
        minimum_characteristic_length_m=minimum_characteristic_length_m,
        event_kind="explosion",
        random_key=random_key,
        backend_seed=backend_seed,
    )


def generate_collision_fragments(
    primary_state: CartesianState,
    primary_properties: PhysicalProperties,
    secondary_state: CartesianState,
    secondary_properties: PhysicalProperties,
    *,
    primary_type: SatelliteType,
    secondary_type: SatelliteType,
    minimum_characteristic_length_m: float,
    random_key: RandomKey,
) -> FragmentationResult:
    """Generate one deterministic NASA SBM collision at a shared state position."""
    if not isinstance(primary_state, CartesianState) or not isinstance(
        secondary_state, CartesianState
    ):
        raise TypeError("collision states must be CartesianState values")
    if not isinstance(primary_properties, PhysicalProperties) or not isinstance(
        secondary_properties, PhysicalProperties
    ):
        raise TypeError("collision properties must be PhysicalProperties values")
    if primary_state.epoch != secondary_state.epoch:
        raise ValueError("collision states must have the same epoch")
    if primary_state.frame != secondary_state.frame:
        raise ValueError("collision states must have the same reference frame")
    if primary_state.position_m != secondary_state.position_m:
        raise ValueError("collision states must have the same event position")
    _require_satellite_type(primary_type, "primary_type")
    _require_satellite_type(secondary_type, "secondary_type")
    _require_positive_finite(minimum_characteristic_length_m, "minimum_characteristic_length_m")
    relative_velocity_m_s = math.sqrt(
        sum(
            (primary - secondary) ** 2
            for primary, secondary in zip(
                primary_state.velocity_m_s,
                secondary_state.velocity_m_s,
                strict=True,
            )
        )
    )
    _require_positive_finite(relative_velocity_m_s, "relative collision velocity")
    backend_seed = nasa_sbm_seed(random_key)
    dataset = _load_backend().collision(
        mass1=primary_properties.mass_kg,
        mass2=secondary_properties.mass_kg,
        velocity_relative=relative_velocity_m_s / 1_000.0,
        sat_type1=primary_type,
        sat_type2=secondary_type,
        cutoff=minimum_characteristic_length_m,
        seed=backend_seed,
        position=tuple(value / 1_000.0 for value in primary_state.position_m),
        velocity1=tuple(value / 1_000.0 for value in primary_state.velocity_m_s),
        velocity2=tuple(value / 1_000.0 for value in secondary_state.velocity_m_s),
    )
    return _convert_result(
        dataset,
        epoch=primary_state.epoch,
        frame=primary_state.frame,
        minimum_characteristic_length_m=minimum_characteristic_length_m,
        event_kind="collision",
        random_key=random_key,
        backend_seed=backend_seed,
    )


def select_fragments_by_size(
    result: FragmentationResult,
    minimum_characteristic_length_m: float,
) -> FragmentationResult:
    """Select a larger size cutoff from one already-generated breakup result."""
    if not isinstance(result, FragmentationResult):
        raise TypeError("result must be a FragmentationResult")
    _require_positive_finite(minimum_characteristic_length_m, "minimum_characteristic_length_m")
    if minimum_characteristic_length_m < result.generated_minimum_characteristic_length_m:
        raise ValueError("selection cutoff cannot be smaller than the generated cutoff")
    indices = tuple(
        index
        for index, length_m in enumerate(result.characteristic_length_m)
        if length_m >= minimum_characteristic_length_m
    )
    population = result.population
    subset = ParticlePopulation(
        epoch=population.epoch,
        frame=population.frame,
        position_x_m=[population.position_x_m[index] for index in indices],
        position_y_m=[population.position_y_m[index] for index in indices],
        position_z_m=[population.position_z_m[index] for index in indices],
        velocity_x_m_s=[population.velocity_x_m_s[index] for index in indices],
        velocity_y_m_s=[population.velocity_y_m_s[index] for index in indices],
        velocity_z_m_s=[population.velocity_z_m_s[index] for index in indices],
        mass_kg=[population.mass_kg[index] for index in indices],
        area_m2=[population.area_m2[index] for index in indices],
    )
    return FragmentationResult(
        population=subset,
        characteristic_length_m=tuple(result.characteristic_length_m[index] for index in indices),
        area_to_mass_ratio_m2_kg=tuple(
            result.area_to_mass_ratio_m2_kg[index] for index in indices
        ),
        delta_velocity_x_m_s=tuple(result.delta_velocity_x_m_s[index] for index in indices),
        delta_velocity_y_m_s=tuple(result.delta_velocity_y_m_s[index] for index in indices),
        delta_velocity_z_m_s=tuple(result.delta_velocity_z_m_s[index] for index in indices),
        minimum_characteristic_length_m=minimum_characteristic_length_m,
        generated_minimum_characteristic_length_m=(
            result.generated_minimum_characteristic_length_m
        ),
        event_kind=result.event_kind,
        random_key=result.random_key,
        backend_seed=result.backend_seed,
    )


__all__ = [
    "FragmentationResult",
    "SatelliteType",
    "generate_collision_fragments",
    "generate_explosion_fragments",
    "nasa_sbm_seed",
    "select_fragments_by_size",
]
