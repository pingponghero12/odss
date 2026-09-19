"""Explicit propagation backends for catalog and particle state data."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import overload

import numpy as np

from ._core import Epoch, ParticlePopulation
from .catalog import Catalog
from .sgp4 import SynchronizedCatalog, synchronize_catalog_sgp4

_earth_gravitational_parameter_m3_s2 = 3.986_004_418e14
_cascade_frame = "GCRS"


@dataclass(frozen=True, slots=True)
class Sgp4PropagationSpec:
    """Propagate catalog mean elements to one explicit absolute epoch."""

    epoch: Epoch

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, Epoch):
            raise TypeError("epoch must be an Epoch")


@dataclass(frozen=True, slots=True)
class CascadePropagationSpec:
    """Numerically propagate GCRS particles with point-mass Earth gravity."""

    duration_s: float
    collisional_timestep_s: float
    gravitational_parameter_m3_s2: float = _earth_gravitational_parameter_m3_s2
    tolerance: float | None = None
    high_accuracy: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.duration_s, "duration_s"),
            (self.collisional_timestep_s, "collisional_timestep_s"),
            (self.gravitational_parameter_m3_s2, "gravitational_parameter_m3_s2"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.tolerance is not None:
            if not isinstance(self.tolerance, (int, float)) or isinstance(self.tolerance, bool):
                raise TypeError("tolerance must be a number or None")
            if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
                raise ValueError("tolerance must be positive and finite")
        if not isinstance(self.high_accuracy, bool):
            raise TypeError("high_accuracy must be a bool")


def _load_cascade() -> object:
    try:
        return importlib.import_module("cascade")
    except ImportError as error:
        raise ImportError(
            "Cascade propagation requires the optional 'cascade' Python package; "
            "install the conda-forge 'cascade' package"
        ) from error


def _cascade_state(population: ParticlePopulation) -> np.ndarray:
    radii_m = np.sqrt(np.asarray(population.area_m2, dtype=np.float64) / math.pi)
    return np.column_stack(
        (
            population.position_x_m,
            population.position_y_m,
            population.position_z_m,
            population.velocity_x_m_s,
            population.velocity_y_m_s,
            population.velocity_z_m_s,
            radii_m,
        )
    )


def _population_from_cascade_state(
    source: ParticlePopulation,
    state: np.ndarray,
    duration_s: float,
) -> ParticlePopulation:
    final_epoch = Epoch(
        source.epoch.offset_s + duration_s,
        source.epoch.reference_epoch,
        source.epoch.time_scale,
    )
    return ParticlePopulation(
        epoch=final_epoch,
        frame=source.frame,
        position_x_m=state[:, 0],
        position_y_m=state[:, 1],
        position_z_m=state[:, 2],
        velocity_x_m_s=state[:, 3],
        velocity_y_m_s=state[:, 4],
        velocity_z_m_s=state[:, 5],
        mass_kg=source.mass_kg,
        area_m2=source.area_m2,
    )


def _propagate_cascade(
    population: ParticlePopulation,
    spec: CascadePropagationSpec,
) -> ParticlePopulation:
    if population.frame.identifier != _cascade_frame:
        raise ValueError("Cascade propagation requires GCRS particle states")
    if population.empty:
        return _population_from_cascade_state(population, np.empty((0, 7)), spec.duration_s)

    cascade = _load_cascade()
    arguments: dict[str, object] = {
        "dyn": cascade.dynamics.kepler(mu=spec.gravitational_parameter_m3_s2),
        "high_accuracy": spec.high_accuracy,
    }
    if spec.tolerance is not None:
        arguments["tol"] = spec.tolerance
    simulation = cascade.sim(
        _cascade_state(population),
        spec.collisional_timestep_s,
        **arguments,
    )
    outcome = simulation.propagate_until(spec.duration_s)
    if outcome != cascade.outcome.time_limit:
        raise RuntimeError(f"Cascade propagation stopped before the requested epoch: {outcome}")
    return _population_from_cascade_state(
        population,
        np.asarray(simulation.state, dtype=np.float64),
        spec.duration_s,
    )


@overload
def propagate(source: Catalog, spec: Sgp4PropagationSpec) -> SynchronizedCatalog: ...


@overload
def propagate(source: ParticlePopulation, spec: CascadePropagationSpec) -> ParticlePopulation: ...


def propagate(
    source: Catalog | ParticlePopulation,
    spec: Sgp4PropagationSpec | CascadePropagationSpec,
) -> SynchronizedCatalog | ParticlePopulation:
    """Propagate supported source data with the backend selected by an immutable specification."""
    if isinstance(spec, Sgp4PropagationSpec):
        if not isinstance(source, Catalog):
            raise TypeError("Sgp4PropagationSpec requires a Catalog source")
        return synchronize_catalog_sgp4(source, spec.epoch)
    if isinstance(spec, CascadePropagationSpec):
        if not isinstance(source, ParticlePopulation):
            raise TypeError("CascadePropagationSpec requires a ParticlePopulation source")
        return _propagate_cascade(source, spec)
    raise TypeError("spec must be Sgp4PropagationSpec or CascadePropagationSpec")


__all__ = ["CascadePropagationSpec", "Sgp4PropagationSpec", "propagate"]
