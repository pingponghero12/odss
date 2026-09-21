"""Configurable Cascade dynamics and sensitivity tools for SSO propagation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ._core import ParticlePopulation
from .cascade_screening import _combined_state, _event_from_cascade
from .conjunction import ConjunctionEvent, _require_screening_inputs
from .coordinates import elapsed_time_s, epoch_from_iso
from .propagation import (
    _cascade_state,
    _load_cascade,
    _population_from_cascade_state,
)

_earth_radius_m = 6_378_136.3
_cascade_drag_reference_density_kg_m3 = 0.1570 / _earth_radius_m
_j2000_tt = epoch_from_iso("2000-01-01T12:00:00", "TT")


@dataclass(frozen=True, slots=True)
class SsoForceModelSpec:
    """Terms and particle coefficients for Cascade's EME2000 Earth dynamics."""

    j2: bool = True
    j3: bool = False
    c22_s22: bool = False
    drag: bool = True
    sun: bool = False
    moon: bool = False
    srp: bool = False
    drag_coefficient: float = 2.2
    reflectivity_coefficient: float = 1.3

    def __post_init__(self) -> None:
        for value, name in (
            (self.j2, "j2"),
            (self.j3, "j3"),
            (self.c22_s22, "c22_s22"),
            (self.drag, "drag"),
            (self.sun, "sun"),
            (self.moon, "moon"),
            (self.srp, "srp"),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        for value, name in (
            (self.drag_coefficient, "drag_coefficient"),
            (self.reflectivity_coefficient, "reflectivity_coefficient"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class SsoPropagationSpec:
    """Numerical interval and force model for one SSO propagation."""

    duration_s: float
    collisional_timestep_s: float
    force_model: SsoForceModelSpec
    tolerance: float | None = None
    high_accuracy: bool = False
    collisional_steps_per_batch: int = 1

    def __post_init__(self) -> None:
        for value, name in (
            (self.duration_s, "duration_s"),
            (self.collisional_timestep_s, "collisional_timestep_s"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.force_model, SsoForceModelSpec):
            raise TypeError("force_model must be an SsoForceModelSpec")
        if self.tolerance is not None:
            if not isinstance(self.tolerance, (int, float)) or isinstance(self.tolerance, bool):
                raise TypeError("tolerance must be a number or None")
            if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
                raise ValueError("tolerance must be positive and finite")
        if not isinstance(self.high_accuracy, bool):
            raise TypeError("high_accuracy must be a bool")
        if (
            isinstance(self.collisional_steps_per_batch, bool)
            or not isinstance(self.collisional_steps_per_batch, int)
            or self.collisional_steps_per_batch <= 0
        ):
            raise ValueError("collisional_steps_per_batch must be a positive integer")


@dataclass(frozen=True, slots=True)
class ForceModelSensitivity:
    """Runtime and final-state difference from an explicit reference model."""

    force_model: SsoForceModelSpec
    wall_duration_s: float
    maximum_position_difference_m: float
    rms_position_difference_m: float
    maximum_velocity_difference_m_s: float


@dataclass(frozen=True, slots=True)
class SsoScreeningResult:
    """Final populations and continuous debris-target conjunction events."""

    final_debris: ParticlePopulation
    final_targets: ParticlePopulation
    conjunctions: tuple[ConjunctionEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.final_debris, ParticlePopulation):
            raise TypeError("final_debris must be a ParticlePopulation")
        if not isinstance(self.final_targets, ParticlePopulation):
            raise TypeError("final_targets must be a ParticlePopulation")
        object.__setattr__(self, "conjunctions", tuple(self.conjunctions))
        if any(not isinstance(event, ConjunctionEvent) for event in self.conjunctions):
            raise TypeError("conjunctions must contain only ConjunctionEvent values")


def _particle_parameters(
    population: ParticlePopulation,
    force_model: SsoForceModelSpec,
) -> np.ndarray | None:
    columns: list[np.ndarray] = []
    area_to_mass_m2_kg = np.asarray(population.area_m2) / np.asarray(population.mass_kg)
    if force_model.drag:
        bstar_m_inv = (
            0.5
            * force_model.drag_coefficient
            * area_to_mass_m2_kg
            * _cascade_drag_reference_density_kg_m3
        )
        columns.append(bstar_m_inv)
    if force_model.srp:
        columns.append(force_model.reflectivity_coefficient * area_to_mass_m2_kg)
    return np.column_stack(columns) if columns else None


def propagate_sso(
    population: ParticlePopulation,
    spec: SsoPropagationSpec,
) -> ParticlePopulation:
    """Propagate EME2000 particles with Cascade's configurable Earth model."""
    if not isinstance(population, ParticlePopulation):
        raise TypeError("population must be a ParticlePopulation")
    if not isinstance(spec, SsoPropagationSpec):
        raise TypeError("spec must be an SsoPropagationSpec")
    if population.frame.identifier != "EME2000":
        raise ValueError("SSO dynamics requires EME2000 particle states")
    if population.empty:
        return _population_from_cascade_state(population, np.empty((0, 7)), spec.duration_s)

    cascade = _load_cascade()
    force_model = spec.force_model
    arguments: dict[str, object] = {
        "dyn": cascade.dynamics.simple_earth(
            J2=force_model.j2,
            J3=force_model.j3,
            J4=False,
            C22S22=force_model.c22_s22,
            drag=force_model.drag,
            sun=force_model.sun,
            moon=force_model.moon,
            SRP=force_model.srp,
        ),
        "high_accuracy": spec.high_accuracy,
        "min_coll_radius": math.inf,
        "n_par_ct": spec.collisional_steps_per_batch,
    }
    parameters = _particle_parameters(population, force_model)
    if parameters is not None:
        arguments["pars"] = parameters
    if spec.tolerance is not None:
        arguments["tol"] = spec.tolerance

    simulation = cascade.sim(
        _cascade_state(population),
        spec.collisional_timestep_s,
        **arguments,
    )
    start_time_s = elapsed_time_s(_j2000_tt, population.epoch)
    simulation.time = start_time_s
    outcome = simulation.propagate_until(start_time_s + spec.duration_s)
    if outcome != cascade.outcome.time_limit:
        raise RuntimeError(f"Cascade SSO propagation stopped before the requested epoch: {outcome}")
    return _population_from_cascade_state(
        population,
        np.asarray(simulation.state, dtype=np.float64),
        spec.duration_s,
    )


def _combined_population(
    debris: ParticlePopulation,
    targets: ParticlePopulation,
) -> ParticlePopulation:
    def values(name: str) -> tuple[float, ...]:
        return tuple(getattr(debris, name)) + tuple(getattr(targets, name))

    return ParticlePopulation(
        epoch=debris.epoch,
        frame=debris.frame,
        position_x_m=values("position_x_m"),
        position_y_m=values("position_y_m"),
        position_z_m=values("position_z_m"),
        velocity_x_m_s=values("velocity_x_m_s"),
        velocity_y_m_s=values("velocity_y_m_s"),
        velocity_z_m_s=values("velocity_z_m_s"),
        mass_kg=values("mass_kg"),
        area_m2=values("area_m2"),
    )


def propagate_and_screen_sso(
    debris: ParticlePopulation,
    targets: ParticlePopulation,
    spec: SsoPropagationSpec,
    *,
    threshold_m: float,
) -> SsoScreeningResult:
    """Propagate both roles with one SSO model and continuously screen cross-role events."""
    if not isinstance(spec, SsoPropagationSpec):
        raise TypeError("spec must be an SsoPropagationSpec")
    _require_screening_inputs(debris, targets, spec.duration_s, threshold_m)
    if debris.frame.identifier != "EME2000":
        raise ValueError("SSO dynamics requires EME2000 particle states")
    if debris.empty or targets.empty:
        return SsoScreeningResult(
            final_debris=propagate_sso(debris, spec),
            final_targets=propagate_sso(targets, spec),
            conjunctions=(),
        )

    cascade = _load_cascade()
    force_model = spec.force_model
    combined = _combined_population(debris, targets)
    arguments: dict[str, object] = {
        "dyn": cascade.dynamics.simple_earth(
            J2=force_model.j2,
            J3=force_model.j3,
            J4=False,
            C22S22=force_model.c22_s22,
            drag=force_model.drag,
            sun=force_model.sun,
            moon=force_model.moon,
            SRP=force_model.srp,
        ),
        "high_accuracy": spec.high_accuracy,
        "min_coll_radius": math.inf,
        "n_par_ct": spec.collisional_steps_per_batch,
        "conj_thresh": np.nextafter(float(threshold_m), math.inf),
    }
    debris_count = len(debris)
    target_count = len(targets)
    if debris_count <= target_count:
        arguments["conj_whitelist"] = set(range(debris_count))
    else:
        arguments["conj_whitelist"] = set(range(debris_count, debris_count + target_count))
    parameters = _particle_parameters(combined, force_model)
    if parameters is not None:
        arguments["pars"] = parameters
    if spec.tolerance is not None:
        arguments["tol"] = spec.tolerance

    simulation = cascade.sim(
        _combined_state(debris, targets),
        spec.collisional_timestep_s,
        **arguments,
    )
    start_time_s = elapsed_time_s(_j2000_tt, debris.epoch)
    simulation.time = start_time_s
    outcome = simulation.propagate_until(start_time_s + spec.duration_s)
    if outcome != cascade.outcome.time_limit:
        raise RuntimeError(f"Cascade SSO propagation stopped before the requested epoch: {outcome}")

    events = []
    for backend_event in simulation.conjunctions:
        event = _event_from_cascade(backend_event, debris_count)
        if event is not None:
            events.append(
                ConjunctionEvent(
                    debris_index=event.debris_index,
                    target_index=event.target_index,
                    tca_s=event.tca_s - start_time_s,
                    miss_distance_m=event.miss_distance_m,
                    relative_velocity_m_s=event.relative_velocity_m_s,
                )
            )
    events.sort(key=lambda event: (event.tca_s, event.debris_index, event.target_index))
    final_state = np.asarray(simulation.state, dtype=np.float64)
    return SsoScreeningResult(
        final_debris=_population_from_cascade_state(
            debris,
            final_state[:debris_count],
            spec.duration_s,
        ),
        final_targets=_population_from_cascade_state(
            targets,
            final_state[debris_count:],
            spec.duration_s,
        ),
        conjunctions=tuple(events),
    )


def evaluate_force_model_sensitivity(
    population: ParticlePopulation,
    *,
    duration_s: float,
    collisional_timestep_s: float,
    candidates: tuple[SsoForceModelSpec, ...],
    reference: SsoForceModelSpec,
) -> tuple[ForceModelSensitivity, ...]:
    """Compare candidate final states and runtimes with one explicit reference model."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    reference_start = time.perf_counter()
    reference_population = propagate_sso(
        population,
        SsoPropagationSpec(duration_s, collisional_timestep_s, reference),
    )
    reference_wall_duration_s = time.perf_counter() - reference_start
    reference_position = np.column_stack(
        (
            reference_population.position_x_m,
            reference_population.position_y_m,
            reference_population.position_z_m,
        )
    )
    reference_velocity = np.column_stack(
        (
            reference_population.velocity_x_m_s,
            reference_population.velocity_y_m_s,
            reference_population.velocity_z_m_s,
        )
    )
    results: list[ForceModelSensitivity] = []
    for candidate in candidates:
        if candidate == reference:
            propagated = reference_population
            wall_duration_s = reference_wall_duration_s
        else:
            start = time.perf_counter()
            propagated = propagate_sso(
                population,
                SsoPropagationSpec(duration_s, collisional_timestep_s, candidate),
            )
            wall_duration_s = time.perf_counter() - start
        position = np.column_stack(
            (propagated.position_x_m, propagated.position_y_m, propagated.position_z_m)
        )
        velocity = np.column_stack(
            (
                propagated.velocity_x_m_s,
                propagated.velocity_y_m_s,
                propagated.velocity_z_m_s,
            )
        )
        position_difference_m = np.linalg.norm(position - reference_position, axis=1)
        velocity_difference_m_s = np.linalg.norm(velocity - reference_velocity, axis=1)
        results.append(
            ForceModelSensitivity(
                force_model=candidate,
                wall_duration_s=wall_duration_s,
                maximum_position_difference_m=float(np.max(position_difference_m, initial=0.0)),
                rms_position_difference_m=float(
                    np.sqrt(np.mean(position_difference_m**2))
                    if len(position_difference_m)
                    else 0.0
                ),
                maximum_velocity_difference_m_s=float(np.max(velocity_difference_m_s, initial=0.0)),
            )
        )
    return tuple(results)


def fastest_converged_force_model(
    sensitivities: tuple[ForceModelSensitivity, ...],
    *,
    maximum_position_difference_m: float,
    maximum_velocity_difference_m_s: float,
) -> SsoForceModelSpec:
    """Select the fastest measured candidate within explicit state-error limits."""
    converged = tuple(
        item
        for item in sensitivities
        if item.maximum_position_difference_m <= maximum_position_difference_m
        and item.maximum_velocity_difference_m_s <= maximum_velocity_difference_m_s
    )
    if not converged:
        raise ValueError("no force model satisfies the requested convergence limits")
    return min(converged, key=lambda item: item.wall_duration_s).force_model


__all__ = [
    "ForceModelSensitivity",
    "SsoForceModelSpec",
    "SsoPropagationSpec",
    "SsoScreeningResult",
    "evaluate_force_model_sensitivity",
    "fastest_converged_force_model",
    "propagate_and_screen_sso",
    "propagate_sso",
]
