"""Pure orbital decay, escape, and flux-derived impact-risk observables."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._core import ParticlePopulation
from .coordinates import elapsed_time_s
from .flux import FluxResult

OrbitClassification = Literal["retained", "removal", "reentry", "escape"]


@dataclass(frozen=True, slots=True)
class OrbitalDecaySpec:
    """Earth constants and altitude boundaries used by decay diagnostics."""

    gravitational_parameter_m3_s2: float = 3.986_004_418e14
    earth_radius_m: float = 6_378_136.3
    reentry_altitude_m: float = 100_000.0
    removal_altitude_m: float = 200_000.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.gravitational_parameter_m3_s2, "gravitational_parameter_m3_s2"),
            (self.earth_radius_m, "earth_radius_m"),
            (self.reentry_altitude_m, "reentry_altitude_m"),
            (self.removal_altitude_m, "removal_altitude_m"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.gravitational_parameter_m3_s2 == 0.0 or self.earth_radius_m == 0.0:
            raise ValueError("gravitational_parameter_m3_s2 and earth_radius_m must be positive")
        if self.reentry_altitude_m > self.removal_altitude_m:
            raise ValueError("reentry_altitude_m must not exceed removal_altitude_m")


@dataclass(frozen=True, slots=True)
class OrbitalDecayDiagnostic:
    """Osculating two-body orbital changes for one particle."""

    particle_index: int
    initial_semimajor_axis_m: float | None
    final_semimajor_axis_m: float | None
    semimajor_axis_change_m: float | None
    initial_perigee_altitude_m: float | None
    final_perigee_altitude_m: float | None
    perigee_altitude_change_m: float | None
    initial_apogee_altitude_m: float | None
    final_apogee_altitude_m: float | None
    apogee_altitude_change_m: float | None
    classification: OrbitClassification


@dataclass(frozen=True, slots=True)
class OrbitalDecayResult:
    """Particle diagnostics and population classification fractions."""

    spec: OrbitalDecaySpec
    diagnostics: tuple[OrbitalDecayDiagnostic, ...]
    retained_fraction: float
    removal_fraction: float
    reentry_fraction: float
    escape_fraction: float


@dataclass(frozen=True, slots=True)
class EscapeResult:
    """Two-body specific energy and Earth-unbound classification."""

    gravitational_parameter_m3_s2: float
    specific_orbital_energy_j_kg: tuple[float, ...]
    escaped: tuple[bool, ...]
    escaped_fraction: float


@dataclass(frozen=True, slots=True)
class ImpactRiskValue:
    """Flux-derived impact expectation for one target and reference area."""

    target_index: int
    reference_area_m2: float
    expected_impacts: float
    model_impact_probability: float


@dataclass(frozen=True, slots=True)
class ImpactRiskResult:
    """Impact-risk values under the rare independent-impact approximation."""

    values: tuple[ImpactRiskValue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))


def _states(population: ParticlePopulation) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.column_stack(
            (population.position_x_m, population.position_y_m, population.position_z_m)
        ),
        np.column_stack(
            (
                population.velocity_x_m_s,
                population.velocity_y_m_s,
                population.velocity_z_m_s,
            )
        ),
    )


def _specific_energy(
    population: ParticlePopulation,
    gravitational_parameter_m3_s2: float,
) -> np.ndarray:
    position_m, velocity_m_s = _states(population)
    radius_m = np.linalg.norm(position_m, axis=1)
    if np.any(radius_m == 0.0):
        raise ValueError("orbital diagnostics require non-zero geocentric position")
    return 0.5 * np.sum(velocity_m_s**2, axis=1) - gravitational_parameter_m3_s2 / radius_m


def evaluate_escape(
    population: ParticlePopulation,
    *,
    gravitational_parameter_m3_s2: float = 3.986_004_418e14,
) -> EscapeResult:
    """Classify particles with non-negative two-body specific orbital energy as escaped."""
    if not isinstance(population, ParticlePopulation):
        raise TypeError("population must be a ParticlePopulation")
    if (
        not isinstance(gravitational_parameter_m3_s2, (int, float))
        or isinstance(gravitational_parameter_m3_s2, bool)
        or not math.isfinite(gravitational_parameter_m3_s2)
        or gravitational_parameter_m3_s2 <= 0.0
    ):
        raise ValueError("gravitational_parameter_m3_s2 must be positive and finite")
    energy_j_kg = _specific_energy(population, gravitational_parameter_m3_s2)
    escaped = tuple(bool(value >= 0.0) for value in energy_j_kg)
    return EscapeResult(
        gravitational_parameter_m3_s2=float(gravitational_parameter_m3_s2),
        specific_orbital_energy_j_kg=tuple(float(value) for value in energy_j_kg),
        escaped=escaped,
        escaped_fraction=sum(escaped) / len(escaped) if escaped else 0.0,
    )


def _bound_orbit(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    spec: OrbitalDecaySpec,
) -> tuple[float, float, float] | None:
    radius_m = float(np.linalg.norm(position_m))
    if radius_m == 0.0:
        raise ValueError("orbital diagnostics require non-zero geocentric position")
    energy_j_kg = float(np.dot(velocity_m_s, velocity_m_s) / 2.0)
    energy_j_kg -= spec.gravitational_parameter_m3_s2 / radius_m
    if energy_j_kg >= 0.0:
        return None
    semimajor_axis_m = -spec.gravitational_parameter_m3_s2 / (2.0 * energy_j_kg)
    angular_momentum_m2_s = np.cross(position_m, velocity_m_s)
    eccentricity_squared = 1.0 + (
        2.0
        * energy_j_kg
        * float(np.dot(angular_momentum_m2_s, angular_momentum_m2_s))
        / spec.gravitational_parameter_m3_s2**2
    )
    eccentricity = math.sqrt(max(0.0, eccentricity_squared))
    perigee_altitude_m = semimajor_axis_m * (1.0 - eccentricity) - spec.earth_radius_m
    apogee_altitude_m = semimajor_axis_m * (1.0 + eccentricity) - spec.earth_radius_m
    return semimajor_axis_m, perigee_altitude_m, apogee_altitude_m


def _change(initial: float | None, final: float | None) -> float | None:
    return None if initial is None or final is None else final - initial


def evaluate_decay(
    initial: ParticlePopulation,
    final: ParticlePopulation,
    spec: OrbitalDecaySpec = OrbitalDecaySpec(),
) -> OrbitalDecayResult:
    """Compare initial and final osculating two-body orbital geometry."""
    if not isinstance(initial, ParticlePopulation) or not isinstance(final, ParticlePopulation):
        raise TypeError("initial and final must be ParticlePopulation values")
    if not isinstance(spec, OrbitalDecaySpec):
        raise TypeError("spec must be an OrbitalDecaySpec")
    if len(initial) != len(final):
        raise ValueError("initial and final populations must have equal size")
    if initial.frame != final.frame:
        raise ValueError("initial and final populations must use the same reference frame")
    if elapsed_time_s(initial.epoch, final.epoch) < 0.0:
        raise ValueError("final population epoch must not precede the initial epoch")

    initial_position_m, initial_velocity_m_s = _states(initial)
    final_position_m, final_velocity_m_s = _states(final)
    diagnostics: list[OrbitalDecayDiagnostic] = []
    classifications: list[OrbitClassification] = []
    for index in range(len(initial)):
        initial_orbit = _bound_orbit(initial_position_m[index], initial_velocity_m_s[index], spec)
        final_orbit = _bound_orbit(final_position_m[index], final_velocity_m_s[index], spec)
        if final_orbit is None:
            classification: OrbitClassification = "escape"
        elif final_orbit[1] <= spec.reentry_altitude_m:
            classification = "reentry"
        elif final_orbit[1] <= spec.removal_altitude_m:
            classification = "removal"
        else:
            classification = "retained"
        classifications.append(classification)
        initial_values = initial_orbit or (None, None, None)
        final_values = final_orbit or (None, None, None)
        diagnostics.append(
            OrbitalDecayDiagnostic(
                particle_index=index,
                initial_semimajor_axis_m=initial_values[0],
                final_semimajor_axis_m=final_values[0],
                semimajor_axis_change_m=_change(initial_values[0], final_values[0]),
                initial_perigee_altitude_m=initial_values[1],
                final_perigee_altitude_m=final_values[1],
                perigee_altitude_change_m=_change(initial_values[1], final_values[1]),
                initial_apogee_altitude_m=initial_values[2],
                final_apogee_altitude_m=final_values[2],
                apogee_altitude_change_m=_change(initial_values[2], final_values[2]),
                classification=classification,
            )
        )
    denominator = len(classifications)

    def fraction(value: OrbitClassification) -> float:
        return classifications.count(value) / denominator if denominator else 0.0

    return OrbitalDecayResult(
        spec=spec,
        diagnostics=tuple(diagnostics),
        retained_fraction=fraction("retained"),
        removal_fraction=fraction("removal"),
        reentry_fraction=fraction("reentry"),
        escape_fraction=fraction("escape"),
    )


def evaluate_impact_risk(
    flux: FluxResult,
    reference_areas_m2: Sequence[float],
) -> ImpactRiskResult:
    """Integrate number flux into a reference-area impact probability model."""
    if not isinstance(flux, FluxResult):
        raise TypeError("flux must be a FluxResult")
    areas_m2 = tuple(float(value) for value in reference_areas_m2)
    if not areas_m2:
        raise ValueError("reference_areas_m2 must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in areas_m2):
        raise ValueError("reference_areas_m2 values must be positive and finite")

    integrated_number_flux_m2 = [0.0] * flux.target_count
    for value in flux.bins:
        integrated_number_flux_m2[value.target_index] += value.number_flux_m2_s * (
            value.time_end_s - value.time_start_s
        )
    values = tuple(
        ImpactRiskValue(
            target_index=target_index,
            reference_area_m2=area_m2,
            expected_impacts=area_m2 * integrated_number_flux_m2[target_index],
            model_impact_probability=-math.expm1(
                -area_m2 * integrated_number_flux_m2[target_index]
            ),
        )
        for area_m2 in areas_m2
        for target_index in range(flux.target_count)
    )
    return ImpactRiskResult(values)


__all__ = [
    "EscapeResult",
    "ImpactRiskResult",
    "ImpactRiskValue",
    "OrbitClassification",
    "OrbitalDecayDiagnostic",
    "OrbitalDecayResult",
    "OrbitalDecaySpec",
    "evaluate_decay",
    "evaluate_escape",
    "evaluate_impact_risk",
]
