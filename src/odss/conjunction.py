"""Continuous local conjunction screening for small particle populations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._core import ParticlePopulation


@dataclass(frozen=True, slots=True)
class ConjunctionEvent:
    """Closest approach for one debris-target pair over a local time interval."""

    debris_index: int
    target_index: int
    tca_s: float
    miss_distance_m: float
    relative_velocity_m_s: float

    def __post_init__(self) -> None:
        if self.debris_index < 0 or self.target_index < 0:
            raise ValueError("particle indices must be non-negative")
        for value, name in (
            (self.tca_s, "tca_s"),
            (self.miss_distance_m, "miss_distance_m"),
            (self.relative_velocity_m_s, "relative_velocity_m_s"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")


def _require_screening_inputs(
    debris: ParticlePopulation,
    targets: ParticlePopulation,
    duration_s: float,
    threshold_m: float,
) -> None:
    if not isinstance(debris, ParticlePopulation):
        raise TypeError("debris must be a ParticlePopulation")
    if not isinstance(targets, ParticlePopulation):
        raise TypeError("targets must be a ParticlePopulation")
    if debris.epoch != targets.epoch:
        raise ValueError("debris and targets must use the same epoch")
    if debris.frame != targets.frame:
        raise ValueError("debris and targets must use the same reference frame")
    for value, name, allow_zero in (
        (duration_s, "duration_s", False),
        (threshold_m, "threshold_m", True),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier} and finite")


def _vector_fields(
    population: ParticlePopulation,
    index: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (
            population.position_x_m[index],
            population.position_y_m[index],
            population.position_z_m[index],
        ),
        (
            population.velocity_x_m_s[index],
            population.velocity_y_m_s[index],
            population.velocity_z_m_s[index],
        ),
    )


def _local_tca(
    relative_position_m: tuple[float, float, float],
    relative_velocity_m_s: tuple[float, float, float],
    duration_s: float,
) -> tuple[float, float, float] | None:
    speed_squared_m2_s2 = sum(value * value for value in relative_velocity_m_s)
    if speed_squared_m2_s2 == 0.0:
        return None
    tca_s = -sum(
        position * velocity
        for position, velocity in zip(
            relative_position_m,
            relative_velocity_m_s,
            strict=True,
        )
    ) / speed_squared_m2_s2
    if not 0.0 < tca_s < duration_s:
        return None
    displacement_m = tuple(
        position + velocity * tca_s
        for position, velocity in zip(
            relative_position_m,
            relative_velocity_m_s,
            strict=True,
        )
    )
    return (
        tca_s,
        math.sqrt(sum(value * value for value in displacement_m)),
        math.sqrt(speed_squared_m2_s2),
    )


def screen_conjunctions_reference(
    debris: ParticlePopulation,
    targets: ParticlePopulation,
    *,
    duration_s: float,
    threshold_m: float,
) -> tuple[ConjunctionEvent, ...]:
    """Brute-force all debris-target pairs using exact constant-velocity local TCA."""
    _require_screening_inputs(debris, targets, duration_s, threshold_m)
    events: list[ConjunctionEvent] = []
    for debris_index in range(len(debris)):
        debris_position_m, debris_velocity_m_s = _vector_fields(debris, debris_index)
        for target_index in range(len(targets)):
            target_position_m, target_velocity_m_s = _vector_fields(targets, target_index)
            relative_position_m = tuple(
                debris_value - target_value
                for debris_value, target_value in zip(
                    debris_position_m,
                    target_position_m,
                    strict=True,
                )
            )
            relative_velocity_m_s = tuple(
                debris_value - target_value
                for debris_value, target_value in zip(
                    debris_velocity_m_s,
                    target_velocity_m_s,
                    strict=True,
                )
            )
            closest_approach = _local_tca(
                relative_position_m,
                relative_velocity_m_s,
                duration_s,
            )
            if closest_approach is None:
                continue
            tca_s, miss_distance_m, relative_velocity_m_s_magnitude = closest_approach
            if miss_distance_m <= threshold_m:
                events.append(
                    ConjunctionEvent(
                        debris_index=debris_index,
                        target_index=target_index,
                        tca_s=tca_s,
                        miss_distance_m=miss_distance_m,
                        relative_velocity_m_s=relative_velocity_m_s_magnitude,
                    )
                )
    return tuple(events)


__all__ = ["ConjunctionEvent", "screen_conjunctions_reference"]
