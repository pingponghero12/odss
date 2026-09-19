"""Cascade-backed continuous conjunction screening for local trajectory intervals."""

from __future__ import annotations

import importlib
import math

import numpy as np

from ._core import ParticlePopulation
from .conjunction import ConjunctionEvent, _require_screening_inputs


def _load_screening_backends() -> tuple[object, object]:
    try:
        cascade = importlib.import_module("cascade")
        heyoka = importlib.import_module("heyoka")
    except ImportError as error:
        raise ImportError(
            "Cascade screening requires the optional conda-forge 'cascade' package"
        ) from error
    return cascade, heyoka


def _constant_velocity_dynamics(heyoka: object) -> list[tuple[object, object]]:
    x, y, z, velocity_x, velocity_y, velocity_z = heyoka.make_vars(
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
    )
    zero = heyoka.expression(0.0)
    return [
        (x, velocity_x),
        (y, velocity_y),
        (z, velocity_z),
        (velocity_x, zero),
        (velocity_y, zero),
        (velocity_z, zero),
    ]


def _combined_state(
    debris: ParticlePopulation,
    targets: ParticlePopulation,
) -> np.ndarray:
    def field(name: str) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(getattr(debris, name), dtype=np.float64),
                np.asarray(getattr(targets, name), dtype=np.float64),
            )
        )

    radii_m = np.sqrt(field("area_m2") / math.pi)
    return np.column_stack(
        (
            field("position_x_m"),
            field("position_y_m"),
            field("position_z_m"),
            field("velocity_x_m_s"),
            field("velocity_y_m_s"),
            field("velocity_z_m_s"),
            radii_m,
        )
    )


def _event_from_cascade(
    event: object,
    debris_count: int,
) -> ConjunctionEvent | None:
    first_index = int(event["i"])
    second_index = int(event["j"])
    first_is_debris = first_index < debris_count
    second_is_debris = second_index < debris_count
    if first_is_debris == second_is_debris:
        return None
    if first_is_debris:
        debris_index = first_index
        target_index = second_index - debris_count
        debris_state = event["state_i"]
        target_state = event["state_j"]
    else:
        debris_index = second_index
        target_index = first_index - debris_count
        debris_state = event["state_j"]
        target_state = event["state_i"]
    relative_velocity_m_s = math.sqrt(
        sum(
            (float(debris_state[index]) - float(target_state[index])) ** 2
            for index in range(3, 6)
        )
    )
    return ConjunctionEvent(
        debris_index=debris_index,
        target_index=target_index,
        tca_s=float(event["time"]),
        miss_distance_m=float(event["dist"]),
        relative_velocity_m_s=relative_velocity_m_s,
    )


def screen_conjunctions_cascade(
    debris: ParticlePopulation,
    targets: ParticlePopulation,
    *,
    duration_s: float,
    threshold_m: float,
    collisional_timestep_s: float,
) -> tuple[ConjunctionEvent, ...]:
    """Screen debris-target local trajectories with Cascade's continuous event detector."""
    _require_screening_inputs(debris, targets, duration_s, threshold_m)
    if not isinstance(collisional_timestep_s, (int, float)) or isinstance(
        collisional_timestep_s, bool
    ):
        raise TypeError("collisional_timestep_s must be a number")
    if not math.isfinite(collisional_timestep_s) or collisional_timestep_s <= 0.0:
        raise ValueError("collisional_timestep_s must be positive and finite")
    if debris.empty or targets.empty:
        return ()

    cascade, heyoka = _load_screening_backends()
    debris_count = len(debris)
    if debris_count <= len(targets):
        conjunction_whitelist = set(range(debris_count))
    else:
        conjunction_whitelist = set(range(debris_count, debris_count + len(targets)))
    simulation = cascade.sim(
        _combined_state(debris, targets),
        collisional_timestep_s,
        dyn=_constant_velocity_dynamics(heyoka),
        conj_thresh=np.nextafter(float(threshold_m), math.inf),
        min_coll_radius=math.inf,
        conj_whitelist=conjunction_whitelist,
    )
    outcome = simulation.propagate_until(duration_s)
    if outcome != cascade.outcome.time_limit:
        raise RuntimeError(f"Cascade screening stopped before the interval ended: {outcome}")

    events = tuple(
        converted
        for event in simulation.conjunctions
        if (converted := _event_from_cascade(event, debris_count)) is not None
    )
    return tuple(
        sorted(
            events,
            key=lambda event: (event.tca_s, event.debris_index, event.target_index),
        )
    )


__all__ = ["screen_conjunctions_cascade"]
