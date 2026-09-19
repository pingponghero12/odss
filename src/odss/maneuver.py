"""Simple geometric maneuver-demand proxy for trackable debris encounters."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from ._core import ParticlePopulation
from .conjunction import ConjunctionEvent


@dataclass(frozen=True, slots=True)
class ManeuverDemandSpec:
    """Trackability, geometric action, time-horizon, and deduplication assumptions."""

    trackability_size_threshold_m: float
    miss_distance_threshold_m: float
    duration_s: float
    deduplication_tolerance_s: float = 1.0e-6

    def __post_init__(self) -> None:
        for value, name, allow_zero in (
            (self.trackability_size_threshold_m, "trackability_size_threshold_m", False),
            (self.miss_distance_threshold_m, "miss_distance_threshold_m", False),
            (self.duration_s, "duration_s", False),
            (self.deduplication_tolerance_s, "deduplication_tolerance_s", True),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {qualifier} and finite")


@dataclass(frozen=True, slots=True)
class ManeuverDemandResult:
    """Gross geometric action demand without maneuver design or repropagation."""

    spec: ManeuverDemandSpec
    actionable_encounters: tuple[ConjunctionEvent, ...]
    affected_target_indices: tuple[int, ...]
    events_per_target: tuple[int, ...]
    event_rate_s: float
    probability_at_least_one_actionable_encounter: float

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ManeuverDemandSpec):
            raise TypeError("spec must be a ManeuverDemandSpec")
        object.__setattr__(self, "actionable_encounters", tuple(self.actionable_encounters))
        object.__setattr__(self, "affected_target_indices", tuple(self.affected_target_indices))
        object.__setattr__(self, "events_per_target", tuple(self.events_per_target))


def _deduplicate(
    events: Sequence[ConjunctionEvent],
    tolerance_s: float,
) -> tuple[ConjunctionEvent, ...]:
    by_pair: dict[tuple[int, int], list[ConjunctionEvent]] = defaultdict(list)
    for event in events:
        by_pair[(event.debris_index, event.target_index)].append(event)

    retained: list[ConjunctionEvent] = []
    for pair_events in by_pair.values():
        ordered = sorted(pair_events, key=lambda event: (event.tca_s, event.miss_distance_m))
        cluster_start_s = ordered[0].tca_s
        best = ordered[0]
        for event in ordered[1:]:
            if event.tca_s - cluster_start_s <= tolerance_s:
                if (event.miss_distance_m, event.tca_s) < (
                    best.miss_distance_m,
                    best.tca_s,
                ):
                    best = event
            else:
                retained.append(best)
                cluster_start_s = event.tca_s
                best = event
        retained.append(best)
    return tuple(
        sorted(
            retained,
            key=lambda event: (event.tca_s, event.target_index, event.debris_index),
        )
    )


def evaluate_maneuver_demand(
    events: Sequence[ConjunctionEvent],
    debris: ParticlePopulation,
    targets: ParticlePopulation,
    characteristic_length_m: Sequence[float],
    spec: ManeuverDemandSpec,
) -> ManeuverDemandResult:
    """Evaluate gross action demand from trackability and geometric miss-distance thresholds."""
    if not isinstance(debris, ParticlePopulation):
        raise TypeError("debris must be a ParticlePopulation")
    if not isinstance(targets, ParticlePopulation):
        raise TypeError("targets must be a ParticlePopulation")
    if debris.epoch != targets.epoch or debris.frame != targets.frame:
        raise ValueError("debris and targets must share one epoch and reference frame")
    if not isinstance(spec, ManeuverDemandSpec):
        raise TypeError("spec must be a ManeuverDemandSpec")
    sizes_m = tuple(float(value) for value in characteristic_length_m)
    if len(sizes_m) != len(debris):
        raise ValueError("characteristic_length_m must match debris population size")
    if any(not math.isfinite(value) or value <= 0.0 for value in sizes_m):
        raise ValueError("characteristic_length_m values must be positive and finite")

    candidates: list[ConjunctionEvent] = []
    for event in events:
        if not isinstance(event, ConjunctionEvent):
            raise TypeError("events must contain only ConjunctionEvent values")
        if event.debris_index >= len(debris) or event.target_index >= len(targets):
            raise ValueError("event particle index is outside its population")
        if event.tca_s > spec.duration_s:
            continue
        if (
            sizes_m[event.debris_index] >= spec.trackability_size_threshold_m
            and event.miss_distance_m <= spec.miss_distance_threshold_m
        ):
            candidates.append(event)

    actionable = _deduplicate(candidates, spec.deduplication_tolerance_s)
    events_per_target = [0] * len(targets)
    for event in actionable:
        events_per_target[event.target_index] += 1
    affected_targets = tuple(
        index for index, event_count in enumerate(events_per_target) if event_count > 0
    )
    event_rate_s = len(actionable) / spec.duration_s
    probability = -math.expm1(-event_rate_s * spec.duration_s)
    return ManeuverDemandResult(
        spec=spec,
        actionable_encounters=actionable,
        affected_target_indices=affected_targets,
        events_per_target=tuple(events_per_target),
        event_rate_s=event_rate_s,
        probability_at_least_one_actionable_encounter=probability,
    )


__all__ = ["ManeuverDemandResult", "ManeuverDemandSpec", "evaluate_maneuver_demand"]
