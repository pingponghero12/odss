"""Target-centered number, mass, and kinetic-energy flux observables."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._core import ParticlePopulation
from .conjunction import ConjunctionEvent


def _validated_edges(
    values: Sequence[float],
    name: str,
    *,
    require_zero_start: bool,
) -> tuple[float, ...]:
    edges = tuple(float(value) for value in values)
    if len(edges) < 2:
        raise ValueError(f"{name} must contain at least two edges")
    if require_zero_start and edges[0] != 0.0:
        raise ValueError(f"{name} must start at zero")
    if any(math.isnan(value) or value < 0.0 for value in edges):
        raise ValueError(f"{name} must contain non-negative values")
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    if not math.isfinite(edges[-1]) and edges[-1] != math.inf:
        raise ValueError(f"{name} has an invalid final edge")
    if any(not math.isfinite(value) for value in edges[:-1]):
        raise ValueError(f"only the final {name} edge may be infinite")
    return edges


@dataclass(frozen=True, slots=True)
class FluxSpec:
    """Sampling sphere and explicit SI time/size bin edges."""

    sampling_radius_m: float
    time_bin_edges_s: tuple[float, ...]
    size_bin_edges_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sampling_radius_m, (int, float)) or isinstance(
            self.sampling_radius_m, bool
        ):
            raise TypeError("sampling_radius_m must be a number")
        if not math.isfinite(self.sampling_radius_m) or self.sampling_radius_m <= 0.0:
            raise ValueError("sampling_radius_m must be positive and finite")
        object.__setattr__(
            self,
            "time_bin_edges_s",
            _validated_edges(self.time_bin_edges_s, "time_bin_edges_s", require_zero_start=True),
        )
        object.__setattr__(
            self,
            "size_bin_edges_m",
            _validated_edges(self.size_bin_edges_m, "size_bin_edges_m", require_zero_start=True),
        )
        if not math.isfinite(self.time_bin_edges_s[-1]):
            raise ValueError("the final time_bin_edges_s edge must be finite")


@dataclass(frozen=True, slots=True)
class FluxBin:
    """Flux values for one target, time interval, and fragment-size interval."""

    target_index: int
    time_start_s: float
    time_end_s: float
    size_min_m: float
    size_max_m: float
    encounter_count: int
    number_flux_m2_s: float
    mass_flux_kg_m2_s: float
    kinetic_energy_flux_w_m2: float


@dataclass(frozen=True, slots=True)
class FluxResult:
    """Representation-independent, target-wise binned flux output."""

    spec: FluxSpec
    target_count: int
    bins: tuple[FluxBin, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, FluxSpec):
            raise TypeError("spec must be a FluxSpec")
        if self.target_count < 0:
            raise ValueError("target_count must be non-negative")
        object.__setattr__(self, "bins", tuple(self.bins))


def _bin_index(value: float, edges: tuple[float, ...]) -> int | None:
    if value < edges[0] or value > edges[-1]:
        return None
    index = bisect.bisect_right(edges, value) - 1
    if index == len(edges) - 1:
        index -= 1
    return index


def evaluate_flux(
    events: Sequence[ConjunctionEvent],
    debris: ParticlePopulation,
    targets: ParticlePopulation,
    characteristic_length_m: Sequence[float],
    spec: FluxSpec,
) -> FluxResult:
    """Estimate flux from unique local encounters through target-centered sampling disks."""
    if not isinstance(debris, ParticlePopulation):
        raise TypeError("debris must be a ParticlePopulation")
    if not isinstance(targets, ParticlePopulation):
        raise TypeError("targets must be a ParticlePopulation")
    if debris.epoch != targets.epoch or debris.frame != targets.frame:
        raise ValueError("debris and targets must share one epoch and reference frame")
    if not isinstance(spec, FluxSpec):
        raise TypeError("spec must be a FluxSpec")
    sizes_m = tuple(float(value) for value in characteristic_length_m)
    if len(sizes_m) != len(debris):
        raise ValueError("characteristic_length_m must match debris population size")
    if any(not math.isfinite(value) or value <= 0.0 for value in sizes_m):
        raise ValueError("characteristic_length_m values must be positive and finite")

    time_bin_count = len(spec.time_bin_edges_s) - 1
    size_bin_count = len(spec.size_bin_edges_m) - 1
    counts = [0] * (len(targets) * time_bin_count * size_bin_count)
    masses_kg = [0.0] * len(counts)
    energies_j = [0.0] * len(counts)

    def flat_index(target_index: int, time_index: int, size_index: int) -> int:
        return (target_index * time_bin_count + time_index) * size_bin_count + size_index

    for event in events:
        if not isinstance(event, ConjunctionEvent):
            raise TypeError("events must contain only ConjunctionEvent values")
        if event.debris_index >= len(debris) or event.target_index >= len(targets):
            raise ValueError("event particle index is outside its population")
        if event.miss_distance_m > spec.sampling_radius_m:
            continue
        time_index = _bin_index(event.tca_s, spec.time_bin_edges_s)
        size_index = _bin_index(sizes_m[event.debris_index], spec.size_bin_edges_m)
        if time_index is None or size_index is None:
            continue
        index = flat_index(event.target_index, time_index, size_index)
        mass_kg = debris.mass_kg[event.debris_index]
        counts[index] += 1
        masses_kg[index] += mass_kg
        energies_j[index] += 0.5 * mass_kg * event.relative_velocity_m_s**2

    sampling_area_m2 = math.pi * spec.sampling_radius_m**2
    bins: list[FluxBin] = []
    for target_index in range(len(targets)):
        for time_index in range(time_bin_count):
            time_start_s = spec.time_bin_edges_s[time_index]
            time_end_s = spec.time_bin_edges_s[time_index + 1]
            denominator_m2_s = sampling_area_m2 * (time_end_s - time_start_s)
            for size_index in range(size_bin_count):
                index = flat_index(target_index, time_index, size_index)
                bins.append(
                    FluxBin(
                        target_index=target_index,
                        time_start_s=time_start_s,
                        time_end_s=time_end_s,
                        size_min_m=spec.size_bin_edges_m[size_index],
                        size_max_m=spec.size_bin_edges_m[size_index + 1],
                        encounter_count=counts[index],
                        number_flux_m2_s=counts[index] / denominator_m2_s,
                        mass_flux_kg_m2_s=masses_kg[index] / denominator_m2_s,
                        kinetic_energy_flux_w_m2=energies_j[index] / denominator_m2_s,
                    )
                )
    return FluxResult(spec=spec, target_count=len(targets), bins=tuple(bins))


def evaluate_flux_radius_convergence(
    events: Sequence[ConjunctionEvent],
    debris: ParticlePopulation,
    targets: ParticlePopulation,
    characteristic_length_m: Sequence[float],
    *,
    sampling_radii_m: Sequence[float],
    time_bin_edges_s: Sequence[float],
    size_bin_edges_m: Sequence[float],
) -> tuple[FluxResult, ...]:
    """Evaluate identical events at multiple sampling radii for convergence analysis."""
    radii_m = tuple(float(value) for value in sampling_radii_m)
    if not radii_m:
        raise ValueError("sampling_radii_m must not be empty")
    return tuple(
        evaluate_flux(
            events,
            debris,
            targets,
            characteristic_length_m,
            FluxSpec(
                sampling_radius_m=radius_m,
                time_bin_edges_s=tuple(time_bin_edges_s),
                size_bin_edges_m=tuple(size_bin_edges_m),
            ),
        )
        for radius_m in radii_m
    )


__all__ = [
    "FluxBin",
    "FluxResult",
    "FluxSpec",
    "evaluate_flux",
    "evaluate_flux_radius_convergence",
]
