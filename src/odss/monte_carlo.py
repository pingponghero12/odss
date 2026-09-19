"""Deterministic Monte Carlo execution and run-level statistical summaries."""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from statistics import NormalDist
from typing import Generic, TypeVar

from ._core import RandomKey, uniform_01
from .rng import _require_uint64, named_random_key

ResultType = TypeVar("ResultType")


def _require_probability(value: float, name: str, *, open_interval: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite probability")
    valid = 0.0 < value < 1.0 if open_interval else 0.0 <= value <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")


@dataclass(frozen=True, slots=True)
class MonteCarloRun:
    """One independently reproducible realization of one study variant."""

    master_seed: int
    scenario_id: int
    run_id: int
    variant: str
    backend_threads: int

    def __post_init__(self) -> None:
        _require_uint64(self.master_seed, "master_seed")
        _require_uint64(self.scenario_id, "scenario_id")
        _require_uint64(self.run_id, "run_id")
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("variant must not be empty")
        if isinstance(self.backend_threads, bool) or not isinstance(self.backend_threads, int):
            raise ValueError("backend_threads must be a positive integer")
        if self.backend_threads <= 0:
            raise ValueError("backend_threads must be a positive integer")

    def random_key(self, stream_name: str, *, object_id: int = 0) -> RandomKey:
        """Return a schedule-independent key shared by paired variants."""
        return named_random_key(
            master_seed=self.master_seed,
            scenario_id=self.scenario_id,
            run_id=self.run_id,
            object_id=object_id,
            stream_name=stream_name,
        )


@dataclass(frozen=True, slots=True)
class MonteCarloSpec:
    """Run identities and an explicit, non-oversubscribed CPU allocation."""

    master_seed: int
    scenario_id: int
    run_ids: tuple[int, ...]
    variants: tuple[str, ...] = ("nominal",)
    max_workers: int = 1
    backend_threads_per_worker: int = 1
    cpu_budget: int | None = None

    def __post_init__(self) -> None:
        _require_uint64(self.master_seed, "master_seed")
        _require_uint64(self.scenario_id, "scenario_id")
        object.__setattr__(self, "run_ids", tuple(self.run_ids))
        object.__setattr__(self, "variants", tuple(self.variants))
        if not self.run_ids:
            raise ValueError("run_ids must not be empty")
        for run_id in self.run_ids:
            _require_uint64(run_id, "run_id")
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("run_ids must be unique")
        if not self.variants or any(
            not isinstance(variant, str) or not variant.strip() for variant in self.variants
        ):
            raise ValueError("variants must contain non-empty names")
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must be unique")
        for value, name in (
            (self.max_workers, "max_workers"),
            (self.backend_threads_per_worker, "backend_threads_per_worker"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.cpu_budget is not None and (
            isinstance(self.cpu_budget, bool)
            or not isinstance(self.cpu_budget, int)
            or self.cpu_budget <= 0
        ):
            raise ValueError("cpu_budget must be a positive integer or None")
        cpu_budget = self.cpu_budget or os.cpu_count() or 1
        if self.max_workers * self.backend_threads_per_worker > cpu_budget:
            raise ValueError(
                "max_workers * backend_threads_per_worker must not exceed the CPU budget"
            )

    @property
    def runs(self) -> tuple[MonteCarloRun, ...]:
        """Return stable variant-major execution addresses."""
        return tuple(
            MonteCarloRun(
                master_seed=self.master_seed,
                scenario_id=self.scenario_id,
                run_id=run_id,
                variant=variant,
                backend_threads=self.backend_threads_per_worker,
            )
            for variant in self.variants
            for run_id in self.run_ids
        )


@dataclass(frozen=True, slots=True)
class MonteCarloOutcome(Generic[ResultType]):
    """Value returned for one explicit Monte Carlo run address."""

    run: MonteCarloRun
    value: ResultType


def _set_backend_thread_limits(thread_count: int) -> None:
    value = str(thread_count)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = value


def _execute_one(
    arguments: tuple[Callable[[MonteCarloRun], ResultType], MonteCarloRun],
) -> MonteCarloOutcome[ResultType]:
    function, run = arguments
    return MonteCarloOutcome(run=run, value=function(run))


def execute_monte_carlo(
    function: Callable[[MonteCarloRun], ResultType],
    spec: MonteCarloSpec,
) -> tuple[MonteCarloOutcome[ResultType], ...]:
    """Execute independent runs serially or in non-oversubscribed worker processes."""
    if not callable(function):
        raise TypeError("function must be callable")
    if not isinstance(spec, MonteCarloSpec):
        raise TypeError("spec must be a MonteCarloSpec")
    arguments = tuple((function, run) for run in spec.runs)
    if spec.max_workers == 1:
        return tuple(_execute_one(argument) for argument in arguments)
    with ProcessPoolExecutor(
        max_workers=spec.max_workers,
        initializer=_set_backend_thread_limits,
        initargs=(spec.backend_threads_per_worker,),
    ) as executor:
        return tuple(executor.map(_execute_one, arguments))


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A two-sided confidence interval for one scalar estimate."""

    estimate: float
    lower: float
    upper: float
    confidence: float


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Return the Wilson score interval for a run-level Bernoulli probability."""
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    _require_probability(confidence, "confidence", open_interval=True)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    estimate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (estimate + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(estimate * (1.0 - estimate) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return ConfidenceInterval(
        estimate,
        max(0.0, center - half_width),
        min(1.0, center + half_width),
        confidence,
    )


def zero_event_upper_bound(trials: int, *, confidence: float = 0.95) -> float:
    """Return the exact one-sided binomial upper bound after zero observed events."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    _require_probability(confidence, "confidence", open_interval=True)
    return -math.expm1(math.log1p(-confidence) / trials)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    """A fixed-seed percentile bootstrap summary of a run-level statistic."""

    statistic: str
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int


def bootstrap_summary(
    values: Sequence[float],
    *,
    statistic: str = "mean",
    confidence: float = 0.95,
    resamples: int = 2_000,
    master_seed: int = 0,
) -> BootstrapSummary:
    """Summarize a mean or median with deterministic counter-based resampling."""
    samples = tuple(float(value) for value in values)
    if not samples or any(not math.isfinite(value) for value in samples):
        raise ValueError("values must contain finite numbers")
    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be mean or median")
    _require_probability(confidence, "confidence", open_interval=True)
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    _require_uint64(master_seed, "master_seed")
    calculate = statistics.fmean if statistic == "mean" else statistics.median
    key = named_random_key(
        master_seed=master_seed,
        scenario_id=0,
        run_id=0,
        object_id=0,
        stream_name=f"bootstrap_{statistic}",
    )
    sample_count = len(samples)
    estimates = []
    for resample_index in range(resamples):
        resampled = tuple(
            samples[
                min(
                    int(
                        uniform_01(key, resample_index * sample_count + sample_index)
                        * sample_count
                    ),
                    sample_count - 1,
                )
            ]
            for sample_index in range(sample_count)
        )
        estimates.append(float(calculate(resampled)))
    tail = (1.0 - confidence) / 2.0
    return BootstrapSummary(
        statistic=statistic,
        estimate=float(calculate(samples)),
        lower=_quantile(estimates, tail),
        upper=_quantile(estimates, 1.0 - tail),
        confidence=confidence,
        resamples=resamples,
    )


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """One comparison-minus-baseline difference using a shared realization."""

    run_id: int
    baseline: float
    comparison: float
    difference: float


@dataclass(frozen=True, slots=True)
class PairedSensitivity:
    """Run-level paired differences for one study sensitivity."""

    baseline_variant: str
    comparison_variant: str
    differences: tuple[PairedDifference, ...]
    mean_difference: float
    median_difference: float


def paired_sensitivity(
    outcomes: Sequence[MonteCarloOutcome[float]],
    *,
    baseline_variant: str,
    comparison_variant: str,
) -> PairedSensitivity:
    """Compare variants only where master seed, scenario and run ID match."""
    if baseline_variant == comparison_variant:
        raise ValueError("baseline_variant and comparison_variant must differ")
    selected: dict[tuple[str, int], MonteCarloOutcome[float]] = {}
    identity: tuple[int, int] | None = None
    for outcome in outcomes:
        if not isinstance(outcome, MonteCarloOutcome):
            raise TypeError("outcomes must contain only MonteCarloOutcome values")
        if not isinstance(outcome.value, (int, float)) or isinstance(outcome.value, bool):
            raise TypeError("paired outcome values must be numeric")
        if not math.isfinite(float(outcome.value)):
            raise ValueError("paired outcome values must be finite")
        current_identity = (outcome.run.master_seed, outcome.run.scenario_id)
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError("paired outcomes must share one master seed and scenario")
        key = (outcome.run.variant, outcome.run.run_id)
        if key in selected:
            raise ValueError("paired outcomes must have unique variant and run IDs")
        selected[key] = outcome
    baseline_ids = {
        run_id for variant, run_id in selected if variant == baseline_variant
    }
    comparison_ids = {
        run_id for variant, run_id in selected if variant == comparison_variant
    }
    if not baseline_ids or baseline_ids != comparison_ids:
        raise ValueError("paired variants must contain the same non-empty run ID set")
    differences = tuple(
        PairedDifference(
            run_id=run_id,
            baseline=float(selected[(baseline_variant, run_id)].value),
            comparison=float(selected[(comparison_variant, run_id)].value),
            difference=float(selected[(comparison_variant, run_id)].value)
            - float(selected[(baseline_variant, run_id)].value),
        )
        for run_id in sorted(baseline_ids)
    )
    values = tuple(value.difference for value in differences)
    return PairedSensitivity(
        baseline_variant=baseline_variant,
        comparison_variant=comparison_variant,
        differences=differences,
        mean_difference=statistics.fmean(values),
        median_difference=statistics.median(values),
    )


def required_probability_runs(
    pilot_probability: float,
    absolute_margin: float,
    *,
    confidence: float = 0.95,
) -> int:
    """Estimate Bernoulli runs needed for an absolute normal-approximation margin."""
    _require_probability(pilot_probability, "pilot_probability")
    if not math.isfinite(absolute_margin) or absolute_margin <= 0.0:
        raise ValueError("absolute_margin must be positive and finite")
    _require_probability(confidence, "confidence", open_interval=True)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    variance = pilot_probability * (1.0 - pilot_probability)
    return max(1, math.ceil(z**2 * variance / absolute_margin**2))


def required_mean_runs(
    pilot_values: Sequence[float],
    absolute_margin: float,
    *,
    confidence: float = 0.95,
) -> int:
    """Estimate runs needed for a mean from a pilot sample standard deviation."""
    values = tuple(float(value) for value in pilot_values)
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("pilot_values must contain at least two finite values")
    if not math.isfinite(absolute_margin) or absolute_margin <= 0.0:
        raise ValueError("absolute_margin must be positive and finite")
    _require_probability(confidence, "confidence", open_interval=True)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return max(1, math.ceil((z * statistics.stdev(values) / absolute_margin) ** 2))


__all__ = [
    "BootstrapSummary",
    "ConfidenceInterval",
    "MonteCarloOutcome",
    "MonteCarloRun",
    "MonteCarloSpec",
    "PairedDifference",
    "PairedSensitivity",
    "bootstrap_summary",
    "execute_monte_carlo",
    "paired_sensitivity",
    "required_mean_runs",
    "required_probability_runs",
    "wilson_interval",
    "zero_event_upper_bound",
]
