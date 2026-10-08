"""Summarize the preliminary SSO Monte Carlo campaign with run-level uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import xarray as xr

import odss

_reported_metrics = (
    "fragment_count_ge_5_cm",
    "fragment_count_ge_10_cm",
    "retained_fraction",
    "removal_fraction",
    "removed_below_200_km_fraction",
    "conjunction_count_le_10000_m",
    "conjunction_count_le_1000_m",
    "maneuver_event_count_le_1000_m",
    "maneuver_affected_target_count_le_1000_m",
    "summed_target_integrated_number_flux_r_5000_m",
    "maximum_model_impact_probability_area_10_m2",
)
_binary_metrics = (
    "has_conjunction_le_1000_m",
    "has_maneuver_event_le_1000_m",
)


def _bootstrap_seed(family: str, metric: str) -> int:
    digest = hashlib.sha256(f"{family}/{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _continuous_summary(
    values: tuple[float, ...],
    *,
    family: str,
    metric: str,
) -> dict[str, float | int | None]:
    bootstrap = odss.bootstrap_summary(
        values,
        statistic="mean",
        confidence=0.95,
        resamples=10_000,
        master_seed=_bootstrap_seed(family, metric),
    )
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    required_runs: int | None = None
    if mean != 0.0 and len(values) > 1:
        required_runs = odss.required_mean_runs(values, 0.2 * abs(mean))
    return {
        "bootstrap_95_lower": bootstrap.lower,
        "bootstrap_95_upper": bootstrap.upper,
        "mean": mean,
        "median": statistics.median(values),
        "p05": _quantile(values, 0.05),
        "p95": _quantile(values, 0.95),
        "required_runs_for_20_percent_relative_mean_margin": required_runs,
        "runs": len(values),
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(len(values)),
    }


def _binary_summary(values: tuple[float, ...]) -> dict[str, float | int]:
    successes = sum(value != 0.0 for value in values)
    interval = odss.wilson_interval(successes, len(values))
    return {
        "runs": len(values),
        "successes": successes,
        "probability": interval.estimate,
        "wilson_95_lower": interval.lower,
        "wilson_95_upper": interval.upper,
    }


def _read_summary(path: Path) -> dict[str, float]:
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        names = tuple(str(value) for value in dataset["summary_name"].values)
        values = tuple(float(value) for value in dataset["summary_value"].values)
        summary = dict(zip(names, values, strict=True))
        summary["conjunction_count_le_10000_m"] = float(dataset.sizes.get("conjunction_event", 0))
        summary["removed_below_200_km_fraction"] = (
            summary["removal_fraction"] + summary["reentry_fraction"]
        )
    return summary


def analyze(directory: Path, *, expected_runs: int = 10) -> dict[str, object]:
    """Return deterministic run-level summaries for a complete campaign directory."""
    files = tuple(sorted(directory.glob("*/*/run_*.nc")))
    if not files:
        raise FileNotFoundError(f"no run files found below {directory}")
    grouped: dict[str, list[tuple[int, dict[str, float]]]] = defaultdict(list)
    for path in files:
        family = f"{path.parent.parent.name}/{path.parent.name}"
        run_id = int(path.stem.removeprefix("run_"))
        grouped[family].append((run_id, _read_summary(path)))
    incomplete = {
        family: len(runs) for family, runs in grouped.items() if len(runs) != expected_runs
    }
    if incomplete:
        raise ValueError(f"campaign families do not contain {expected_runs} runs: {incomplete}")

    families: dict[str, object] = {}
    for family, runs in sorted(grouped.items()):
        ordered = tuple(value for _, value in sorted(runs))
        families[family] = {
            "continuous": {
                metric: _continuous_summary(
                    tuple(run[metric] for run in ordered),
                    family=family,
                    metric=metric,
                )
                for metric in _reported_metrics
            },
            "binary": {
                metric: _binary_summary(tuple(run[metric] for run in ordered))
                for metric in _binary_metrics
            },
        }

    return {
        "families": families,
        "family_count": len(grouped),
        "files": len(files),
        "schema": "odss.iac_2026_sso.preliminary_analysis.v1",
    }


def main(arguments: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("results/iac_2026_sso/preliminary"),
    )
    parser.add_argument("--expected-runs", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parsed = parser.parse_args(arguments)
    report = analyze(parsed.directory, expected_runs=parsed.expected_runs)
    output = parsed.output or parsed.directory / "analysis.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
