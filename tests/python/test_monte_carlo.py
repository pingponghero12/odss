import math
import os

import pytest

import odss


def deterministic_run(run: odss.MonteCarloRun) -> tuple[int, int, str]:
    value = odss.random_u64(run.random_key("breakup"), 0)
    return run.run_id, value, os.environ.get("OMP_NUM_THREADS", "")


def scalar_run(run: odss.MonteCarloRun) -> float:
    offset = 10.0 if run.variant == "high_drag" else 0.0
    return float(run.run_id) + offset


def test_serial_and_parallel_execution_have_identical_run_rng() -> None:
    serial = odss.execute_monte_carlo(
        deterministic_run,
        odss.MonteCarloSpec(2026, 4, (8, 3, 5), max_workers=1, cpu_budget=2),
    )
    parallel = odss.execute_monte_carlo(
        deterministic_run,
        odss.MonteCarloSpec(2026, 4, (8, 3, 5), max_workers=2, cpu_budget=2),
    )

    assert [value.value[:2] for value in serial] == [value.value[:2] for value in parallel]
    assert [value.run.run_id for value in parallel] == [8, 3, 5]
    assert all(value.value[2] == "1" for value in parallel)


def test_paired_variants_reuse_rng_and_report_differences() -> None:
    spec = odss.MonteCarloSpec(
        7,
        9,
        (2, 4),
        variants=("nominal", "high_drag"),
        cpu_budget=1,
    )
    runs = spec.runs

    assert odss.random_u64(runs[0].random_key("breakup"), 0) == odss.random_u64(
        runs[2].random_key("breakup"), 0
    )
    outcomes = odss.execute_monte_carlo(scalar_run, spec)
    sensitivity = odss.paired_sensitivity(
        outcomes,
        baseline_variant="nominal",
        comparison_variant="high_drag",
    )

    assert [value.run_id for value in sensitivity.differences] == [2, 4]
    assert [value.difference for value in sensitivity.differences] == [10.0, 10.0]
    assert sensitivity.mean_difference == 10.0
    assert sensitivity.median_difference == 10.0


def test_cpu_allocation_rejects_nested_oversubscription() -> None:
    with pytest.raises(ValueError, match="CPU budget"):
        odss.MonteCarloSpec(
            1,
            1,
            (0,),
            max_workers=8,
            backend_threads_per_worker=4,
            cpu_budget=16,
        )


def test_wilson_interval_matches_reference_values_and_zero_events() -> None:
    interval = odss.wilson_interval(40, 100)

    assert interval.estimate == 0.4
    assert interval.lower == pytest.approx(0.3094012864)
    assert interval.upper == pytest.approx(0.4979974132)
    zero = odss.wilson_interval(0, 100)
    assert zero.lower == 0.0
    assert zero.upper > 0.0


def test_zero_event_bound_is_exact_and_approaches_rule_of_three() -> None:
    bound = odss.zero_event_upper_bound(1_000)

    assert (1.0 - bound) ** 1_000 == pytest.approx(0.05)
    assert bound == pytest.approx(3.0 / 1_000, rel=0.003)


def test_fixed_seed_bootstrap_is_reproducible() -> None:
    first = odss.bootstrap_summary((1.0, 2.0, 3.0, 10.0), resamples=200, master_seed=42)
    second = odss.bootstrap_summary((1.0, 2.0, 3.0, 10.0), resamples=200, master_seed=42)
    median = odss.bootstrap_summary(
        (1.0, 2.0, 3.0, 10.0), statistic="median", resamples=200, master_seed=42
    )

    assert first == second
    assert first.estimate == 4.0
    assert first.lower <= first.estimate <= first.upper
    assert median.estimate == 2.5


def test_pilot_sample_size_helpers_match_reference_equations() -> None:
    probability_runs = odss.required_probability_runs(0.5, 0.05)
    mean_runs = odss.required_mean_runs((8.0, 10.0, 12.0), 1.0)

    assert probability_runs == 385
    expected = math.ceil((1.959963984540054 * 2.0) ** 2)
    assert mean_runs == expected


def test_monte_carlo_and_statistical_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="unique"):
        odss.MonteCarloSpec(1, 1, (0, 0))
    with pytest.raises(ValueError, match="successes"):
        odss.wilson_interval(2, 1)
    with pytest.raises(ValueError, match="statistic"):
        odss.bootstrap_summary((1.0,), statistic="mode")
    with pytest.raises(ValueError, match="same non-empty"):
        odss.paired_sensitivity(
            (odss.MonteCarloOutcome(odss.MonteCarloRun(1, 1, 0, "a", 1), 1.0),),
            baseline_variant="a",
            comparison_variant="b",
        )
