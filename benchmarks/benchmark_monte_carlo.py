"""Measure independent-run executor scaling without nested backend threads."""

from __future__ import annotations

import argparse
import time

import odss


def cpu_work(run: odss.MonteCarloRun) -> int:
    value = odss.random_u64(run.random_key("benchmark"), 0)
    for _ in range(250_000):
        value = (value * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & (
            (1 << 64) - 1
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 2, 4))
    arguments = parser.parse_args()
    cpu_budget = max(arguments.workers)
    baseline_s: float | None = None
    for workers in arguments.workers:
        start = time.perf_counter()
        outcomes = odss.execute_monte_carlo(
            cpu_work,
            odss.MonteCarloSpec(
                master_seed=2026,
                scenario_id=0,
                run_ids=tuple(range(arguments.runs)),
                max_workers=workers,
                backend_threads_per_worker=1,
                cpu_budget=cpu_budget,
            ),
        )
        wall_s = time.perf_counter() - start
        baseline_s = wall_s if baseline_s is None else baseline_s
        print(
            f"workers={workers} runs={len(outcomes)} wall_s={wall_s:.6f} "
            f"runs_s={len(outcomes) / wall_s:.3f} speedup={baseline_s / wall_s:.3f}"
        )


if __name__ == "__main__":
    main()
