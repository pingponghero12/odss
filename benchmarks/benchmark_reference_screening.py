"""Measure the brute-force local conjunction correctness baseline."""

from __future__ import annotations

import argparse
import time

import numpy as np

import odss


def population(count: int, seed: int) -> odss.ParticlePopulation:
    generator = np.random.default_rng(seed)
    positions_m = generator.uniform(-100_000.0, 100_000.0, size=(count, 3))
    velocities_m_s = generator.uniform(-8_000.0, 8_000.0, size=(count, 3))
    return odss.ParticlePopulation(
        epoch=odss.epoch_from_iso("2026-06-20T12:00:00", "TAI"),
        frame=odss.ReferenceFrame("GCRS"),
        position_x_m=positions_m[:, 0],
        position_y_m=positions_m[:, 1],
        position_z_m=positions_m[:, 2],
        velocity_x_m_s=velocities_m_s[:, 0],
        velocity_y_m_s=velocities_m_s[:, 1],
        velocity_z_m_s=velocities_m_s[:, 2],
        mass_kg=np.ones(count),
        area_m2=np.ones(count),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debris", type=int, default=200)
    parser.add_argument("--targets", type=int, default=200)
    arguments = parser.parse_args()
    if arguments.debris < 1 or arguments.targets < 1:
        parser.error("--debris and --targets must be positive")

    debris = population(arguments.debris, seed=1)
    targets = population(arguments.targets, seed=2)
    start = time.perf_counter()
    events = odss.screen_conjunctions_reference(
        debris,
        targets,
        duration_s=60.0,
        threshold_m=1_000.0,
    )
    duration_s = time.perf_counter() - start
    pairs = arguments.debris * arguments.targets
    print(f"pairs: {pairs}")
    print(f"events: {len(events)}")
    print(f"duration_s: {duration_s:.6f}")
    print(f"pairs_per_s: {pairs / duration_s:.0f}")


if __name__ == "__main__":
    main()
