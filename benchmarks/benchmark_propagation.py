"""Measure representative SSO propagation throughput with Cascade."""

from __future__ import annotations

import argparse
import math
import time

import odss


def sso_population(count: int) -> odss.ParticlePopulation:
    radius_m = 7_078_000.0
    speed_m_s = math.sqrt(3.986_004_418e14 / radius_m)
    phases = [2.0 * math.pi * index / count for index in range(count)]
    return odss.ParticlePopulation(
        epoch=odss.epoch_from_iso("2026-06-20T12:00:00", "TAI"),
        frame=odss.ReferenceFrame("GCRS"),
        position_x_m=[radius_m * math.cos(phase) for phase in phases],
        position_y_m=[radius_m * math.sin(phase) for phase in phases],
        position_z_m=[0.0] * count,
        velocity_x_m_s=[-speed_m_s * math.sin(phase) for phase in phases],
        velocity_y_m_s=[speed_m_s * math.cos(phase) for phase in phases],
        velocity_z_m_s=[0.0] * count,
        mass_kg=[100.0] * count,
        area_m2=[1.0] * count,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=int, default=10_000)
    parser.add_argument("--duration-s", type=float, default=5_400.0)
    arguments = parser.parse_args()
    if arguments.objects < 1:
        parser.error("--objects must be positive")
    if arguments.duration_s <= 0.0:
        parser.error("--duration-s must be positive")

    population = sso_population(arguments.objects)
    spec = odss.CascadePropagationSpec(
        duration_s=arguments.duration_s,
        collisional_timestep_s=60.0,
    )
    start = time.perf_counter()
    result = odss.propagate(population, spec)
    elapsed_s = time.perf_counter() - start
    print(f"objects: {len(result)}")
    print(f"propagated_duration_s: {arguments.duration_s:.3f}")
    print(f"wall_duration_s: {elapsed_s:.6f}")
    throughput = len(result) * arguments.duration_s / elapsed_s
    print(f"object_simulated_seconds_per_wall_s: {throughput:.0f}")


if __name__ == "__main__":
    main()
