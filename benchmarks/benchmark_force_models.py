"""Compare SSO force-model accuracy and runtime candidates."""

from __future__ import annotations

import argparse
import math

import odss


def population(count: int) -> odss.ParticlePopulation:
    radius_m = 7_078_000.0
    speed_m_s = math.sqrt(3.986_004_407_799_724e14 / radius_m)
    phases = [2.0 * math.pi * index / count for index in range(count)]
    return odss.ParticlePopulation(
        epoch=odss.epoch_from_iso("2026-06-20T12:00:00", "TT"),
        frame=odss.ReferenceFrame("EME2000"),
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
    parser.add_argument("--objects", type=int, default=1_000)
    parser.add_argument("--duration-s", type=float, default=86_400.0)
    arguments = parser.parse_args()
    if arguments.objects < 1 or arguments.duration_s <= 0.0:
        parser.error("--objects and --duration-s must be positive")

    point_mass = odss.SsoForceModelSpec(j2=False, drag=False)
    j2_drag = odss.SsoForceModelSpec(j2=True, drag=True)
    zonal_third_body = odss.SsoForceModelSpec(
        j2=True,
        j3=True,
        drag=True,
        sun=True,
        moon=True,
    )
    reference = odss.SsoForceModelSpec(
        j2=True,
        j3=True,
        c22_s22=True,
        drag=True,
        sun=True,
        moon=True,
        srp=True,
    )
    results = odss.evaluate_force_model_sensitivity(
        population(arguments.objects),
        duration_s=arguments.duration_s,
        collisional_timestep_s=60.0,
        candidates=(point_mass, j2_drag, zonal_third_body, reference),
        reference=reference,
    )
    for index, result in enumerate(results):
        print(
            f"candidate={index} wall_s={result.wall_duration_s:.6f} "
            f"max_position_m={result.maximum_position_difference_m:.6f} "
            f"rms_position_m={result.rms_position_difference_m:.6f} "
            f"max_velocity_m_s={result.maximum_velocity_difference_m_s:.9f}"
        )


if __name__ == "__main__":
    main()
