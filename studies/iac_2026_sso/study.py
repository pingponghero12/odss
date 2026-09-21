"""Frozen scientific configuration for the SSO fragmentation campaign."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import odss

_day_s = 86_400.0
_sun_mean_rate_deg_day = 360.0 / 365.242_189_7

MASTER_SEED = 20_260_921
SSO_PRECESSION_TOLERANCE_DEG_DAY = 0.05 * _sun_mean_rate_deg_day
GENERATED_MINIMUM_SIZE_M = 0.01
ANALYSIS_SIZE_THRESHOLDS_M = (0.01, 0.05, 0.10)
FLUX_SIZE_BIN_EDGES_M = (0.0, 0.01, 0.05, 0.10, math.inf)
FLUX_RADII_M = (1_000.0, 2_000.0, 5_000.0, 10_000.0)
MANEUVER_DISTANCE_THRESHOLDS_M = (500.0, 1_000.0, 2_000.0, 5_000.0)
TRACKABILITY_SIZE_THRESHOLD_M = 0.10
REFERENCE_AREAS_M2 = (1.0, 5.0, 10.0, 20.0)

CATALOG_SNAPSHOT_PATH = Path("results/iac_2026_sso/input/catalog_snapshot.json")


@dataclass(frozen=True, slots=True)
class Scenario:
    """One controlled breakup altitude and mechanism."""

    scenario_id: int
    altitude_m: float
    inclination_deg: float
    breakup_mode: Literal["explosion", "collision"]
    primary_mass_kg: float = 1_000.0
    primary_area_m2: float = 10.0
    secondary_mass_kg: float = 100.0
    secondary_area_m2: float = 1.0
    collision_relative_velocity_m_s: float = 10_000.0

    @property
    def name(self) -> str:
        altitude_km = round(self.altitude_m / 1_000.0)
        return f"{altitude_km:03d}_km_{self.breakup_mode}"


@dataclass(frozen=True, slots=True)
class ModelVariant:
    """One paired propagation-model choice."""

    name: str
    force_model: odss.SsoForceModelSpec


@dataclass(frozen=True, slots=True)
class Campaign:
    """Runtime and output limits for one launch script."""

    name: str
    duration_s: float
    time_bin_edges_s: tuple[float, ...]
    screening_threshold_m: float
    stored_flux_radius_m: float
    default_runs_per_family: int
    maximum_runs_per_family: int
    maximum_workers: int
    wall_time_budget_s: float


SCENARIOS = (
    Scenario(0, 500_000.0, 97.401_806_730_4, "explosion"),
    Scenario(1, 500_000.0, 97.401_806_730_4, "collision"),
    Scenario(2, 700_000.0, 98.187_980_822_1, "explosion"),
    Scenario(3, 700_000.0, 98.187_980_822_1, "collision"),
    Scenario(4, 800_000.0, 98.603_109_604_9, "explosion"),
    Scenario(5, 800_000.0, 98.603_109_604_9, "collision"),
)

NOMINAL_VARIANT = ModelVariant(
    "nominal",
    odss.SsoForceModelSpec(j2=True, drag=True, drag_coefficient=2.2),
)
SENSITIVITY_VARIANTS = (
    ModelVariant(
        "enhanced_forces",
        odss.SsoForceModelSpec(
            j2=True,
            j3=True,
            c22_s22=True,
            drag=True,
            sun=True,
            moon=True,
            drag_coefficient=2.2,
        ),
    ),
    ModelVariant(
        "drag_coefficient_2_0",
        odss.SsoForceModelSpec(j2=True, drag=True, drag_coefficient=2.0),
    ),
    ModelVariant(
        "drag_coefficient_2_4",
        odss.SsoForceModelSpec(j2=True, drag=True, drag_coefficient=2.4),
    ),
)

PRELIMINARY_CAMPAIGN = Campaign(
    name="preliminary",
    duration_s=7.0 * _day_s,
    time_bin_edges_s=(
        0.0,
        6.0 * 3_600.0,
        1.0 * _day_s,
        3.0 * _day_s,
        7.0 * _day_s,
    ),
    screening_threshold_m=10_000.0,
    stored_flux_radius_m=5_000.0,
    default_runs_per_family=2,
    maximum_runs_per_family=4,
    maximum_workers=32,
    wall_time_budget_s=3_600.0,
)

SMOKE_CAMPAIGN = Campaign(
    name="smoke",
    duration_s=600.0,
    time_bin_edges_s=(0.0, 600.0),
    screening_threshold_m=10_000.0,
    stored_flux_radius_m=5_000.0,
    default_runs_per_family=1,
    maximum_runs_per_family=1,
    maximum_workers=1,
    wall_time_budget_s=600.0,
)

PRODUCTION_CAMPAIGN = Campaign(
    name="production",
    duration_s=365.25 * _day_s,
    time_bin_edges_s=(
        0.0,
        6.0 * 3_600.0,
        1.0 * _day_s,
        3.0 * _day_s,
        7.0 * _day_s,
        14.0 * _day_s,
        30.0 * _day_s,
        90.0 * _day_s,
        180.0 * _day_s,
        365.25 * _day_s,
    ),
    screening_threshold_m=10_000.0,
    stored_flux_radius_m=5_000.0,
    default_runs_per_family=16,
    maximum_runs_per_family=64,
    maximum_workers=32,
    wall_time_budget_s=48.0 * 3_600.0,
)

COLLISIONAL_TIMESTEP_S = 120.0
COLLISIONAL_STEPS_PER_BATCH = 120
INTEGRATOR_TOLERANCE = 1.0e-12
TARGET_PROXY_MASS_KG = 1_000.0
TARGET_PROXY_AREA_M2 = 10.0


def variants_for(scenario: Scenario) -> tuple[ModelVariant, ...]:
    """Return nominal physics everywhere and paired sensitivities at 700 km."""
    if scenario.altitude_m == 700_000.0:
        return (NOMINAL_VARIANT, *SENSITIVITY_VARIANTS)
    return (NOMINAL_VARIANT,)


def family_count() -> int:
    """Return the number of independently scheduled scenario/model families."""
    return sum(len(variants_for(scenario)) for scenario in SCENARIOS)


__all__ = [
    "ANALYSIS_SIZE_THRESHOLDS_M",
    "CATALOG_SNAPSHOT_PATH",
    "COLLISIONAL_TIMESTEP_S",
    "COLLISIONAL_STEPS_PER_BATCH",
    "Campaign",
    "FLUX_RADII_M",
    "FLUX_SIZE_BIN_EDGES_M",
    "GENERATED_MINIMUM_SIZE_M",
    "INTEGRATOR_TOLERANCE",
    "MANEUVER_DISTANCE_THRESHOLDS_M",
    "MASTER_SEED",
    "ModelVariant",
    "NOMINAL_VARIANT",
    "PRELIMINARY_CAMPAIGN",
    "PRODUCTION_CAMPAIGN",
    "REFERENCE_AREAS_M2",
    "SCENARIOS",
    "SMOKE_CAMPAIGN",
    "SSO_PRECESSION_TOLERANCE_DEG_DAY",
    "Scenario",
    "TARGET_PROXY_AREA_M2",
    "TARGET_PROXY_MASS_KG",
    "TRACKABILITY_SIZE_THRESHOLD_M",
    "family_count",
    "variants_for",
]
