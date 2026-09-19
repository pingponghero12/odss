"""Deterministic scientific validation cases and machine-readable reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ._core import CartesianState, ParticlePopulation, PhysicalProperties, ReferenceFrame
from .cascade_screening import screen_conjunctions_cascade
from .catalog import Catalog, OmmRecord
from .conjunction import ConjunctionEvent, screen_conjunctions_reference
from .coordinates import bundled_iers_b, elapsed_time_s, epoch_from_iso, transform_state
from .flux import FluxSpec, evaluate_flux, evaluate_flux_radius_convergence
from .fragmentation import generate_explosion_fragments
from .maneuver import ManeuverDemandSpec, evaluate_maneuver_demand
from .orbital_observables import evaluate_impact_risk
from .rng import named_random_key
from .sgp4 import propagate_omm_sgp4, synchronize_catalog_sgp4
from .sso_dynamics import SsoForceModelSpec, SsoPropagationSpec, propagate_sso

ValidationStatus = Literal["pass", "fail", "skip"]


@dataclass(frozen=True, slots=True)
class ValidationMetric:
    """One numerical comparison with an explicit absolute tolerance."""

    name: str
    value: float
    reference: float
    absolute_tolerance: float
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must not be empty")
        for value, name in (
            (self.value, "value"),
            (self.reference, "reference"),
            (self.absolute_tolerance, "absolute_tolerance"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance must be non-negative")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must not be empty")

    @property
    def absolute_error(self) -> float:
        return abs(self.value - self.reference)

    @property
    def passed(self) -> bool:
        return self.absolute_error <= self.absolute_tolerance


@dataclass(frozen=True, slots=True)
class ValidationCase:
    """A named validation result containing numerical evidence or a skip reason."""

    name: str
    status: ValidationStatus
    metrics: tuple[ValidationMetric, ...]
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must not be empty")
        if self.status not in ("pass", "fail", "skip"):
            raise ValueError("status must be pass, fail, or skip")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if any(not isinstance(metric, ValidationMetric) for metric in self.metrics):
            raise TypeError("metrics must contain only ValidationMetric values")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("detail must not be empty")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """A deterministic collection of component and end-to-end validation cases."""

    cases: tuple[ValidationCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        if any(not isinstance(case, ValidationCase) for case in self.cases):
            raise TypeError("cases must contain only ValidationCase values")

    @property
    def passed(self) -> bool:
        return all(case.status != "fail" for case in self.cases)

    @property
    def skipped_case_names(self) -> tuple[str, ...]:
        return tuple(case.name for case in self.cases if case.status == "skip")


def _case(name: str, metrics: tuple[ValidationMetric, ...], detail: str) -> ValidationCase:
    status: ValidationStatus = "pass" if all(metric.passed for metric in metrics) else "fail"
    return ValidationCase(name=name, status=status, metrics=metrics, detail=detail)


def _skip(name: str, detail: str) -> ValidationCase:
    return ValidationCase(name=name, status="skip", metrics=(), detail=detail)


def _reference_omm() -> OmmRecord:
    return OmmRecord(
        catalog_id=25544,
        object_name="ISS (ZARYA)",
        international_designator="1998-067A",
        epoch=epoch_from_iso("2026-06-19T12:16:41.638656", "UTC"),
        mean_motion_rev_day=15.49315858,
        eccentricity=0.00045965,
        inclination_deg=51.6332,
        raan_deg=288.5889,
        argument_of_pericenter_deg=205.0015,
        mean_anomaly_deg=155.0751,
        ephemeris_type=0,
        classification="U",
        element_set_number=999,
        revolution_number=57211,
        bstar_1_earth_radii=0.0001560528,
        mean_motion_dot_rev_day2=0.00008252,
        mean_motion_ddot_rev_day3=0.0,
    )


def _sgp4_validation() -> ValidationCase:
    state = propagate_omm_sgp4(_reference_omm(), _reference_omm().epoch).state
    expected_position_m = (2_167_810.4309, -6_445_660.7017, -4.7725)
    expected_velocity_m_s = (4_501.4572680, 1_519.3295486, 6_005.6301219)
    position_error_m = max(
        abs(actual - expected)
        for actual, expected in zip(state.position_m, expected_position_m, strict=True)
    )
    velocity_error_m_s = max(
        abs(actual - expected)
        for actual, expected in zip(state.velocity_m_s, expected_velocity_m_s, strict=True)
    )
    return _case(
        "sgp4_reference_state",
        (
            ValidationMetric("maximum_position_error", position_error_m, 0.0, 0.1, "m"),
            ValidationMetric("maximum_velocity_error", velocity_error_m_s, 0.0, 1.0e-4, "m/s"),
        ),
        "Known TEME state at the OMM epoch using the independent sgp4 package.",
    )


def _frame_time_validation() -> ValidationCase:
    before = epoch_from_iso("2016-12-31T23:59:59", "UTC")
    after = epoch_from_iso("2017-01-01T00:00:00", "UTC")
    original = CartesianState(
        (-6_102_443.276428913, -986_332.0160861297, -2_820_313.0707199224),
        (-1_455.2527284474308, -5_527.413835655969, 5_101.042029427083),
        epoch_from_iso("2019-12-09T20:42:09.072", "UTC"),
        ReferenceFrame("TEME"),
    )
    gcrs = transform_state(original, ReferenceFrame("GCRS"), bundled_iers_b())
    round_trip = transform_state(gcrs, ReferenceFrame("TEME"), bundled_iers_b())
    return _case(
        "frame_and_time",
        (
            ValidationMetric(
                "leap_second_elapsed_time", elapsed_time_s(before, after), 2.0, 1.0e-11, "s"
            ),
            ValidationMetric(
                "round_trip_position_error",
                max(
                    abs(actual - expected)
                    for actual, expected in zip(
                        round_trip.position_m, original.position_m, strict=True
                    )
                ),
                0.0,
                1.0e-3,
                "m",
            ),
            ValidationMetric(
                "round_trip_velocity_error",
                max(
                    abs(actual - expected)
                    for actual, expected in zip(
                        round_trip.velocity_m_s, original.velocity_m_s, strict=True
                    )
                ),
                0.0,
                1.0e-6,
                "m/s",
            ),
        ),
        "Explicit UTC leap-second handling and TEME/GCRS round trip with bundled IERS-B data.",
    )


def _common_epoch_validation() -> ValidationCase:
    record = _reference_omm()
    target_epoch = epoch_from_iso("2026-06-20T00:00:00", "UTC")
    synchronized = synchronize_catalog_sgp4(Catalog((record,), ()), target_epoch).objects[0]
    scalar = propagate_omm_sgp4(record, target_epoch)
    return _case(
        "common_epoch",
        (
            ValidationMetric(
                "position_difference",
                max(
                    abs(first - second)
                    for first, second in zip(
                        synchronized.state.position_m, scalar.state.position_m, strict=True
                    )
                ),
                0.0,
                1.0e-6,
                "m",
            ),
            ValidationMetric(
                "velocity_difference",
                max(
                    abs(first - second)
                    for first, second in zip(
                        synchronized.state.velocity_m_s, scalar.state.velocity_m_s, strict=True
                    )
                ),
                0.0,
                1.0e-9,
                "m/s",
            ),
        ),
        "Batch synchronization agrees with scalar SGP4 propagation at one absolute epoch.",
    )


def _local_population(
    positions_m: tuple[tuple[float, float, float], ...],
    velocities_m_s: tuple[tuple[float, float, float], ...],
    *,
    frame: str = "GCRS",
) -> ParticlePopulation:
    positions = np.asarray(positions_m)
    velocities = np.asarray(velocities_m_s)
    count = len(positions)
    return ParticlePopulation(
        epoch=epoch_from_iso("2026-06-20T12:00:00", "TT"),
        frame=ReferenceFrame(frame),
        position_x_m=positions[:, 0],
        position_y_m=positions[:, 1],
        position_z_m=positions[:, 2],
        velocity_x_m_s=velocities[:, 0],
        velocity_y_m_s=velocities[:, 1],
        velocity_z_m_s=velocities[:, 2],
        mass_kg=np.ones(count),
        area_m2=np.ones(count),
    )


def _tca_validation() -> ValidationCase:
    debris = _local_population(((7_000_000.0 - 10.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),))
    targets = _local_population(((7_000_000.0 + 10.0, 3.0, 0.0),), ((-2.0, 0.0, 0.0),))
    event = screen_conjunctions_reference(
        debris, targets, duration_s=10.0, threshold_m=3.0
    )[0]
    return _case(
        "continuous_tca",
        (
            ValidationMetric("tca", event.tca_s, 5.0, 1.0e-12, "s"),
            ValidationMetric("miss_distance", event.miss_distance_m, 3.0, 1.0e-12, "m"),
            ValidationMetric(
                "relative_velocity", event.relative_velocity_m_s, 4.0, 1.0e-12, "m/s"
            ),
        ),
        "Analytical constant-velocity head-on encounter with a nonzero cross-track offset.",
    )


def _flux_validation() -> ValidationCase:
    debris = _local_population(
        ((7_000_000.0, 0.0, 0.0),) * 9,
        ((0.0, 7_500.0, 0.0),) * 9,
    )
    targets = _local_population(((7_000_000.0, 0.0, 0.0),), ((0.0, 7_500.0, 0.0),))
    distances_m = (0.5,) + (1.5,) * 3 + (2.5,) * 5
    events = tuple(
        ConjunctionEvent(index, 0, 5.0, distance, 10.0)
        for index, distance in enumerate(distances_m)
    )
    results = evaluate_flux_radius_convergence(
        events,
        debris,
        targets,
        (0.1,) * 9,
        sampling_radii_m=(1.0, 2.0, 3.0),
        time_bin_edges_s=(0.0, 10.0),
        size_bin_edges_m=(0.0, math.inf),
    )
    expected_flux = 1.0 / (10.0 * math.pi)
    return _case(
        "flux_and_radius_convergence",
        tuple(
            ValidationMetric(
                f"number_flux_radius_{index}",
                result.bins[0].number_flux_m2_s,
                expected_flux,
                1.0e-15,
                "m^-2 s^-1",
            )
            for index, result in enumerate(results)
        ),
        "Synthetic counts scale with sampling area, giving radius-independent analytical flux.",
    )


def _impact_risk_validation() -> ValidationCase:
    debris = _local_population(((7_000_000.0, 0.0, 0.0),), ((0.0, 7_500.0, 0.0),))
    targets = _local_population(((7_000_000.0, 0.0, 0.0),), ((0.0, 7_500.0, 0.0),))
    flux = evaluate_flux(
        (ConjunctionEvent(0, 0, 5.0, 1.0, 10.0),),
        debris,
        targets,
        (0.1,),
        FluxSpec(10.0, (0.0, 10.0), (0.0, math.inf)),
    )
    value = evaluate_impact_risk(flux, (2.0,)).values[0]
    expected_impacts = 2.0 / (math.pi * 10.0**2)
    return _case(
        "impact_risk",
        (
            ValidationMetric(
                "expected_impacts", value.expected_impacts, expected_impacts, 1.0e-15, "1"
            ),
            ValidationMetric(
                "model_probability",
                value.model_impact_probability,
                1.0 - math.exp(-expected_impacts),
                1.0e-15,
                "1",
            ),
        ),
        "Flux integration agrees with the rare independent-impact Poisson equation.",
    )


def _end_to_end_validation() -> ValidationCase:
    debris = _local_population(((0.0, 0.0, 0.0),), ((1.0, 0.0, 0.0),))
    targets = _local_population(((10.0, 2.0, 0.0),), ((-1.0, 0.0, 0.0),))

    def evaluate() -> tuple[int, float, int, float]:
        events = screen_conjunctions_reference(
            debris, targets, duration_s=10.0, threshold_m=5.0
        )
        flux = evaluate_flux(
            events,
            debris,
            targets,
            (0.2,),
            FluxSpec(5.0, (0.0, 10.0), (0.0, math.inf)),
        )
        demand = evaluate_maneuver_demand(
            events,
            debris,
            targets,
            (0.2,),
            ManeuverDemandSpec(0.1, 5.0, 10.0),
        )
        risk = evaluate_impact_risk(flux, (1.0,)).values[0]
        return (
            len(events),
            flux.bins[0].number_flux_m2_s,
            len(demand.actionable_encounters),
            risk.model_impact_probability,
        )

    first = evaluate()
    second = evaluate()
    return _case(
        "deterministic_end_to_end",
        tuple(
            ValidationMetric(f"output_{index}", actual, expected, 0.0, "1")
            for index, (actual, expected) in enumerate(zip(first, second, strict=True))
        ),
        "Screening, flux, maneuver demand, and impact risk reproduce exactly on rerun.",
    )


def _nasa_sbm_validation() -> ValidationCase:
    if importlib.util.find_spec("nasa_sbm") is None:
        return _skip("nasa_sbm_repeatability", "Optional nasa-sbm-py backend is not installed.")
    parent = CartesianState(
        (7_000_000.0, 0.0, 0.0),
        (0.0, 7_500.0, 0.0),
        epoch_from_iso("2026-06-20T12:00:00", "UTC"),
        ReferenceFrame("GCRS"),
    )
    key = named_random_key(
        master_seed=2026,
        scenario_id=1,
        run_id=2,
        object_id=0,
        stream_name="nasa_sbm_validation",
    )
    arguments = {
        "parent_state": parent,
        "parent_properties": PhysicalProperties(20.0, 4.0),
        "satellite_type": "spacecraft",
        "minimum_characteristic_length_m": 0.2,
        "random_key": key,
    }
    first = generate_explosion_fragments(**arguments)
    second = generate_explosion_fragments(**arguments)
    return _case(
        "nasa_sbm_repeatability",
        (
            ValidationMetric("exact_repeatability", float(first == second), 1.0, 0.0, "1"),
            ValidationMetric("positive_fragment_count", float(len(first) > 0), 1.0, 0.0, "1"),
        ),
        "Identical NASA SBM seed and inputs reproduce the complete fragment result.",
    )


def _cascade_validations() -> tuple[ValidationCase, ValidationCase]:
    if importlib.util.find_spec("cascade") is None or importlib.util.find_spec("heyoka") is None:
        reason = "Optional Cascade/heyoka backend is not installed."
        return (
            _skip("propagation_convergence", reason),
            _skip("cascade_vs_brute_force", reason),
        )
    radius_m = 7_078_000.0
    speed_m_s = math.sqrt(3.986_004_407_799_724e14 / radius_m)
    population = _local_population(
        ((radius_m, 0.0, 0.0),),
        ((0.0, speed_m_s, 0.0),),
        frame="EME2000",
    )
    force_model = SsoForceModelSpec(j2=False, drag=False)
    loose = propagate_sso(
        population,
        SsoPropagationSpec(600.0, 60.0, force_model, tolerance=1.0e-10),
    )
    tight = propagate_sso(
        population,
        SsoPropagationSpec(600.0, 60.0, force_model, tolerance=1.0e-12),
    )
    loose_position_m = np.column_stack(
        (loose.position_x_m, loose.position_y_m, loose.position_z_m)
    )
    tight_position_m = np.column_stack(
        (tight.position_x_m, tight.position_y_m, tight.position_z_m)
    )
    position_error_m = float(
        np.max(np.linalg.norm(loose_position_m - tight_position_m, axis=1), initial=0.0)
    )
    propagation = _case(
        "propagation_convergence",
        (ValidationMetric("final_position_difference", position_error_m, 0.0, 1.0, "m"),),
        "Point-mass Cascade propagation agrees under tighter integration tolerance.",
    )
    debris = _local_population(((radius_m - 10.0, 0.0, 0.0),), ((2.0, 0.0, 0.0),))
    targets = _local_population(((radius_m + 10.0, 3.0, 0.0),), ((-2.0, 0.0, 0.0),))
    reference = screen_conjunctions_reference(
        debris, targets, duration_s=10.0, threshold_m=3.0
    )[0]
    cascade = screen_conjunctions_cascade(
        debris,
        targets,
        duration_s=10.0,
        threshold_m=3.0,
        collisional_timestep_s=1.0,
    )[0]
    screening = _case(
        "cascade_vs_brute_force",
        (
            ValidationMetric("tca_difference", cascade.tca_s, reference.tca_s, 1.0e-8, "s"),
            ValidationMetric(
                "miss_distance_difference",
                cascade.miss_distance_m,
                reference.miss_distance_m,
                1.0e-6,
                "m",
            ),
            ValidationMetric(
                "relative_velocity_difference",
                cascade.relative_velocity_m_s,
                reference.relative_velocity_m_s,
                1.0e-10,
                "m/s",
            ),
        ),
        "Cascade continuous screening agrees with the analytical brute-force case.",
    )
    return propagation, screening


def run_scientific_validation(*, include_optional_backends: bool = False) -> ValidationReport:
    """Run deterministic validation and optionally exercise installed specialized backends."""
    cases = [
        _sgp4_validation(),
        _frame_time_validation(),
        _common_epoch_validation(),
        _tca_validation(),
        _flux_validation(),
        _impact_risk_validation(),
        _end_to_end_validation(),
    ]
    if include_optional_backends:
        cases.append(_nasa_sbm_validation())
        cases.extend(_cascade_validations())
    else:
        cases.extend(
            (
                _skip(
                    "nasa_sbm_repeatability",
                    "Optional backend validation was not requested.",
                ),
                _skip("propagation_convergence", "Optional backend validation was not requested."),
                _skip("cascade_vs_brute_force", "Optional backend validation was not requested."),
            )
        )
    return ValidationReport(tuple(cases))


def validation_report_json(report: ValidationReport) -> str:
    """Return a deterministic JSON report with numerical failure evidence."""
    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    representation = {
        "cases": [
            {
                "detail": case.detail,
                "metrics": [
                    {
                        **asdict(metric),
                        "absolute_error": metric.absolute_error,
                        "passed": metric.passed,
                    }
                    for metric in case.metrics
                ],
                "name": case.name,
                "status": case.status,
            }
            for case in report.cases
        ],
        "passed": report.passed,
        "schema": "odss.validation.v1",
    }
    return json.dumps(representation, allow_nan=False, indent=2, sort_keys=True) + "\n"


def write_validation_report(path: str | Path, report: ValidationReport) -> None:
    """Write a deterministic validation report without timestamps or random identifiers."""
    Path(path).write_text(validation_report_json(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-optional-backends", action="store_true")
    arguments = parser.parse_args()
    report = run_scientific_validation(
        include_optional_backends=arguments.include_optional_backends
    )
    rendered = validation_report_json(report)
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0 if report.passed else 1


__all__ = [
    "ValidationCase",
    "ValidationMetric",
    "ValidationReport",
    "run_scientific_validation",
    "validation_report_json",
    "write_validation_report",
]
