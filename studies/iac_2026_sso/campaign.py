"""Execution orchestration for the short and production SSO campaigns."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import statistics
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import odss

from . import study

_earth_gravitational_parameter_m3_s2 = 3.986_004_418e14
_earth_radius_m = 6_378_136.3
_prepared_inputs: PreparedInputs | None = None


@dataclass(frozen=True, slots=True)
class ExecutionInputs:
    """Resolved external inputs and execution controls."""

    catalog_path: Path
    catalog_source_uri: str
    catalog_acquired_at_utc: str
    common_epoch_utc: str
    code_version: str
    output_directory: Path
    workers: int
    resume: bool


@dataclass(frozen=True, slots=True)
class PreparedInputs:
    """Catalog targets and immutable provenance shared by local worker processes."""

    targets: odss.ParticlePopulation
    assets: tuple[odss.InputAssetMetadata, ...]
    common_epoch: odss.Epoch
    catalog_record_count: int
    selected_target_count: int
    target_catalog_ids: tuple[int, ...]
    study_hash: str
    code_version: str


@dataclass(frozen=True, slots=True)
class StudyTask:
    """One deterministic campaign realization."""

    campaign: study.Campaign
    scenario: study.Scenario
    variant: study.ModelVariant
    run_id: int
    output_path: Path


@dataclass(frozen=True, slots=True)
class TaskReport:
    """Small scheduling record returned without retaining scientific arrays."""

    scenario_id: int
    variant: str
    run_id: int
    wall_duration_s: float
    fragment_count: int
    conjunction_count: int
    output_path: str


def _canonical_value(value: object) -> object:
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return "+infinity" if value > 0.0 else "-infinity"
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_configuration(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _asset_for_file(path: Path, logical_name: str) -> odss.InputAssetMetadata:
    content = path.read_bytes()
    return odss.InputAssetMetadata(
        logical_name=logical_name,
        content_sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def _require_resolved_inputs(inputs: ExecutionInputs) -> None:
    unresolved = tuple(
        name
        for name, value in (
            ("catalog", str(inputs.catalog_path)),
            ("catalog source URI", inputs.catalog_source_uri),
            ("catalog acquisition epoch", inputs.catalog_acquired_at_utc),
            ("common simulation epoch", inputs.common_epoch_utc),
            ("code version", inputs.code_version),
        )
        if value == study.XX
    )
    if unresolved:
        raise ValueError("replace XX for: " + ", ".join(unresolved))
    if not inputs.catalog_path.is_file():
        raise FileNotFoundError(f"catalog does not exist: {inputs.catalog_path}")
    if inputs.workers <= 0:
        raise ValueError("workers must be positive")


def _prepare_inputs(inputs: ExecutionInputs) -> PreparedInputs:
    _require_resolved_inputs(inputs)
    acquisition_epoch = odss.epoch_from_iso(inputs.catalog_acquired_at_utc, "UTC")
    catalog = odss.parse_omm_json(
        inputs.catalog_path.read_bytes(),
        logical_name=inputs.catalog_path.name,
        source_uri=inputs.catalog_source_uri,
        acquired_at=acquisition_epoch,
    )
    selected = odss.filter_sso(
        catalog,
        max_precession_error_deg_day=study.SSO_PRECESSION_TOLERANCE_DEG_DAY,
    )
    if not selected.records:
        raise ValueError(
            "the supplied active EO catalog contains no objects passing the SSO filter"
        )
    common_epoch = odss.epoch_from_iso(inputs.common_epoch_utc, "UTC")
    synchronized = odss.synchronize_catalog_sgp4(selected, common_epoch)
    earth_orientation = odss.bundled_iers_a()
    frame = odss.ReferenceFrame("EME2000")
    states = tuple(
        odss.transform_state(item.state, frame, earth_orientation) for item in synchronized.objects
    )
    target_count = len(states)
    targets = odss.ParticlePopulation(
        epoch=common_epoch,
        frame=frame,
        position_x_m=tuple(state.position_m[0] for state in states),
        position_y_m=tuple(state.position_m[1] for state in states),
        position_z_m=tuple(state.position_m[2] for state in states),
        velocity_x_m_s=tuple(state.velocity_m_s[0] for state in states),
        velocity_y_m_s=tuple(state.velocity_m_s[1] for state in states),
        velocity_z_m_s=tuple(state.velocity_m_s[2] for state in states),
        mass_kg=(study.TARGET_PROXY_MASS_KG,) * target_count,
        area_m2=(study.TARGET_PROXY_AREA_M2,) * target_count,
    )
    study_directory = Path(__file__).resolve().parent
    assets = tuple(acquisition.asset for acquisition in catalog.acquisitions) + (
        earth_orientation.metadata,
        _asset_for_file(study_directory / "study.py", "iac_2026_sso/study.py"),
        _asset_for_file(study_directory / "campaign.py", "iac_2026_sso/campaign.py"),
    )
    study_identity = {
        "analysis_size_thresholds_m": study.ANALYSIS_SIZE_THRESHOLDS_M,
        "catalog_assets": tuple(asset.content_sha256 for asset in assets),
        "catalog_acquired_at_utc": inputs.catalog_acquired_at_utc,
        "catalog_source_uri": inputs.catalog_source_uri,
        "collisional_timestep_s": study.COLLISIONAL_TIMESTEP_S,
        "collisional_steps_per_batch": study.COLLISIONAL_STEPS_PER_BATCH,
        "common_epoch_utc": inputs.common_epoch_utc,
        "flux_radii_m": study.FLUX_RADII_M,
        "generated_minimum_size_m": study.GENERATED_MINIMUM_SIZE_M,
        "integrator_tolerance": study.INTEGRATOR_TOLERANCE,
        "maneuver_distance_thresholds_m": study.MANEUVER_DISTANCE_THRESHOLDS_M,
        "master_seed": study.MASTER_SEED,
        "reference_areas_m2": study.REFERENCE_AREAS_M2,
        "scenarios": study.SCENARIOS,
        "sso_precession_tolerance_deg_day": study.SSO_PRECESSION_TOLERANCE_DEG_DAY,
        "target_proxy_area_m2": study.TARGET_PROXY_AREA_M2,
        "target_proxy_mass_kg": study.TARGET_PROXY_MASS_KG,
        "trackability_size_threshold_m": study.TRACKABILITY_SIZE_THRESHOLD_M,
        "variants": tuple(
            variant for scenario in study.SCENARIOS for variant in study.variants_for(scenario)
        ),
    }
    return PreparedInputs(
        targets=targets,
        assets=assets,
        common_epoch=common_epoch,
        catalog_record_count=len(catalog.records),
        selected_target_count=target_count,
        target_catalog_ids=tuple(item.record.catalog_id for item in synchronized.objects),
        study_hash=_sha256_configuration(study_identity),
        code_version=inputs.code_version,
    )


def _breakup_angles(run: odss.MonteCarloRun) -> tuple[float, float]:
    return (
        2.0 * math.pi * odss.uniform_01(run.random_key("breakup_raan"), 0),
        2.0 * math.pi * odss.uniform_01(run.random_key("breakup_phase"), 0),
    )


def _rotated_circular_state(
    scenario: study.Scenario,
    run: odss.MonteCarloRun,
    epoch: odss.Epoch,
) -> odss.CartesianState:
    raan_rad, argument_of_latitude_rad = _breakup_angles(run)
    inclination_rad = math.radians(scenario.inclination_deg)
    radius_m = _earth_radius_m + scenario.altitude_m
    speed_m_s = math.sqrt(_earth_gravitational_parameter_m3_s2 / radius_m)
    cosine_u = math.cos(argument_of_latitude_rad)
    sine_u = math.sin(argument_of_latitude_rad)
    cosine_i = math.cos(inclination_rad)
    sine_i = math.sin(inclination_rad)
    cosine_raan = math.cos(raan_rad)
    sine_raan = math.sin(raan_rad)

    def rotate(x_value: float, y_value: float) -> tuple[float, float, float]:
        inclined_y = y_value * cosine_i
        return (
            cosine_raan * x_value - sine_raan * inclined_y,
            sine_raan * x_value + cosine_raan * inclined_y,
            y_value * sine_i,
        )

    position_unit = rotate(cosine_u, sine_u)
    velocity_unit = rotate(-sine_u, cosine_u)
    return odss.CartesianState(
        position_m=tuple(radius_m * value for value in position_unit),
        velocity_m_s=tuple(speed_m_s * value for value in velocity_unit),
        epoch=epoch,
        frame=odss.ReferenceFrame("EME2000"),
    )


def _collision_secondary_state(
    primary: odss.CartesianState,
    relative_velocity_m_s: float,
) -> odss.CartesianState:
    radius_m = math.sqrt(sum(value * value for value in primary.position_m))
    speed_m_s = math.sqrt(sum(value * value for value in primary.velocity_m_s))
    if relative_velocity_m_s > 2.0 * speed_m_s:
        raise ValueError("collision relative velocity exceeds two circular-orbit speeds")
    radial = tuple(value / radius_m for value in primary.position_m)
    tangent = tuple(value / speed_m_s for value in primary.velocity_m_s)
    normal = (
        radial[1] * tangent[2] - radial[2] * tangent[1],
        radial[2] * tangent[0] - radial[0] * tangent[2],
        radial[0] * tangent[1] - radial[1] * tangent[0],
    )
    separation_angle_rad = 2.0 * math.asin(relative_velocity_m_s / (2.0 * speed_m_s))
    secondary_velocity_m_s = tuple(
        speed_m_s
        * (
            math.cos(separation_angle_rad) * tangent[index]
            + math.sin(separation_angle_rad) * normal[index]
        )
        for index in range(3)
    )
    return odss.CartesianState(
        primary.position_m,
        secondary_velocity_m_s,
        primary.epoch,
        primary.frame,
    )


def _generate_fragments(
    task: StudyTask,
    run: odss.MonteCarloRun,
    epoch: odss.Epoch,
) -> odss.FragmentationResult:
    scenario = task.scenario
    primary_state = _rotated_circular_state(scenario, run, epoch)
    primary_properties = odss.PhysicalProperties(
        scenario.primary_mass_kg,
        scenario.primary_area_m2,
    )
    random_key = run.random_key("nasa_sbm")
    if scenario.breakup_mode == "explosion":
        return odss.generate_explosion_fragments(
            primary_state,
            primary_properties,
            satellite_type="spacecraft",
            minimum_characteristic_length_m=study.GENERATED_MINIMUM_SIZE_M,
            random_key=random_key,
        )
    secondary_state = _collision_secondary_state(
        primary_state,
        scenario.collision_relative_velocity_m_s,
    )
    return odss.generate_collision_fragments(
        primary_state,
        primary_properties,
        secondary_state,
        odss.PhysicalProperties(scenario.secondary_mass_kg, scenario.secondary_area_m2),
        primary_type="spacecraft",
        secondary_type="spacecraft",
        minimum_characteristic_length_m=study.GENERATED_MINIMUM_SIZE_M,
        random_key=random_key,
    )


def _events_within(
    events: tuple[odss.ConjunctionEvent, ...],
    threshold_m: float,
) -> tuple[odss.ConjunctionEvent, ...]:
    return tuple(event for event in events if event.miss_distance_m <= threshold_m)


def _integrated_flux(flux: odss.FluxResult, field_name: str) -> float:
    return sum(
        float(getattr(value, field_name)) * (value.time_end_s - value.time_start_s)
        for value in flux.bins
    )


def _finite_mean(values: Sequence[float | None]) -> float:
    finite = tuple(float(value) for value in values if value is not None and math.isfinite(value))
    return statistics.fmean(finite) if finite else 0.0


def _package_version(distribution: str, module_name: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "unknown"))


def _run_address(task: StudyTask) -> odss.MonteCarloRun:
    return odss.MonteCarloRun(
        master_seed=study.MASTER_SEED,
        scenario_id=task.scenario.scenario_id,
        run_id=task.run_id,
        variant=task.variant.name,
        backend_threads=1,
    )


def _run_manifest(
    task: StudyTask,
    run: odss.MonteCarloRun,
    prepared: PreparedInputs,
) -> odss.RunManifest:
    return odss.RunManifest(
        experiment_hash=_experiment_hash(task),
        study_hash=prepared.study_hash,
        master_seed=run.master_seed,
        scenario_id=run.scenario_id,
        run_id=run.run_id,
        input_assets=prepared.assets,
        software=(
            odss.odss_software_metadata(),
            odss.SoftwareMetadata("odss_git", prepared.code_version),
            odss.SoftwareMetadata("cascade", _package_version("esa-cascade", "cascade")),
            odss.SoftwareMetadata("nasa-sbm-py", _package_version("nasa-sbm-py", "nasa_sbm")),
        ),
    )


def _scientific_result(
    task: StudyTask,
    run: odss.MonteCarloRun,
    prepared: PreparedInputs,
) -> tuple[odss.RunResult, int, int]:
    fragments = _generate_fragments(task, run, prepared.common_epoch)
    initial_escape = odss.evaluate_escape(fragments.population)
    propagation_start = time.perf_counter()
    evolution = odss.propagate_and_screen_sso(
        fragments.population,
        prepared.targets,
        odss.SsoPropagationSpec(
            duration_s=task.campaign.duration_s,
            collisional_timestep_s=study.COLLISIONAL_TIMESTEP_S,
            force_model=task.variant.force_model,
            tolerance=study.INTEGRATOR_TOLERANCE,
            high_accuracy=False,
            collisional_steps_per_batch=study.COLLISIONAL_STEPS_PER_BATCH,
        ),
        threshold_m=task.campaign.screening_threshold_m,
    )
    propagation_wall_duration_s = time.perf_counter() - propagation_start
    decay = odss.evaluate_decay(fragments.population, evolution.final_debris)

    flux_by_radius: dict[float, odss.FluxResult] = {}
    for radius_m in study.FLUX_RADII_M:
        if radius_m <= task.campaign.screening_threshold_m:
            flux_by_radius[radius_m] = odss.evaluate_flux(
                _events_within(evolution.conjunctions, radius_m),
                fragments.population,
                prepared.targets,
                fragments.characteristic_length_m,
                odss.FluxSpec(
                    sampling_radius_m=radius_m,
                    time_bin_edges_s=task.campaign.time_bin_edges_s,
                    size_bin_edges_m=study.FLUX_SIZE_BIN_EDGES_M,
                ),
            )
    stored_flux = flux_by_radius[task.campaign.stored_flux_radius_m]
    impact = odss.evaluate_impact_risk(stored_flux, study.REFERENCE_AREAS_M2)

    demand_by_distance = {
        distance_m: odss.evaluate_maneuver_demand(
            _events_within(evolution.conjunctions, distance_m),
            fragments.population,
            prepared.targets,
            fragments.characteristic_length_m,
            odss.ManeuverDemandSpec(
                trackability_size_threshold_m=study.TRACKABILITY_SIZE_THRESHOLD_M,
                miss_distance_threshold_m=distance_m,
                duration_s=task.campaign.duration_s,
            ),
        )
        for distance_m in study.MANEUVER_DISTANCE_THRESHOLDS_M
        if distance_m <= task.campaign.screening_threshold_m
    }
    stored_demand = demand_by_distance[1_000.0]

    summaries = [
        odss.ScalarRunResult("fragment_count", float(len(fragments)), "1"),
        odss.ScalarRunResult("nasa_sbm_seed", float(fragments.backend_seed), "1"),
        odss.ScalarRunResult("breakup_raan_rad", _breakup_angles(run)[0], "rad"),
        odss.ScalarRunResult("breakup_argument_of_latitude_rad", _breakup_angles(run)[1], "rad"),
        odss.ScalarRunResult("initial_escaped_fraction", initial_escape.escaped_fraction, "1"),
        odss.ScalarRunResult("retained_fraction", decay.retained_fraction, "1"),
        odss.ScalarRunResult("removal_fraction", decay.removal_fraction, "1"),
        odss.ScalarRunResult("reentry_fraction", decay.reentry_fraction, "1"),
        odss.ScalarRunResult("final_escaped_fraction", decay.escape_fraction, "1"),
        odss.ScalarRunResult(
            "mean_semimajor_axis_change_m",
            _finite_mean(tuple(item.semimajor_axis_change_m for item in decay.diagnostics)),
            "m",
        ),
        odss.ScalarRunResult(
            "mean_perigee_altitude_change_m",
            _finite_mean(tuple(item.perigee_altitude_change_m for item in decay.diagnostics)),
            "m",
        ),
        odss.ScalarRunResult("propagation_wall_duration_s", propagation_wall_duration_s, "s"),
        odss.ScalarRunResult("selected_target_count", float(prepared.selected_target_count), "1"),
        odss.ScalarRunResult(
            "has_screened_conjunction",
            float(bool(evolution.conjunctions)),
            "1",
        ),
        odss.ScalarRunResult(
            "censored_minimum_miss_distance_m",
            min(
                (event.miss_distance_m for event in evolution.conjunctions),
                default=task.campaign.screening_threshold_m,
            ),
            "m",
        ),
    ]
    for threshold_m in study.ANALYSIS_SIZE_THRESHOLDS_M:
        label_cm = round(threshold_m * 100.0)
        count = sum(length_m >= threshold_m for length_m in fragments.characteristic_length_m)
        summaries.append(
            odss.ScalarRunResult(f"fragment_count_ge_{label_cm}_cm", float(count), "1")
        )
    for distance_m in study.MANEUVER_DISTANCE_THRESHOLDS_M:
        if distance_m > task.campaign.screening_threshold_m:
            continue
        label_m = round(distance_m)
        encounters = _events_within(evolution.conjunctions, distance_m)
        demand = demand_by_distance[distance_m]
        summaries.extend(
            (
                odss.ScalarRunResult(
                    f"conjunction_count_le_{label_m}_m", float(len(encounters)), "1"
                ),
                odss.ScalarRunResult(
                    f"has_conjunction_le_{label_m}_m",
                    float(bool(encounters)),
                    "1",
                ),
                odss.ScalarRunResult(
                    f"maneuver_event_count_le_{label_m}_m",
                    float(len(demand.actionable_encounters)),
                    "1",
                ),
                odss.ScalarRunResult(
                    f"maneuver_affected_target_count_le_{label_m}_m",
                    float(len(demand.affected_target_indices)),
                    "1",
                ),
                odss.ScalarRunResult(
                    f"has_maneuver_event_le_{label_m}_m",
                    float(bool(demand.actionable_encounters)),
                    "1",
                ),
                odss.ScalarRunResult(
                    f"poisson_model_maneuver_probability_le_{label_m}_m",
                    demand.probability_at_least_one_actionable_encounter,
                    "1",
                ),
            )
        )
    for radius_m, flux in flux_by_radius.items():
        label_m = round(radius_m)
        summaries.extend(
            (
                odss.ScalarRunResult(
                    f"summed_target_integrated_number_flux_r_{label_m}_m",
                    _integrated_flux(flux, "number_flux_m2_s"),
                    "m-2",
                ),
                odss.ScalarRunResult(
                    f"summed_target_integrated_mass_flux_r_{label_m}_m",
                    _integrated_flux(flux, "mass_flux_kg_m2_s"),
                    "kg m-2",
                ),
                odss.ScalarRunResult(
                    f"summed_target_integrated_energy_flux_r_{label_m}_m",
                    _integrated_flux(flux, "kinetic_energy_flux_w_m2"),
                    "J m-2",
                ),
            )
        )
    for area_m2 in study.REFERENCE_AREAS_M2:
        values = tuple(
            value.model_impact_probability
            for value in impact.values
            if value.reference_area_m2 == area_m2
        )
        label_m2 = round(area_m2)
        summaries.extend(
            (
                odss.ScalarRunResult(
                    f"mean_model_impact_probability_area_{label_m2}_m2",
                    statistics.fmean(values) if values else 0.0,
                    "1",
                ),
                odss.ScalarRunResult(
                    f"maximum_model_impact_probability_area_{label_m2}_m2",
                    max(values, default=0.0),
                    "1",
                ),
            )
        )

    manifest = _run_manifest(task, run, prepared)
    return (
        odss.RunResult(
            manifest=manifest,
            summary=tuple(summaries),
            conjunctions=evolution.conjunctions,
            flux=stored_flux,
            maneuver_demand=stored_demand,
        ),
        len(fragments),
        len(evolution.conjunctions),
    )


def _set_worker_thread_limits() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        cascade = importlib.import_module("cascade")
    except ImportError:
        return
    cascade.set_nthreads(1)


def _run_task(task: StudyTask) -> TaskReport:
    prepared = _prepared_inputs
    if prepared is None:
        raise RuntimeError("worker scientific inputs were not prepared")
    start = time.perf_counter()
    run = _run_address(task)
    result, fragment_count, conjunction_count = _scientific_result(task, run, prepared)
    odss.write_run_result(task.output_path, result)
    return TaskReport(
        scenario_id=task.scenario.scenario_id,
        variant=task.variant.name,
        run_id=task.run_id,
        wall_duration_s=time.perf_counter() - start,
        fragment_count=fragment_count,
        conjunction_count=conjunction_count,
        output_path=str(task.output_path),
    )


def _result_path(
    output_directory: Path,
    scenario: study.Scenario,
    variant: study.ModelVariant,
    run_id: int,
) -> Path:
    return output_directory / scenario.name / variant.name / f"run_{run_id:06d}.nc"


def _experiment_hash(task: StudyTask) -> str:
    return _sha256_configuration(
        {
            "campaign": task.campaign,
            "scenario": task.scenario,
            "variant": task.variant,
        }
    )


def _require_matching_existing_result(task: StudyTask, prepared: PreparedInputs) -> None:
    existing = odss.read_run_result(task.output_path)
    expected = _run_manifest(task, _run_address(task), prepared)
    if odss.canonical_manifest(existing.manifest) != odss.canonical_manifest(expected):
        raise ValueError(f"existing result manifest does not match: {task.output_path}")


def _tasks(
    campaign: study.Campaign,
    output_directory: Path,
    runs_per_family: int,
) -> tuple[StudyTask, ...]:
    return tuple(
        StudyTask(
            campaign=campaign,
            scenario=scenario,
            variant=variant,
            run_id=run_id,
            output_path=_result_path(output_directory, scenario, variant, run_id),
        )
        for scenario in study.SCENARIOS
        for variant in study.variants_for(scenario)
        for run_id in range(runs_per_family)
    )


def _plan(
    campaign: study.Campaign,
    runs_per_family: int,
    workers: int,
    wall_time_budget_s: float,
) -> dict[str, object]:
    return {
        "campaign": campaign.name,
        "duration_s": campaign.duration_s,
        "expected_result_files": study.family_count() * runs_per_family,
        "families": study.family_count(),
        "maximum_workers": workers,
        "runs_per_family": runs_per_family,
        "scenario_names": tuple(scenario.name for scenario in study.SCENARIOS),
        "schema": "odss.sso_campaign_plan.v1",
        "wall_time_budget_s": wall_time_budget_s,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(_canonical_value(value), indent=2, sort_keys=True) + "\n")


def _execute_campaign(
    campaign: study.Campaign,
    inputs: ExecutionInputs,
    *,
    runs_per_family: int,
    wall_time_budget_s: float,
) -> int:
    global _prepared_inputs
    _require_resolved_inputs(inputs)
    if not 1 <= runs_per_family <= campaign.maximum_runs_per_family:
        raise ValueError(f"runs per family must be in [1, {campaign.maximum_runs_per_family}]")
    if inputs.workers > campaign.maximum_workers:
        raise ValueError(f"workers must not exceed {campaign.maximum_workers}")
    if not math.isfinite(wall_time_budget_s) or wall_time_budget_s <= 0.0:
        raise ValueError("wall-time budget must be positive and finite")
    output_directory = inputs.output_directory
    if output_directory.exists() and any(output_directory.iterdir()) and not inputs.resume:
        raise FileExistsError(
            f"output directory is not empty: {output_directory}; use --resume or another path"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    _set_worker_thread_limits()
    validation = odss.run_scientific_validation(include_optional_backends=True)
    odss.write_validation_report(output_directory / "validation.json", validation)
    if not validation.passed or validation.skipped_case_names:
        raise RuntimeError(
            "scientific validation must pass without skipped optional backends before execution"
        )

    _prepared_inputs = _prepare_inputs(inputs)
    all_tasks = _tasks(campaign, output_directory, runs_per_family)
    for task in all_tasks:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = tuple(task for task in all_tasks if task.output_path.exists())
    if existing and not inputs.resume:
        raise FileExistsError(f"result already exists: {existing[0].output_path}")
    for task in existing:
        _require_matching_existing_result(task, _prepared_inputs)
    pending = tuple(task for task in all_tasks if not task.output_path.exists())
    plan = _plan(campaign, runs_per_family, inputs.workers, wall_time_budget_s)
    campaign_manifest = {
        **plan,
        "catalog_path": str(inputs.catalog_path),
        "catalog_acquired_at_utc": inputs.catalog_acquired_at_utc,
        "catalog_record_count": _prepared_inputs.catalog_record_count,
        "catalog_source_uri": inputs.catalog_source_uri,
        "code_version": inputs.code_version,
        "common_epoch_utc": inputs.common_epoch_utc,
        "family_experiment_hashes": {
            f"{task.scenario.name}/{task.variant.name}": _experiment_hash(task)
            for task in all_tasks
            if task.run_id == 0
        },
        "selected_target_count": _prepared_inputs.selected_target_count,
        "study_hash": _prepared_inputs.study_hash,
        "target_catalog_ids": _prepared_inputs.target_catalog_ids,
    }
    _write_json(output_directory / "campaign_manifest.json", campaign_manifest)
    if not pending:
        print(f"All {len(all_tasks)} result files already exist; nothing to resume.")
        return 0

    start = time.perf_counter()
    reports: tuple[TaskReport, ...]
    worker_count = min(inputs.workers, len(pending))
    if worker_count == 1:
        _set_worker_thread_limits()
        reports = tuple(_run_task(task) for task in pending)
    else:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_set_worker_thread_limits,
        ) as executor:
            reports = tuple(executor.map(_run_task, pending, chunksize=1))
    campaign_wall_duration_s = time.perf_counter() - start
    ordered_reports = tuple(
        sorted(reports, key=lambda item: (item.scenario_id, item.variant, item.run_id))
    )
    mean_task_wall_duration_s = statistics.fmean(
        report.wall_duration_s for report in ordered_reports
    )
    summary = {
        **plan,
        "campaign_wall_duration_s": campaign_wall_duration_s,
        "completed_existing_files": len(existing),
        "completed_new_files": len(ordered_reports),
        "mean_task_wall_duration_s": mean_task_wall_duration_s,
        "reports": ordered_reports,
    }
    if campaign == study.PRELIMINARY_CAMPAIGN:
        summary["recommended_production_runs_per_family"] = _budgeted_production_runs(
            mean_task_wall_duration_s,
            pilot_duration_s=campaign.duration_s,
            workers=inputs.workers,
            wall_time_budget_s=study.PRODUCTION_CAMPAIGN.wall_time_budget_s,
        )
    _write_json(output_directory / "campaign_summary.json", summary)
    print(
        f"Completed {len(ordered_reports)} new files in "
        f"{campaign_wall_duration_s / 3_600.0:.2f} h; "
        f"results: {output_directory}"
    )
    return 0


def _budgeted_production_runs(
    pilot_task_wall_duration_s: float,
    *,
    pilot_duration_s: float,
    workers: int,
    wall_time_budget_s: float,
) -> int:
    if (
        not math.isfinite(pilot_task_wall_duration_s)
        or not math.isfinite(pilot_duration_s)
        or pilot_task_wall_duration_s <= 0.0
        or pilot_duration_s <= 0.0
    ):
        raise ValueError("pilot timing values must be positive")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not math.isfinite(wall_time_budget_s) or wall_time_budget_s <= 0.0:
        raise ValueError("wall-time budget must be positive and finite")
    full_task_estimate_s = (
        pilot_task_wall_duration_s * study.PRODUCTION_CAMPAIGN.duration_s / pilot_duration_s
    )
    for candidate in range(study.PRODUCTION_CAMPAIGN.maximum_runs_per_family, 0, -1):
        task_count = study.family_count() * candidate
        worker_waves = math.ceil(task_count / workers)
        if worker_waves * full_task_estimate_s * 1.25 <= wall_time_budget_s:
            return candidate
    return 0


def _common_parser(campaign: study.Campaign) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run the {campaign.name} SSO fragmentation campaign."
    )
    parser.add_argument("--catalog", type=Path, default=Path(study.CATALOG_PATH))
    parser.add_argument("--catalog-source-uri", default=study.CATALOG_SOURCE_URI)
    parser.add_argument("--catalog-acquired-at", default=study.CATALOG_ACQUIRED_AT_UTC)
    parser.add_argument("--epoch", default=study.COMMON_EPOCH_UTC)
    parser.add_argument("--code-version", default=study.CODE_VERSION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "iac_2026_sso" / campaign.name,
    )
    parser.add_argument("--workers", type=int, default=campaign.maximum_workers)
    parser.add_argument(
        "--wall-hours",
        type=float,
        default=campaign.wall_time_budget_s / 3_600.0,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _execution_inputs(arguments: argparse.Namespace) -> ExecutionInputs:
    return ExecutionInputs(
        catalog_path=arguments.catalog,
        catalog_source_uri=arguments.catalog_source_uri,
        catalog_acquired_at_utc=arguments.catalog_acquired_at,
        common_epoch_utc=arguments.epoch,
        code_version=arguments.code_version,
        output_directory=arguments.output,
        workers=arguments.workers,
        resume=arguments.resume,
    )


def preliminary_main(arguments: Sequence[str] | None = None) -> int:
    """CLI entry point for the approximately one-hour calibration campaign."""
    campaign = study.PRELIMINARY_CAMPAIGN
    parser = _common_parser(campaign)
    parser.add_argument(
        "--runs-per-family",
        type=int,
        default=campaign.default_runs_per_family,
    )
    parsed = parser.parse_args(arguments)
    wall_time_budget_s = parsed.wall_hours * 3_600.0
    if not 1 <= parsed.runs_per_family <= campaign.maximum_runs_per_family:
        parser.error(f"--runs-per-family must be in [1, {campaign.maximum_runs_per_family}]")
    if not 1 <= parsed.workers <= campaign.maximum_workers:
        parser.error(f"--workers must be in [1, {campaign.maximum_workers}]")
    if not math.isfinite(wall_time_budget_s) or wall_time_budget_s <= 0.0:
        parser.error("--wall-hours must be positive and finite")
    plan = _plan(campaign, parsed.runs_per_family, parsed.workers, wall_time_budget_s)
    if parsed.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return _execute_campaign(
        campaign,
        _execution_inputs(parsed),
        runs_per_family=parsed.runs_per_family,
        wall_time_budget_s=wall_time_budget_s,
    )


def _runs_from_pilot(
    pilot_summary_path: Path,
    *,
    workers: int,
    wall_time_budget_s: float,
) -> int:
    if not pilot_summary_path.is_file():
        raise FileNotFoundError(
            f"pilot summary does not exist: {pilot_summary_path}; run preliminary first or "
            "provide --runs-per-family"
        )
    pilot = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
    runs_per_family = _budgeted_production_runs(
        float(pilot["mean_task_wall_duration_s"]),
        pilot_duration_s=float(pilot["duration_s"]),
        workers=workers,
        wall_time_budget_s=wall_time_budget_s,
    )
    if runs_per_family == 0:
        raise RuntimeError(
            "pilot timing predicts that the minimum production matrix exceeds the wall-time budget"
        )
    return runs_per_family


def production_main(arguments: Sequence[str] | None = None) -> int:
    """CLI entry point for the budget-sized one-year production campaign."""
    campaign = study.PRODUCTION_CAMPAIGN
    parser = _common_parser(campaign)
    parser.add_argument("--runs-per-family", type=int)
    parser.add_argument(
        "--pilot-summary",
        type=Path,
        default=Path("results/iac_2026_sso/preliminary/campaign_summary.json"),
    )
    parsed = parser.parse_args(arguments)
    wall_time_budget_s = parsed.wall_hours * 3_600.0
    if not 1 <= parsed.workers <= campaign.maximum_workers:
        parser.error(f"--workers must be in [1, {campaign.maximum_workers}]")
    if not math.isfinite(wall_time_budget_s) or wall_time_budget_s <= 0.0:
        parser.error("--wall-hours must be positive and finite")
    if parsed.runs_per_family is not None:
        runs_per_family = parsed.runs_per_family
    elif parsed.dry_run and not parsed.pilot_summary.is_file():
        runs_per_family = campaign.default_runs_per_family
    else:
        runs_per_family = _runs_from_pilot(
            parsed.pilot_summary,
            workers=parsed.workers,
            wall_time_budget_s=wall_time_budget_s,
        )
    if not 1 <= runs_per_family <= campaign.maximum_runs_per_family:
        parser.error(f"--runs-per-family must be in [1, {campaign.maximum_runs_per_family}]")
    plan = _plan(campaign, runs_per_family, parsed.workers, wall_time_budget_s)
    if parsed.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return _execute_campaign(
        campaign,
        _execution_inputs(parsed),
        runs_per_family=runs_per_family,
        wall_time_budget_s=wall_time_budget_s,
    )


__all__ = [
    "ExecutionInputs",
    "PreparedInputs",
    "StudyTask",
    "TaskReport",
    "preliminary_main",
    "production_main",
]
