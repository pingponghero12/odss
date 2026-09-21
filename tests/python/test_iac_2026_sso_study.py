import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import odss
from studies.iac_2026_sso import campaign, study
from studies.iac_2026_sso import inputs as study_inputs


def test_principal_matrix_and_nested_analysis_are_explicit() -> None:
    assert {(item.altitude_m, item.breakup_mode) for item in study.SCENARIOS} == {
        (500_000.0, "explosion"),
        (500_000.0, "collision"),
        (700_000.0, "explosion"),
        (700_000.0, "collision"),
        (800_000.0, "explosion"),
        (800_000.0, "collision"),
    }
    assert tuple(item.scenario_id for item in study.SCENARIOS) == tuple(range(6))
    assert study.GENERATED_MINIMUM_SIZE_M == 0.01
    assert study.ANALYSIS_SIZE_THRESHOLDS_M == (0.01, 0.05, 0.10)
    assert study.PRELIMINARY_CAMPAIGN.duration_s == 7.0 * 86_400.0
    assert study.SMOKE_CAMPAIGN.duration_s == 600.0
    assert study.PRODUCTION_CAMPAIGN.duration_s == 365.25 * 86_400.0
    assert study.PRODUCTION_CAMPAIGN.screening_threshold_m == max(study.FLUX_RADII_M)
    assert study.PRODUCTION_CAMPAIGN.maximum_runs_per_family == 64
    assert study.PRODUCTION_CAMPAIGN.wall_time_budget_s == 48.0 * 3_600.0
    assert study.COLLISIONAL_STEPS_PER_BATCH == 120


def test_model_sensitivities_are_paired_only_at_central_altitude() -> None:
    family_counts = {
        scenario.name: len(study.variants_for(scenario)) for scenario in study.SCENARIOS
    }

    assert family_counts["500_km_explosion"] == 1
    assert family_counts["800_km_collision"] == 1
    assert family_counts["700_km_explosion"] == 4
    assert family_counts["700_km_collision"] == 4
    assert study.family_count() == 12


def test_breakup_geometry_is_deterministic_and_variants_share_it() -> None:
    scenario = study.SCENARIOS[2]
    epoch = odss.epoch_from_iso("2026-09-21T00:00:00", "UTC")
    nominal = odss.MonteCarloRun(study.MASTER_SEED, scenario.scenario_id, 7, "nominal", 1)
    enhanced = odss.MonteCarloRun(
        study.MASTER_SEED,
        scenario.scenario_id,
        7,
        "enhanced_forces",
        1,
    )

    first = campaign._rotated_circular_state(scenario, nominal, epoch)
    second = campaign._rotated_circular_state(scenario, enhanced, epoch)

    assert first == second
    assert math.sqrt(sum(value * value for value in first.position_m)) == pytest.approx(
        6_378_136.3 + scenario.altitude_m
    )
    assert math.sqrt(sum(value * value for value in first.velocity_m_s)) == pytest.approx(
        math.sqrt(3.986_004_418e14 / (6_378_136.3 + scenario.altitude_m))
    )


def test_collision_geometry_has_the_requested_relative_speed() -> None:
    scenario = study.SCENARIOS[3]
    run = odss.MonteCarloRun(study.MASTER_SEED, scenario.scenario_id, 0, "nominal", 1)
    primary = campaign._rotated_circular_state(
        scenario,
        run,
        odss.epoch_from_iso("2026-09-21T00:00:00", "UTC"),
    )
    secondary = campaign._collision_secondary_state(
        primary,
        scenario.collision_relative_velocity_m_s,
    )

    relative_speed_m_s = math.sqrt(
        sum(
            (first - second) ** 2
            for first, second in zip(
                primary.velocity_m_s,
                secondary.velocity_m_s,
                strict=True,
            )
        )
    )
    assert secondary.position_m == primary.position_m
    assert relative_speed_m_s == pytest.approx(scenario.collision_relative_velocity_m_s)


def test_catalog_is_filtered_synchronized_and_transformed_to_eme2000(
    tmp_path: Path,
) -> None:
    source = Path("tests/fixtures/omm_catalog_a.json")
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(source.read_bytes())
    inputs = campaign.ExecutionInputs(
        catalog_path=catalog,
        catalog_source_uri="fixture://active-eo",
        catalog_acquired_at_utc="2026-06-20T00:00:00",
        common_epoch_utc="2026-06-20T00:00:00",
        code_version="test-commit",
        catalog_provenance_paths=(),
        output_directory=tmp_path / "results",
        workers=1,
        resume=False,
    )

    prepared = campaign._prepare_inputs(inputs)

    assert prepared.catalog_record_count == 3
    assert prepared.selected_target_count == 2
    assert len(prepared.target_catalog_ids) == prepared.selected_target_count
    assert prepared.targets.frame == odss.ReferenceFrame("EME2000")
    assert len(prepared.study_hash) == 64
    assert {asset.logical_name for asset in prepared.assets} >= {
        "catalog.json",
        "iac_2026_sso/study.py",
        "iac_2026_sso/campaign.py",
        "iac_2026_sso/inputs.py",
    }


def test_production_count_respects_runtime_and_data_caps() -> None:
    fast = campaign._budgeted_production_runs(
        60.0,
        pilot_duration_s=study.PRELIMINARY_CAMPAIGN.duration_s,
        workers=32,
        wall_time_budget_s=48.0 * 3_600.0,
    )
    slower = campaign._budgeted_production_runs(
        3_600.0,
        pilot_duration_s=study.PRELIMINARY_CAMPAIGN.duration_s,
        workers=32,
        wall_time_budget_s=48.0 * 3_600.0,
    )

    assert fast == study.PRODUCTION_CAMPAIGN.maximum_runs_per_family
    assert slower == 0


def test_launch_scripts_expose_dry_run_plans(capsys: pytest.CaptureFixture[str]) -> None:
    assert campaign.smoke_main(("--dry-run",)) == 0
    smoke = json.loads(capsys.readouterr().out)
    assert smoke["expected_result_files"] == 1
    assert smoke["scenario_names"] == ["500_km_explosion"]

    assert campaign.preliminary_main(("--dry-run",)) == 0
    preliminary = json.loads(capsys.readouterr().out)
    assert preliminary["expected_result_files"] == 24
    assert preliminary["wall_time_budget_s"] == 3_600.0

    assert campaign.production_main(("--dry-run",)) == 0
    production = json.loads(capsys.readouterr().out)
    assert production["duration_s"] == 365.25 * 86_400.0
    assert production["expected_result_files"] == 192


def test_worker_setup_disables_nested_cascade_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_counts: list[int] = []
    backend = SimpleNamespace(set_nthreads=thread_counts.append)
    monkeypatch.setattr(campaign.importlib, "import_module", lambda name: backend)

    campaign._set_worker_thread_limits()

    assert thread_counts == [1]
    assert campaign.os.environ["OMP_NUM_THREADS"] == "1"


def test_missing_catalog_fails_before_execution(tmp_path: Path) -> None:
    inputs = campaign.ExecutionInputs(
        catalog_path=tmp_path / "missing.json",
        catalog_source_uri="fixture://active-eo",
        catalog_acquired_at_utc="2026-06-20T00:00:00",
        common_epoch_utc="2026-06-20T00:00:00",
        code_version="test-commit",
        catalog_provenance_paths=(),
        output_directory=tmp_path,
        workers=1,
        resume=False,
    )

    with pytest.raises(FileNotFoundError, match="catalog does not exist"):
        campaign._require_resolved_inputs(inputs)


def test_catalog_snapshot_is_frozen_verified_and_reusable(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/omm_catalog_a.json").read_bytes()
    requested: list[str] = []

    def fetch(url: str) -> bytes:
        requested.append(url)
        return fixture

    snapshot = study_inputs.acquire_catalog_snapshot(
        tmp_path / "input",
        fetcher=fetch,
        acquired_at_utc="2026-09-21T12:00:00",
    )
    loaded = study_inputs.load_catalog_snapshot(snapshot.manifest_path)

    assert loaded == snapshot
    assert len(requested) == 3
    assert {path.name for path in snapshot.source_paths} == {
        "celestrak_weather.json",
        "celestrak_resource.json",
        "celestrak_sar.json",
    }
    merged = json.loads(snapshot.catalog_path.read_text(encoding="utf-8"))
    assert len(merged) == 9
    assert snapshot.common_epoch_utc == "2026-09-21T12:00:00"

    snapshot.source_paths[0].write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        study_inputs.load_catalog_snapshot(snapshot.manifest_path)


def test_one_synthetic_realization_builds_a_round_trip_netcdf_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epoch = odss.epoch_from_iso("2026-09-21T00:00:00", "UTC")
    frame = odss.ReferenceFrame("EME2000")
    initial_debris = odss.ParticlePopulation(
        epoch=epoch,
        frame=frame,
        position_x_m=(7_078_136.3, 7_078_136.3),
        position_y_m=(0.0, 1.0),
        position_z_m=(0.0, 0.0),
        velocity_x_m_s=(0.0, 0.0),
        velocity_y_m_s=(7_500.0, 7_500.0),
        velocity_z_m_s=(0.0, 0.0),
        mass_kg=(0.1, 0.2),
        area_m2=(0.01, 0.04),
    )
    targets = odss.ParticlePopulation(
        epoch=epoch,
        frame=frame,
        position_x_m=(7_078_136.3,),
        position_y_m=(100.0,),
        position_z_m=(0.0,),
        velocity_x_m_s=(0.0,),
        velocity_y_m_s=(7_500.0,),
        velocity_z_m_s=(0.0,),
        mass_kg=(study.TARGET_PROXY_MASS_KG,),
        area_m2=(study.TARGET_PROXY_AREA_M2,),
    )
    run = odss.MonteCarloRun(study.MASTER_SEED, 0, 0, "nominal", 1)
    fragments = odss.FragmentationResult(
        population=initial_debris,
        characteristic_length_m=(0.02, 0.15),
        area_to_mass_ratio_m2_kg=(0.1, 0.2),
        delta_velocity_x_m_s=(0.0, 0.0),
        delta_velocity_y_m_s=(0.0, 0.0),
        delta_velocity_z_m_s=(0.0, 0.0),
        minimum_characteristic_length_m=0.01,
        generated_minimum_characteristic_length_m=0.01,
        event_kind="explosion",
        random_key=run.random_key("nasa_sbm"),
        backend_seed=odss.nasa_sbm_seed(run.random_key("nasa_sbm")),
    )

    def at_end(population: odss.ParticlePopulation) -> odss.ParticlePopulation:
        final_epoch = odss.Epoch(
            population.epoch.offset_s + study.PRELIMINARY_CAMPAIGN.duration_s,
            population.epoch.reference_epoch,
            population.epoch.time_scale,
        )
        return odss.ParticlePopulation(
            epoch=final_epoch,
            frame=population.frame,
            position_x_m=population.position_x_m,
            position_y_m=population.position_y_m,
            position_z_m=population.position_z_m,
            velocity_x_m_s=population.velocity_x_m_s,
            velocity_y_m_s=population.velocity_y_m_s,
            velocity_z_m_s=population.velocity_z_m_s,
            mass_kg=population.mass_kg,
            area_m2=population.area_m2,
        )

    event = odss.ConjunctionEvent(1, 0, 3_600.0, 500.0, 10_000.0)
    evolution = odss.SsoScreeningResult(
        final_debris=at_end(initial_debris),
        final_targets=at_end(targets),
        conjunctions=(event,),
    )
    monkeypatch.setattr(campaign, "_generate_fragments", lambda *args: fragments)
    monkeypatch.setattr(odss, "propagate_and_screen_sso", lambda *args, **kwargs: evolution)
    monkeypatch.setattr(campaign, "_package_version", lambda *args: "test")
    prepared = campaign.PreparedInputs(
        targets=targets,
        assets=(odss.InputAssetMetadata("catalog.json", "a" * 64, 1),),
        common_epoch=epoch,
        catalog_record_count=1,
        selected_target_count=1,
        target_catalog_ids=(12_345,),
        study_hash="b" * 64,
        code_version="test-commit",
    )
    task = campaign.StudyTask(
        campaign=study.PRELIMINARY_CAMPAIGN,
        scenario=study.SCENARIOS[0],
        variant=study.NOMINAL_VARIANT,
        run_id=0,
        output_path=tmp_path / "run.nc",
    )

    result, fragment_count, conjunction_count = campaign._scientific_result(task, run, prepared)
    odss.write_run_result(task.output_path, result)
    restored = odss.read_run_result(task.output_path)
    summaries = {value.name: value.value for value in restored.summary}

    assert fragment_count == 2
    assert conjunction_count == 1
    assert restored.conjunctions == (event,)
    assert restored.manifest.master_seed == study.MASTER_SEED
    assert summaries["fragment_count_ge_10_cm"] == 1.0
    assert summaries["has_conjunction_le_500_m"] == 1.0
    assert summaries["maneuver_event_count_le_500_m"] == 1.0
    assert summaries["has_maneuver_event_le_500_m"] == 1.0
    assert summaries["censored_minimum_miss_distance_m"] == 500.0
    assert summaries["summed_target_integrated_number_flux_r_10000_m"] > 0.0

    campaign._require_matching_existing_result(task, prepared)
    changed = campaign.PreparedInputs(
        targets=prepared.targets,
        assets=prepared.assets,
        common_epoch=prepared.common_epoch,
        catalog_record_count=prepared.catalog_record_count,
        selected_target_count=prepared.selected_target_count,
        target_catalog_ids=prepared.target_catalog_ids,
        study_hash="c" * 64,
        code_version=prepared.code_version,
    )
    with pytest.raises(ValueError, match="manifest does not match"):
        campaign._require_matching_existing_result(task, changed)
