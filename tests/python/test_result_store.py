import math
import sqlite3

import pytest

import odss

_hash_a = "a" * 64
_hash_b = "b" * 64


def run_result() -> odss.RunResult:
    manifest = odss.RunManifest(
        experiment_hash=_hash_a,
        study_hash=_hash_b,
        master_seed=2026,
        scenario_id=4,
        run_id=9,
        input_assets=(odss.InputAssetMetadata("catalog", "c" * 64, 12),),
        software=(odss.SoftwareMetadata("odss", "0.1.0"),),
    )
    events = (
        odss.ConjunctionEvent(0, 0, 2.0, 5.0, 10.0),
        odss.ConjunctionEvent(1, 1, 8.0, 6.0, 20.0),
    )
    flux = odss.FluxResult(
        odss.FluxSpec(10.0, (0.0, 10.0), (0.0, math.inf)),
        2,
        (
            odss.FluxBin(0, 0.0, 10.0, 0.0, math.inf, 1, 1.0, 2.0, 3.0),
            odss.FluxBin(1, 0.0, 10.0, 0.0, math.inf, 0, 0.0, 0.0, 0.0),
        ),
    )
    maneuver = odss.ManeuverDemandResult(
        spec=odss.ManeuverDemandSpec(0.1, 1_000.0, 10.0),
        actionable_encounters=(events[0],),
        affected_target_indices=(0,),
        events_per_target=(1, 0),
        event_rate_s=0.1,
        probability_at_least_one_actionable_encounter=1.0 - math.exp(-1.0),
    )
    return odss.RunResult(
        manifest=manifest,
        summary=(
            odss.ScalarRunResult("fragment_count", 2.0, "1"),
            odss.ScalarRunResult("escaped_fraction", 0.0, "1"),
        ),
        conjunctions=events,
        flux=flux,
        maneuver_demand=maneuver,
    )


def test_event_oriented_result_round_trip(tmp_path: object) -> None:
    path = tmp_path / "results.sqlite"
    expected = run_result()

    odss.write_run_result(path, expected)
    identities = odss.list_run_results(path)
    actual = odss.read_run_result(path, identities[0])

    assert identities == (odss.ResultIdentity(_hash_b, 4, 9),)
    assert actual == expected


def test_tables_carry_common_identity_and_no_trajectory_table(tmp_path: object) -> None:
    path = tmp_path / "results.sqlite"
    odss.write_run_result(path, run_result())

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in (
            "run_summaries",
            "conjunctions",
            "target_flux",
            "maneuver_demand",
        ):
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            assert {"study_id", "scenario_id", "run_id"} <= columns
        assert "trajectories" not in tables


def test_duplicate_identity_is_not_silently_overwritten(tmp_path: object) -> None:
    path = tmp_path / "results.sqlite"
    expected = run_result()
    odss.write_run_result(path, expected)

    with pytest.raises(ValueError, match="already exists"):
        odss.write_run_result(path, expected)

    assert odss.read_run_result(path, expected.identity) == expected


def test_empty_selected_outputs_round_trip_and_missing_run(tmp_path: object) -> None:
    path = tmp_path / "results.sqlite"
    manifest = odss.RunManifest(_hash_a, 1, 2, 3)
    expected = odss.RunResult(manifest)

    odss.write_run_result(path, expected)

    assert odss.read_run_result(path, expected.identity) == expected
    with pytest.raises(KeyError, match="not found"):
        odss.read_run_result(path, odss.ResultIdentity(_hash_a, 2, 4))

    missing_path = tmp_path / "missing.sqlite"
    with pytest.raises(KeyError, match="not found"):
        odss.read_run_result(missing_path, expected.identity)
    assert not missing_path.exists()


def test_result_values_are_validated() -> None:
    with pytest.raises(ValueError, match="finite"):
        odss.ScalarRunResult("flux", math.inf, "m^-2 s^-1")
    with pytest.raises(ValueError, match="unique"):
        odss.RunResult(
            odss.RunManifest(_hash_a, 1, 2, 3),
            summary=(
                odss.ScalarRunResult("count", 1.0, "1"),
                odss.ScalarRunResult("count", 2.0, "1"),
            ),
        )
