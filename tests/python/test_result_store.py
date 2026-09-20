import math
from pathlib import Path

import pytest
import xarray as xr

import odss

_hash_a = "a" * 64
_hash_b = "b" * 64


def _run_result() -> odss.RunResult:
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
        spec=odss.FluxSpec(
            sampling_radius_m=10.0,
            time_bin_edges_s=(0.0, 10.0),
            size_bin_edges_m=(0.0, math.inf),
        ),
        target_count=2,
        bins=(
            odss.FluxBin(0, 0.0, 10.0, 0.0, math.inf, 1, 0.5, 0.25, 1.0),
            odss.FluxBin(1, 0.0, 10.0, 0.0, math.inf, 0, 0.0, 0.0, 0.0),
        ),
    )
    maneuver = odss.ManeuverDemandResult(
        spec=odss.ManeuverDemandSpec(
            trackability_size_threshold_m=0.1,
            miss_distance_threshold_m=1_000.0,
            duration_s=10.0,
        ),
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
            odss.ScalarRunResult("escaped_fraction", 0.25, "1"),
        ),
        conjunctions=events,
        flux=flux,
        maneuver_demand=maneuver,
    )


def test_xarray_dataset_uses_named_scientific_dimensions_and_units() -> None:
    dataset = odss.run_result_dataset(_run_result())

    assert dataset.attrs["schema"] == "odss.run_result.v1"
    assert dataset.coords["study_id"].item() == _hash_b
    assert dataset.coords["scenario_id"].item() == "4"
    assert dataset.coords["run_id"].item() == "9"
    assert dataset["number_flux_m2_s"].dims == ("target", "time_bin", "size_bin")
    assert dataset["number_flux_m2_s"].shape == (2, 1, 1)
    assert dataset["number_flux_m2_s"].attrs["units"] == "m-2 s-1"
    assert dataset["conjunction_tca_s"].dims == ("conjunction_event",)
    assert dataset["maneuver_events_per_target"].dims == ("target",)
    assert not any("trajectory" in name for name in dataset.variables)


def test_netcdf_run_result_round_trip(tmp_path: Path) -> None:
    expected = _run_result()
    path = tmp_path / "run_000009.nc"

    odss.write_run_result(path, expected)

    assert odss.read_run_result(path) == expected
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        assert dataset.attrs["schema"] == "odss.run_result.v1"
        assert dataset["manifest_json"].item() == odss.canonical_manifest(expected.manifest)
        assert dataset["kinetic_energy_flux_w_m2"].attrs["units"] == "W m-2"


def test_existing_run_file_is_not_silently_overwritten(tmp_path: Path) -> None:
    result = _run_result()
    path = tmp_path / "run_000009.nc"
    odss.write_run_result(path, result)

    with pytest.raises(FileExistsError):
        odss.write_run_result(path, result)

    assert odss.read_run_result(path) == result


def test_empty_selected_outputs_round_trip(tmp_path: Path) -> None:
    expected = odss.RunResult(manifest=_run_result().manifest)
    path = tmp_path / "run_000009.nc"

    odss.write_run_result(path, expected)

    assert odss.read_run_result(path) == expected
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        assert dataset.sizes["summary"] == 0
        assert dataset.sizes["conjunction_event"] == 0
        assert dataset["has_flux"].item() == 0


def test_dataset_identity_must_match_canonical_manifest() -> None:
    dataset = odss.run_result_dataset(_run_result()).assign_coords(run_id="10")

    with pytest.raises(ValueError, match="does not match"):
        odss.run_result_from_dataset(dataset)


def test_result_storage_inputs_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        odss.ScalarRunResult("invalid", math.inf, "1")
    with pytest.raises(ValueError, match="unique"):
        odss.RunResult(
            manifest=_run_result().manifest,
            summary=(
                odss.ScalarRunResult("same", 1.0, "1"),
                odss.ScalarRunResult("same", 2.0, "1"),
            ),
        )
    with pytest.raises(ValueError, match="NetCDF"):
        odss.write_run_result(tmp_path / "result.db", _run_result())
    with pytest.raises(FileNotFoundError):
        odss.read_run_result(tmp_path / "missing.nc")
