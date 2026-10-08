"""Synthetic tests for manuscript statistics, integrity checks, and campaign replacement."""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from studies.iac_2026_sso import analyze_preliminary, prepare_paper


def _campaign(directory: Path, runs: int = 3) -> None:
    families = {
        f"{altitude}_km_{event}/nominal": f"hash-{altitude}-{event}"
        for altitude in (500, 700, 800)
        for event in ("explosion", "collision")
    }
    manifest = {
        "study_hash": "synthetic-study",
        "family_experiment_hashes": families,
        "runs_per_family": runs,
        "duration_s": 30.0,
        "expected_result_files": len(families) * runs,
        "selected_target_count": 2,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "campaign_manifest.json").write_text(json.dumps(manifest))
    for family, experiment_hash in families.items():
        for run in range(runs):
            number = np.zeros((2, 2, 3))
            number[:, :, 1:] = (run + 1) * 1e-10
            summary = {name: 0.0 for name in analyze_preliminary._reported_metrics}
            summary.update(
                reentry_fraction=0.1,
                removal_fraction=0.2,
                retained_fraction=0.7,
                fragment_count_ge_5_cm=100.0,
                has_conjunction_le_1000_m=float(run == 0),
                has_maneuver_event_le_1000_m=float(run == 0),
                summed_target_integrated_number_flux_r_5000_m=(run + 1) * 1.2e-8,
                summed_target_integrated_mass_flux_r_5000_m=(run + 1) * 2.4e-8,
                summed_target_integrated_energy_flux_r_5000_m=(run + 1) * 3.6e-8,
            )
            for area in (1, 5, 10, 20):
                summary[f"maximum_model_impact_probability_area_{area}_m2"] = float(
                    -np.expm1(-area * (run + 1) * 6e-9)
                )
            for radius in (1000, 2000, 10000):
                summary[f"summed_target_integrated_number_flux_r_{radius}_m"] = 1e-8
            dataset = xr.Dataset(
                {
                    "manifest_json": xr.DataArray(
                        json.dumps(
                            {
                                "study_hash": manifest["study_hash"],
                                "experiment_hash": experiment_hash,
                                "run_id": run,
                                "scenario_id": 0,
                            }
                        )
                    ),
                    "summary_name": ("summary", list(summary)),
                    "summary_value": ("summary", list(summary.values())),
                    "flux_sampling_radius_m": xr.DataArray(5000.0),
                    "time_bin_start_s": ("time_bin", [0.0, 10.0]),
                    "time_bin_end_s": ("time_bin", [10.0, 30.0]),
                    **{
                        name: (("target", "time_bin", "size_bin"), number * (index + 1))
                        for index, name in enumerate(prepare_paper._flux_variables)
                    },
                },
                coords={"run_id": str(run), "study_id": "synthetic-study", "scenario_id": "0"},
            )
            path = directory / family / f"run_{run:06d}.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_netcdf(path, engine="netcdf4")


def _rewrite(path: Path, modify) -> None:
    with xr.open_dataset(path, engine="netcdf4") as opened:
        dataset = opened.load()
    modify(dataset)
    dataset.to_netcdf(path, engine="netcdf4")


def test_target_and_size_sums_and_unequal_interval_integrals(tmp_path: Path) -> None:
    _campaign(tmp_path)
    _, families, hashes = prepare_paper.read_campaign(tmp_path)
    data = families["500_km_explosion/nominal"]
    assert data.summed_flux.shape == (3, 3, 2)
    np.testing.assert_allclose(data.summed_flux[0, 0], [4e-10, 4e-10])
    np.testing.assert_array_equal(data.time_edges_s, [0.0, 10.0, 30.0])
    assert len(hashes) == 19
    assert all(len(value) == 64 for value in hashes.values())


def test_missing_entire_family_is_rejected(tmp_path: Path) -> None:
    _campaign(tmp_path)
    for path in (tmp_path / "500_km_explosion/nominal").glob("*.nc"):
        path.unlink()
    with pytest.raises(ValueError, match="complete manifest-defined"):
        prepare_paper.read_campaign(tmp_path)


@pytest.mark.parametrize(
    "fault", ["study", "run", "duration", "grid", "integral", "mass", "impact", "nan"]
)
def test_mixed_or_inconsistent_results_are_rejected(tmp_path: Path, fault: str) -> None:
    _campaign(tmp_path)

    def modify(dataset):
        if fault in ("study", "run"):
            metadata = json.loads(str(dataset.manifest_json.item()))
            metadata["study_hash" if fault == "study" else "run_id"] = "wrong"
            dataset["manifest_json"] = xr.DataArray(json.dumps(metadata))
        elif fault == "duration":
            dataset["time_bin_end_s"].values[-1] = 29.0
        elif fault == "grid":
            dataset["time_bin_start_s"].values[-1] = 11.0
        elif fault == "integral":
            dataset["number_flux_m2_s"].values[:] = 0.0
        elif fault == "mass":
            dataset["mass_flux_kg_m2_s"].values[:] = 0.0
        elif fault == "impact":
            index = list(dataset.summary_name.values).index(
                "maximum_model_impact_probability_area_10_m2"
            )
            dataset["summary_value"].values[index] = 0.5
        else:
            dataset["number_flux_m2_s"].values[:] = np.nan

    _rewrite(tmp_path / "500_km_explosion/nominal/run_000000.nc", modify)
    with pytest.raises(ValueError):
        prepare_paper.read_campaign(tmp_path)


def test_statistics_and_generated_counts_follow_selected_campaign(tmp_path: Path) -> None:
    directory, output = tmp_path / "data", tmp_path / "paper"
    _campaign(directory, runs=4)
    manifest, families, _ = prepare_paper.read_campaign(directory)
    report = analyze_preliminary.analyze(directory, expected_runs=4)
    stats = report["families"]["500_km_explosion/nominal"]
    assert stats["continuous"]["removed_below_200_km_fraction"]["median"] == pytest.approx(0.3)
    assert stats["continuous"]["summed_target_integrated_number_flux_r_5000_m"][
        "mean"
    ] == pytest.approx(3e-8)
    assert stats["binary"]["has_maneuver_event_le_1000_m"]["probability"] == 0.25
    output.mkdir()
    prepare_paper.write_tables(output, manifest, report, families)
    first = (output / "results.tex").read_bytes()
    assert b"\\newcommand{\\ProductionRunCount}{24}" in first
    assert b"\\newcommand{\\ProductionRunsPerScenario}{4}" in first
    assert "1/4" in (output / "principal_table.tex").read_text()
    prepare_paper.write_tables(output, manifest, report, families)
    assert (output / "results.tex").read_bytes() == first


def test_figures_and_provenance_generated_from_synthetic_inputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    directory, output = tmp_path / "data", tmp_path / "paper"
    _campaign(directory)
    report = prepare_paper.prepare(directory, output)
    assert report["files"] == 18
    for name in ("low_perigee", "flux", "operations"):
        assert (output / "figures" / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["campaign_manifest"]["runs_per_family"] == 3
    assert len(provenance["input_sha256"]) == 19
