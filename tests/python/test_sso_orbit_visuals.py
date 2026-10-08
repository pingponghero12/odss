"""Known orbital geometry and snapshot-integrity regression tests."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import odss
from studies.iac_2026_sso import prepare_orbit_visuals as visuals


def _population(inclination_deg=98.0, raan_deg=40.0, speed_factor=1.0, radial=False, offset_s=0.0):
    radius = odss.OrbitalDecaySpec().earth_radius_m + 700_000.0
    speed = math.sqrt(odss.OrbitalDecaySpec().gravitational_parameter_m3_s2 / radius)
    node, inclination = math.radians(raan_deg), math.radians(inclination_deg)
    position = np.array([radius * math.cos(node), radius * math.sin(node), 0.0])
    velocity = (
        speed
        * speed_factor
        * np.array(
            [
                -math.sin(node) * math.cos(inclination),
                math.cos(node) * math.cos(inclination),
                math.sin(inclination),
            ]
        )
    )
    if radial:
        velocity = position * 0.001
    return odss.ParticlePopulation(
        epoch=odss.Epoch(offset_s, "2026-09-21T17:51:42.469497", "UTC"),
        frame=odss.ReferenceFrame("EME2000"),
        position_x_m=[position[0]],
        position_y_m=[position[1]],
        position_z_m=[position[2]],
        velocity_x_m_s=[velocity[0]],
        velocity_y_m_s=[velocity[1]],
        velocity_z_m_s=[velocity[2]],
        mass_kg=[1000.0],
        area_m2=[10.0],
    )


@pytest.mark.parametrize("raan", [0.0, 40.0, 180.0, 270.0, 359.0])
def test_circular_orbit_altitude_plane_and_ascending_node(raan: float) -> None:
    population = _population(raan_deg=raan)
    altitude, inclination, node = visuals.orbital_coordinates(population)
    np.testing.assert_allclose(altitude, [700_000.0], atol=1e-8)
    np.testing.assert_allclose(inclination, [98.0], atol=1e-12)
    np.testing.assert_allclose(node, [raan], atol=1e-12)
    assert visuals.positions_m(population).shape == (1, 3)


@pytest.mark.parametrize("kind", ["equatorial", "radial", "unbound"])
def test_undefined_or_unbound_target_geometry_is_rejected(kind: str) -> None:
    population = _population(
        inclination_deg=0.0 if kind == "equatorial" else 98.0,
        radial=kind == "radial",
        speed_factor=2.0 if kind == "unbound" else 1.0,
    )
    with pytest.raises(ValueError):
        visuals.orbital_coordinates(population)


class _Fragments(SimpleNamespace):
    def __len__(self):
        return len(self.population)


def _mock_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault="") -> None:
    source = tmp_path / "500_km_explosion/nominal/run_000000.nc"
    source.parent.mkdir(parents=True)
    summary = {
        "fragment_count": 1.0,
        "nasa_sbm_seed": 7.0,
        "retained_fraction": 0.9 if fault == "classification" else 1.0,
        "removal_fraction": 0.0,
        "reentry_fraction": 0.0,
        "final_escaped_fraction": 0.0,
    }
    xr.Dataset(
        {
            "manifest_json": xr.DataArray(
                json.dumps({"scenario_id": 0, "experiment_hash": "hash"})
            ),
            "summary_name": ("summary", list(summary)),
            "summary_value": ("summary", list(summary.values())),
        },
        coords={"conjunction_event": []},
    ).to_netcdf(source, engine="netcdf4")
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "catalog_path": str(tmp_path / "input/catalog.json"),
                "common_epoch_utc": "2026-09-21T17:51:42.469497",
                "code_version": "source",
                "study_hash": "study",
                "duration_s": 43200.0,
            }
        )
    )
    population = _population()
    snapshot = SimpleNamespace(
        catalog_path=tmp_path / "input/catalog.json",
        source_uri="fixture",
        acquired_at_utc="2026-09-21T17:51:42.469497",
        provenance_paths=(),
    )
    monkeypatch.setattr(visuals.inputs, "load_catalog_snapshot", lambda path: snapshot)
    prepared = SimpleNamespace(
        targets=population,
        target_catalog_ids=(123456,),
        common_epoch=population.epoch,
        study_hash="different" if fault == "inputs" else "study",
    )
    monkeypatch.setattr(visuals.campaign, "_prepare_inputs", lambda config: prepared)
    monkeypatch.setattr(visuals.campaign, "_experiment_hash", lambda task: "hash")
    monkeypatch.setattr(visuals.campaign, "_set_worker_thread_limits", lambda: None)
    fragments = _Fragments(
        population=population,
        backend_seed=8 if fault == "seed" else 7,
        characteristic_length_m=(0.05,),
    )
    monkeypatch.setattr(visuals.campaign, "_generate_fragments", lambda *args: fragments)
    final_population = _population(offset_s=0.0 if fault == "epoch" else 43200.0)
    monkeypatch.setattr(
        odss,
        "propagate_and_screen_sso",
        lambda *args, **kwargs: SimpleNamespace(
            final_debris=final_population, final_targets=final_population, conjunctions=()
        ),
    )
    monkeypatch.setattr(
        visuals.campaign,
        "_run_manifest",
        lambda *args: SimpleNamespace(software=(odss.SoftwareMetadata("fixture", "1"),)),
    )


def test_rerun_snapshot_preserves_coordinates_ids_units_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_run(tmp_path, monkeypatch)
    snapshot = visuals.recover(tmp_path, "500_km_explosion/nominal", 0)
    assert snapshot.sizes == {"target": 1, "debris": 1, "component": 3}
    assert snapshot.target_catalog_id.values.tolist() == [123456]
    assert snapshot.final_debris_position_m.attrs["units"] == "m"
    assert snapshot.initial_target_raan_deg.attrs["units"] == "degree"
    assert json.loads(str(snapshot.metadata_json.item()))["checks"]["final_escaped_fraction"] == 0.0
    path = tmp_path / "snapshot.nc"
    snapshot.to_netcdf(path, engine="netcdf4")
    with xr.open_dataset(path, engine="netcdf4") as restored:
        xr.testing.assert_equal(snapshot, restored)


@pytest.mark.parametrize("fault", ["inputs", "seed", "classification", "epoch"])
def test_rerun_with_mismatched_provenance_or_outputs_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    _mock_run(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError):
        visuals.recover(tmp_path, "500_km_explosion/nominal", 0)


def test_both_figures_render_from_recorded_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    _mock_run(tmp_path, monkeypatch)
    snapshot = visuals.recover(tmp_path, "500_km_explosion/nominal", 0)
    output = tmp_path / "figures"
    visuals.render(snapshot, output)
    for name in ("debris_snapshot", "target_orbit_distribution"):
        assert (output / f"{name}.pdf").read_bytes().startswith(b"%PDF")
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")
