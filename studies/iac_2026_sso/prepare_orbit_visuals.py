"""Recover one seeded endpoint snapshot and visualize the frozen SSO target geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

import odss
from studies.iac_2026_sso import campaign, inputs, study


def positions_m(population: odss.ParticlePopulation) -> np.ndarray:
    """Return Earth-centred EME2000 positions with particle order preserved."""
    return np.column_stack(
        (population.position_x_m, population.position_y_m, population.position_z_m)
    )


def orbital_coordinates(population: odss.ParticlePopulation) -> tuple[np.ndarray, ...]:
    """Compute osculating semi-major-axis altitude, inclination, and node longitude."""
    position = positions_m(population)
    velocity = np.column_stack(
        (population.velocity_x_m_s, population.velocity_y_m_s, population.velocity_z_m_s)
    )
    angular_momentum = np.cross(position, velocity)
    momentum_norm = np.linalg.norm(angular_momentum, axis=1)
    if np.any(momentum_norm == 0.0):
        raise ValueError("orbital plane is undefined for zero angular momentum")
    inclination_deg = np.degrees(
        np.arccos(np.clip(angular_momentum[:, 2] / momentum_norm, -1.0, 1.0))
    )
    node_x, node_y = -angular_momentum[:, 1], angular_momentum[:, 0]
    if np.any(np.hypot(node_x, node_y) < 1e-12 * momentum_norm):
        raise ValueError("ascending node is undefined for an equatorial orbit")
    raan_deg = np.degrees(np.arctan2(node_y, node_x)) % 360.0
    constants = odss.OrbitalDecaySpec()
    energy = 0.5 * np.sum(velocity**2, axis=1)
    energy -= constants.gravitational_parameter_m3_s2 / np.linalg.norm(position, axis=1)
    if np.any(energy >= 0.0):
        raise ValueError("target orbital distribution requires bound orbits")
    semimajor_axis_m = -constants.gravitational_parameter_m3_s2 / (2.0 * energy)
    altitude_m = semimajor_axis_m - constants.earth_radius_m
    return altitude_m, inclination_deg, raan_deg


def recover(directory: Path, family: str, run_id: int) -> xr.Dataset:
    """Rerun an archived address with matching inputs and check its scientific summaries."""
    manifest = json.loads((directory / "campaign_manifest.json").read_text())
    source = directory / family / f"run_{run_id:06d}.nc"
    with xr.open_dataset(source, engine="netcdf4") as dataset:
        run_manifest = json.loads(str(dataset.manifest_json.item()))
        stored = dict(
            zip(
                map(str, dataset.summary_name.values),
                map(float, dataset.summary_value.values),
                strict=True,
            )
        )
        stored_conjunctions = dataset.sizes["conjunction_event"]
    snapshot = inputs.load_catalog_snapshot(
        Path(manifest["catalog_path"]).parent / "catalog_snapshot.json"
    )
    prepared = campaign._prepare_inputs(
        campaign.ExecutionInputs(
            catalog_path=snapshot.catalog_path,
            catalog_source_uri=snapshot.source_uri,
            catalog_acquired_at_utc=snapshot.acquired_at_utc,
            common_epoch_utc=manifest["common_epoch_utc"],
            code_version=manifest["code_version"],
            catalog_provenance_paths=snapshot.provenance_paths,
            output_directory=source.parent,
            workers=1,
            resume=False,
        )
    )
    if prepared.study_hash != manifest["study_hash"]:
        raise ValueError("current scientific inputs/configuration do not match the archived study")
    scenario = next(s for s in study.SCENARIOS if s.scenario_id == run_manifest["scenario_id"])
    task = campaign.StudyTask(
        study.PRODUCTION_CAMPAIGN, scenario, study.NOMINAL_VARIANT, run_id, source
    )
    if (
        f"{scenario.name}/nominal" != family
        or campaign._experiment_hash(task) != run_manifest["experiment_hash"]
        or manifest["duration_s"] != task.campaign.duration_s
    ):
        raise ValueError("snapshot scenario/model/duration does not match the archived run")
    campaign._set_worker_thread_limits()
    run = campaign._run_address(task)
    fragments = campaign._generate_fragments(task, run, prepared.common_epoch)
    if (
        len(fragments) != stored["fragment_count"]
        or fragments.backend_seed != stored["nasa_sbm_seed"]
    ):
        raise ValueError("regenerated fragment population does not match the archived seed/count")
    print(
        f"Propagating {len(fragments)} fragments and {len(prepared.targets)} targets; one thread",
        flush=True,
    )
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
    for final_population in (evolution.final_debris, evolution.final_targets):
        if final_population.frame != prepared.targets.frame or not np.isclose(
            odss.elapsed_time_s(prepared.common_epoch, final_population.epoch),
            task.campaign.duration_s,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("snapshot frame or final epoch does not match the requested endpoint")
    decay = odss.evaluate_decay(fragments.population, evolution.final_debris)
    checks = {
        "retained_fraction": decay.retained_fraction,
        "removal_fraction": decay.removal_fraction,
        "reentry_fraction": decay.reentry_fraction,
        "final_escaped_fraction": decay.escape_fraction,
    }
    if len(evolution.conjunctions) != stored_conjunctions:
        raise ValueError("rerun conjunction count disagrees with the archived run")
    for name, value in checks.items():
        if not np.isclose(value, stored[name], rtol=0.0, atol=1e-14):
            raise ValueError(f"rerun orbital classification disagrees: {name}")
    initial_altitude, initial_inclination, initial_raan = orbital_coordinates(prepared.targets)
    metadata = {
        "source_run_manifest": run_manifest,
        "source_run_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "family": family,
        "run_id": run_id,
        "duration_s": task.campaign.duration_s,
        "frame": "EME2000",
        "initial_epoch_utc": odss.epoch_to_iso(prepared.targets.epoch, "UTC"),
        "final_epoch_utc": odss.epoch_to_iso(evolution.final_targets.epoch, "UTC"),
        "checks": checks,
        "conjunction_count": len(evolution.conjunctions),
        "rerun_software": [
            {"name": item.name, "version": item.version}
            for item in campaign._run_manifest(task, run, prepared).software
        ],
        "note": "Seeded rerun; endpoint positions were not retained in the original run file.",
    }
    result = xr.Dataset(
        {
            "initial_target_position_m": (("target", "component"), positions_m(prepared.targets)),
            "final_target_position_m": (
                ("target", "component"),
                positions_m(evolution.final_targets),
            ),
            "final_debris_position_m": (
                ("debris", "component"),
                positions_m(evolution.final_debris),
            ),
            "initial_target_semimajor_axis_altitude_m": ("target", initial_altitude),
            "initial_target_inclination_deg": ("target", initial_inclination),
            "initial_target_raan_deg": ("target", initial_raan),
            "fragment_characteristic_length_m": (
                "debris",
                np.asarray(fragments.characteristic_length_m),
            ),
            "metadata_json": xr.DataArray(json.dumps(metadata, sort_keys=True)),
        },
        coords={
            "component": ["x", "y", "z"],
            "target_catalog_id": ("target", np.asarray(prepared.target_catalog_ids)),
        },
        attrs={"schema": "odss.sso_orbit_visuals.v1", "frame": "EME2000"},
    )
    for name in result.data_vars:
        if name.endswith("_m"):
            result[name].attrs["units"] = "m"
        elif name.endswith("_deg"):
            result[name].attrs["units"] = "degree"
    return result


def render(snapshot: xr.Dataset, output: Path) -> None:
    """Render only recorded coordinates; marker sizes do not represent physical sizes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(str(snapshot.metadata_json.item()))
    plt.rcParams.update({"font.size": 9})
    debris = snapshot.final_debris_position_m.values / 1000.0
    targets = snapshot.final_target_position_m.values / 1000.0
    radius_km = odss.OrbitalDecaySpec().earth_radius_m / 1000.0
    fig = plt.figure(figsize=(6.8, 4.7), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    longitude = np.linspace(0.0, 2.0 * np.pi, 55)
    latitude = np.linspace(-np.pi / 2.0, np.pi / 2.0, 29)
    x = radius_km * np.outer(np.cos(longitude), np.cos(latitude))
    y = radius_km * np.outer(np.sin(longitude), np.cos(latitude))
    z = radius_km * np.outer(np.ones_like(longitude), np.sin(latitude))
    ax.plot_surface(x, y, z, color="#a9cbdc", alpha=0.32, linewidth=0, rasterized=True)
    ax.plot_wireframe(x, y, z, rstride=5, cstride=4, color="#527b92", linewidth=0.35, alpha=0.3)
    ax.scatter(
        *debris.T,
        s=4,
        color="#c43d3d",
        alpha=0.55,
        depthshade=False,
        label=f"Debris fragments ({len(debris)})",
    )
    ax.scatter(
        *targets.T,
        s=18,
        marker="^",
        color="#153f72",
        edgecolors="white",
        linewidths=0.3,
        depthshade=False,
        label=f"Selected EO targets ({len(targets)})",
    )
    extent = max(radius_km, float(np.abs(debris).max()), float(np.abs(targets).max())) * 1.06
    ax.set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        zlim=(-extent, extent),
        xlabel="EME2000 x (km)",
        ylabel="EME2000 y (km)",
        zlabel="EME2000 z (km)",
    )
    ax.set_box_aspect((1, 1, 1))
    tick_km = 2000.0 * np.floor(extent / 2000.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticks([-tick_km, 0.0, tick_km])
    ax.tick_params(labelsize=8)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=36)
    ax.set_title(
        f"{metadata['family'].split('/')[0].replace('_', ' ')}"
        f" · run {metadata['run_id']} · {metadata['duration_s'] / 3600:g} h"
    )
    ax.legend(loc="upper left", frameon=False, markerscale=1.5)
    fig.savefig(output / "debris_snapshot.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / "debris_snapshot.png", dpi=200)
    plt.close(fig)

    altitude = snapshot.initial_target_semimajor_axis_altitude_m.values / 1000.0
    inclination = snapshot.initial_target_inclination_deg.values
    raan = snapshot.initial_target_raan_deg.values
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), layout="constrained")
    scatter = axes[0].scatter(
        raan, altitude, c=inclination, cmap="viridis", s=18, edgecolors="white", linewidths=0.25
    )
    axes[0].set(
        xlabel="Ascending-node longitude (deg)",
        ylabel="Semi-major-axis altitude (km)",
        xlim=(0, 360),
        xticks=[0, 90, 180, 270, 360],
    )
    fig.colorbar(scatter, ax=axes[0], label="Inclination (deg)", shrink=0.9)
    axes[1].hist(
        altitude,
        bins=np.arange(200, max(1200, altitude.max() + 50), 50),
        color="#287a9d",
        edgecolor="white",
    )
    axes[1].set(xlabel="Semi-major-axis altitude (km)", ylabel="Selected target count")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(
        output / "target_orbit_distribution.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    fig.savefig(output / "target_orbit_distribution.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--family", default="500_km_explosion/nominal")
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("docs/paper_sso/generated"))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    snapshot_path = args.output / "orbit_snapshot.nc"
    if args.render_only:
        with xr.open_dataset(snapshot_path, engine="netcdf4") as opened:
            snapshot = opened.load()
    else:
        snapshot = recover(args.directory, args.family, args.run_id)
        args.output.mkdir(parents=True, exist_ok=True)
        snapshot.to_netcdf(snapshot_path, engine="netcdf4")
    render(snapshot, args.output / "figures")
    print(f"Orbit visuals saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
