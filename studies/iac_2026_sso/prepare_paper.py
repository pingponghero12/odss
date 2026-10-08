"""Regenerate manuscript tables and figures from a complete frozen NetCDF campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import xarray as xr

from studies.iac_2026_sso.analyze_preliminary import analyze

_flux_variables = ("number_flux_m2_s", "mass_flux_kg_m2_s", "kinetic_energy_flux_w_m2")
_colors = {500: "#0072B2", 700: "#D55E00", 800: "#009E73"}


@dataclass(frozen=True)
class FamilyData:
    """Small run-level arrays; trajectories and individual encounters are not loaded."""

    summaries: tuple[dict[str, float], ...]
    time_edges_s: np.ndarray
    summed_flux: np.ndarray  # run, observable, time_bin; summed over targets and sizes


def read_campaign(directory: Path) -> tuple[dict, dict[str, FamilyData], dict[str, str]]:
    """Reject missing runs, mixed campaigns, inconsistent grids, and corrupt flux sums."""
    manifest_path = directory / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["family_experiment_hashes"]
    count = manifest["runs_per_family"]
    if count < 2 or manifest["expected_result_files"] != count * len(expected):
        raise ValueError("inconsistent campaign size or fewer than two runs per family")
    paths = sorted(directory.glob("*/*/run_*.nc"))
    expected_paths = {
        directory / family / f"run_{run_id:06d}.nc"
        for family in expected
        for run_id in range(count)
    }
    if set(paths) != expected_paths:
        raise ValueError("campaign run files do not match the complete manifest-defined ensemble")
    hashes = {"campaign_manifest.json": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    families = {}
    common_edges = None
    for family, experiment_hash in sorted(expected.items()):
        summaries, fluxes = [], []
        for run_id in range(count):
            path = directory / family / f"run_{run_id:06d}.nc"
            hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with xr.open_dataset(path, engine="netcdf4") as dataset:
                run_manifest = json.loads(str(dataset.manifest_json.item()))
                if (
                    run_manifest["study_hash"] != manifest["study_hash"]
                    or run_manifest["experiment_hash"] != experiment_hash
                    or run_manifest["run_id"] != run_id
                    or int(dataset.run_id.item()) != run_id
                    or str(dataset.study_id.item()) != manifest["study_hash"]
                    or int(dataset.scenario_id.item()) != run_manifest["scenario_id"]
                    or dataset.sizes["target"] != manifest["selected_target_count"]
                    or float(dataset.flux_sampling_radius_m.item()) != 5000.0
                ):
                    raise ValueError(f"inconsistent run metadata: {path}")
                starts = dataset.time_bin_start_s.values
                ends = dataset.time_bin_end_s.values
                if (
                    starts[0] != 0.0
                    or ends[-1] != manifest["duration_s"]
                    or np.any(ends <= starts)
                    or not np.array_equal(starts[1:], ends[:-1])
                ):
                    raise ValueError(f"invalid time coverage: {path}")
                edges = np.concatenate((starts[:1], ends))
                if common_edges is not None and not np.array_equal(edges, common_edges):
                    raise ValueError(f"mixed reporting time grids: {path}")
                common_edges = edges
                summary = dict(
                    zip(
                        map(str, dataset.summary_name.values),
                        map(float, dataset.summary_value.values),
                        strict=True,
                    )
                )
                arrays = tuple(
                    dataset[name].transpose("target", "time_bin", "size_bin").values
                    for name in _flux_variables
                )
                flux = np.stack([values.sum(axis=(0, 2)) for values in arrays])
                if not np.all(np.isfinite(flux)) or np.any(flux < 0.0):
                    raise ValueError(f"invalid flux: {path}")
                for index, name in enumerate(("number", "mass", "energy")):
                    integral = np.dot(flux[index], ends - starts)
                    if not np.isclose(
                        integral,
                        summary[f"summed_target_integrated_{name}_flux_r_5000_m"],
                        rtol=1e-12,
                        atol=1e-20,
                    ):
                        raise ValueError(f"flux integral disagrees with stored summary: {path}")
                maximum_target_integral = np.max(arrays[0].sum(axis=2) @ (ends - starts))
                for area in (1, 5, 10, 20):
                    probability = -np.expm1(-area * maximum_target_integral)
                    if not np.isclose(
                        probability,
                        summary[f"maximum_model_impact_probability_area_{area}_m2"],
                        rtol=1e-12,
                        atol=1e-20,
                    ):
                        raise ValueError(f"target-level impact indicator disagrees: {path}")
                summaries.append(summary)
                fluxes.append(flux)
        families[family] = FamilyData(tuple(summaries), common_edges, np.stack(fluxes))
    return manifest, families, hashes


def _label(family: str) -> tuple[int, str]:
    altitude, _, event = family.split("/")[0].split("_")
    if int(altitude) not in _colors or event not in ("explosion", "collision"):
        raise ValueError(f"unsupported paper scenario: {family}")
    return int(altitude), event


def write_tables(output: Path, manifest: dict, report: dict, families: dict) -> None:
    """Keep sample size, duration, headline values, and intervals out of hand-written prose."""
    peak = max(
        report["families"],
        key=lambda f: report["families"][f]["continuous"][
            "summed_target_integrated_number_flux_r_5000_m"
        ]["mean"],
    )
    altitude, event = _label(peak)
    peak_summary = report["families"][peak]
    peak_flux = peak_summary["continuous"]["summed_target_integrated_number_flux_r_5000_m"]
    impact_case = max(
        report["families"],
        key=lambda f: report["families"][f]["continuous"][
            "maximum_model_impact_probability_area_10_m2"
        ]["median"],
    )
    impact_altitude, impact_event = _label(impact_case)
    peak_impact = report["families"][impact_case]["continuous"][
        "maximum_model_impact_probability_area_10_m2"
    ]
    peak_binary = peak_summary["binary"]["has_maneuver_event_le_1000_m"]
    collision_counts = [
        f["continuous"]["fragment_count_ge_5_cm"]["mean"]
        for name, f in report["families"].items()
        if _label(name)[1] == "collision"
    ]
    explosion_counts = [
        f["continuous"]["fragment_count_ge_5_cm"]["mean"]
        for name, f in report["families"].items()
        if _label(name)[1] == "explosion"
    ]
    values = {
        "ProductionRunCount": report["files"],
        "ProductionRunsPerScenario": manifest["runs_per_family"],
        "ProductionDurationHours": f"{manifest['duration_s'] / 3600:g}",
        "TargetCount": manifest["selected_target_count"],
        "PeakFluxCase": f"{altitude} km {event}",
        "PeakFluxScaled": f"{peak_flux['mean'] * 1e6:.2f}",
        "PeakImpactCase": f"{impact_altitude} km {impact_event}",
        "PeakImpactScaled": f"{peak_impact['median'] * 1e6:.2f}",
        "PeakOccurrenceSuccesses": peak_binary["successes"],
        "PeakOccurrence": f"{peak_binary['probability']:.2f}",
        "PeakOccurrenceLower": f"{peak_binary['wilson_95_lower']:.2f}",
        "PeakOccurrenceUpper": f"{peak_binary['wilson_95_upper']:.2f}",
        "ExplosionFragmentMeanRange": (
            f"{min(explosion_counts):.0f}--{max(explosion_counts):.0f}"
            if min(explosion_counts) != max(explosion_counts)
            else f"{explosion_counts[0]:.0f}"
        ),
        "CollisionFragmentMeanRange": f"{min(collision_counts):.0f}--{max(collision_counts):.0f}",
    }
    text = "% Generated by studies.iac_2026_sso.prepare_paper; do not edit.\n"
    text += "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items())
    (output / "results.tex").write_text(text)
    rows, radii = [], []
    for family in sorted(report["families"]):
        altitude, event = _label(family)
        summary = report["families"][family]
        continuous = summary["continuous"]
        low = continuous["removed_below_200_km_fraction"]
        flux = continuous["summed_target_integrated_number_flux_r_5000_m"]
        impact = continuous["maximum_model_impact_probability_area_10_m2"]
        binary = summary["binary"]["has_maneuver_event_le_1000_m"]
        rows.append(
            f"{altitude} & {event} & "
            f"{100 * low['median']:.1f} [{100 * low['p05']:.1f}--{100 * low['p95']:.1f}] & "
            f"{1e7 * flux['mean']:.2f} "
            f"[{1e7 * flux['bootstrap_95_lower']:.2f}--{1e7 * flux['bootstrap_95_upper']:.2f}] & "
            f"{1e6 * impact['median']:.2f} "
            f"[{1e6 * impact['p05']:.2f}--{1e6 * impact['p95']:.2f}] & "
            f"{binary['successes']}/{binary['runs']} "
            f"[{binary['wilson_95_lower']:.2f}--{binary['wilson_95_upper']:.2f}] \\\\"
        )
        means = [
            np.mean(
                [
                    s[f"summed_target_integrated_number_flux_r_{radius}_m"]
                    for s in families[family].summaries
                ]
            )
            for radius in (1000, 2000, 5000, 10000)
        ]
        radii.append(
            f"{altitude} & {event} & " + " & ".join(f"{1e7 * v:.2f}" for v in means) + r" \\"
        )
    principal_header = (
        r"\begin{tabular}{@{}rlrrrr@{}}" + "\n"
        r"\toprule" + "\n"
        r"Altitude & Event & Low perigee & Integrated number flux & "
        r"Maximum-target $P_{\rm model}$ & Close approach \\" + "\n"
        r"(km) & & (\%) & ($10^{-7}$ m$^{-2}$) & ($10^{-6}$, $A=10$ m$^2$) & occurrence \\"
        + "\n"
        + r"\midrule"
        + "\n"
    )
    radius_header = (
        r"\begin{tabular}{@{}rlrrrr@{}}" + "\n"
        r"\toprule" + "\n"
        r"Altitude & Event & 1 km & 2 km & 5 km & 10 km \\" + "\n"
        r"\midrule" + "\n"
    )
    table_end = "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    (output / "principal_table.tex").write_text(principal_header + "\n".join(rows) + table_end)
    (output / "radius_table.tex").write_text(radius_header + "\n".join(radii) + table_end)


def write_figures(output: Path, report: dict, families: dict[str, FamilyData]) -> None:
    """Show empirical run spread, not confidence intervals for uncomputed statistics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(output / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.7), layout="constrained")
    for event, offset, marker, color in (
        ("explosion", -9, "o", "#0072B2"),
        ("collision", 9, "s", "#D55E00"),
    ):
        for altitude in _colors:
            stat = report["families"][f"{altitude}_km_{event}/nominal"]["continuous"][
                "removed_below_200_km_fraction"
            ]
            ax.vlines(altitude + offset, 100 * stat["p05"], 100 * stat["p95"], color=color)
            ax.plot(
                altitude + offset,
                100 * stat["median"],
                marker,
                color=color,
                label=event.capitalize() if altitude == 500 else None,
            )
    ax.set(
        xlabel="Breakup altitude (km)",
        ylabel="Final low-perigee fraction (%)",
        xticks=list(_colors),
        ylim=(0, None),
    )
    ax.legend(frameon=False)
    save(fig, "low_perigee")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), layout="constrained")
    for family, data in families.items():
        altitude, event = _label(family)
        edges_h = data.time_edges_s / 3600.0
        mean = data.summed_flux.mean(axis=0)
        for index, ax in enumerate(axes):
            ax.stairs(
                mean[index],
                edges_h,
                color=_colors[altitude],
                linestyle="-" if event == "collision" else "--",
                label=f"{altitude} km {event}",
            )
    for ax, ylabel in zip(
        axes,
        (
            "Number flux (m$^{-2}$ s$^{-1}$)",
            "Mass flux (kg m$^{-2}$ s$^{-1}$)",
            "Energy flux (W m$^{-2}$)",
        ),
        strict=True,
    ):
        ax.set(
            xlabel="Time since breakup (h)", ylabel=ylabel, xlim=(0, edges_h[-1]), ylim=(0, None)
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=8)
    save(fig, "flux")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), layout="constrained")
    for family, data in families.items():
        altitude, event = _label(family)
        areas = np.array([1, 5, 10, 20])
        probabilities = np.array(
            [
                [s[f"maximum_model_impact_probability_area_{area}_m2"] for area in areas]
                for s in data.summaries
            ]
        )
        lower, median, upper = np.quantile(probabilities, [0.05, 0.5, 0.95], axis=0)
        axes[0].plot(
            areas,
            median * 1e6,
            color=_colors[altitude],
            linestyle="-" if event == "collision" else "--",
            label=f"{altitude} km {event}",
        )
        axes[0].fill_between(areas, lower * 1e6, upper * 1e6, color=_colors[altitude], alpha=0.07)
        binary = report["families"][family]["binary"]["has_maneuver_event_le_1000_m"]
        offset = -9 if event == "explosion" else 9
        color = "#0072B2" if event == "explosion" else "#D55E00"
        axes[1].vlines(
            altitude + offset, binary["wilson_95_lower"], binary["wilson_95_upper"], color=color
        )
        axes[1].plot(
            altitude + offset,
            binary["probability"],
            "o" if event == "explosion" else "s",
            color=color,
            label=event.capitalize() if altitude == 500 else None,
        )
    axes[0].set(
        xlabel="Reference area (m$^2$)",
        ylabel=r"Median maximum-target $P_{\rm model}$ ($\times10^{-6}$)",
        xticks=[1, 5, 10, 20],
        ylim=(0, None),
    )
    axes[1].set(
        xlabel="Breakup altitude (km)",
        ylabel="Run-level close-approach occurrence",
        xticks=list(_colors),
        ylim=(0, 1),
    )
    axes[1].legend(frameon=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=8)
    save(fig, "operations")


def prepare(directory: Path, output: Path) -> dict:
    """Regenerate auditable paper assets without changing or combining raw campaigns."""
    manifest, families, hashes = read_campaign(directory)
    required = {
        f"{altitude}_km_{event}/nominal"
        for altitude in _colors
        for event in ("explosion", "collision")
    }
    if set(families) != required:
        raise ValueError("paper requires all six nominal altitude/event scenarios")
    report = analyze(directory, expected_runs=manifest["runs_per_family"])
    output.mkdir(parents=True, exist_ok=True)
    write_tables(output, manifest, report, families)
    write_figures(output / "figures", report, families)
    (output / "analysis.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    provenance = {
        "schema": "odss.sso_paper_assets.v1",
        "campaign_manifest": manifest,
        "input_sha256": hashes,
        "mean_interval": "95% percentile bootstrap, 10000 fixed-seed run-level resamples",
        "spread": "empirical run-level 5th and 95th percentiles, linear interpolation",
        "binary_interval": "95% Wilson score interval",
        "flux_figure": "mean across runs; summed over targets and generated size bins",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_software": {
            name: version(name) for name in ("numpy", "xarray", "netCDF4", "matplotlib", "odss")
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/paper_sso/generated"))
    args = parser.parse_args()
    report = prepare(args.directory, args.output)
    print(f"Prepared paper assets from {report['files']} runs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
