"""Xarray-native NetCDF persistence for selected scientific run results."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import xarray as xr

from .conjunction import ConjunctionEvent
from .flux import FluxBin, FluxResult, FluxSpec
from .maneuver import ManeuverDemandResult, ManeuverDemandSpec
from .provenance import (
    InputAssetMetadata,
    RunManifest,
    SoftwareMetadata,
    canonical_manifest,
)
from .rng import _require_uint64

_schema = "odss.run_result.v1"


@dataclass(frozen=True, slots=True)
class ResultIdentity:
    """Identity shared by a run file and every scientific value it contains."""

    study_id: str
    scenario_id: int
    run_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise ValueError("study_id must not be empty")
        _require_uint64(self.scenario_id, "scenario_id")
        _require_uint64(self.run_id, "run_id")


@dataclass(frozen=True, slots=True)
class ScalarRunResult:
    """One named finite scalar and its physical unit."""

    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must not be empty")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise ValueError("value must be a finite number")
        if not math.isfinite(self.value):
            raise ValueError("value must be a finite number")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must not be empty")


@dataclass(frozen=True, slots=True)
class RunResult:
    """Selected event-oriented outputs for one reproducible realization."""

    manifest: RunManifest
    summary: tuple[ScalarRunResult, ...] = ()
    conjunctions: tuple[ConjunctionEvent, ...] = ()
    flux: FluxResult | None = None
    maneuver_demand: ManeuverDemandResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RunManifest):
            raise TypeError("manifest must be a RunManifest")
        summary = tuple(self.summary)
        object.__setattr__(self, "conjunctions", tuple(self.conjunctions))
        if any(not isinstance(value, ScalarRunResult) for value in summary):
            raise TypeError("summary must contain only ScalarRunResult values")
        object.__setattr__(self, "summary", tuple(sorted(summary, key=lambda value: value.name)))
        if len({value.name for value in self.summary}) != len(self.summary):
            raise ValueError("summary result names must be unique")
        if any(not isinstance(value, ConjunctionEvent) for value in self.conjunctions):
            raise TypeError("conjunctions must contain only ConjunctionEvent values")
        if self.flux is not None and not isinstance(self.flux, FluxResult):
            raise TypeError("flux must be a FluxResult or None")
        if self.maneuver_demand is not None and not isinstance(
            self.maneuver_demand, ManeuverDemandResult
        ):
            raise TypeError("maneuver_demand must be a ManeuverDemandResult or None")

    @property
    def identity(self) -> ResultIdentity:
        return ResultIdentity(
            study_id=self.manifest.study_hash or self.manifest.experiment_hash,
            scenario_id=self.manifest.scenario_id,
            run_id=self.manifest.run_id,
        )


def _event_variables(
    prefix: str,
    dimension: str,
    events: tuple[ConjunctionEvent, ...],
) -> dict[str, tuple[tuple[str], np.ndarray]]:
    return {
        f"{prefix}_debris_index": (
            (dimension,),
            np.asarray([event.debris_index for event in events], dtype=np.int64),
        ),
        f"{prefix}_target_index": (
            (dimension,),
            np.asarray([event.target_index for event in events], dtype=np.int64),
        ),
        f"{prefix}_tca_s": (
            (dimension,),
            np.asarray([event.tca_s for event in events], dtype=np.float64),
        ),
        f"{prefix}_miss_distance_m": (
            (dimension,),
            np.asarray([event.miss_distance_m for event in events], dtype=np.float64),
        ),
        f"{prefix}_relative_velocity_m_s": (
            (dimension,),
            np.asarray([event.relative_velocity_m_s for event in events], dtype=np.float64),
        ),
    }


def _require_canonical_flux_bins(flux: FluxResult) -> tuple[int, int]:
    time_count = len(flux.spec.time_bin_edges_s) - 1
    size_count = len(flux.spec.size_bin_edges_m) - 1
    expected_count = flux.target_count * time_count * size_count
    if len(flux.bins) != expected_count:
        raise ValueError("flux bins must cover every target, time bin, and size bin")
    for flat_index, value in enumerate(flux.bins):
        target_index = flat_index // (time_count * size_count)
        remainder = flat_index % (time_count * size_count)
        time_index = remainder // size_count
        size_index = remainder % size_count
        expected = (
            target_index,
            flux.spec.time_bin_edges_s[time_index],
            flux.spec.time_bin_edges_s[time_index + 1],
            flux.spec.size_bin_edges_m[size_index],
            flux.spec.size_bin_edges_m[size_index + 1],
        )
        actual = (
            value.target_index,
            value.time_start_s,
            value.time_end_s,
            value.size_min_m,
            value.size_max_m,
        )
        if actual != expected:
            raise ValueError("flux bins must use canonical target, time, and size order")
    return time_count, size_count


def _set_units(dataset: xr.Dataset) -> None:
    units = {
        "conjunction_tca_s": "s",
        "conjunction_miss_distance_m": "m",
        "conjunction_relative_velocity_m_s": "m s-1",
        "flux_sampling_radius_m": "m",
        "time_bin_start_s": "s",
        "time_bin_end_s": "s",
        "size_bin_min_m": "m",
        "size_bin_max_m": "m",
        "number_flux_m2_s": "m-2 s-1",
        "mass_flux_kg_m2_s": "kg m-2 s-1",
        "kinetic_energy_flux_w_m2": "W m-2",
        "maneuver_trackability_size_threshold_m": "m",
        "maneuver_miss_distance_threshold_m": "m",
        "maneuver_duration_s": "s",
        "maneuver_deduplication_tolerance_s": "s",
        "maneuver_event_rate_s": "s-1",
        "maneuver_probability_at_least_one": "1",
        "maneuver_tca_s": "s",
        "maneuver_miss_distance_m": "m",
        "maneuver_relative_velocity_m_s": "m s-1",
    }
    for name, unit in units.items():
        if name in dataset:
            dataset[name].attrs["units"] = unit


def run_result_dataset(result: RunResult) -> xr.Dataset:
    """Represent one run as a self-describing xarray dataset."""
    if not isinstance(result, RunResult):
        raise TypeError("result must be a RunResult")
    identity = result.identity
    coordinates: dict[str, object] = {
        "study_id": identity.study_id,
        "scenario_id": str(identity.scenario_id),
        "run_id": str(identity.run_id),
        "summary": np.arange(len(result.summary), dtype=np.int64),
        "conjunction_event": np.arange(len(result.conjunctions), dtype=np.int64),
    }
    variables: dict[str, object] = {
        "manifest_json": canonical_manifest(result.manifest),
        "summary_name": (
            ("summary",),
            np.asarray([value.name for value in result.summary], dtype=str),
        ),
        "summary_value": (
            ("summary",),
            np.asarray([value.value for value in result.summary], dtype=np.float64),
        ),
        "summary_unit": (
            ("summary",),
            np.asarray([value.unit for value in result.summary], dtype=str),
        ),
        "has_flux": np.int8(result.flux is not None),
        "has_maneuver_demand": np.int8(result.maneuver_demand is not None),
        **_event_variables("conjunction", "conjunction_event", result.conjunctions),
    }

    target_count: int | None = None
    if result.flux is not None:
        flux = result.flux
        time_count, size_count = _require_canonical_flux_bins(flux)
        target_count = flux.target_count
        coordinates.update(
            {
                "target": np.arange(target_count, dtype=np.int64),
                "time_bin": np.arange(time_count, dtype=np.int64),
                "size_bin": np.arange(size_count, dtype=np.int64),
            }
        )
        shape = (target_count, time_count, size_count)
        variables.update(
            {
                "flux_sampling_radius_m": flux.spec.sampling_radius_m,
                "time_bin_start_s": (
                    ("time_bin",),
                    np.asarray(flux.spec.time_bin_edges_s[:-1], dtype=np.float64),
                ),
                "time_bin_end_s": (
                    ("time_bin",),
                    np.asarray(flux.spec.time_bin_edges_s[1:], dtype=np.float64),
                ),
                "size_bin_min_m": (
                    ("size_bin",),
                    np.asarray(flux.spec.size_bin_edges_m[:-1], dtype=np.float64),
                ),
                "size_bin_max_m": (
                    ("size_bin",),
                    np.asarray(flux.spec.size_bin_edges_m[1:], dtype=np.float64),
                ),
                "flux_encounter_count": (
                    ("target", "time_bin", "size_bin"),
                    np.asarray(
                        [value.encounter_count for value in flux.bins], dtype=np.int64
                    ).reshape(shape),
                ),
                "number_flux_m2_s": (
                    ("target", "time_bin", "size_bin"),
                    np.asarray(
                        [value.number_flux_m2_s for value in flux.bins], dtype=np.float64
                    ).reshape(shape),
                ),
                "mass_flux_kg_m2_s": (
                    ("target", "time_bin", "size_bin"),
                    np.asarray(
                        [value.mass_flux_kg_m2_s for value in flux.bins], dtype=np.float64
                    ).reshape(shape),
                ),
                "kinetic_energy_flux_w_m2": (
                    ("target", "time_bin", "size_bin"),
                    np.asarray(
                        [value.kinetic_energy_flux_w_m2 for value in flux.bins],
                        dtype=np.float64,
                    ).reshape(shape),
                ),
            }
        )

    if result.maneuver_demand is not None:
        demand = result.maneuver_demand
        maneuver_target_count = len(demand.events_per_target)
        if target_count is not None and maneuver_target_count != target_count:
            raise ValueError("flux and maneuver demand must use the same target count")
        if target_count is None:
            target_count = maneuver_target_count
            coordinates["target"] = np.arange(target_count, dtype=np.int64)
        coordinates.update(
            {
                "maneuver_event": np.arange(len(demand.actionable_encounters), dtype=np.int64),
                "affected_target": np.arange(len(demand.affected_target_indices), dtype=np.int64),
            }
        )
        variables.update(
            {
                "maneuver_trackability_size_threshold_m": (
                    demand.spec.trackability_size_threshold_m
                ),
                "maneuver_miss_distance_threshold_m": demand.spec.miss_distance_threshold_m,
                "maneuver_duration_s": demand.spec.duration_s,
                "maneuver_deduplication_tolerance_s": (demand.spec.deduplication_tolerance_s),
                "maneuver_event_rate_s": demand.event_rate_s,
                "maneuver_probability_at_least_one": (
                    demand.probability_at_least_one_actionable_encounter
                ),
                "maneuver_events_per_target": (
                    ("target",),
                    np.asarray(demand.events_per_target, dtype=np.int64),
                ),
                "maneuver_affected_target_index": (
                    ("affected_target",),
                    np.asarray(demand.affected_target_indices, dtype=np.int64),
                ),
                **_event_variables(
                    "maneuver",
                    "maneuver_event",
                    demand.actionable_encounters,
                ),
            }
        )

    dataset = xr.Dataset(
        data_vars=variables,
        coords=coordinates,
        attrs={"schema": _schema, "title": "ODSS scientific run result"},
    )
    _set_units(dataset)
    dataset["summary_value"].attrs["units"] = "see summary_unit"
    return dataset


def _manifest_from_json(value: str) -> RunManifest:
    item = json.loads(value)
    if item.get("schema") != "odss.run_manifest.v2":
        raise ValueError("unsupported stored manifest schema")
    return RunManifest(
        experiment_hash=item["experiment_hash"],
        master_seed=item["master_seed"],
        scenario_id=item["scenario_id"],
        run_id=item["run_id"],
        rng_algorithm=item["rng_algorithm"],
        study_hash=item["study_hash"],
        input_assets=tuple(InputAssetMetadata(**asset) for asset in item["input_assets"]),
        software=tuple(SoftwareMetadata(**software) for software in item["software"]),
    )


def _scalar_string(dataset: xr.Dataset, name: str) -> str:
    if name not in dataset:
        raise ValueError(f"stored run result is missing {name}")
    value = dataset[name].item()
    if not isinstance(value, str):
        raise ValueError(f"stored {name} must be a string")
    return value


def _events_from_dataset(
    dataset: xr.Dataset,
    prefix: str,
    dimension: str,
) -> tuple[ConjunctionEvent, ...]:
    count = dataset.sizes.get(dimension, 0)
    names = (
        f"{prefix}_debris_index",
        f"{prefix}_target_index",
        f"{prefix}_tca_s",
        f"{prefix}_miss_distance_m",
        f"{prefix}_relative_velocity_m_s",
    )
    if any(name not in dataset for name in names):
        raise ValueError(f"stored run result has incomplete {prefix} event fields")
    return tuple(
        ConjunctionEvent(
            debris_index=int(dataset[names[0]].values[index]),
            target_index=int(dataset[names[1]].values[index]),
            tca_s=float(dataset[names[2]].values[index]),
            miss_distance_m=float(dataset[names[3]].values[index]),
            relative_velocity_m_s=float(dataset[names[4]].values[index]),
        )
        for index in range(count)
    )


def _edges_from_dataset(
    dataset: xr.Dataset,
    start_name: str,
    end_name: str,
) -> tuple[float, ...]:
    starts = np.asarray(dataset[start_name].values, dtype=np.float64)
    ends = np.asarray(dataset[end_name].values, dtype=np.float64)
    if starts.ndim != 1 or ends.ndim != 1 or len(starts) == 0 or len(starts) != len(ends):
        raise ValueError(f"stored {start_name} and {end_name} must be equal non-empty vectors")
    if not np.array_equal(starts[1:], ends[:-1]):
        raise ValueError(f"stored {start_name} and {end_name} must be contiguous")
    return tuple(float(value) for value in np.concatenate((starts[:1], ends)))


def _flux_from_dataset(dataset: xr.Dataset) -> FluxResult | None:
    if int(dataset["has_flux"].item()) == 0:
        return None
    time_edges_s = _edges_from_dataset(dataset, "time_bin_start_s", "time_bin_end_s")
    size_edges_m = _edges_from_dataset(dataset, "size_bin_min_m", "size_bin_max_m")
    target_count = dataset.sizes.get("target", 0)
    time_count = len(time_edges_s) - 1
    size_count = len(size_edges_m) - 1
    expected_shape = (target_count, time_count, size_count)
    names = (
        "flux_encounter_count",
        "number_flux_m2_s",
        "mass_flux_kg_m2_s",
        "kinetic_energy_flux_w_m2",
    )
    if any(name not in dataset or dataset[name].shape != expected_shape for name in names):
        raise ValueError("stored flux fields do not match target, time, and size dimensions")
    bins = tuple(
        FluxBin(
            target_index=target_index,
            time_start_s=time_edges_s[time_index],
            time_end_s=time_edges_s[time_index + 1],
            size_min_m=size_edges_m[size_index],
            size_max_m=size_edges_m[size_index + 1],
            encounter_count=int(
                dataset["flux_encounter_count"].values[target_index, time_index, size_index]
            ),
            number_flux_m2_s=float(
                dataset["number_flux_m2_s"].values[target_index, time_index, size_index]
            ),
            mass_flux_kg_m2_s=float(
                dataset["mass_flux_kg_m2_s"].values[target_index, time_index, size_index]
            ),
            kinetic_energy_flux_w_m2=float(
                dataset["kinetic_energy_flux_w_m2"].values[target_index, time_index, size_index]
            ),
        )
        for target_index in range(target_count)
        for time_index in range(time_count)
        for size_index in range(size_count)
    )
    return FluxResult(
        spec=FluxSpec(
            sampling_radius_m=float(dataset["flux_sampling_radius_m"].item()),
            time_bin_edges_s=time_edges_s,
            size_bin_edges_m=size_edges_m,
        ),
        target_count=target_count,
        bins=bins,
    )


def _maneuver_from_dataset(dataset: xr.Dataset) -> ManeuverDemandResult | None:
    if int(dataset["has_maneuver_demand"].item()) == 0:
        return None
    return ManeuverDemandResult(
        spec=ManeuverDemandSpec(
            trackability_size_threshold_m=float(
                dataset["maneuver_trackability_size_threshold_m"].item()
            ),
            miss_distance_threshold_m=float(dataset["maneuver_miss_distance_threshold_m"].item()),
            duration_s=float(dataset["maneuver_duration_s"].item()),
            deduplication_tolerance_s=float(dataset["maneuver_deduplication_tolerance_s"].item()),
        ),
        actionable_encounters=_events_from_dataset(dataset, "maneuver", "maneuver_event"),
        affected_target_indices=tuple(
            int(value) for value in dataset["maneuver_affected_target_index"].values
        ),
        events_per_target=tuple(
            int(value) for value in dataset["maneuver_events_per_target"].values
        ),
        event_rate_s=float(dataset["maneuver_event_rate_s"].item()),
        probability_at_least_one_actionable_encounter=float(
            dataset["maneuver_probability_at_least_one"].item()
        ),
    )


def run_result_from_dataset(dataset: xr.Dataset) -> RunResult:
    """Reconstruct immutable ODSS result values from one loaded xarray dataset."""
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset")
    if dataset.attrs.get("schema") != _schema:
        raise ValueError("unsupported run-result dataset schema")
    identity = ResultIdentity(
        study_id=_scalar_string(dataset, "study_id"),
        scenario_id=int(_scalar_string(dataset, "scenario_id")),
        run_id=int(_scalar_string(dataset, "run_id")),
    )
    manifest = _manifest_from_json(_scalar_string(dataset, "manifest_json"))
    summary_count = dataset.sizes.get("summary", 0)
    summary = tuple(
        ScalarRunResult(
            name=str(dataset["summary_name"].values[index]),
            value=float(dataset["summary_value"].values[index]),
            unit=str(dataset["summary_unit"].values[index]),
        )
        for index in range(summary_count)
    )
    result = RunResult(
        manifest=manifest,
        summary=summary,
        conjunctions=_events_from_dataset(dataset, "conjunction", "conjunction_event"),
        flux=_flux_from_dataset(dataset),
        maneuver_demand=_maneuver_from_dataset(dataset),
    )
    if result.identity != identity:
        raise ValueError("stored manifest does not match the dataset identity coordinates")
    return result


def _netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}
    for name, variable in dataset.data_vars.items():
        if variable.ndim > 0 and variable.size > 0 and variable.dtype.kind in "biufc":
            encoding[name] = {"zlib": True, "complevel": 4, "shuffle": True}
    return encoding


def write_run_result(path: str | Path, result: RunResult) -> None:
    """Atomically write one immutable run to a new NetCDF file."""
    destination = Path(path)
    if destination.suffix.lower() not in (".nc", ".nc4"):
        raise ValueError("NetCDF run result path must use a .nc or .nc4 suffix")
    if not destination.parent.exists():
        raise FileNotFoundError(f"result directory does not exist: {destination.parent}")
    if destination.exists():
        raise FileExistsError(f"run result already exists: {destination}")
    dataset = run_result_dataset(result)
    temporary_handle = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp.nc",
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        dataset.to_netcdf(
            temporary_path,
            engine="netcdf4",
            format="NETCDF4",
            encoding=_netcdf_encoding(dataset),
        )
        try:
            os.link(temporary_path, destination)
        except FileExistsError as error:
            raise FileExistsError(f"run result already exists: {destination}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def read_run_result(path: str | Path) -> RunResult:
    """Read one NetCDF run file and reconstruct its immutable ODSS values."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"run result does not exist: {source}")
    with xr.open_dataset(source, engine="netcdf4") as opened:
        dataset = cast(xr.Dataset, opened.load())
    return run_result_from_dataset(dataset)


__all__ = [
    "ResultIdentity",
    "RunResult",
    "ScalarRunResult",
    "read_run_result",
    "run_result_dataset",
    "run_result_from_dataset",
    "write_run_result",
]
