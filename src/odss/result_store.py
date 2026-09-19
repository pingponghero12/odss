"""Transactional event-oriented persistence for scientific run results."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class ResultIdentity:
    """Join identity shared by every persistent result table."""

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
    """One named finite scalar stored in the run-summary table."""

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


_schema = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS manifests (
    study_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    canonical_json TEXT NOT NULL,
    PRIMARY KEY (study_id, scenario_id, run_id)
);
CREATE TABLE IF NOT EXISTS run_summaries (
    study_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    PRIMARY KEY (study_id, scenario_id, run_id, name),
    FOREIGN KEY (study_id, scenario_id, run_id)
        REFERENCES manifests (study_id, scenario_id, run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS conjunctions (
    study_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    event_index INTEGER NOT NULL,
    debris_index INTEGER NOT NULL,
    target_index INTEGER NOT NULL,
    tca_s REAL NOT NULL,
    miss_distance_m REAL NOT NULL,
    relative_velocity_m_s REAL NOT NULL,
    PRIMARY KEY (study_id, scenario_id, run_id, event_index),
    FOREIGN KEY (study_id, scenario_id, run_id)
        REFERENCES manifests (study_id, scenario_id, run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS flux_specs (
    study_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    target_count INTEGER NOT NULL,
    sampling_radius_m REAL NOT NULL,
    time_bin_edges_hex TEXT NOT NULL,
    size_bin_edges_hex TEXT NOT NULL,
    PRIMARY KEY (study_id, scenario_id, run_id),
    FOREIGN KEY (study_id, scenario_id, run_id)
        REFERENCES manifests (study_id, scenario_id, run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS target_flux (
    study_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    bin_index INTEGER NOT NULL,
    target_index INTEGER NOT NULL,
    time_start_s REAL NOT NULL,
    time_end_s REAL NOT NULL,
    size_min_hex TEXT NOT NULL,
    size_max_hex TEXT NOT NULL,
    encounter_count INTEGER NOT NULL,
    number_flux_m2_s REAL NOT NULL,
    mass_flux_kg_m2_s REAL NOT NULL,
    kinetic_energy_flux_w_m2 REAL NOT NULL,
    PRIMARY KEY (study_id, scenario_id, run_id, bin_index),
    FOREIGN KEY (study_id, scenario_id, run_id)
        REFERENCES manifests (study_id, scenario_id, run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS maneuver_demand (
    study_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    target_count INTEGER NOT NULL,
    trackability_size_threshold_m REAL NOT NULL,
    miss_distance_threshold_m REAL NOT NULL,
    duration_s REAL NOT NULL,
    deduplication_tolerance_s REAL NOT NULL,
    event_rate_s REAL NOT NULL,
    probability_at_least_one REAL NOT NULL,
    PRIMARY KEY (study_id, scenario_id, run_id),
    FOREIGN KEY (study_id, scenario_id, run_id)
        REFERENCES manifests (study_id, scenario_id, run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS maneuver_events (
    study_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    event_index INTEGER NOT NULL,
    debris_index INTEGER NOT NULL,
    target_index INTEGER NOT NULL,
    tca_s REAL NOT NULL,
    miss_distance_m REAL NOT NULL,
    relative_velocity_m_s REAL NOT NULL,
    PRIMARY KEY (study_id, scenario_id, run_id, event_index),
    FOREIGN KEY (study_id, scenario_id, run_id)
        REFERENCES manifests (study_id, scenario_id, run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS maneuver_target_counts (
    study_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    target_index INTEGER NOT NULL,
    event_count INTEGER NOT NULL,
    PRIMARY KEY (study_id, scenario_id, run_id, target_index),
    FOREIGN KEY (study_id, scenario_id, run_id)
        REFERENCES manifests (study_id, scenario_id, run_id) ON DELETE CASCADE
);
"""


def _identity_values(identity: ResultIdentity) -> tuple[str, str, str]:
    return identity.study_id, str(identity.scenario_id), str(identity.run_id)


def _float_tuple_json(values: tuple[float, ...]) -> str:
    return json.dumps([float(value).hex() for value in values], separators=(",", ":"))


def _decode_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float.fromhex(item) for item in json.loads(value))


def _event_values(
    identity: tuple[str, str, str],
    index: int,
    event: ConjunctionEvent,
) -> tuple[object, ...]:
    return (
        *identity,
        index,
        event.debris_index,
        event.target_index,
        event.tca_s,
        event.miss_distance_m,
        event.relative_velocity_m_s,
    )


def write_run_result(database_path: str | Path, result: RunResult) -> None:
    """Atomically append one run; existing run identities are never overwritten."""
    if not isinstance(result, RunResult):
        raise TypeError("result must be a RunResult")
    path = Path(database_path)
    if not path.parent.exists():
        raise FileNotFoundError(f"result directory does not exist: {path.parent}")
    identity = _identity_values(result.identity)
    try:
        with sqlite3.connect(path) as connection:
            connection.executescript(_schema)
            connection.execute(
                "INSERT INTO manifests VALUES (?, ?, ?, ?)",
                (*identity, canonical_manifest(result.manifest)),
            )
            connection.executemany(
                "INSERT INTO run_summaries VALUES (?, ?, ?, ?, ?, ?)",
                ((*identity, value.name, value.value, value.unit) for value in result.summary),
            )
            connection.executemany(
                "INSERT INTO conjunctions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _event_values(identity, index, event)
                    for index, event in enumerate(result.conjunctions)
                ),
            )
            if result.flux is not None:
                flux = result.flux
                connection.execute(
                    "INSERT INTO flux_specs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        *identity,
                        flux.target_count,
                        flux.spec.sampling_radius_m,
                        _float_tuple_json(flux.spec.time_bin_edges_s),
                        _float_tuple_json(flux.spec.size_bin_edges_m),
                    ),
                )
                connection.executemany(
                    "INSERT INTO target_flux VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        (
                            *identity,
                            index,
                            value.target_index,
                            value.time_start_s,
                            value.time_end_s,
                            value.size_min_m.hex(),
                            value.size_max_m.hex(),
                            value.encounter_count,
                            value.number_flux_m2_s,
                            value.mass_flux_kg_m2_s,
                            value.kinetic_energy_flux_w_m2,
                        )
                        for index, value in enumerate(flux.bins)
                    ),
                )
            if result.maneuver_demand is not None:
                demand = result.maneuver_demand
                connection.execute(
                    "INSERT INTO maneuver_demand VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        *identity,
                        len(demand.events_per_target),
                        demand.spec.trackability_size_threshold_m,
                        demand.spec.miss_distance_threshold_m,
                        demand.spec.duration_s,
                        demand.spec.deduplication_tolerance_s,
                        demand.event_rate_s,
                        demand.probability_at_least_one_actionable_encounter,
                    ),
                )
                connection.executemany(
                    "INSERT INTO maneuver_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _event_values(identity, index, event)
                        for index, event in enumerate(demand.actionable_encounters)
                    ),
                )
                connection.executemany(
                    "INSERT INTO maneuver_target_counts VALUES (?, ?, ?, ?, ?)",
                    (
                        (*identity, target_index, event_count)
                        for target_index, event_count in enumerate(demand.events_per_target)
                    ),
                )
    except sqlite3.IntegrityError as error:
        message = f"run result already exists or is internally inconsistent: {result.identity}"
        raise ValueError(message) from error


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


def _event_from_row(row: tuple[object, ...]) -> ConjunctionEvent:
    return ConjunctionEvent(
        debris_index=int(row[0]),
        target_index=int(row[1]),
        tca_s=float(row[2]),
        miss_distance_m=float(row[3]),
        relative_velocity_m_s=float(row[4]),
    )


def read_run_result(database_path: str | Path, identity: ResultIdentity) -> RunResult:
    """Reconstruct one stored run and its selected scientific outputs."""
    if not isinstance(identity, ResultIdentity):
        raise TypeError("identity must be a ResultIdentity")
    path = Path(database_path)
    if not path.exists():
        raise KeyError(f"run result not found: {identity}")
    keys = _identity_values(identity)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        manifest_row = connection.execute(
            "SELECT canonical_json FROM manifests WHERE study_id=? AND scenario_id=? AND run_id=?",
            keys,
        ).fetchone()
        if manifest_row is None:
            raise KeyError(f"run result not found: {identity}")
        summary = tuple(
            ScalarRunResult(str(name), float(value), str(unit))
            for name, value, unit in connection.execute(
                "SELECT name, value, unit FROM run_summaries "
                "WHERE study_id=? AND scenario_id=? AND run_id=? ORDER BY name",
                keys,
            )
        )
        conjunctions = tuple(
            _event_from_row(row)
            for row in connection.execute(
                "SELECT debris_index, target_index, tca_s, miss_distance_m, "
                "relative_velocity_m_s FROM conjunctions "
                "WHERE study_id=? AND scenario_id=? AND run_id=? ORDER BY event_index",
                keys,
            )
        )
        flux_spec_row = connection.execute(
            "SELECT target_count, sampling_radius_m, time_bin_edges_hex, size_bin_edges_hex "
            "FROM flux_specs WHERE study_id=? AND scenario_id=? AND run_id=?",
            keys,
        ).fetchone()
        flux = None
        if flux_spec_row is not None:
            bins = tuple(
                FluxBin(
                    target_index=int(row[0]),
                    time_start_s=float(row[1]),
                    time_end_s=float(row[2]),
                    size_min_m=float.fromhex(str(row[3])),
                    size_max_m=float.fromhex(str(row[4])),
                    encounter_count=int(row[5]),
                    number_flux_m2_s=float(row[6]),
                    mass_flux_kg_m2_s=float(row[7]),
                    kinetic_energy_flux_w_m2=float(row[8]),
                )
                for row in connection.execute(
                    "SELECT target_index, time_start_s, time_end_s, size_min_hex, size_max_hex, "
                    "encounter_count, number_flux_m2_s, mass_flux_kg_m2_s, "
                    "kinetic_energy_flux_w_m2 FROM target_flux "
                    "WHERE study_id=? AND scenario_id=? AND run_id=? ORDER BY bin_index",
                    keys,
                )
            )
            flux = FluxResult(
                spec=FluxSpec(
                    sampling_radius_m=float(flux_spec_row[1]),
                    time_bin_edges_s=_decode_float_tuple(str(flux_spec_row[2])),
                    size_bin_edges_m=_decode_float_tuple(str(flux_spec_row[3])),
                ),
                target_count=int(flux_spec_row[0]),
                bins=bins,
            )
        demand_row = connection.execute(
            "SELECT target_count, trackability_size_threshold_m, miss_distance_threshold_m, "
            "duration_s, deduplication_tolerance_s, event_rate_s, probability_at_least_one "
            "FROM maneuver_demand WHERE study_id=? AND scenario_id=? AND run_id=?",
            keys,
        ).fetchone()
        demand = None
        if demand_row is not None:
            actionable = tuple(
                _event_from_row(row)
                for row in connection.execute(
                    "SELECT debris_index, target_index, tca_s, miss_distance_m, "
                    "relative_velocity_m_s FROM maneuver_events "
                    "WHERE study_id=? AND scenario_id=? AND run_id=? ORDER BY event_index",
                    keys,
                )
            )
            counts = tuple(
                int(row[0])
                for row in connection.execute(
                    "SELECT event_count FROM maneuver_target_counts "
                    "WHERE study_id=? AND scenario_id=? AND run_id=? ORDER BY target_index",
                    keys,
                )
            )
            if len(counts) != int(demand_row[0]):
                raise ValueError("stored maneuver target counts are incomplete")
            demand = ManeuverDemandResult(
                spec=ManeuverDemandSpec(
                    trackability_size_threshold_m=float(demand_row[1]),
                    miss_distance_threshold_m=float(demand_row[2]),
                    duration_s=float(demand_row[3]),
                    deduplication_tolerance_s=float(demand_row[4]),
                ),
                actionable_encounters=actionable,
                affected_target_indices=tuple(
                    index for index, count in enumerate(counts) if count > 0
                ),
                events_per_target=counts,
                event_rate_s=float(demand_row[5]),
                probability_at_least_one_actionable_encounter=float(demand_row[6]),
            )
    result = RunResult(
        manifest=_manifest_from_json(str(manifest_row[0])),
        summary=summary,
        conjunctions=conjunctions,
        flux=flux,
        maneuver_demand=demand,
    )
    if result.identity != identity:
        raise ValueError("stored manifest does not match its table identity")
    return result


def list_run_results(database_path: str | Path) -> tuple[ResultIdentity, ...]:
    """List stored run identities without loading scientific event rows."""
    path = Path(database_path)
    if not path.exists():
        return ()
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT study_id, scenario_id, run_id FROM manifests"
        )
        identities = tuple(
            ResultIdentity(str(study), int(scenario), int(run))
            for study, scenario, run in rows
        )
        return tuple(
            sorted(
                identities,
                key=lambda identity: (
                    identity.study_id,
                    identity.scenario_id,
                    identity.run_id,
                ),
            )
        )


__all__ = [
    "ResultIdentity",
    "RunResult",
    "ScalarRunResult",
    "list_run_results",
    "read_run_result",
    "write_run_result",
]
