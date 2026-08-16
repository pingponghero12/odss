"""Deterministic experiment identity and provenance metadata."""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Sequence
from dataclasses import dataclass

from ._core import ExperimentSpec, ParticlePopulation, version

_SHA256_HEX_LENGTH = 64
_HEX_DIGITS = frozenset(string.hexdigits)


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _normalize_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")
    normalized = value.lower()
    if len(normalized) != _SHA256_HEX_LENGTH or any(
        character not in _HEX_DIGITS for character in normalized
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")
    return normalized


def _canonical_float(value: float) -> str:
    # Positive and negative zero compare equal and must have the same identity.
    return (0.0 if value == 0.0 else value).hex()


def _canonical_floats(values: Sequence[float]) -> list[str]:
    return [_canonical_float(value) for value in values]


def _canonical_json(value: dict[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _population_representation(population: ParticlePopulation) -> dict[str, object]:
    return {
        "area_m2": _canonical_floats(population.area_m2),
        "epoch": {
            "offset_s": _canonical_float(population.epoch.offset_s),
            "reference_epoch": population.epoch.reference_epoch,
            "time_scale": population.epoch.time_scale,
        },
        "frame": population.frame.identifier,
        "mass_kg": _canonical_floats(population.mass_kg),
        "position_x_m": _canonical_floats(population.position_x_m),
        "position_y_m": _canonical_floats(population.position_y_m),
        "position_z_m": _canonical_floats(population.position_z_m),
        "velocity_x_m_s": _canonical_floats(population.velocity_x_m_s),
        "velocity_y_m_s": _canonical_floats(population.velocity_y_m_s),
        "velocity_z_m_s": _canonical_floats(population.velocity_z_m_s),
    }


def canonical_experiment(experiment: ExperimentSpec) -> str:
    """Return the canonical internal JSON representation of an experiment."""
    if not isinstance(experiment, ExperimentSpec):
        raise TypeError("experiment must be an ExperimentSpec")
    representation = {
        "backend": {"kind": experiment.backend.kind},
        "observables": [
            {"kind": item.kind}
            for item in sorted(experiment.observables, key=lambda item: item.kind)
        ],
        "population": _population_representation(experiment.population),
        "schema": "odss.experiment.v1",
    }
    return _canonical_json(representation)


def experiment_hash(experiment: ExperimentSpec) -> str:
    """Return the SHA-256 identity of an experiment's canonical representation."""
    return hashlib.sha256(canonical_experiment(experiment).encode("utf-8")).hexdigest()


def canonical_study(experiments: Sequence[ExperimentSpec]) -> str:
    """Return canonical JSON for an order-independent collection of experiments."""
    if isinstance(experiments, (str, bytes)) or not isinstance(experiments, Sequence):
        raise TypeError("experiments must be a sequence of ExperimentSpec values")
    if any(not isinstance(experiment, ExperimentSpec) for experiment in experiments):
        raise TypeError("experiments must contain only ExperimentSpec values")
    representation = {
        "experiment_hashes": sorted(experiment_hash(experiment) for experiment in experiments),
        "schema": "odss.study.v1",
    }
    return _canonical_json(representation)


def study_hash(experiments: Sequence[ExperimentSpec]) -> str:
    """Return the SHA-256 identity of a study's canonical representation."""
    return hashlib.sha256(canonical_study(experiments).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class InputAssetMetadata:
    """Content identity and size of an external scientific input."""

    logical_name: str
    content_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _require_text(self.logical_name, "logical_name")
        object.__setattr__(
            self,
            "content_sha256",
            _normalize_sha256(self.content_sha256, "content_sha256"),
        )
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ValueError("size_bytes must be a non-negative integer")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SoftwareMetadata:
    """Name and version of software relevant to a scientific run."""

    name: str
    version: str

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.version, "version")


def odss_software_metadata() -> SoftwareMetadata:
    """Return metadata for the installed odss compiled core."""
    return SoftwareMetadata(name="odss", version=version())


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Minimal immutable identity and provenance record for a run."""

    experiment_hash: str
    study_hash: str | None = None
    input_assets: tuple[InputAssetMetadata, ...] = ()
    software: tuple[SoftwareMetadata, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_hash",
            _normalize_sha256(self.experiment_hash, "experiment_hash"),
        )
        if self.study_hash is not None:
            object.__setattr__(
                self,
                "study_hash",
                _normalize_sha256(self.study_hash, "study_hash"),
            )
        object.__setattr__(self, "input_assets", tuple(self.input_assets))
        object.__setattr__(self, "software", tuple(self.software))
        if any(not isinstance(asset, InputAssetMetadata) for asset in self.input_assets):
            raise TypeError("input_assets must contain only InputAssetMetadata values")
        if any(not isinstance(item, SoftwareMetadata) for item in self.software):
            raise TypeError("software must contain only SoftwareMetadata values")


def canonical_manifest(manifest: RunManifest) -> str:
    """Return canonical JSON for a run manifest."""
    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    assets = sorted(
        manifest.input_assets,
        key=lambda item: (item.logical_name, item.content_sha256, item.size_bytes),
    )
    software = sorted(manifest.software, key=lambda item: (item.name, item.version))
    representation = {
        "experiment_hash": manifest.experiment_hash,
        "input_assets": [
            {
                "content_sha256": asset.content_sha256,
                "logical_name": asset.logical_name,
                "size_bytes": asset.size_bytes,
            }
            for asset in assets
        ],
        "schema": "odss.run_manifest.v1",
        "software": [{"name": item.name, "version": item.version} for item in software],
        "study_hash": manifest.study_hash,
    }
    return _canonical_json(representation)
