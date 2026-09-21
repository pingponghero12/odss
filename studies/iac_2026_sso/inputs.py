"""Acquire and verify the frozen CelesTrak input snapshot for the SSO study."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_schema = "odss.iac_2026_sso.catalog_snapshot.v1"
_groups = ("weather", "resource", "sar")
_url_template = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=json"


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """Verified paths and metadata for one immutable catalog acquisition."""

    manifest_path: Path
    catalog_path: Path
    acquired_at_utc: str
    common_epoch_utc: str
    source_uri: str
    source_paths: tuple[Path, ...]

    @property
    def provenance_paths(self) -> tuple[Path, ...]:
        return (self.manifest_path, *self.source_paths)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "odss/0.1 scientific study"})
    with urllib.request.urlopen(request, timeout=60.0) as response:
        return response.read()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "")


def _records(data: bytes, group: str) -> list[object]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"CelesTrak {group} response is not valid JSON") from error
    if not isinstance(value, list) or not value:
        raise ValueError(f"CelesTrak {group} response must be a non-empty JSON array")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"CelesTrak {group} response contains a non-object record")
    return value


def _file_record(path: Path, data: bytes, **fields: object) -> dict[str, object]:
    return {
        **fields,
        "filename": path.name,
        "sha256": _sha256(data),
        "size_bytes": len(data),
    }


def _merged_bytes(records: list[object]) -> bytes:
    return (
        json.dumps(records, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def acquire_catalog_snapshot(
    directory: Path,
    *,
    fetcher: Callable[[str], bytes] = _download,
    acquired_at_utc: str | None = None,
) -> CatalogSnapshot:
    """Download the three official OMM groups into a new immutable directory."""
    destination = Path(directory)
    manifest_path = destination / "catalog_snapshot.json"
    if manifest_path.is_file():
        return load_catalog_snapshot(manifest_path)
    if destination.exists():
        raise FileExistsError(
            f"catalog snapshot directory exists without a manifest: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        acquisition = acquired_at_utc or _utc_now()
        merged: list[object] = []
        source_records: list[dict[str, object]] = []
        for group in _groups:
            url = _url_template.format(group=group)
            data = fetcher(url)
            records = _records(data, group)
            path = temporary / f"celestrak_{group}.json"
            path.write_bytes(data)
            merged.extend(records)
            source_records.append(
                _file_record(
                    path,
                    data,
                    group=group,
                    record_count=len(records),
                    source_uri=url,
                )
            )

        merged_data = _merged_bytes(merged)
        catalog_path = temporary / "celestrak_earth_observation.json"
        catalog_path.write_bytes(merged_data)
        manifest = {
            "acquired_at_utc": acquisition,
            "common_epoch_utc": acquisition,
            "merged_catalog": _file_record(
                catalog_path,
                merged_data,
                record_count=len(merged),
            ),
            "schema": _schema,
            "sources": source_records,
        }
        (temporary / manifest_path.name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load_catalog_snapshot(manifest_path)


def _verified_file(directory: Path, record: object) -> Path:
    if not isinstance(record, dict):
        raise ValueError("catalog snapshot contains an invalid file record")
    filename = record.get("filename")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("catalog snapshot contains an invalid filename")
    path = directory / filename
    data = path.read_bytes()
    if len(data) != expected_size or _sha256(data) != expected_hash:
        raise ValueError(f"catalog snapshot asset does not match its manifest: {path}")
    return path


def load_catalog_snapshot(manifest_path: Path) -> CatalogSnapshot:
    """Load a snapshot only after verifying every archived byte against its hash."""
    manifest_file = Path(manifest_path)
    value = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != _schema:
        raise ValueError("unsupported catalog snapshot schema")
    acquired_at_utc = value.get("acquired_at_utc")
    common_epoch_utc = value.get("common_epoch_utc")
    if not isinstance(acquired_at_utc, str) or not isinstance(common_epoch_utc, str):
        raise ValueError("catalog snapshot epochs must be strings")
    catalog_path = _verified_file(manifest_file.parent, value.get("merged_catalog"))
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != len(_groups):
        raise ValueError("catalog snapshot must contain all configured sources")
    if tuple(item.get("group") for item in sources if isinstance(item, dict)) != _groups:
        raise ValueError("catalog snapshot source groups are invalid")
    source_paths = tuple(_verified_file(manifest_file.parent, item) for item in sources)
    source_uris = tuple(item.get("source_uri") for item in sources)
    if any(not isinstance(uri, str) for uri in source_uris):
        raise ValueError("catalog snapshot source URIs must be strings")
    expected_uris = tuple(_url_template.format(group=group) for group in _groups)
    if source_uris != expected_uris:
        raise ValueError("catalog snapshot source URIs are invalid")
    source_records = [
        record
        for path, group in zip(source_paths, _groups, strict=True)
        for record in _records(path.read_bytes(), group)
    ]
    if catalog_path.read_bytes() != _merged_bytes(source_records):
        raise ValueError("merged catalog does not match the archived source responses")
    return CatalogSnapshot(
        manifest_path=manifest_file,
        catalog_path=catalog_path,
        acquired_at_utc=acquired_at_utc,
        common_epoch_utc=common_epoch_utc,
        source_uri=" | ".join(source_uris),
        source_paths=source_paths,
    )


__all__ = [
    "CatalogSnapshot",
    "acquire_catalog_snapshot",
    "load_catalog_snapshot",
]
