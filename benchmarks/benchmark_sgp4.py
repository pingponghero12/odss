"""Measure common-epoch SGP4 catalog throughput."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import odss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=int, default=10_000)
    arguments = parser.parse_args()
    if arguments.objects < 1:
        parser.error("--objects must be positive")

    fixture = Path(__file__).parents[1] / "tests" / "fixtures" / "omm_catalog_a.json"
    source = odss.parse_omm_json(
        fixture.read_bytes(),
        logical_name=fixture.name,
        source_uri="https://celestrak.org/test/omm_catalog_a.json",
        acquired_at=odss.epoch_from_iso("2026-06-19T12:30:00", "UTC"),
    ).records[0]
    records = tuple(
        replace(source, catalog_id=index + 1, object_name=None, international_designator=None)
        for index in range(arguments.objects)
    )
    catalog = odss.Catalog(records, ())
    epoch = odss.epoch_from_iso("2026-06-20T12:00:00", "UTC")

    start = time.perf_counter()
    synchronized = odss.synchronize_catalog_sgp4(catalog, epoch)
    duration_s = time.perf_counter() - start
    print(f"objects: {len(synchronized)}")
    print(f"duration_s: {duration_s:.6f}")
    print(f"objects_per_s: {len(synchronized) / duration_s:.0f}")


if __name__ == "__main__":
    main()
