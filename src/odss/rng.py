"""Deterministic named random-stream helpers."""

from __future__ import annotations

import hashlib

from ._core import RNG_ALGORITHM, RandomKey, random_u64, uniform_01

_STREAM_DOMAIN = b"odss.named_stream.v1\0"
_UINT64_MAX = (1 << 64) - 1


def _require_uint64(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"{field_name} must be an unsigned 64-bit integer")


def named_stream_id(name: str) -> int:
    """Return the stable numeric ID of a non-empty UTF-8 stream name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must not be empty")
    digest = hashlib.sha256(_STREAM_DOMAIN + name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def named_random_key(
    *,
    master_seed: int,
    scenario_id: int,
    run_id: int,
    object_id: int,
    stream_name: str,
) -> RandomKey:
    """Build a random key whose stream ID is derived from a stable name."""
    _require_uint64(master_seed, "master_seed")
    _require_uint64(scenario_id, "scenario_id")
    _require_uint64(run_id, "run_id")
    _require_uint64(object_id, "object_id")
    return RandomKey(
        master_seed=master_seed,
        scenario_id=scenario_id,
        run_id=run_id,
        object_id=object_id,
        stream_id=named_stream_id(stream_name),
    )


__all__ = [
    "RNG_ALGORITHM",
    "RandomKey",
    "named_random_key",
    "named_stream_id",
    "random_u64",
    "uniform_01",
]
