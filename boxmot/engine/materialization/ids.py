"""Stable identifiers and fingerprints for immutable materialization builds."""

from __future__ import annotations

import hashlib
from typing import Any

from boxmot.datasets.manifest import canonical_json_bytes
from boxmot.datasets.validation import expected_instance_id


def fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-like configuration."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def make_build_id(value: Any) -> str:
    """Return the full SHA-256 digest of a complete build definition."""

    return fingerprint(value)


def make_instance_id(build_id: str, sample_id: str, detection_index: int) -> str:
    """Return ``{build_id}:{sample_id}:{detection_index}``."""

    return expected_instance_id(build_id, sample_id, detection_index)


def make_stage_fingerprint(
    name: str,
    *,
    config: Any,
    component: Any = None,
    upstream: tuple[str, ...] = (),
    batch_size: int = 1,
) -> str:
    """Fingerprint a stage including every upstream content identity."""

    return fingerprint(
        {
            "name": name,
            "config": config,
            "component": component,
            "upstream": list(upstream),
            "batch_size": batch_size,
        }
    )


__all__ = ("fingerprint", "make_build_id", "make_instance_id", "make_stage_fingerprint")
