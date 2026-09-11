"""Stable identifiers and fingerprints for immutable materialization builds."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from boxmot.datasets.manifest import StageProvenance, canonical_json_bytes
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


def component_content(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return component semantics while retaining execution device in provenance."""

    content = dict(provenance)
    spec = content.get("spec")
    if isinstance(spec, Mapping):
        content["spec"] = {key: value for key, value in spec.items() if key != "device"}
    return content


def stage_content(stages: tuple[StageProvenance, ...]) -> tuple[StageProvenance, ...]:
    """Normalize device-dependent stage identities without hiding fingerprint drift.

    Original fingerprints must match their recorded inputs before they can be
    recomputed with normalized components and upstream identities. Opaque or
    mismatched fingerprints remain unchanged, preserving mismatch detection.
    Returned stages retain the supplied order and leave original provenance intact.
    """

    original = {stage.name: stage for stage in stages}
    if len(original) != len(stages):
        raise ValueError("Materialization stage names must be unique.")
    for stage in stages:
        unknown = set(stage.inputs) - original.keys()
        if unknown:
            raise ValueError(f"Stage {stage.name!r} has unknown dependencies: {sorted(unknown)!r}.")
    normalized: dict[str, StageProvenance] = {}
    while len(normalized) < len(original):
        ready = [stage for stage in stages if stage.name not in normalized and set(stage.inputs) <= normalized.keys()]
        if not ready:
            raise ValueError("Materialization stage graph contains a cycle.")
        for stage in ready:
            component = component_content(stage.component)
            original_fingerprint = make_stage_fingerprint(
                stage.name,
                config=stage.config,
                component=stage.component,
                upstream=tuple(original[name].fingerprint for name in stage.inputs),
                batch_size=stage.batch_size,
            )
            resolved_fingerprint = stage.fingerprint
            if stage.fingerprint == original_fingerprint:
                resolved_fingerprint = make_stage_fingerprint(
                    stage.name,
                    config=stage.config,
                    component=component,
                    upstream=tuple(normalized[name].fingerprint for name in stage.inputs),
                    batch_size=stage.batch_size,
                )
            normalized[stage.name] = replace(stage, component=component, fingerprint=resolved_fingerprint)
    return tuple(normalized[stage.name] for stage in stages)


__all__ = (
    "component_content",
    "fingerprint",
    "make_build_id",
    "make_instance_id",
    "make_stage_fingerprint",
    "stage_content",
)
