"""Segmentor configuration and immutable specification resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from boxmot.components.artifacts import resolve_artifact
from boxmot.components.resolution import (
    ArtifactResolver,
    artifact_provenance,
    component_options,
    load_component_mapping,
    required_artifact_values,
    resolve_component_artifact,
)
from boxmot.segmentors.specs import SegmentorSpec
from boxmot.utils.config import ConfigurationError


def resolve_segmentor_spec(
    reference: str | Path | Mapping[str, Any],
    *,
    geometry: str,
    allow_download: bool = True,
    artifact_resolver: ArtifactResolver = resolve_artifact,
) -> tuple[SegmentorSpec, dict[str, Any]]:
    """Resolve a segmentor YAML or mapping into a hashed specification."""

    config_path: Path | None = None
    if isinstance(reference, Mapping):
        payload = dict(reference)
    else:
        authored = load_component_mapping(reference)
        if authored is None:
            raise FileNotFoundError(
                f"Segmentor {reference!r} is not a YAML config; SAM requires an explicit checkpoint config."
            )
        payload, config_path = authored
    configured_geometry = str(payload.get("geometry_mode") or geometry)
    if configured_geometry not in {"auto", geometry}:
        raise ConfigurationError(
            f"Segmentor geometry_mode {configured_geometry!r} does not match build geometry {geometry!r}."
        )
    path, uri, expected_hash = required_artifact_values(payload, component="Segmentor")
    artifact = resolve_component_artifact(
        path,
        uri=uri,
        expected_sha256=expected_hash,
        config_path=config_path,
        allow_download=allow_download,
        artifact_resolver=artifact_resolver,
    )
    backend = str(payload.get("backend") or "")
    if backend in {"sam", "maskrcnn"} and not artifact.path.is_file():
        raise ConfigurationError(f"Segmentor backend {backend!r} requires a checkpoint file artifact.")
    normalized_options = dict(payload.get("options") or {})
    normalized_options.setdefault("mask_threshold", 0.5)
    if backend == "maskrcnn":
        normalized_options.setdefault("class_mapping", ())
        normalized_options.setdefault("match_iou", 0.5)
        normalized_options.setdefault("score_threshold", 0.0)
    spec = SegmentorSpec(
        backend=backend,
        artifact=str(artifact.path),
        artifact_sha256=artifact.sha256,
        device=str(payload.get("device") or "cpu"),
        precision=str(payload.get("precision") or "fp32"),
        options=component_options(normalized_options),
        preprocessing=str(payload.get("preprocessing") or "default"),
        geometry_mode=geometry,
    )
    return spec, {"spec": asdict(spec), "artifact": artifact_provenance(artifact)}


__all__ = ("resolve_segmentor_spec",)
