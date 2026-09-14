"""Output identities and provenance for optional temporal-mask evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from boxmot.components.artifacts import sha256_artifact
from boxmot.engine.materialization import fingerprint
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.config import load_tracker_config
from boxmot.utils.devices import normalize_device


def _association_policy(name: str, options: dict[str, Any]) -> dict[str, Any]:
    """Record the cost convention and configured gates of each guided stage."""
    if name == "bytetrack":
        return {
            "candidate_matrix": "stage_cost",
            "high_threshold": options["match_thresh"],
            "low_threshold": 0.5,
            "unconfirmed_threshold": 0.7,
        }
    if name == "botsort":
        return {
            "candidate_matrix": "stage_cost_after_appearance_and_confidence_fusion",
            "high_threshold": options["match_thresh"],
            "low_threshold": options["second_match_thresh"],
            "unconfirmed_threshold": options["unconfirmed_match_thresh"],
        }
    if name == "strongsort":
        return {
            "candidate_matrix": "stage_cost",
            "appearance_threshold": options["max_cos_dist"],
            "fallback_threshold": options["max_iou_dist"],
            "motion_gate": "original_mahalanobis_gate_preserved",
        }
    if name == "sfsort":
        from boxmot.trackers.sfsort.tracker import SFSORT

        resolve = SFSORT._resolve_or_default
        high_confidence = resolve(options["high_th"], 0.6, 0.0, 1.0)
        low_confidence = resolve(options["low_th"], 0.1, 0.0, high_confidence)
        dynamic = bool(options["dynamic_tuning"])
        return {
            "candidate_matrix": "stage_cost",
            "high_threshold": resolve(options["match_th_first"], 0.67, 0.0, 0.67),
            "low_threshold": resolve(options["match_th_second"], 0.3, 0.0, 1.0),
            "dynamic_threshold": {
                "enabled": dynamic,
                "rule": "clamp(high_threshold - multiplier * log10(max(count_above_cutoff, 1)), 0, 0.67)",
                "multiplier": resolve(options["match_th_first_m"], 0.0, 0.02, 0.08) if dynamic else 0.0,
                "confidence_cutoff": resolve(options["cth"], 0.5, low_confidence, 1.0),
            },
        }
    policy: dict[str, Any] = {
        "candidate_matrix": "one_minus_original_geometric_similarity",
        "similarity_threshold": options.get("iou_threshold", 0.3),
        "adjustment": "add_fill_to_geometry_and_fused_ranking",
        "appearance_and_motion_terms": "preserved",
    }
    if name == "occluboost":
        policy.update(
            recovery_threshold=options["recovery_iou_thresh"],
            low_threshold=options["second_iou_thresh"],
            appearance_gates="original_recovery_and_low_cosine_gates_preserved",
        )
    elif name == "hybridsort":
        policy["appearance_gates"] = "original_longterm_reid_correction_gates_preserved"
    return policy


def _mask_guidance_identity(
    checkpoint: str | Path,
    device: str,
    *,
    max_objects: int | None = None,
    tracker_spec: TrackerSpec | None = None,
) -> dict[str, Any]:
    """Fingerprint the actual model, bounded memory policy, and matching rules."""
    from boxmot.segmentors.propagation.model import (
        EDGETAM_REVISION,
        effective_precision,
        postprocessing_metadata,
    )
    from boxmot.trackers.common.mask_guidance import mask_guidance_config_from_options, validate_mask_guidance_spec

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mask guidance requires an existing EdgeTAM checkpoint: {path}")
    tracker_spec = tracker_spec or TrackerSpec("bytetrack")
    validate_mask_guidance_spec(tracker_spec)
    options = load_tracker_config(tracker_spec.name, None, tracker_spec.option_dict)
    if max_objects is not None:
        options["edgetam.max_objects"] = max_objects
    config = mask_guidance_config_from_options(path, device, options)
    device = normalize_device(device)
    return {
        "method": f"{tracker_spec.name}-edgetam",
        "policy_version": 4,
        "resolved_tracker_options": options,
        "reference": {
            "repository": "https://github.com/facebookresearch/EdgeTAM",
            "commit": EDGETAM_REVISION,
        },
        "checkpoint_sha256": sha256_artifact(path),
        "device": device,
        "precision": effective_precision(device),
        "postprocessing": postprocessing_metadata(device),
        "propagation": {
            "input": "incoming_frames",
            "history": {"conditioning": 1, "spatial": 6, "pointers": 15},
            "max_objects": config.max_objects,
            "prompt_overlap": config.prompt_overlap,
            "admission": "visible_residents_then_visible_new_then_recent_lost/v1",
            "retirement": "absent_from_tracker_association_pools_or_evicted",
        },
        "matching": {
            "min_mask_coverage": config.min_coverage,
            "min_mask_fill": config.min_fill,
            **_association_policy(tracker_spec.name, options),
            "isolation_recovery": "both_endpoints_without_admissible_partner",
            "rasterization": "clipped_floor_ceil",
        },
    }


def mask_guidance_output_path(
    base: str | Path,
    *,
    checkpoint: str | Path,
    device: str,
    max_objects: int | None = None,
    tracker_spec: TrackerSpec | None = None,
) -> Path:
    """Separate guided evaluation outputs from ordinary trackers and other models.

    Checkpoint content, execution device, memory budget, and matching policy determine the
    suffix. Moving identical weights does not change their output identity.
    This does not reuse tracking results; every evaluation replays its inputs.
    """
    base = Path(base)
    identity = _mask_guidance_identity(checkpoint, device, max_objects=max_objects, tracker_spec=tracker_spec)
    return base.with_name(f"{base.name}-{identity['method']}-{fingerprint(identity)[:12]}")


def write_mask_guidance_provenance(
    output_dir: str | Path,
    *,
    checkpoint: str | Path,
    device: str,
    build: str | Path,
    tracker_spec: TrackerSpec,
    sequence_names: tuple[str, ...] | None = None,
    max_objects: int | None = None,
) -> Path:
    """Atomically record the model and tracker used by a completed guided replay."""
    identity = _mask_guidance_identity(checkpoint, device, max_objects=max_objects, tracker_spec=tracker_spec)
    metadata = {
        "schema": "boxmot.mask-guidance-evaluation/v4",
        **identity,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "build": str(Path(build).expanduser().resolve()),
        "tracker": asdict(tracker_spec),
        "sequence_names": None if sequence_names is None else list(sequence_names),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "mask-guidance.json"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".mask-guidance-", suffix=".json", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
