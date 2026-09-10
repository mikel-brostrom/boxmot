"""Shared tracker support code.

The names exported here are loaded lazily so importing a small common helper
does not pull in optional dependencies used by unrelated tracker subsystems.
"""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    "appearance",
    "association",
    "base",
    "box",
    "config",
    "detections",
    "factory",
    "geometry",
    "input",
    "manifest",
    "motion",
    "native",
    "protocols",
    "registry",
    "specs",
    "track_state",
    "tracking",
}

_EXPORTS = {
    "AssociationResult": "tracking",
    "AssociationFunction": "association",
    "AssociationStage": "association",
    "AssociationStageResult": "association",
    "MotionModelAdapter": "motion.models",
    "MotionModelKind": "motion.models",
    "TrackIdAllocator": "tracking",
    "TrackLifecycleMixin": "tracking",
    "TrackMeta": "tracking",
    "TrackRecord": "tracking",
    "TrackState": "tracking",
    "VisualizationMixin": "tracking",
    "align_obb_measurement": "geometry.obb",
    "all_indices": "association",
    "apply_cmc_to_tracks": "motion.cmc.integration",
    "blend_embeddings": "appearance",
    "cmc_detection_boxes": "motion.cmc.integration",
    "confidence_aware_alpha": "appearance",
    "create_cmc": "motion.cmc.registry",
    "create_motion_model": "motion.models",
    "detection_track_similarity_assignment": "association",
    "embedding_distance": "association",
    "feature_distance": "association",
    "ema_update_embedding": "appearance",
    "fuse_score": "association",
    "iou_distance": "association",
    "joint_stracks": "tracking",
    "linear_assignment": "association",
    "normalize_embedding": "appearance",
    "normalize_angle": "geometry.obb",
    "order_corners": "geometry.obb",
    "placeholder_embeddings": "appearance",
    "remove_duplicate_stracks": "tracking",
    "reset_cmc": "motion.cmc.integration",
    "resolve_batch_embeddings": "appearance",
    "run_association_stage": "association",
    "solve_assignment": "association",
    "smooth_display_angle": "geometry.obb",
    "smooth_obb_corners": "geometry.obb",
    "sub_stracks": "tracking",
    "sync_track_meta": "tracking",
    "track_duration": "tracking",
    "track_id": "tracking",
    "wrap_pi_periodic": "geometry.obb",
    "xywha_to_corners": "geometry.obb",
    "xywha_to_xyxy": "geometry.obb",
}

__all__ = tuple(sorted([*_SUBMODULES, *_EXPORTS]))


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
