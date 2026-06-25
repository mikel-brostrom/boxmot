"""Shared tracker support code.

The names exported here are loaded lazily so importing a small common helper
does not pull in optional dependencies used by unrelated tracker subsystems.
"""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    "appearance",
    "association",
    "detections",
    "geometry",
    "motion",
    "tracking",
    "tracks",
}

_EXPORTS = {
    "AABB_DETECTIONS": "detections",
    "AssociationResult": "tracking",
    "AssociationFunction": "association",
    "AssociationStage": "association",
    "AssociationStageResult": "association",
    "AxisAlignedDetections": "detections",
    "BoxTrack": "tracks",
    "DetectionBatch": "detections",
    "DetectionLayout": "detections",
    "DetectionRecord": "tracking",
    "MotionModelAdapter": "motion",
    "MotionModelKind": "motion",
    "OBB_DETECTIONS": "detections",
    "OrientedDetections": "detections",
    "SortBoxTrack": "tracks",
    "TrackIdAllocator": "tracking",
    "TrackLifecycleMixin": "tracking",
    "TrackMeta": "tracking",
    "TrackRecord": "tracking",
    "TrackState": "tracking",
    "TrackerProtocol": "tracking",
    "VisualizationMixin": "tracking",
    "align_obb_measurement": "geometry.obb",
    "all_indices": "association",
    "apply_cmc_to_tracks": "motion.cmc",
    "blend_embeddings": "appearance",
    "cmc_detection_boxes": "motion.cmc",
    "confidence_aware_alpha": "appearance",
    "create_cmc": "motion.cmc",
    "create_motion_model": "motion",
    "detection_track_iou_assignment": "association",
    "detection_track_tuple_to_association_result": "association",
    "embedding_distance": "association",
    "ema_update_embedding": "appearance",
    "empty_output": "tracking",
    "fuse_iou": "association",
    "fuse_motion": "association",
    "fuse_score": "association",
    "format_output_row": "tracking",
    "format_output_rows": "tracking",
    "get_detection_layout": "detections",
    "infer_detection_layout": "detections",
    "iou_distance": "association",
    "iou_obb_pair": "association",
    "joint_stracks": "tracking",
    "linear_assignment": "association",
    "normalize_embedding": "appearance",
    "normalize_angle": "geometry.obb",
    "order_corners": "geometry.obb",
    "placeholder_embeddings": "appearance",
    "remove_duplicate_stracks": "tracking",
    "reset_cmc": "motion.cmc",
    "resolve_batch_embeddings": "appearance",
    "run_association_stage": "association",
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
