"""Image-only camera analysis shared by joint camera/Kalman selection candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from boxmot.datasets.cached import CachedVisionDataset, DatasetSample
from boxmot.datasets.readers.images import (
    NUMPY_IMAGE_EXTENSIONS,
    _local_image_path,
    _read_video_frame,
    read_numpy_bgr_uint8,
)
from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE, SequenceCMCPolicy, validate_cmc_analysis
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.cmc.integration import resolve_cmc_min_conf
from boxmot.trackers.common.motion.cmc.registry import sof_options
from boxmot.trackers.common.motion.cmc.sof import SOF
from boxmot.trackers.common.tracking.classes import ClassCatalog

_MIN_ACCEPTED_FRACTION = 0.6
_MIN_MEDIAN_NORMALIZED_MOTION = 0.001
_GRID_FRACTIONS = (0.1, 0.5, 0.9)


def cmc_analysis_settings(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Describe the resolved SOF estimator and sequence gate in saved JSON."""
    estimator = SOF(**sof_options(options or {}))
    return {
        "scale": estimator.scale,
        "target_width": estimator.target_width,
        "resize_before_grayscale": estimator.resize_before_grayscale,
        "feature_params": dict(estimator.feature_params),
        "feature_refresh_interval": estimator.feature_refresh_interval,
        "lk_params": {
            "winSize": list(estimator.lk_params["winSize"]),
            "maxLevel": estimator.lk_params["maxLevel"],
            "criteria": list(estimator.lk_params["criteria"]),
        },
        "min_inliers": estimator.min_inliers,
        "min_inlier_ratio": estimator.min_inlier_ratio,
        "ransac_reproj_threshold_resized_px": estimator.ransac_reproj_threshold,
        "opencv_threads": 1,
        "opencv_rng_seed_per_sequence": 0,
        "grid_fractions": list(_GRID_FRACTIONS),
        "minimum_accepted_fraction": _MIN_ACCEPTED_FRACTION,
        "minimum_median_motion_fraction_image_diagonal": _MIN_MEDIAN_NORMALIZED_MOTION,
    }


@dataclass(frozen=True)
class PreparedCMCCandidates:
    """Frozen camera behavior reused for every filter, profile, and fold.

    ``warps[sequence][frame]`` maps the preceding delivered frame to the
    current frame, with identity at frame zero and on disabled sequences.
    The NPZ artifact retains raw estimates even for disabled sequences.
    """

    warps: Mapping[str, np.ndarray]
    policy: SequenceCMCPolicy
    report_path: Path
    warps_path: Path
    report: Mapping[str, Any]
    mask_min_conf: float


class _DiagnosticSOF(SOF):
    """Observe production SOF checks without altering its estimated transforms."""

    def apply(self, img: np.ndarray, dets: np.ndarray | None = None) -> np.ndarray:
        self.was_initializing = not self.initialized or self.prev_frame is None or self.prev_keypoints is None
        self.match_count: int | None = None
        self.inlier_count: int | None = None
        self.inliers_accepted = False
        self.transform_valid = False
        return super().apply(img, dets)

    def _has_enough_inliers(self, inliers: np.ndarray | None, match_count: int) -> bool:
        self.match_count = int(match_count)
        self.inlier_count = 0 if inliers is None else int(np.count_nonzero(inliers))
        self.inliers_accepted = super()._has_enough_inliers(inliers, match_count)
        return self.inliers_accepted

    def is_valid_transform(self, transform: np.ndarray, **kwargs: float) -> bool:
        self.transform_valid = super().is_valid_transform(transform, **kwargs)
        return self.transform_valid


def _bgr_image(sample: DatasetSample, image_root: str | Path) -> np.ndarray:
    """Decode the same BGR pixels as tracker replay, without intermediate tensors."""
    if sample.image_ref is None:
        raise ValueError(f"Joint CMC/KF selection requires source image references: {sample.sample_id}.")
    path, video_frame = _local_image_path(sample.image_ref, image_root)
    if path.suffix.lower() in NUMPY_IMAGE_EXTENSIONS:
        image = read_numpy_bgr_uint8(path)
    elif video_frame is not None:
        image = _read_video_frame(path, video_frame)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or tuple(image.shape[:2]) != tuple(sample.image_size):
        raise ValueError(f"Cannot decode source image at its published size: {sample.sample_id} ({path}).")
    return image


def _motion_statistics(values: np.ndarray) -> dict[str, float | None]:
    """Report supported motion without treating unavailable fits as static evidence."""
    if not len(values):
        return {"median": None, "p90": None}
    return {"median": float(np.median(values)), "p90": float(np.percentile(values, 90))}


def _automatic_decision(accepted: np.ndarray, normalized_motion: np.ndarray) -> tuple[bool, str]:
    """Apply the fixed image-only gate, excluding the initialization frame."""
    supported = accepted[1:]
    if not len(supported):
        return False, "no_camera_transitions"
    if float(supported.mean()) < _MIN_ACCEPTED_FRACTION:
        return False, "insufficient_accepted_transforms"
    if float(np.median(normalized_motion[1:][supported])) < _MIN_MEDIAN_NORMALIZED_MOTION:
        return False, "camera_motion_below_threshold"
    return True, "reliable_camera_motion"


def _analyze_sequence(
    args: argparse.Namespace,
    sequence_id: str,
    *,
    mask_min_conf: float,
    estimator_options: Mapping[str, int | float | None],
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Estimate every delivered transition once with the runtime detection mask."""
    dataset = CachedVisionDataset._for_sequence(
        args.build_path, sequence_id=sequence_id, split=args.split, load_images=False
    )
    if dataset.manifest.box_type != args.geometry:
        raise ValueError("Joint CMC/KF selection geometry must match the cached build.")
    expected_count = int(args.seq_info[sequence_id])
    if len(dataset) != expected_count or expected_count < 1:
        raise ValueError(f"Camera analysis sequence {sequence_id!r} has a different cached and selected frame count.")
    estimator = _DiagnosticSOF(**estimator_options)
    catalog = ClassCatalog.from_metadata(
        class_ids=getattr(args, "tracker_class_ids", None),
        class_names=dict(getattr(args, "tracker_class_names", ()) or ()),
    )
    cv2.setRNGSeed(0)
    warps, accepted, motions, frame_indices, diagnostics = [], [], [], [], []
    image_sizes: set[tuple[int, int]] = set()
    input_digest = hashlib.sha256()
    for expected_index, sample in enumerate(dataset):
        if sample.frame_index != expected_index:
            raise ValueError(f"Camera analysis requires contiguous zero-based delivered frames in {sequence_id!r}.")
        image = _bgr_image(sample, dataset._image_root)
        # The public tracker boundary packs geometry and scores into float32.
        # Keep that comparison precision even for higher-precision cache tensors.
        detections = sample.detections
        catalog.validate_ids(detections.class_ids.tolist())
        boxes = detections.geometry.values.numpy().astype(np.float32, copy=False)
        scores = detections.scores.numpy().astype(np.float32, copy=False)
        masked_boxes = boxes[scores >= mask_min_conf]
        warp = np.asarray(estimator.apply(image, masked_boxes), dtype=np.float32)
        if warp.shape != (2, 3) or not np.isfinite(warp).all():
            raise ValueError(f"Camera estimator produced an invalid affine transform in {sample.sample_id!r}.")
        if expected_index == 0 and not np.array_equal(warp, np.eye(2, 3, dtype=np.float32)):
            raise ValueError("The first delivered camera transform must be identity.")
        supported = bool(estimator.inliers_accepted and estimator.transform_valid)
        height, width = sample.image_size
        image_sizes.add((height, width))
        grid = np.asarray([(x * (width - 1), y * (height - 1)) for y in _GRID_FRACTIONS for x in _GRID_FRACTIONS])
        displacement = np.linalg.norm(grid @ warp[:, :2].T + warp[:, 2] - grid, axis=1)
        motion_px = float(np.median(displacement))
        motions.append((motion_px, motion_px / np.hypot(width, height)))
        warps.append(warp.copy())
        accepted.append(supported)
        frame_indices.append(sample.frame_index)
        diagnostics.append(
            {
                "frame_index": sample.frame_index,
                "status": "accepted" if supported else "initialization" if estimator.was_initializing else "rejected",
                "match_count": estimator.match_count,
                "inlier_count": estimator.inlier_count,
                "masked_detections": len(masked_boxes),
            }
        )
        # Include the decoded pixels and actual masking input, not only paths:
        # source images live outside the immutable materialization directory.
        input_digest.update(json.dumps([sample.sample_id, sample.image_ref, sample.image_size]).encode())
        input_digest.update(np.ascontiguousarray(image).tobytes())
        input_digest.update(np.asarray(masked_boxes.shape, dtype=np.int64).tobytes())
        input_digest.update(np.ascontiguousarray(masked_boxes).tobytes())
        if progress is not None and (expected_index + 1) % 100 == 0:
            progress(f"CMC analysis: {sequence_id}, frame {expected_index + 1}/{expected_count}…")
    if len(warps) != expected_count:
        raise ValueError(f"Camera analysis did not receive every delivered frame in {sequence_id!r}.")
    accepted_array = np.asarray(accepted, dtype=bool)
    motion_array = np.asarray(motions)
    enabled, reason = _automatic_decision(accepted_array, motion_array[:, 1])
    supported = accepted_array[1:]
    stats = {
        "frames": len(warps),
        "transitions": len(warps) - 1,
        "accepted_transitions": int(supported.sum()),
        "accepted_fraction": float(supported.mean()) if len(supported) else None,
        "image_sizes_hw": sorted(image_sizes),
        "grid_motion_px_per_delivered_frame": _motion_statistics(motion_array[1:][supported, 0]),
        "grid_motion_fraction_image_diagonal": _motion_statistics(motion_array[1:][supported, 1]),
        "enabled": enabled,
        "reason": reason,
        "source_image_root": str(dataset._image_root),
        "source_pixels_and_masks_sha256": input_digest.hexdigest(),
        "frame_diagnostics": diagnostics,
    }
    arrays = {
        "warps": np.asarray(warps, dtype=np.float32),
        "accepted": accepted_array,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "motion_normalized": motion_array[:, 1],
    }
    return stats, arrays


def prepare_cmc_candidates(
    args: argparse.Namespace,
    baseline: Mapping[str, Any],
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
) -> PreparedCMCCandidates:
    """Prepare image-only ``off`` and automatic SOF behavior for joint selection.

    ``args`` must already contain the evaluator's resolved ``build_path``,
    ``split``, ``seq_info``, and optional ``sequence_names``. No ground-truth
    coordinates, tracker predictions, or tracking metrics enter the decision.
    The caller freezes the returned masking threshold in subsequent tuning.
    """
    validate_cmc_analysis(args.tracker, getattr(args, "tracker_backend", "python"))
    if args.geometry not in {"aabb", "obb"}:
        raise ValueError("Joint CMC/KF selection requires AABB or OBB geometry.")
    options = load_tracker_config(args.tracker, None, baseline)
    mask_min_conf = resolve_cmc_min_conf(options)
    estimator_options = sof_options(options)
    sequences = tuple(sorted(getattr(args, "sequence_names", None) or args.seq_info))
    if not sequences or set(sequences) - set(args.seq_info):
        raise ValueError("Camera analysis requires selected sequences present in the resolved dataset.")
    directory = Path(output_dir)
    report_path, warps_path = directory / "analysis.json", directory / "warps.npz"
    if report_path.exists() or warps_path.exists():
        raise ValueError("Camera analysis artifacts already exist; use a new selection output directory.")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.build_path) / "manifest.json"
    report: dict[str, Any] = {
        "schema": "boxmot.cmc-analysis/v1",
        "status": "complete",
        "build_path": str(Path(args.build_path).resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "split": args.split,
        "tracker": args.tracker,
        "geometry": args.geometry,
        "ground_truth_used": False,
        "method": "sof",
        "rule": CMC_ANALYSIS_RULE,
        "opencv_version": cv2.__version__,
        "mask_min_conf": mask_min_conf,
        "class_ids": getattr(args, "tracker_class_ids", None),
        "settings": cmc_analysis_settings(options),
        "definitions": {
            "accepted": "Production SOF accepted both RANSAC support and the finite, nondegenerate transform.",
            "motion": (
                "Median displacement of nine grid points divided by image diagonal; quantiles use accepted transitions."
            ),
            "initialization": "Frame zero has identity and is excluded from transition statistics.",
            "unavailable": (
                "Rejected or unavailable transforms use runtime identity fallback, not static-camera evidence."
            ),
            "warps": "Raw previous-to-current full-resolution transforms; archive keys are recorded for each sequence.",
            "scope": (
                "Offline sequence analysis; uses full-sequence images and detections, "
                "never ground truth or tracking metrics."
            ),
        },
        "sequences": {},
    }
    arrays, effective_warps, decisions = {}, {}, []
    previous_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(1)
        for number, sequence_id in enumerate(sequences):
            if progress is not None:
                progress(f"CMC analysis: estimating camera motion for {sequence_id} ({number + 1}/{len(sequences)})…")
            stats, sequence_arrays = _analyze_sequence(
                args, sequence_id, mask_min_conf=mask_min_conf, estimator_options=estimator_options, progress=progress
            )
            prefix = f"sequence_{number:04d}"
            stats["archive_keys"] = {name: f"{prefix}_{name}" for name in sequence_arrays}
            report["sequences"][sequence_id] = stats
            arrays.update({f"{prefix}_{name}": values for name, values in sequence_arrays.items()})
            enabled = stats["enabled"]
            decisions.append((sequence_id, enabled))
            effective = sequence_arrays["warps"].copy()
            if not enabled:
                effective[:] = np.eye(2, 3, dtype=np.float32)
            effective.flags.writeable = False
            effective_warps[sequence_id] = effective
    finally:
        cv2.setNumThreads(previous_threads)
    policy = SequenceCMCPolicy(method="sof", default=False, sequences=tuple(decisions))
    report["policy"] = policy.to_dict()
    report["total_frames"] = sum(stats["frames"] for stats in report["sequences"].values())
    np.savez_compressed(warps_path, **arrays)
    report["warps_sha256"] = hashlib.sha256(warps_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return PreparedCMCCandidates(effective_warps, policy, report_path, warps_path, report, mask_min_conf)
