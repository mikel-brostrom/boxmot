"""Image-only camera analysis shared by joint camera/Kalman selection candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from boxmot.datasets.cached import CachedVisionDataset, DatasetSample
from boxmot.datasets.manifest import DatasetManifest
from boxmot.datasets.readers.images import (
    NUMPY_IMAGE_EXTENSIONS,
    _local_image_path,
    _read_video_frame,
    read_numpy_bgr_uint8,
)
from boxmot.datasets.schema import MANIFEST_FILENAME, SUCCESS_FILENAME
from boxmot.datasets.storage import artifact_files, resolve_artifact_path
from boxmot.engine.calibration.cmc_workers import CMCSequenceTask, analyze_parallel
from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE, SequenceCMCPolicy, validate_cmc_analysis
from boxmot.engine.config.runtime import resolve_sequence_workers
from boxmot.engine.materialization.cmc import materialize_cmc
from boxmot.trackers import create_tracker
from boxmot.trackers.common.config import get_tracker_config_class, load_tracker_config
from boxmot.trackers.common.motion.cmc.integration import resolve_cmc_min_conf
from boxmot.trackers.common.motion.cmc.registry import sof_options
from boxmot.trackers.common.motion.cmc.sof import SOF
from boxmot.trackers.common.tracking.classes import ClassCatalog

_MIN_ACCEPTED_FRACTION = 0.6
_MIN_MEDIAN_NORMALIZED_MOTION = 0.001
_GRID_FRACTIONS = (0.1, 0.5, 0.9)
CMC_ANALYSIS_SCHEMA = "boxmot.cmc-analysis/v2"


def cmc_file_signature(path: Path) -> list[int]:
    """Bind camera evidence to a regular file's identity and mutation epoch."""
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Camera inputs require a regular file: {path}")
    return [metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns]


def cmc_build_signature(build: Path) -> dict[str, list[int]]:
    """Snapshot published artifacts without traversing source images or workflow outputs."""
    build = build.resolve()
    manifest = DatasetManifest.load(build)
    signatures = {name: cmc_file_signature(build / name) for name in (MANIFEST_FILENAME, SUCCESS_FILENAME)}
    for artifact in manifest.artifacts:
        files = artifact_files(resolve_artifact_path(build, artifact.path))
        actual = tuple(path.relative_to(build).as_posix() for path in files)
        if actual != tuple(shard.path for shard in artifact.shards):
            raise ValueError(f"Camera source artifact {artifact.name!r} differs from its published shard list.")
        signatures.update((path.relative_to(build).as_posix(), cmc_file_signature(path)) for path in files)
    return signatures


def cmc_image_signature(references: tuple[str, ...], image_root: str) -> dict[str, list[int]]:
    """Resolve references again so replaced images and redirected links invalidate reuse."""
    result = {}
    for reference in references:
        path, _ = _local_image_path(reference, image_root)
        if str(path) not in result:
            result[str(path)] = cmc_file_signature(path)
    return result


def cmc_runtime_mask_settings(options: Mapping[str, Any]) -> dict[str, float | None]:
    """Distinguish native runtime masking from an explicit frozen confidence."""
    threshold = options.get("cmc_min_conf")
    return {
        "cmc_min_conf": None if threshold is None else float(threshold),
        "fallback_min_conf": resolve_cmc_min_conf(options) if threshold is None else None,
    }


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
    runtime_masker: Callable[[np.ndarray], np.ndarray],
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Prepare policy and runtime transforms from one decode of each frame."""
    dataset = CachedVisionDataset._for_sequence(
        args.build_path, sequence_id=sequence_id, split=args.split,
        load_images=False, load_masks=False, load_embeddings=False,
    )
    if dataset.manifest.box_type != args.geometry:
        raise ValueError("Joint CMC/KF selection geometry must match the cached build.")
    expected_count = int(args.seq_info[sequence_id])
    if len(dataset) != expected_count or expected_count < 1:
        raise ValueError(f"Camera analysis sequence {sequence_id!r} has a different cached and selected frame count.")
    estimator = _DiagnosticSOF(**estimator_options)
    runtime_estimator = None
    catalog = ClassCatalog.from_metadata(
        class_ids=getattr(args, "tracker_class_ids", None),
        class_names=dict(getattr(args, "tracker_class_names", ()) or ()),
    )
    cv2.setRNGSeed(0)
    warps, accepted, motions, frame_indices, diagnostics = [], [], [], [], []
    runtime_warps = []
    sample_ids, image_refs = [], []
    image_signatures: dict[str, list[int]] = {}
    image_sizes: set[tuple[int, int]] = set()
    input_digest = hashlib.sha256()
    for expected_index, sample in enumerate(dataset):
        if sample.frame_index != expected_index:
            raise ValueError(f"Camera analysis requires contiguous zero-based delivered frames in {sequence_id!r}.")
        if not sample.image_ref:
            raise ValueError(f"Camera analysis requires source image references: {sample.sample_id}.")
        image_refs.append(sample.image_ref)
        sample_ids.append(sample.sample_id)
        current_images = cmc_image_signature((sample.image_ref,), str(dataset._image_root))
        for path, signature in current_images.items():
            if path in image_signatures and image_signatures[path] != signature:
                raise ValueError("Camera source images changed during analysis.")
            image_signatures[path] = signature
        image = _bgr_image(sample, dataset._image_root)
        # The public tracker boundary packs geometry and scores into float32.
        # Keep that comparison precision even for higher-precision cache tensors.
        detections = sample.detections
        catalog.validate_ids(detections.class_ids.tolist())
        boxes = detections.geometry.values.numpy().astype(np.float32, copy=False)
        scores = detections.scores.numpy().astype(np.float32, copy=False)
        masked_boxes = boxes[scores >= mask_min_conf]
        packed = np.column_stack((boxes, scores, detections.class_ids.numpy())).astype(np.float32, copy=False)
        runtime_boxes = runtime_masker(packed)
        if runtime_estimator is None and not np.array_equal(runtime_boxes, masked_boxes):
            # Until this frame both estimators have seen identical pixels and
            # masks. Fork that state before advancing either camera history.
            runtime_estimator = deepcopy(estimator)
        warp = np.asarray(estimator.apply(image, masked_boxes), dtype=np.float32)
        if warp.shape != (2, 3) or not np.isfinite(warp).all():
            raise ValueError(f"Camera estimator produced an invalid affine transform in {sample.sample_id!r}.")
        if expected_index == 0 and not np.array_equal(warp, np.eye(2, 3, dtype=np.float32)):
            raise ValueError("The first delivered camera transform must be identity.")
        runtime_warp = warp if runtime_estimator is None else np.asarray(
            runtime_estimator.apply(image, runtime_boxes), dtype=np.float32
        )
        if runtime_warp.shape != (2, 3) or not np.isfinite(runtime_warp).all():
            raise ValueError(f"Runtime camera estimator produced an invalid affine transform in {sample.sample_id!r}.")
        runtime_warps.append(runtime_warp.copy())
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
    if cmc_image_signature(tuple(image_refs), str(dataset._image_root)) != image_signatures:
        raise ValueError("Camera source images changed during analysis.")
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
        "sample_ids": sample_ids,
        "image_refs": image_refs,
        "image_signature": image_signatures,
        "source_pixels_and_masks_sha256": input_digest.hexdigest(),
        "frame_diagnostics": diagnostics,
    }
    arrays = {
        "warps": np.asarray(warps, dtype=np.float32),
        "runtime_warps": np.asarray(runtime_warps, dtype=np.float32),
        "accepted": accepted_array,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "motion_normalized": motion_array[:, 1],
    }
    return stats, arrays


def cmc_analysis_inputs(args: Any, baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Describe camera content independently of workers and workflow output paths."""
    options = load_tracker_config(args.tracker, None, baseline)
    build = Path(args.build_path).resolve()
    sequences = sorted(getattr(args, "sequence_names", None) or args.seq_info)
    class_ids = getattr(args, "tracker_class_ids", None)
    return {
        "build": str(build),
        "manifest_sha256": hashlib.sha256((build / "manifest.json").read_bytes()).hexdigest(),
        "split": args.split,
        "sequences": [(name, args.seq_info[name]) for name in sequences],
        "tracker": args.tracker,
        "backend": getattr(args, "tracker_backend", "python"),
        "geometry": args.geometry,
        "per_class": bool(getattr(args, "per_class", False)),
        "class_ids": None if class_ids is None else list(class_ids),
        "class_names": [list(entry) for entry in (getattr(args, "tracker_class_names", ()) or ())],
        "rule": options.get("calibration.cmc_rule", CMC_ANALYSIS_RULE),
        "method": "sof",
        "opencv_version": cv2.__version__,
        "settings": cmc_analysis_settings(options),
        "mask_min_conf": float(resolve_cmc_min_conf(options)),
        "runtime_mask": cmc_runtime_mask_settings(options),
    }


def prepare_cmc_candidates(
    args: argparse.Namespace,
    baseline: Mapping[str, Any],
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
) -> PreparedCMCCandidates:
    """Reuse or materialize camera evidence, then copy it into this workflow's output.

    Estimation is deferred until requested because the detection mask depends on
    tracker settings. Independent eval, tuning, and joint-selection runs share
    the validated artifacts under ``build/cmc/<key>``.
    """
    # The validator uses this module's source signature helpers. Import it only
    # after module initialization to keep calibration independent of evaluation.
    from boxmot.engine.calibration.cmc_validation import validate_cmc_artifacts

    validate_cmc_analysis(args.tracker, getattr(args, "tracker_backend", "python"))
    if args.geometry not in {"aabb", "obb"}:
        raise ValueError("Joint CMC/KF selection requires AABB or OBB geometry.")
    sequences = tuple(sorted(getattr(args, "sequence_names", None) or args.seq_info))
    if not sequences or set(sequences) - set(args.seq_info):
        raise ValueError("Camera analysis requires selected sequences present in the resolved dataset.")
    resolve_sequence_workers(len(sequences), getattr(args, "sequence_workers", None))
    directory = Path(output_dir)
    report_path, warps_path = directory / "analysis.json", directory / "warps.npz"
    if report_path.exists() or warps_path.exists():
        raise ValueError("Camera analysis artifacts already exist; use a new selection output directory.")
    inputs = cmc_analysis_inputs(args, baseline)

    def generate(path: Path) -> None:
        _prepare_cmc_candidates_uncached(args, baseline, path, progress)

    def validate(report: bytes, warps: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        return validate_cmc_artifacts(report, warps, inputs)

    entry = materialize_cmc(
        args.build_path, inputs=inputs, analysis_schema=CMC_ANALYSIS_SCHEMA,
        generate=generate, validate=validate,
    )
    report, arrays = entry.validated
    decisions, effective_warps = [], {}
    for sequence in sequences:
        stats = report["sequences"][sequence]
        enabled = stats["enabled"]
        decisions.append((sequence, enabled))
        effective = arrays[stats["archive_keys"]["warps"]].copy()
        if not enabled:
            effective[:] = np.eye(2, 3, dtype=np.float32)
        effective.flags.writeable = False
        effective_warps[sequence] = effective
    directory.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(entry.report)
    warps_path.write_bytes(entry.warps)
    if progress is not None:
        action = "reusing" if entry.reused else "saved"
        progress(f"CMC setup: {action} materialized camera analysis at {entry.path}…")
    return PreparedCMCCandidates(
        effective_warps, SequenceCMCPolicy(sequences=tuple(decisions)),
        report_path, warps_path, report, inputs["mask_min_conf"],
    )


def _prepare_cmc_candidates_uncached(
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
    ``sequence_workers`` controls sequence-level CPU parallelism; each sequence
    retains its own ordered frame history and deterministic estimator seed.
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
    workers = resolve_sequence_workers(len(sequences), getattr(args, "sequence_workers", None))
    directory = Path(output_dir)
    report_path, warps_path = directory / "analysis.json", directory / "warps.npz"
    if report_path.exists() or warps_path.exists():
        raise ValueError("Camera analysis artifacts already exist; use a new selection output directory.")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.build_path) / "manifest.json"
    source_signature = cmc_build_signature(Path(args.build_path))
    report: dict[str, Any] = {
        "schema": CMC_ANALYSIS_SCHEMA,
        "status": "complete",
        "build_path": str(Path(args.build_path).resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_signature": source_signature,
        "split": args.split,
        "tracker": args.tracker,
        "backend": getattr(args, "tracker_backend", "python"),
        "geometry": args.geometry,
        "ground_truth_used": False,
        "method": "sof",
        "rule": CMC_ANALYSIS_RULE,
        "opencv_version": cv2.__version__,
        "mask_min_conf": mask_min_conf,
        "runtime_mask": cmc_runtime_mask_settings(options),
        "class_ids": getattr(args, "tracker_class_ids", None),
        "class_names": list(getattr(args, "tracker_class_names", ()) or ()),
        "per_class": bool(getattr(args, "per_class", False)),
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
            "runtime_warps": "Transforms using the tracker's exact runtime detection mask, before sequence gating.",
            "scope": (
                "Offline sequence analysis; uses full-sequence images and detections, "
                "never ground truth or tracking metrics."
            ),
        },
        "sequences": {},
    }
    arrays, effective_warps, decisions = {}, {}, []
    config_type = get_tracker_config_class(args.tracker)
    mask_config = config_type.from_mapping({name: options[name] for name in config_type.fields() if name in options})
    if workers > 1:
        class_ids = getattr(args, "tracker_class_ids", None)
        config_json = json.dumps(mask_config.to_dict(), allow_nan=False)
        tasks = tuple(
            CMCSequenceTask(
                build_path=str(Path(args.build_path).resolve()),
                split=args.split,
                sequence_id=sequence_id,
                ordinal=number,
                total_sequences=len(sequences),
                frame_count=int(args.seq_info[sequence_id]),
                tracker=args.tracker,
                geometry=args.geometry,
                per_class=bool(getattr(args, "per_class", False)),
                class_ids=None if class_ids is None else tuple(class_ids),
                class_names=tuple(tuple(item) for item in (getattr(args, "tracker_class_names", ()) or ())),
                tracker_config_json=config_json,
                mask_min_conf=mask_min_conf,
                estimator_options=tuple(estimator_options.items()),
            )
            for number, sequence_id in enumerate(sequences)
        )
        results = analyze_parallel(tasks, workers=workers, progress=progress)
    else:
        mask_tracker = create_tracker(
            args.tracker, config=mask_config, geometry=args.geometry,
            per_class=bool(getattr(args, "per_class", False)),
        )
        previous_threads = cv2.getNumThreads()
        results = []
        try:
            cv2.setNumThreads(1)
            for number, sequence_id in enumerate(sequences):
                if progress is not None:
                    progress(
                        f"CMC analysis: estimating camera motion for {sequence_id} ({number + 1}/{len(sequences)})…"
                    )
                results.append(
                    _analyze_sequence(
                        args, sequence_id, mask_min_conf=mask_min_conf, estimator_options=estimator_options,
                        runtime_masker=mask_tracker.cmc_detection_boxes, progress=progress,
                    )
                )
        finally:
            cv2.setNumThreads(previous_threads)
            close = getattr(mask_tracker, "close", None)
            if callable(close):
                close()
    for number, (sequence_id, (stats, sequence_arrays)) in enumerate(zip(sequences, results, strict=True)):
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
    if cmc_build_signature(Path(args.build_path)) != source_signature:
        raise ValueError("Camera source build changed during analysis.")
    policy = SequenceCMCPolicy(method="sof", default=False, sequences=tuple(decisions))
    report["policy"] = policy.to_dict()
    report["total_frames"] = sum(stats["frames"] for stats in report["sequences"].values())
    np.savez_compressed(warps_path, **arrays)
    report["warps_sha256"] = hashlib.sha256(warps_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return PreparedCMCCandidates(effective_warps, policy, report_path, warps_path, report, mask_min_conf)
