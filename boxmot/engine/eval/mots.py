"""KITTI MOTS segmentation evaluation using BoxMOT's metric implementation.

Mask matching and ignore-region preprocessing are adapted from TrackEval's
``trackeval/datasets/kitti_mots.py`` and ``_base_dataset.py`` (MIT license;
copyright (c) 2020 Jonathon Luiten). See ``licenses/TrackEval.txt`` beside this module.
TrackEval is not required at runtime. HOTA, CLEAR, Identity, and Count share the
same arithmetic and report format as BoxMOT's box evaluation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.engine.eval.motmetrics import (
    MetricBundle,
    SequenceData,
    _append_aggregate_results,
    _benchmark_config,
    _combine_bundles,
    _eval_bundle,
    _format_results,
    _load_eval_cfg,
    _relabel_ids,
    _sequence_names_from_paths,
)

if TYPE_CHECKING:
    from boxmot.datasets.sensor_cache import SensorReplaySequence

_CLASS_IDS = {"car": 1, "pedestrian": 2}
_FLOAT_EPS = np.finfo(float).eps
GroundTruthFrame = tuple[int, Path, int, int]
EncodedMask = dict[str, Any]
GroundTruthMasks = tuple[np.ndarray, np.ndarray, list[EncodedMask], EncodedMask | None]


def _mask_ious(masks1: Sequence[EncodedMask], masks2: Sequence[EncodedMask], *, do_ioa: bool = False) -> np.ndarray:
    """Return pixel IoU, or intersection divided by the first mask's area."""
    from pycocotools import mask as mask_utils

    if not masks1 or not masks2:
        return np.zeros((len(masks1), len(masks2)), dtype=float)
    return np.asarray(mask_utils.iou(list(masks1), list(masks2), [do_ioa] * len(masks2)), dtype=float)


def _load_gt_labels(path: Path, *, cache_inputs: bool = False, cache_root: Path | None = None) -> np.ndarray:
    """Load annotation labels, optionally reusing their immutable decoded array."""

    def decode(source: Path) -> np.ndarray:
        labels = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if labels is None:
            raise ValueError(f"Unable to decode KITTI MOTS instance PNG: {source}")
        if labels.dtype != np.uint16 or labels.ndim != 2:
            raise ValueError(
                f"KITTI MOTS instance PNG must be single-channel uint16, got {labels.shape}, {labels.dtype}: {source}"
            )
        return labels

    if cache_inputs:
        from boxmot.datasets.annotation_cache import load_cached_annotation

        return load_cached_annotation(path, reader=decode, format="instance-png-uint16/v1", cache_root=cache_root)
    return decode(path)


def _read_gt_frame(
    path: Path,
    height: int,
    width: int,
    *,
    ignore_ids: Sequence[int] = (10000,),
    ignore_class_ids: Sequence[int] = (),
    cache_inputs: bool = False,
    cache_root: Path | None = None,
) -> GroundTruthMasks:
    """Encode one label PNG without retaining dense masks across frames."""
    from pycocotools import mask as mask_utils

    labels = _load_gt_labels(path, cache_inputs=cache_inputs, cache_root=cache_root)
    if labels.shape != (height, width):
        raise ValueError(
            f"KITTI MOTS instance PNG dimensions {labels.shape} do not match image dimensions {(height, width)}: {path}"
        )
    object_ids = np.unique(labels)
    ignored_ids = object_ids[np.isin(object_ids, ignore_ids) | np.isin(object_ids // 1000, ignore_class_ids)]
    excluded = (object_ids == 0) | np.isin(object_ids, ignored_ids)
    valid = excluded | ((object_ids >= 1000) & (object_ids < 3000))
    if not valid.all():
        raise ValueError(f"KITTI MOTS instance PNG contains unsupported labels {object_ids[~valid].tolist()}: {path}")
    object_ids = object_ids[~excluded].astype(np.int64)
    masks = [mask_utils.encode(np.asfortranarray(labels == object_id, dtype=np.uint8)) for object_id in object_ids]
    ignore_mask = (
        mask_utils.encode(np.asfortranarray(np.isin(labels, ignored_ids), dtype=np.uint8)) if len(ignored_ids) else None
    )
    return object_ids, object_ids // 1000, masks, ignore_mask


def _preprocess_frame(
    gt_ids: np.ndarray,
    gt_masks: Sequence[EncodedMask],
    tracker_ids: np.ndarray,
    tracker_masks: Sequence[EncodedMask],
    ignore_mask: EncodedMask | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match one class and remove only unmatched masks mostly inside ignore."""
    similarity = _mask_ious(gt_masks, tracker_masks)
    unmatched = np.arange(len(tracker_ids))
    if len(gt_ids) and len(tracker_ids):
        matching_scores = similarity.copy()
        matching_scores[matching_scores < 0.5 - _FLOAT_EPS] = -10000
        rows, cols = linear_sum_assignment(-matching_scores)
        matched = cols[matching_scores[rows, cols] > _FLOAT_EPS]
        unmatched = np.delete(unmatched, matched)

    if ignore_mask is not None and len(unmatched):
        ignore_ioa = _mask_ious([tracker_masks[index] for index in unmatched], [ignore_mask], do_ioa=True)
        removed = unmatched[np.any(ignore_ioa > 0.5 + _FLOAT_EPS, axis=1)]
        tracker_ids = np.delete(tracker_ids, removed)
        similarity = np.delete(similarity, removed, axis=1)
    return tracker_ids, similarity


def _resolve_class_pairs(args: argparse.Namespace, config: Mapping[str, Any]) -> list[tuple[str, int]]:
    """Resolve selected classes while preserving native KITTI class IDs."""
    names = getattr(args, "remapped_class_names", None)
    ids = getattr(args, "remapped_class_ids", None)
    if names is None and ids is None:
        names = getattr(args, "tracker_class_names", None)
        ids = getattr(args, "tracker_class_ids", None)
    if names is not None or ids is not None:
        if not names or not ids or len(names) != len(ids):
            raise ValueError("KITTI MOTS evaluation class names and IDs must be non-empty and aligned")
        pairs = [(str(name).lower(), int(class_id)) for name, class_id in zip(names, ids)]
    else:
        configured = _benchmark_config(config).get("eval_classes")
        pairs = (
            [(str(name).lower(), int(class_id)) for class_id, name in configured.items()]
            if configured
            else list(_CLASS_IDS.items())
        )
        selected = getattr(args, "classes", None)
        if selected is not None:
            wanted = {int(value) for value in selected}
            if not wanted.issubset(_CLASS_IDS.values()):
                raise ValueError("KITTI MOTS evaluates only car (1) and pedestrian (2)")
            pairs = [(name, class_id) for name, class_id in pairs if class_id in wanted]
    if not pairs or any(_CLASS_IDS.get(name) != class_id for name, class_id in pairs):
        raise ValueError("KITTI MOTS evaluates only car (1) and pedestrian (2)")
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError("KITTI MOTS evaluation classes must not contain duplicates")
    return sorted(pairs, key=lambda pair: pair[1])


def _validated_gt_frames(
    seq_name: str, entries: Sequence[Any], num_timesteps: int | None, *, check_files: bool = True
) -> tuple[tuple[GroundTruthFrame, ...], int]:
    """Validate catalog-provided paths, frame numbers, and image dimensions."""
    frames: dict[int, GroundTruthFrame] = {}
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            raise ValueError(f"Invalid KITTI MOTS ground-truth frame metadata for {seq_name}: {entry!r}")
        frame_index, raw_path, height, width = entry
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in (frame_index, height, width)
        ):
            raise ValueError(f"KITTI MOTS frame indices and dimensions must be integers for {seq_name}")
        if frame_index < 0 or height <= 0 or width <= 0:
            raise ValueError(f"KITTI MOTS frame indices must be non-negative and dimensions positive for {seq_name}")
        if frame_index in frames:
            raise ValueError(f"Duplicate KITTI MOTS ground-truth frame {frame_index} for {seq_name}")
        path = Path(raw_path)
        if not path.is_absolute() or path.suffix.lower() != ".png":
            raise ValueError(f"KITTI MOTS ground-truth annotations must use absolute PNG paths: {path}")
        if check_files and not path.is_file():
            raise FileNotFoundError(f"Missing KITTI MOTS instance PNG: {path}")
        frames[int(frame_index)] = (int(frame_index), path, int(height), int(width))
    if not frames:
        raise ValueError(f"No KITTI MOTS ground-truth frames selected for {seq_name}")
    minimum_length = max(frames) + 1
    if num_timesteps is None:
        num_timesteps = minimum_length
    if (
        isinstance(num_timesteps, bool)
        or not isinstance(num_timesteps, (int, np.integer))
        or num_timesteps < minimum_length
    ):
        raise ValueError(f"KITTI MOTS sequence length must cover selected frame indices for {seq_name}")
    return tuple(frames[index] for index in sorted(frames)), int(num_timesteps)


def _build_mots_sequence_data(
    seq_name: str,
    gt_frames: Sequence[GroundTruthFrame],
    tracker_path: Path,
    class_pairs: Sequence[tuple[str, int]],
    num_timesteps: int,
    *,
    ground_truth: Callable[[int], GroundTruthMasks | None] | None = None,
    ground_truth_options: Mapping[str, Any] | None = None,
    cache_inputs: bool = False,
    cache_root: Path | None = None,
) -> dict[str, SequenceData]:
    """Decode GT frame by frame and retain only metric-ready IDs and IoUs."""
    from boxmot.engine.eval.mots_io import read_mots_results

    frame_shapes = {frame_index: (height, width) for frame_index, _, height, width in gt_frames}
    tracker_frames = read_mots_results(tracker_path, frame_shapes=frame_shapes)
    gt_ids_by_class = {name: [np.empty(0, dtype=int) for _ in range(num_timesteps)] for name, _ in class_pairs}
    tracker_ids_by_class = {name: [np.empty(0, dtype=int) for _ in range(num_timesteps)] for name, _ in class_pairs}
    similarities = {name: [np.empty((0, 0), dtype=float) for _ in range(num_timesteps)] for name, _ in class_pairs}
    for frame_index, path, height, width in gt_frames:
        if ground_truth is None:
            options = dict(ground_truth_options or {})
            if cache_inputs:
                options.update(cache_inputs=True, cache_root=cache_root)
            annotation = _read_gt_frame(path, height, width, **options)
        else:
            annotation = ground_truth(frame_index)
        if annotation is None:
            raise ValueError(f"No cached ground truth for {seq_name!r} frame {frame_index}.")
        gt_ids, gt_classes, gt_masks, ignore_mask = annotation
        rows = tracker_frames.pop(frame_index, ())
        for name, class_id in class_pairs:
            gt_selected = np.flatnonzero(gt_classes == class_id)
            selected_rows = [row for row in rows if row.class_id == class_id]
            frame_gt_ids = gt_ids[gt_selected]
            frame_tracker_ids, similarity = _preprocess_frame(
                frame_gt_ids,
                [gt_masks[index] for index in gt_selected],
                np.asarray([row.track_id for row in selected_rows], dtype=np.int64),
                [row.rle for row in selected_rows],
                ignore_mask,
            )
            gt_ids_by_class[name][frame_index] = frame_gt_ids
            tracker_ids_by_class[name][frame_index] = frame_tracker_ids
            similarities[name][frame_index] = similarity

    data: dict[str, SequenceData] = {}
    for name, _ in class_pairs:
        sequence_gt_ids, num_gt_ids = _relabel_ids(gt_ids_by_class[name])
        sequence_tracker_ids, num_tracker_ids = _relabel_ids(tracker_ids_by_class[name])
        data[name] = SequenceData(
            seq=seq_name,
            gt_ids=sequence_gt_ids,
            tracker_ids=sequence_tracker_ids,
            similarity_scores=similarities[name],
            num_timesteps=num_timesteps,
            num_gt_dets=sum(map(len, sequence_gt_ids)),
            num_tracker_dets=sum(map(len, sequence_tracker_ids)),
            num_gt_ids=num_gt_ids,
            num_tracker_ids=num_tracker_ids,
        )
    return data


def run_mots_metrics(
    args: argparse.Namespace,
    seq_paths: Sequence[Path],
    save_dir: Path,
    gt_folder: Path,
    *,
    seq_info: Mapping[str, int | None] | None = None,
    cached_ground_truth: Mapping[str, SensorReplaySequence] | None = None,
    ground_truth_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate zero-based KITTI MOTS RLE results against selected label PNGs.

    The engine supplies ``evaluation_config['mots_gt_frames']`` as a mapping
    from sequence names to ``(frame_index, absolute_png_path, height, width)``
    tuples. These are the selected catalog frames, including any FPS remapping.
    Result rows on unselected frames are rejected. Sparse timeline gaps retain
    their frame indices and count toward the sequence's frame total.
    """
    del save_dir
    config = _load_eval_cfg(args)
    sequences = _sequence_names_from_paths(seq_paths, seq_info)
    if not sequences:
        raise ValueError("No KITTI MOTS sequences selected for evaluation")
    annotations = config.get("mots_gt_frames")
    if not isinstance(annotations, Mapping):
        raise ValueError("KITTI MOTS evaluation requires catalog-resolved mots_gt_frames")
    class_pairs = _resolve_class_pairs(args, config)
    per_class_sequence: dict[str, dict[str, MetricBundle]] = {name: {} for name, _ in class_pairs}
    for seq_name, num_timesteps in sorted(sequences.items()):
        if seq_name not in annotations:
            raise ValueError(f"No KITTI MOTS ground-truth frame metadata for {seq_name}")
        frames, num_timesteps = _validated_gt_frames(
            seq_name, annotations[seq_name], num_timesteps, check_files=cached_ground_truth is None
        )
        options = {}
        if ground_truth_options is not None:
            options["ground_truth_options"] = ground_truth_options[seq_name]
        if getattr(args, "cache_inputs", False):
            options.update(cache_inputs=True, cache_root=Path(gt_folder) / ".boxmot/replay_cache/annotations")
        if cached_ground_truth is not None:
            if seq_name not in cached_ground_truth:
                raise ValueError(f"No cached ground-truth sequence for {seq_name!r}.")
            cached = cached_ground_truth[seq_name]
            cached.validate()
            if cached.sequence_id != seq_name or len(cached) != num_timesteps:
                raise ValueError(f"Cached ground truth does not match sequence {seq_name!r}.")
            options["ground_truth"] = cached.ground_truth
        sequence_data = _build_mots_sequence_data(
            seq_name, frames, Path(args.exp_dir) / f"{seq_name}.txt", class_pairs, num_timesteps, **options
        )
        for name, data in sequence_data.items():
            per_class_sequence[name][seq_name] = _eval_bundle(data)
    combined = {name: _combine_bundles(per_class_sequence[name]) for name, _ in class_pairs}
    results = _format_results(combined, per_class_sequence)
    _append_aggregate_results(results, combined, include_obb_super_categories=False)
    return results


__all__ = ("run_mots_metrics",)
