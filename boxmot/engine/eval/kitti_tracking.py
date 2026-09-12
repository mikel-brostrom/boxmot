"""Score image trackers against native KITTI 2D tracking annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.datasets.config import validate_sequence_names
from boxmot.datasets.readers.boxes2d import read_kitti_tracking_labels_2d
from boxmot.engine.eval.motmetrics import (
    MetricBundle,
    SequenceData,
    _combine_bundles,
    _combine_bundles_class_averaged,
    _eval_bundle,
    _relabel_ids,
    _summary_from_bundle,
)
from boxmot.engine.eval.trackeval_reference import normalize_kitti_tracking_row

_INTEGER = re.compile(r"[+-]?[0-9]+")
_RESULT_LABELS = {1: "Car", 2: "Pedestrian"}
_FLOAT_EPS = np.finfo(float).eps


@dataclass(frozen=True, slots=True)
class _TrackingBox:
    """Keep native identities separate from floating-point image geometry."""

    identity: int
    label: str
    bounds: tuple[float, ...]
    truncation: int
    occlusion: int


def _index_kitti_rows(rows: Sequence[str], frame_count: int) -> list[list[_TrackingBox]]:
    """Index already validated tracking rows without converting identities to floats."""
    frames: list[list[_TrackingBox]] = [[] for _ in range(frame_count)]
    for row in rows:
        fields = row.split()
        frames[int(fields[0])].append(
            _TrackingBox(
                identity=int(fields[1]),
                label=fields[2].casefold(),
                bounds=tuple(map(float, fields[6:10])),
                truncation=int(fields[3]),
                occlusion=int(fields[4]),
            )
        )
    return frames


def _image_overlap(first: np.ndarray, second: np.ndarray, *, ioa: bool = False) -> np.ndarray:
    """Calculate continuous KITTI image-box IoU or intersection over the first box area."""
    intersection_edges = np.maximum(
        0.0, np.minimum(first[:, None, 2:], second[None, :, 2:]) - np.maximum(first[:, None, :2], second[None, :, :2])
    )
    intersection = intersection_edges[..., 0] * intersection_edges[..., 1]
    first_area = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    denominator = first_area[:, None]
    valid = denominator > _FLOAT_EPS
    if not ioa:
        second_area = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
        denominator = denominator + second_area[None, :] - intersection
        valid = valid & (second_area[None, :] > _FLOAT_EPS) & (denominator > _FLOAT_EPS)
    return np.divide(intersection, denominator, out=np.zeros_like(intersection), where=valid)


def _preprocess_kitti_frame(
    truth: Sequence[_TrackingBox], predictions: Sequence[_TrackingBox], class_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply KITTI class/visibility matching before unmatched height and DontCare suppression."""
    distractor = "van" if class_name == "car" else "person"
    gt = [box for box in truth if box.label in (class_name, distractor)]
    tracked = [box for box in predictions if box.label == class_name]
    gt_boxes = np.asarray([box.bounds for box in gt], dtype=float).reshape(-1, 4)
    tracker_boxes = np.asarray([box.bounds for box in tracked], dtype=float).reshape(-1, 4)
    similarity = _image_overlap(gt_boxes, tracker_boxes)
    valid_gt = np.asarray(
        [box.label == class_name and box.occlusion <= 2 and box.truncation <= 0 for box in gt], dtype=bool
    )
    keep_predictions = np.ones(len(tracked), dtype=bool)
    unmatched = np.ones(len(tracked), dtype=bool)
    if gt and tracked:
        matching_scores = similarity.copy()
        matching_scores[matching_scores < 0.5 - _FLOAT_EPS] = 0.0
        rows, columns = linear_sum_assignment(-matching_scores)
        matched = matching_scores[rows, columns] > _FLOAT_EPS
        rows, columns = rows[matched], columns[matched]
        unmatched[columns] = False
        keep_predictions[columns[~valid_gt[rows]]] = False
    unmatched_indices = np.flatnonzero(unmatched)
    if len(unmatched_indices):
        boxes = tracker_boxes[unmatched_indices]
        too_small = boxes[:, 3] - boxes[:, 1] <= 25 + _FLOAT_EPS
        ignore_boxes = np.asarray([box.bounds for box in truth if box.label == "dontcare"], dtype=float).reshape(-1, 4)
        inside_ignore = np.any(_image_overlap(boxes, ignore_boxes, ioa=True) > 0.5 + _FLOAT_EPS, axis=1)
        keep_predictions[unmatched_indices[too_small | inside_ignore]] = False
    gt_ids = np.asarray([box.identity for box in gt], dtype=np.int64)[valid_gt]
    tracker_ids = np.asarray([box.identity for box in tracked], dtype=np.int64)[keep_predictions]
    return gt_ids, tracker_ids, similarity[valid_gt][:, keep_predictions]


def _kitti_sequence_data(
    sequence_id: str,
    *,
    gt_rows: Sequence[str],
    prediction_rows: Sequence[str],
    frame_count: int,
    class_names: Sequence[str],
) -> dict[str, SequenceData]:
    """Adapt validated KITTI rows to BoxMOT's HOTA, CLEAR, Identity and Count kernels."""
    gt_frames = _index_kitti_rows(gt_rows, frame_count)
    tracker_frames = _index_kitti_rows(prediction_rows, frame_count)
    results: dict[str, SequenceData] = {}
    for class_name in class_names:
        gt_ids, tracker_ids, similarities = [], [], []
        for truth, predictions in zip(gt_frames, tracker_frames):
            gt, tracked, similarity = _preprocess_kitti_frame(truth, predictions, class_name)
            gt_ids.append(gt)
            tracker_ids.append(tracked)
            similarities.append(similarity)
        gt_ids, gt_count = _relabel_ids(gt_ids)
        tracker_ids, tracker_count = _relabel_ids(tracker_ids)
        results[class_name] = SequenceData(
            seq=sequence_id,
            gt_ids=gt_ids,
            tracker_ids=tracker_ids,
            similarity_scores=similarities,
            num_timesteps=frame_count,
            num_gt_dets=sum(map(len, gt_ids)),
            num_tracker_dets=sum(map(len, tracker_ids)),
            num_gt_ids=gt_count,
            num_tracker_ids=tracker_count,
        )
    return results


def _summarize_kitti_metrics(
    bundles: Mapping[str, Mapping[str, MetricBundle]], *, frame_count: int
) -> dict[str, dict[str, Any]]:
    """Combine sequence and class metrics with the existing BoxMOT report contract."""
    combined = {name: _combine_bundles(values) for name, values in bundles.items()}
    results = {
        name: {
            **_summary_from_bundle(combined[name]),
            "per_sequence": {sequence: _summary_from_bundle(bundle) for sequence, bundle in values.items()},
        }
        for name, values in bundles.items()
    }
    for name, combine in (
        ("cls_comb_cls_av", _combine_bundles_class_averaged),
        ("cls_comb_det_av", _combine_bundles),
    ):
        bundle = combine(combined)
        bundle["Count"]["Frames"] = frame_count
        results[name] = _summary_from_bundle(bundle)
    return results


def _evaluation_classes(args: argparse.Namespace, config: Mapping[str, Any]) -> tuple[str, ...]:
    """Respect an experiment's selected native class bridge when scoring."""
    configured = {
        name.casefold(): value["id"]
        for name, value in config.get("classes", {}).items()
        if value.get("evaluation") == "target"
    }
    native = {name.casefold(): identity for identity, name in _RESULT_LABELS.items()}
    if not configured or any(
        type(identity) is not int or native.get(name) != identity for name, identity in configured.items()
    ):
        raise ValueError("KITTI tracking evaluation requires native target class pairs car (1) and/or pedestrian (2).")
    names = getattr(args, "remapped_class_names", None)
    identities = getattr(args, "remapped_class_ids", None)
    if names is None and identities is None:
        return tuple(sorted(configured, key=configured.get))
    if (
        not isinstance(names, (list, tuple))
        or not isinstance(identities, (list, tuple))
        or not names
        or len(names) != len(identities)
        or any(
            not isinstance(name, str) or type(identity) is not int or configured.get(name.casefold()) != identity
            for name, identity in zip(names, identities)
        )
    ):
        raise ValueError("KITTI evaluation class names and IDs must be aligned native car (1)/pedestrian (2) pairs.")
    selected = tuple(name.casefold() for name in names)
    if len(set(selected)) != len(selected):
        raise ValueError("KITTI evaluation classes must not contain duplicates.")
    return selected


def _integer(text: str, *, minimum: int, name: str) -> int:
    """Parse integer fields directly so identities never pass through floats."""
    if _INTEGER.fullmatch(text) is None:
        raise ValueError(f"{name} must be an integer")
    value = int(text)
    if not minimum <= value <= (1 << 63) - 1:
        raise ValueError(f"{name} must be an int64 integer >= {minimum}")
    return value


def _frame_map(selection: Mapping[str, Any], *, frame_count: int) -> dict[int, int]:
    """Validate the catalog's native-to-evaluation timeline without inventing frames."""
    native_count = selection.get("frame_count")
    if type(native_count) is not int or native_count <= 0:
        raise ValueError("KITTI annotation selection requires a positive native frame_count.")
    pairs = selection.get("frames")
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise ValueError("KITTI annotation selection requires selected evaluation/source frame pairs.")
    mapping: dict[int, int] = {}
    evaluated: set[int] = set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("KITTI frame selections must contain evaluation/source frame pairs.")
        target, source = pair
        if (
            type(target) is not int
            or type(source) is not int
            or not 0 <= target < frame_count
            or not 0 <= source < native_count
            or source in mapping
            or target in evaluated
        ):
            raise ValueError("KITTI selected frames must be unique integers inside both sequence timelines.")
        mapping[source] = target
        evaluated.add(target)
    if max(evaluated) + 1 != frame_count:
        raise ValueError("KITTI evaluation frame count must agree with its selected frames.")
    if sorted(mapping, key=mapping.get) != sorted(mapping):
        raise ValueError("KITTI selected frames must preserve chronological source order.")
    return mapping


def _prediction_rows(
    path: Path, *, frame_count: int, selected_frames: set[int]
) -> tuple[list[str], dict[int, int], str]:
    """Convert AABB9 replay output to 18-field KITTI rows with unused 3D placeholders."""
    payload = path.read_bytes()
    rows: list[str] = []
    identities: dict[int, int] = {}
    seen: set[tuple[int, int]] = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 9:
                raise ValueError("expected nine AABB replay columns")
            frame = _integer(fields[0], minimum=1, name="frame number") - 1
            identity = _integer(fields[1], minimum=0, name="track identity")
            class_id = _integer(fields[7], minimum=0, name="class ID")
            _integer(fields[8], minimum=-1, name="detection index")
            if not frame < frame_count or frame not in selected_frames:
                raise ValueError("frame number is outside the selected image timeline")
            if class_id not in _RESULT_LABELS:
                raise ValueError("KITTI tracking classes must be car (1) or pedestrian (2)")
            left, top, width, height, score = map(float, fields[2:7])
            right, bottom = left + width, top + height
            if not all(math.isfinite(value) for value in (left, top, width, height, right, bottom, score)):
                raise ValueError("prediction image bounds and confidence must be finite")
            if width <= 0 or height <= 0 or right <= left or bottom <= top:
                raise ValueError("prediction image bounds must have positive width and height")
            if not 0 <= score <= 1:
                raise ValueError("prediction confidence must be between 0 and 1")
            if (frame, identity) in seen:
                raise ValueError(f"duplicate track identity {identity} in frame {frame}")
            seen.add((frame, identity))
            native = [
                str(frame),
                str(identity),
                _RESULT_LABELS[class_id],
                "-1",
                "-1",
                "-10",
                *(format(value, ".17g") for value in (left, top, right, bottom)),
                "-1",
                "-1",
                "-1",
                "-1000",
                "-1000",
                "-1000",
                "-10",
                format(score, ".17g"),
            ]
            rows.append(normalize_kitti_tracking_row(native, identities))
        except ValueError as error:
            raise ValueError(f"Invalid KITTI 2D replay results at {path}:{line_number}: {error}.") from error
    return rows, identities, hashlib.sha256(payload).hexdigest()


def run_kitti_tracking_metrics(
    args: argparse.Namespace,
    seq_paths: Sequence[Path],
    save_dir: Path,
    gt_folder: Path,
    *,
    seq_info: Mapping[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Score selected image boxes using built-in metrics and KITTI tracking preprocessing."""
    del seq_paths, gt_folder
    config = getattr(args, "evaluation_config", {})
    selections = config.get("kitti_gt_sequences")
    if not isinstance(selections, Mapping) or not selections or seq_info is None or set(selections) != set(seq_info):
        raise ValueError("KITTI tracking evaluation requires matching annotation and frame-count sequence sets.")
    validate_sequence_names(tuple(selections))
    if any(type(count) is not int or count <= 0 for count in seq_info.values()):
        raise ValueError("KITTI tracking evaluation requires positive integer sequence frame counts.")
    class_names = _evaluation_classes(args, config)
    output = Path(save_dir)
    root = output / "protocol_inputs" / "tracking"
    native_gt, native_predictions = root / "ground_truth", root / "predictions"
    (native_gt / "label_02").mkdir(parents=True, exist_ok=True)
    native_predictions.mkdir(parents=True, exist_ok=True)
    hashes, predicted_hashes, identity_maps, frame_maps = {}, {}, {}, {}
    bundles: dict[str, dict[str, MetricBundle]] = {name: {} for name in class_names}
    for sequence_id, selection in selections.items():
        mapping = _frame_map(selection, frame_count=seq_info[sequence_id])
        truth = read_kitti_tracking_labels_2d(
            Path(selection["path"]),
            frame_count=selection["frame_count"],
            cache_inputs=bool(getattr(args, "cache_inputs", False)),
        )
        gt_ids: dict[int, int] = {}
        gt_rows: list[str] = []
        for row in truth.source_rows:
            fields = row.split()
            target = mapping.get(int(fields[0]))
            if target is not None:
                fields[0] = str(target)
                gt_rows.append(normalize_kitti_tracking_row(fields, gt_ids))
        rows, predicted_ids, prediction_sha256 = _prediction_rows(
            Path(args.exp_dir) / f"{sequence_id}.txt",
            frame_count=seq_info[sequence_id],
            selected_frames=set(mapping.values()),
        )
        (native_gt / "label_02" / f"{sequence_id}.txt").write_text(
            "".join(row + "\n" for row in gt_rows), encoding="utf-8"
        )
        (native_predictions / f"{sequence_id}.txt").write_text("".join(row + "\n" for row in rows), encoding="utf-8")
        sequence_data = _kitti_sequence_data(
            sequence_id,
            gt_rows=gt_rows,
            prediction_rows=rows,
            frame_count=seq_info[sequence_id],
            class_names=class_names,
        )
        for name, data in sequence_data.items():
            bundles[name][sequence_id] = _eval_bundle(data)
        hashes[sequence_id] = truth.source_sha256
        predicted_hashes[sequence_id] = prediction_sha256
        identity_maps[sequence_id] = {"ground_truth": gt_ids, "predictions": predicted_ids}
        frame_maps[sequence_id] = [
            {"frame_index": target, "source_frame_index": source} for source, target in mapping.items()
        ]
    results = _summarize_kitti_metrics(bundles, frame_count=sum(seq_info.values()))
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = [name for name in results[class_names[0]] if name != "per_sequence"]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", *fields])
        writer.writeheader()
        for name, values in results.items():
            writer.writerow({"class": name, **{key: values[key] for key in fields}})
    protocol = {
        "protocol": "kitti-2d-tracking",
        "tracking": {
            "evaluator": "boxmot",
            "implementation": "boxmot.engine.eval.motmetrics",
            "dataset": "KITTI 2D tracking",
            "geometry": "2d",
            "classes": list(class_names),
            "preprocessing": {
                "matching_iou": 0.5,
                "max_occlusion": 2,
                "max_truncation": 0,
                "unmatched_min_height": 25,
                "unmatched_dontcare_ioa": 0.5,
                "distractor_classes": {"car": "van", "pedestrian": "person"},
            },
        },
        "prediction_geometry": "Tracked image boxes; unused KITTI 3D fields contain placeholders.",
        "tracking_class_alias": {"Person_sitting": "Person"},
        "tracking_ground_truth_sha256": hashes,
        "prediction_sha256": predicted_hashes,
        "tracking_identity_maps": identity_maps,
        "frames": frame_maps,
    }
    (output / "evaluation.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return results


__all__ = ("run_kitti_tracking_metrics",)
