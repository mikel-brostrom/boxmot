"""Score image trackers against native KITTI 2D tracking annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from boxmot.datasets.config import validate_sequence_names
from boxmot.datasets.readers.boxes2d import read_kitti_tracking_labels_2d
from boxmot.engine.eval.trackeval_reference import evaluate_trackeval_kitti, normalize_kitti_tracking_row

_INTEGER = re.compile(r"[+-]?[0-9]+")
_RESULT_LABELS = {1: "Car", 2: "Pedestrian"}


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
    """Run TrackEval's KITTI adapter on selected image boxes and unfiltered native GT."""
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
        hashes[sequence_id] = truth.source_sha256
        predicted_hashes[sequence_id] = prediction_sha256
        identity_maps[sequence_id] = {"ground_truth": gt_ids, "predictions": predicted_ids}
        frame_maps[sequence_id] = [
            {"frame_index": target, "source_frame_index": source} for source, target in mapping.items()
        ]
    results = evaluate_trackeval_kitti(
        gt_folder=native_gt, tracker_folder=native_predictions, seq_info=seq_info, class_names=class_names
    )
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = [name for name in results[class_names[0]] if name != "per_sequence"]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", *fields])
        writer.writeheader()
        for name, values in results.items():
            writer.writerow({"class": name, **{key: values[key] for key in fields}})
    protocol = {
        "protocol": "kitti-trackeval-2d-tracking",
        "tracking": {
            "evaluator": "trackeval==1.3.0",
            "dataset": "Kitti2DBox",
            "geometry": "2d",
            "classes": list(class_names),
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
