"""Installed TrackEval dataset pipelines for MOTChallenge and KITTI tracking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from boxmot.engine.eval.motmetrics import _summary_from_bundle


def _load_trackeval():
    try:
        import trackeval
    except ImportError as exc:
        raise RuntimeError(
            "TrackEval is required for the requested reference evaluation. "
            "Install it with `boxmot install --extra trackeval`."
        ) from exc
    return trackeval


def validate_trackeval_kitti_dependencies() -> None:
    """Require the installed, pinned evaluator for optional KITTI reference reports."""
    _load_trackeval()
    try:
        installed = version("trackeval")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "KITTI reference evaluation requires the installed trackeval==1.3.0 distribution. "
            "Run `boxmot install --extra trackeval`."
        ) from error
    if installed != "1.3.0":
        raise RuntimeError(
            f"KITTI reference evaluation requires trackeval==1.3.0; found {installed}. "
            "Run `boxmot install --extra trackeval`."
        )


def _build_metrics(trackeval: Any) -> dict[str, Any]:
    config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    return {
        "HOTA": trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
        "CLEAR": trackeval.metrics.CLEAR(config),
        "Identity": trackeval.metrics.Identity(config),
        "Count": trackeval.metrics.Count(),
    }


def normalize_kitti_tracking_row(fields: list[str], identity_map: dict[int, int]) -> str:
    """Compact identities losslessly before TrackEval's floating-point parser."""
    fields = fields.copy()
    identity = int(fields[1])
    if identity >= 0:
        fields[1] = str(identity_map.setdefault(identity, len(identity_map)))
    if fields[2].casefold() == "person_sitting":
        fields[2] = "Person"
    return " ".join(fields)


def evaluate_trackeval_motchallenge(
    *,
    gt_folder: Path,
    tracker_folder: Path,
    seq_info: Mapping[str, int | None],
    benchmark: str,
) -> dict[str, Any]:
    """Evaluate one BoxMOT result directory through TrackEval's dataset pipeline.

    TrackEval reads both the run-local ground truth and tracker files itself. This
    intentionally covers its MOTChallenge preprocessing, including removal of
    detections matched to distractor ground truth, rather than only comparing
    metric formulas on data preprocessed by BoxMOT.
    """
    trackeval = _load_trackeval()
    tracker_folder = Path(tracker_folder).resolve()
    gt_folder = Path(gt_folder).resolve()
    normalized_seq_info = {str(name): int(length or 0) for name, length in seq_info.items()}
    if not normalized_seq_info or any(length <= 0 for length in normalized_seq_info.values()):
        raise ValueError("TrackEval comparison requires a positive frame count for every sequence")

    dataset = trackeval.datasets.MotChallenge2DBox(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(tracker_folder.parent),
            "OUTPUT_FOLDER": str(tracker_folder.parent),
            "TRACKERS_TO_EVAL": [tracker_folder.name],
            "TRACKER_DISPLAY_NAMES": [tracker_folder.name],
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_SUB_FOLDER": "",
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": benchmark,
            "SPLIT_TO_EVAL": "train",
            "SEQ_INFO": normalized_seq_info,
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt_temp.txt",
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": True,
            "PRINT_CONFIG": False,
        }
    )
    metrics = _build_metrics(trackeval)
    per_sequence_bundles: dict[str, dict[str, dict[str, Any]]] = {}
    for seq_name in normalized_seq_info:
        raw_data = dataset.get_raw_seq_data(tracker_folder.name, seq_name)
        data = dataset.get_preprocessed_seq_data(raw_data, "pedestrian")
        per_sequence_bundles[seq_name] = {family: metric.eval_sequence(data) for family, metric in metrics.items()}

    combined = {
        family: metric.combine_sequences(
            {seq_name: bundle[family] for seq_name, bundle in per_sequence_bundles.items()}
        )
        for family, metric in metrics.items()
    }
    combined["Count"]["Frames"] = sum(normalized_seq_info.values())
    return {
        **_summary_from_bundle(combined),
        "per_sequence": {seq_name: _summary_from_bundle(bundle) for seq_name, bundle in per_sequence_bundles.items()},
    }


def evaluate_trackeval_kitti(
    *,
    gt_folder: Path,
    tracker_folder: Path,
    seq_info: Mapping[str, int],
    class_names: Sequence[str] = ("car", "pedestrian"),
) -> dict[str, dict[str, Any]]:
    """Run pinned KITTI 2D tracking preprocessing and metrics on saved KITTI rows.

    Ground truth retains distractors, truncation, occlusion, and DontCare rows.
    The installed dataset adapter owns every filtering and association rule.
    """
    if (
        isinstance(class_names, (str, bytes))
        or not isinstance(class_names, Sequence)
        or not class_names
        or any(name not in ("car", "pedestrian") for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError("KITTI tracking evaluation classes must be a nonempty, unique subset of car and pedestrian.")
    classes = tuple(class_names)
    validate_trackeval_kitti_dependencies()
    trackeval = _load_trackeval()
    tracker_folder = Path(tracker_folder).resolve()
    gt_folder = Path(gt_folder).resolve()
    if not seq_info or any(type(count) is not int or count <= 0 for count in seq_info.values()):
        raise ValueError("KITTI tracking evaluation requires a positive frame count for every sequence.")
    (gt_folder / "evaluate_tracking.seqmap.training").write_text(
        "".join(f"{name} empty 0 {count}\n" for name, count in seq_info.items()), encoding="utf-8"
    )
    dataset = trackeval.datasets.Kitti2DBox(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(tracker_folder.parent),
            "OUTPUT_FOLDER": str(tracker_folder.parent),
            "TRACKERS_TO_EVAL": [tracker_folder.name],
            "TRACKER_DISPLAY_NAMES": [tracker_folder.name],
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_SUB_FOLDER": "",
            "CLASSES_TO_EVAL": list(classes),
            "SPLIT_TO_EVAL": "training",
            "PRINT_CONFIG": False,
        }
    )
    metrics = _build_metrics(trackeval)
    bundles: dict[str, dict[str, Any]] = {name: {} for name in classes}
    for sequence_id in seq_info:
        raw_data = dataset.get_raw_seq_data(tracker_folder.name, sequence_id)
        for name in classes:
            data = dataset.get_preprocessed_seq_data(raw_data, name)
            bundles[name][sequence_id] = {family: metric.eval_sequence(data) for family, metric in metrics.items()}
    combined = {
        name: {
            family: metric.combine_sequences({sequence: bundle[family] for sequence, bundle in values.items()})
            for family, metric in metrics.items()
        }
        for name, values in bundles.items()
    }
    frame_count = sum(seq_info.values())
    for bundle in combined.values():
        bundle["Count"]["Frames"] = frame_count
    results = {
        name: {
            **_summary_from_bundle(combined[name]),
            "per_sequence": {sequence: _summary_from_bundle(bundle) for sequence, bundle in values.items()},
        }
        for name, values in bundles.items()
    }
    for name, method in (
        ("cls_comb_cls_av", "combine_classes_class_averaged"),
        ("cls_comb_det_av", "combine_classes_det_averaged"),
    ):
        bundle = {
            family: getattr(metric, method)({class_name: values[family] for class_name, values in combined.items()})
            for family, metric in metrics.items()
        }
        bundle["Count"]["Frames"] = frame_count
        results[name] = _summary_from_bundle(bundle)
    return results


__all__ = [
    "evaluate_trackeval_kitti",
    "evaluate_trackeval_motchallenge",
    "normalize_kitti_tracking_row",
    "validate_trackeval_kitti_dependencies",
]
