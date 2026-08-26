"""Independent TrackEval reference evaluation for MOTChallenge results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from boxmot.engine.eval.motmetrics import _summary_from_bundle


def _load_trackeval():
    try:
        import trackeval
    except ImportError as exc:
        raise RuntimeError(
            "TrackEval is required for --compare-trackeval. "
            "Install it with `uv sync --extra trackeval`."
        ) from exc
    return trackeval


def _build_metrics(trackeval: Any) -> dict[str, Any]:
    config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    return {
        "HOTA": trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
        "CLEAR": trackeval.metrics.CLEAR(config),
        "Identity": trackeval.metrics.Identity(config),
        "Count": trackeval.metrics.Count(),
    }


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
        per_sequence_bundles[seq_name] = {
            family: metric.eval_sequence(data) for family, metric in metrics.items()
        }

    combined = {
        family: metric.combine_sequences(
            {seq_name: bundle[family] for seq_name, bundle in per_sequence_bundles.items()}
        )
        for family, metric in metrics.items()
    }
    combined["Count"]["Frames"] = sum(normalized_seq_info.values())
    return {
        **_summary_from_bundle(combined),
        "per_sequence": {
            seq_name: _summary_from_bundle(bundle)
            for seq_name, bundle in per_sequence_bundles.items()
        },
    }


__all__ = ["evaluate_trackeval_motchallenge"]
