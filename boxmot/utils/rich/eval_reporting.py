"""Rich workflow reporter for the ``eval`` command."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import boxmot.utils.rich.ui as ui
from boxmot.trackers.specs import normalize_tracker_backend, parse_tracker_spec
from boxmot.trackers.tracker_zoo import get_tracker_config
from boxmot.utils.rich.reporting import RichWorkflowReporter


EVAL_SETUP_STEP = "Set up"
EVAL_GENERATE_STEP = "Generate detections and embeddings"
EVAL_TRACK_STEP = "Run tracker"
EVAL_EVALUATE_STEP = "Evaluate results"


def _effective_eval_tracker_backend(args: argparse.Namespace) -> str | None:
    tracking_backend = str(getattr(args, "tracking_backend", "") or "").strip().lower()
    if tracking_backend == "cpp":
        return "cpp"

    raw_tracker_backend = getattr(args, "tracker_backend", None)
    if raw_tracker_backend in {None, ""}:
        return None

    return normalize_tracker_backend(raw_tracker_backend, default="python")


def _format_eval_param_label(name: str) -> str:
    label = str(name).replace("_", " ").title()
    replacements = {
        "Id": "ID",
        "Idsw": "IDSW",
        "Reid": "ReID",
        "Cmc": "CMC",
        "Fps": "FPS",
        "Imgsz": "Image Size",
    }
    for source, target in replacements.items():
        label = label.replace(source, target)
    return label


def _read_yaml_mapping(cfg_path: Path | None) -> dict[str, object]:
    if cfg_path is None:
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _build_eval_tracker_parameter_fields(args: argparse.Namespace) -> list[tuple[str, object]]:
    try:
        tracker_name = parse_tracker_spec(getattr(args, "tracker", "")).name
    except Exception:
        return []

    try:
        with open(get_tracker_config(tracker_name), "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception:
        return []

    params: list[tuple[str, object]] = []
    for param_name, details in raw.items():
        value = getattr(args, param_name, details.get("default"))
        if value is None:
            value = details.get("default")
        params.append((_format_eval_param_label(param_name), value))
    return params


def _build_eval_pipeline_parameter_fields(
    args: argparse.Namespace,
    *,
    tracker_backend: str | None,
    replay_backend: str,
) -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    if tracker_backend:
        items.append(("Tracker backend", tracker_backend))
    if replay_backend not in {"", None}:
        items.append(("Replay backend", replay_backend))

    device = getattr(args, "device", None)
    if device not in {None, ""}:
        items.append(("Device", device))

    items.append(("Precision", "fp16" if bool(getattr(args, "half", False)) else "fp32"))

    imgsz = getattr(args, "imgsz", None)
    if imgsz is not None:
        items.append(("Image size", imgsz))

    conf = getattr(args, "conf", None)
    if conf is not None:
        items.append(("Confidence", conf))

    n_threads = getattr(args, "n_threads", None)
    if n_threads is not None:
        items.append(("Threads", n_threads))

    postprocessing = getattr(args, "postprocessing", None)
    if postprocessing not in {None, ""}:
        items.append(("Postprocessing", postprocessing))

    return items


def _build_eval_workflow_fields(args: argparse.Namespace) -> list[tuple[str, object]]:
    dataset = (
        getattr(args, "data", None)
        or getattr(args, "benchmark", None)
        or getattr(args, "dataset_id", None)
        or getattr(args, "benchmark_id", None)
        or getattr(args, "source", None)
    )

    fields: list[tuple[str, object]] = []

    detector = getattr(args, "detector", None)
    if detector:
        fields.append(("Detector", detector[0]))

    reid = getattr(args, "reid", None)
    if reid:
        fields.append(("ReID", reid[0]))

    tracker = getattr(args, "tracker", None)
    if tracker not in {None, ""}:
        fields.append(("Tracker", tracker))

    tracker_backend = _effective_eval_tracker_backend(args)
    if tracker_backend:
        fields.append(("Tracker backend", tracker_backend))

    replay_backend = str(getattr(args, "tracking_backend", "") or "").strip().lower()
    if replay_backend not in {"", "cpp", "process"}:
        fields.append(("Replay backend", replay_backend))

    fields.append(("Dataset", dataset))

    imgsz = getattr(args, "imgsz", None)
    if imgsz is not None:
        fields.append(("Image size", imgsz))

    tracker_params = _build_eval_tracker_parameter_fields(args)
    if tracker_params:
        fields.append(("__panel__:Tracker Parameters", tracker_params))

    pipeline_params = _build_eval_pipeline_parameter_fields(
        args, tracker_backend=tracker_backend, replay_backend=replay_backend
    )
    if pipeline_params:
        fields.append(("__panel__:Pipeline Parameters", pipeline_params))

    return fields


def _refresh_eval_pipeline_intro(
    workflow: ui.WorkflowProgress | None,
    args: argparse.Namespace,
) -> None:
    if workflow is None:
        return

    updated_fields = _build_eval_workflow_fields(args)
    if hasattr(workflow, "set_fields"):
        workflow.set_fields(updated_fields)
        return

    if hasattr(workflow, "fields"):
        workflow.fields = updated_fields


class EvalWorkflowReporter(RichWorkflowReporter):
    title = "Evaluation"
    prefer_compact_layout = True
    steps = (
        (EVAL_SETUP_STEP, "active"),
        (EVAL_GENERATE_STEP, "todo"),
        (EVAL_TRACK_STEP, "todo"),
        (EVAL_EVALUATE_STEP, "todo"),
    )

    def fields(self) -> list[tuple[str, object]]:
        return _build_eval_workflow_fields(self.args)


def log_eval_pipeline_intro(args: argparse.Namespace) -> ui.WorkflowProgress:
    # Engine module is responsible for normalizing args before constructing
    # the reporter; keep this function side-effect free here so it can be
    # imported without pulling engine internals.
    return EvalWorkflowReporter(args).create()


__all__ = [
    "EVAL_SETUP_STEP",
    "EVAL_GENERATE_STEP",
    "EVAL_TRACK_STEP",
    "EVAL_EVALUATE_STEP",
    "EvalWorkflowReporter",
    "log_eval_pipeline_intro",
    "_refresh_eval_pipeline_intro",
]
