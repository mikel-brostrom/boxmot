"""Rich workflow reporter for the ``track`` command."""

from __future__ import annotations

from typing import Any

import yaml

import boxmot.utils.rich.ui as ui
from boxmot.trackers.tracker_zoo import get_tracker_config
from boxmot.utils.rich.reporting import RichWorkflowReporter, format_param_label, primary_model_ref
from boxmot.utils.rich.steps import (
    SETUP as TRACK_SETUP_STEP,
)
from boxmot.utils.rich.steps import (
    TRACK as TRACK_RUN_STEP,
)
from boxmot.utils.rich.steps import (
    TRACK_STEPS,
)


def _tracker_name_from_spec_safe(spec):
    """Best-effort tracker name extraction without importing engine internals."""
    from boxmot.trackers.specs import parse_tracker_spec

    try:
        return parse_tracker_spec(spec).name
    except Exception:
        return None


# Use shared format_param_label from reporting module


def _build_track_tracker_parameter_fields(args) -> list[tuple[str, object]]:
    tracker_name = _tracker_name_from_spec_safe(getattr(args, "tracker", None))
    if tracker_name is None:
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
        params.append((format_param_label(param_name), value))
    return params


def _build_track_pipeline_parameter_fields(args) -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    device = getattr(args, "device", None)
    if device not in {None, ""}:
        items.append(("Device", device))

    items.append(("Precision", "fp16" if bool(getattr(args, "half", False)) else "fp32"))

    tracker_backend = getattr(args, "tracker_backend", None)
    if tracker_backend not in {None, ""}:
        items.append(("Tracker backend", tracker_backend))

    imgsz = getattr(args, "imgsz", None)
    if imgsz is not None:
        items.append(("Image size", imgsz))

    conf = getattr(args, "conf", None)
    if conf is not None:
        items.append(("Confidence", conf))

    iou = getattr(args, "iou", None)
    if iou is not None:
        items.append(("IoU", iou))

    items.append(("Show", bool(getattr(args, "show", False))))
    items.append(("Save video", bool(getattr(args, "save", False))))
    items.append(("Save txt", bool(getattr(args, "save_txt", False))))

    return items


def _build_track_workflow_fields(args) -> list[tuple[str, object]]:
    fields: list[tuple[str, object]] = []

    detector = primary_model_ref(getattr(args, "detector", None))
    if detector is not None:
        fields.append(("Detector", detector))

    reid = primary_model_ref(getattr(args, "reid", None))
    if reid is not None:
        fields.append(("ReID", reid))

    tracker = getattr(args, "tracker", None)
    if tracker not in {None, ""}:
        fields.append(("Tracker", _tracker_name_from_spec_safe(tracker) or tracker))

    source = getattr(args, "source", None)
    if source not in {None, ""}:
        fields.append(("Source", source))

    tracker_params = _build_track_tracker_parameter_fields(args)
    if tracker_params:
        fields.append(("__panel__:Tracker Parameters", tracker_params))

    pipeline_params = _build_track_pipeline_parameter_fields(args)
    if pipeline_params:
        fields.append(("__panel__:Pipeline Parameters", pipeline_params))

    return fields


class TrackWorkflowReporter(RichWorkflowReporter):
    title = "Tracking"
    SETUP = 0
    RUN = 1
    steps = TRACK_STEPS

    def fields(self) -> list[tuple[str, object]]:
        return _build_track_workflow_fields(self.args)


def log_track_pipeline_intro(args: Any) -> ui.WorkflowProgress:
    return TrackWorkflowReporter(args).create()


__all__ = [
    "TRACK_SETUP_STEP",
    "TRACK_RUN_STEP",
    "TrackWorkflowReporter",
    "log_track_pipeline_intro",
]
