"""Rich workflow reporter for the ``track`` command."""

from __future__ import annotations

from typing import Any

import boxmot.utils.rich.core.ui as ui
from boxmot.utils.rich.workflow.fields import bool_glyph, compact_model_name, image_size_text, panel_field
from boxmot.utils.rich.workflow.reporting import RichWorkflowReporter
from boxmot.utils.rich.workflow.steps import (
    SETUP as TRACK_SETUP_STEP,
)
from boxmot.utils.rich.workflow.steps import (
    TRACK as TRACK_RUN_STEP,
)
from boxmot.utils.rich.workflow.steps import (
    TRACK_STEPS,
)


def _tracker_name_from_spec_safe(spec: Any) -> str | None:
    """Best-effort tracker name extraction without importing engine internals."""
    from boxmot.trackers.specs import parse_tracker_spec

    try:
        return parse_tracker_spec(spec).name
    except Exception:
        return None


def _build_track_workflow_fields(args: Any) -> list[tuple[str, object]]:
    """Build workflow fields as compact subsystem cards (like eval view)."""
    fields: list[tuple[str, object]] = []

    # ── Tracker card ──────────────────────────────────────────────
    tracker = getattr(args, "tracker", None)
    tracker_backend = getattr(args, "tracker_backend", None)
    with_reid = getattr(args, "with_reid", None)
    cmc_method = getattr(args, "cmc_method", None)

    tracker_items: list[tuple[str, object]] = []
    if tracker:
        tracker_items.append(("Name", _tracker_name_from_spec_safe(tracker) or tracker))
    if tracker_backend not in {None, ""}:
        tracker_items.append(("Backend", tracker_backend))
    if with_reid is not None:
        tracker_items.append(("ReID", bool_glyph(with_reid)))
    if cmc_method not in {None, "", "none"}:
        tracker_items.append(("CMC", cmc_method))
    if tracker_items:
        fields.append(panel_field("Tracker", tracker_items))

    # ── Detector card ─────────────────────────────────────────────
    detector = getattr(args, "detector", None)
    detector_items: list[tuple[str, object]] = []
    if detector:
        detector_items.append(("Model", compact_model_name(detector)))
    imgsz = getattr(args, "imgsz", None)
    if imgsz is not None:
        detector_items.append(("Size", image_size_text(imgsz)))
    conf = getattr(args, "conf", None)
    if conf is not None:
        detector_items.append(("Conf", f"≥ {conf}"))
    if detector_items:
        fields.append(panel_field("Detector", detector_items))

    # ── ReID card ─────────────────────────────────────────────────
    reid = getattr(args, "reid", None)
    reid_items: list[tuple[str, object]] = []
    if reid:
        reid_items.append(("Model", compact_model_name(reid)))
    if reid_items:
        fields.append(panel_field("ReID", reid_items))

    # ── Source card ───────────────────────────────────────────────
    source = getattr(args, "source", None)
    source_items: list[tuple[str, object]] = []
    if source not in {None, ""}:
        source_items.append(("Input", source))
    if source_items:
        fields.append(panel_field("Source", source_items))

    # ── Runtime card ──────────────────────────────────────────────
    runtime_items: list[tuple[str, object]] = []
    device = getattr(args, "device", None)
    if device not in {None, ""}:
        runtime_items.append(("Device", device))
    runtime_items.append(("Precision", "fp16" if bool(getattr(args, "half", False)) else "fp32"))
    iou = getattr(args, "iou", None)
    if iou is not None:
        runtime_items.append(("IoU", iou))
    runtime_items.append(("Show", bool(getattr(args, "show", False))))
    runtime_items.append(("Save video", bool(getattr(args, "save", False))))
    runtime_items.append(("Save txt", bool(getattr(args, "save_txt", False))))
    if runtime_items:
        fields.append(panel_field("Runtime", runtime_items))

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
