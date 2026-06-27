"""Rich workflow reporter for the ``eval`` command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.console import Group, RenderableType
from rich.text import Text

import boxmot.utils.rich.core.ui as ui
from boxmot.trackers.specs import normalize_tracker_backend, parse_tracker_spec
from boxmot.trackers.registry import get_tracker_config
from boxmot.utils.rich.workflow.fields import bool_glyph, compact_model_name, image_size_text, panel_field
from boxmot.utils.rich.workflow.reporting import RichWorkflowReporter
from boxmot.utils.rich.workflow.steps import (
    EVALUATE as EVAL_EVALUATE_STEP,
)
from boxmot.utils.rich.workflow.steps import (
    GENERATE as EVAL_GENERATE_STEP,
)
from boxmot.utils.rich.workflow.steps import (
    SETUP as EVAL_SETUP_STEP,
)
from boxmot.utils.rich.workflow.steps import (
    TRACK as EVAL_TRACK_STEP,
)
from boxmot.utils.rich.workflow.steps import (
    eval_steps,
)


def _effective_eval_tracker_backend(args: argparse.Namespace) -> str | None:
    tracking_backend = str(getattr(args, "tracking_backend", "") or "").strip().lower()
    if tracking_backend == "cpp":
        return "cpp"

    raw_tracker_backend = getattr(args, "tracker_backend", None)
    if raw_tracker_backend in {None, ""}:
        return None

    return normalize_tracker_backend(raw_tracker_backend, default="python")


def _build_eval_workflow_fields(args: argparse.Namespace) -> list[tuple[str, object]]:
    """Build workflow fields as subsystem summary cards.

    Instead of dumping every tracker parameter, each subsystem gets a
    compact one-line summary with only the most relevant settings.
    """
    dataset = (
        getattr(args, "data", None)
        or getattr(args, "benchmark", None)
        or getattr(args, "dataset_id", None)
        or getattr(args, "benchmark_id", None)
        or getattr(args, "source", None)
    )

    fields: list[tuple[str, object]] = []

    # ── Tracker card ──────────────────────────────────────────────
    tracker = getattr(args, "tracker", None)
    tracker_backend = _effective_eval_tracker_backend(args)
    with_reid = getattr(args, "with_reid", None)
    cmc_method = getattr(args, "cmc_method", None)

    tracker_items: list[tuple[str, object]] = []
    if tracker:
        tracker_items.append(("Name", tracker))
    if tracker_backend:
        tracker_items.append(("Backend", tracker_backend))
    if with_reid is not None:
        tracker_items.append(("ReID", bool_glyph(with_reid)))
    if cmc_method not in {None, "", "none"}:
        tracker_items.append(("CMC", cmc_method))
    # Key thresholds only
    det_thresh = getattr(args, "det_thresh", None)
    if det_thresh is not None:
        tracker_items.append(("Det thresh", f"{det_thresh:.2f}"))
    new_track = getattr(args, "new_track_thresh", None)
    if new_track is not None:
        tracker_items.append(("New track", f"{new_track:.2f}"))
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

    # ── Dataset card ──────────────────────────────────────────────
    dataset_items: list[tuple[str, object]] = []
    if dataset:
        dataset_items.append(("Benchmark", dataset))
    split = getattr(args, "split", None)
    if split:
        dataset_items.append(("Split", split))
    if dataset_items:
        fields.append(panel_field("Dataset", dataset_items))

    # ── Runtime card ──────────────────────────────────────────────
    replay_backend = str(getattr(args, "tracking_backend", "") or "").strip().lower()
    runtime_items: list[tuple[str, object]] = []
    device = getattr(args, "device", None)
    if device not in {None, ""}:
        runtime_items.append(("Device", device))
    runtime_items.append(("Precision", "fp16" if bool(getattr(args, "half", False)) else "fp32"))
    n_threads = getattr(args, "n_threads", None)
    if n_threads is not None:
        runtime_items.append(("Threads", n_threads))
    if replay_backend not in {"", None, "cpp"}:
        runtime_items.append(("Replay", replay_backend))
    postprocessing = getattr(args, "postprocessing", None)
    if postprocessing not in {None, "", "none"}:
        runtime_items.append(("Postproc", postprocessing))
    if runtime_items:
        fields.append(panel_field("Runtime", runtime_items))

    return fields


def _config_link(label: str, path: Path) -> Text:
    """Build a single-line ``Label  file://…`` link for a config file."""
    uri = path.resolve().as_uri()
    t = Text()
    t.append(f"  {label:<10}", style=ui.STYLE_TEXT_STRONG)
    t.append(f"{path.name}", style=f"link {uri}")
    return t


def build_setup_configs_renderable(args: argparse.Namespace) -> RenderableType | None:
    """Build a renderable listing the config files used in this run.

    Each entry is a ``file://`` link so the user can click to open it.
    Returns ``None`` when no config files can be resolved.
    """
    from boxmot.configs.benchmark import (
        resolve_benchmark_cfg_path,
        resolve_detector_cfg_path,
        resolve_reid_cfg_path,
    )

    lines: list[Text] = []
    lines.append(Text("  Configs used in this pipeline:", style=ui.STYLE_TEXT_STRONG))

    # Benchmark / dataset config
    benchmark = getattr(args, "data", None) or getattr(args, "benchmark", None)
    if benchmark:
        try:
            path = resolve_benchmark_cfg_path(str(benchmark))
            lines.append(_config_link("Benchmark", path))
        except Exception:
            pass

    # Detector config
    det_cfg_ref = getattr(args, "detector_config", None)
    if det_cfg_ref is None:
        try:
            from boxmot.configs.benchmark import load_benchmark_only_cfg

            bench = load_benchmark_only_cfg(str(benchmark)) if benchmark else {}
            det_cfg_ref = bench.get("detector_config") or bench.get("detector")
        except Exception:
            pass
    if det_cfg_ref:
        try:
            path = resolve_detector_cfg_path(str(det_cfg_ref), benchmark=str(benchmark) if benchmark else None)
            lines.append(_config_link("Detector", path))
        except Exception:
            pass

    # ReID config
    reid_cfg_ref = getattr(args, "reid_config", None)
    if reid_cfg_ref is None:
        try:
            from boxmot.configs.benchmark import load_benchmark_only_cfg

            bench = load_benchmark_only_cfg(str(benchmark)) if benchmark else {}
            reid_cfg_ref = bench.get("reid_config") or bench.get("reid")
        except Exception:
            pass
    if reid_cfg_ref:
        try:
            path = resolve_reid_cfg_path(str(reid_cfg_ref), benchmark=str(benchmark) if benchmark else None)
            lines.append(_config_link("ReID", path))
        except Exception:
            pass

    # Tracker config
    tracker = getattr(args, "tracker", None)
    if tracker:
        try:
            tracker_name = parse_tracker_spec(tracker).name
            path = get_tracker_config(tracker_name)
            lines.append(_config_link("Tracker", path))
        except Exception:
            pass

    if len(lines) <= 1:
        return None

    return Group(*lines)


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
    SETUP = 0
    GENERATE = 1
    TRACK = 2
    EVALUATE = 3

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        tune_kf = bool(getattr(args, "tune_kf", False))
        pp_raw = getattr(args, "postprocessing", "none") or "none"
        has_postprocess = pp_raw.strip().lower() not in ("none", "")
        self.steps = eval_steps(tune_kf=tune_kf, postprocess=has_postprocess)

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
    "build_setup_configs_renderable",
    "log_eval_pipeline_intro",
    "_refresh_eval_pipeline_intro",
]
