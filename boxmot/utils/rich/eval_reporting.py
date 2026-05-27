"""Rich workflow reporter for the ``eval`` command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from rich.console import Group, RenderableType
from rich.text import Text

import boxmot.utils.rich.ui as ui
from boxmot.trackers.specs import normalize_tracker_backend, parse_tracker_spec
from boxmot.trackers.tracker_zoo import get_tracker_config
from boxmot.utils.rich.reporting import RichWorkflowReporter, format_param_label
from boxmot.utils.rich.steps import (
    eval_steps,
)
from boxmot.utils.rich.steps import (
    EVALUATE as EVAL_EVALUATE_STEP,
)
from boxmot.utils.rich.steps import (
    GENERATE as EVAL_GENERATE_STEP,
)
from boxmot.utils.rich.steps import (
    SETUP as EVAL_SETUP_STEP,
)
from boxmot.utils.rich.steps import (
    TRACK as EVAL_TRACK_STEP,
)


def _effective_eval_tracker_backend(args: argparse.Namespace) -> str | None:
    tracking_backend = str(getattr(args, "tracking_backend", "") or "").strip().lower()
    if tracking_backend == "cpp":
        return "cpp"

    raw_tracker_backend = getattr(args, "tracker_backend", None)
    if raw_tracker_backend in {None, ""}:
        return None

    return normalize_tracker_backend(raw_tracker_backend, default="python")


# Use shared format_param_label from reporting module


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
        params.append((format_param_label(param_name), value))
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


def _read_yaml_text(path: Path) -> str | None:
    """Read a YAML file and return its text, or None on failure."""
    try:
        return path.read_text(encoding="utf-8").rstrip()
    except Exception:
        return None


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
    benchmark = (
        getattr(args, "data", None)
        or getattr(args, "benchmark", None)
    )
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
            path = resolve_detector_cfg_path(str(det_cfg_ref))
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
            path = resolve_reid_cfg_path(str(reid_cfg_ref))
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
