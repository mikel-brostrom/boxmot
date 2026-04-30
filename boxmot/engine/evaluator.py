# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import yaml

from boxmot.data.benchmark import (
    COCO_CLASSES,
    _ordered_benchmark_eval_class_names,
    build_gt_class_remap,
    configure_benchmark_runtime,
    eval_init,
    load_benchmark_cfg_from_args,
    prepare_aabb_eval_gt,
    resolve_eval_box_type,
)
from boxmot.data.cache import (
    AppendableNpyWriter,
    _collect_seq_info,
    _existing_cache_path,
    _existing_embedding_cache_path,
    _load_embedding_cache_array,
    _load_numeric_cache_array,
    _max_frame_id,
    _migrate_legacy_embedding_cache,
    _migrate_legacy_numeric_cache,
    _saved_detection_column_count,
)
from boxmot.detectors import get_runtime_detector_cfg
from boxmot.engine.cache import generate_dets_embs_batched, run_generate_dets_embs
from boxmot.engine.replay import process_sequence, run_generate_mot_results
from boxmot.engine.workflow_reporting import extract_summary, timing_summary_from_stats
from boxmot.engine.workflow_results import ValidationResult
from boxmot.engine.workflow_support import suppress_boxmot_logs
from boxmot.trackers.specs import normalize_tracker_backend, parse_tracker_spec
from boxmot.trackers.tracker_zoo import get_tracker_config
from boxmot.utils import (
    BENCHMARK_CONFIGS,
    logger as LOGGER,
)
from boxmot.utils.benchmark_config import (
    ensure_benchmark_detector_model,
    ensure_benchmark_reid_model,
    load_benchmark_cfg,
    should_use_benchmark_detector,
    should_use_benchmark_reid,
)
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.evaluation.results import (
    _filter_obb_trackeval_results,
    _known_trackeval_class_names,
    _select_plot_metrics_data,
    log_trackeval_report,
    parse_mot_results,
    render_trackeval_report,
)
from boxmot.utils.evaluation.trackeval import (
    _load_obb_gt_matrix,
    trackeval_aabb,
    trackeval_obb,
)
from boxmot.utils.misc import resolve_model_path
from boxmot.utils.plots import MetricsPlotter
from boxmot.utils.rich.reporting import RichWorkflowReporter, WorkflowDetailCallback
from boxmot.utils.timing import TimingStats
import boxmot.utils.rich.ui as ui

_EVAL_DEPENDENCIES_READY = False
EVAL_GENERATE_STEP = "Generate detections and embeddings"
EVAL_TRACK_STEP = "Run tracker"
EVAL_EVALUATE_STEP = "Evaluate results"

__all__ = [
    "AppendableNpyWriter",
    "_configure_benchmark_runtime",
    "_ensure_eval_dependencies",
    "_existing_cache_path",
    "_existing_embedding_cache_path",
    "_load_benchmark_cfg",
    "_load_embedding_cache_array",
    "_load_numeric_cache_array",
    "_load_obb_gt_matrix",
    "_max_frame_id",
    "_migrate_legacy_embedding_cache",
    "_migrate_legacy_numeric_cache",
    "_ordered_benchmark_eval_class_names",
    "_saved_detection_column_count",
    "_select_plot_metrics_data",
    "apply_class_remap",
    "eval_setup",
    "generate_dets_embs_batched",
    "main",
    "parse_mot_results",
    "process_sequence",
    "run_eval",
    "run_generate_dets_embs",
    "run_generate_mot_results",
    "run_trackeval",
]


def _ensure_eval_dependencies() -> None:
    global _EVAL_DEPENDENCIES_READY
    if _EVAL_DEPENDENCIES_READY:
        return
    checker = RequirementsChecker()
    checker.check_packages(("ultralytics",))
    _EVAL_DEPENDENCIES_READY = True


def _load_benchmark_cfg(args: argparse.Namespace) -> dict:
    return load_benchmark_cfg_from_args(args)


def _resolve_eval_box_type(args: argparse.Namespace, bench_cfg: Optional[dict] = None) -> str:
    return resolve_eval_box_type(args, bench_cfg)


def _configure_benchmark_runtime(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    return configure_benchmark_runtime(
        args,
        load_benchmark_cfg_fn=_load_benchmark_cfg,
        should_use_benchmark_detector_fn=should_use_benchmark_detector,
        should_use_benchmark_reid_fn=should_use_benchmark_reid,
        ensure_benchmark_detector_model_fn=ensure_benchmark_detector_model,
        ensure_benchmark_reid_model_fn=ensure_benchmark_reid_model,
    )


def run_trackeval(args: argparse.Namespace, verbose: bool = True) -> dict:
    """
    Evaluate tracking results via TrackEval and print a summary.
    """
    seq_paths, seq_info = _collect_seq_info(args.source)
    annotations_dir = args.source.parent / "annotations"
    gt_folder = annotations_dir if annotations_dir.exists() else args.source

    if not seq_paths:
        raise ValueError(f"No sequences with images found under {args.source}")

    if annotations_dir.exists():
        for seq_name in list(seq_info.keys()):
            ann_file = annotations_dir / f"{seq_name}.txt"
            if not ann_file.exists():
                continue
            try:
                with open(ann_file, "r") as handle:
                    max_frame = 0
                    for line in handle:
                        if not line.strip():
                            continue
                        frame_id = int(float(line.split(",", 1)[0]))
                        if frame_id > max_frame:
                            max_frame = frame_id
                    if max_frame:
                        seq_info[seq_name] = max(seq_info.get(seq_name, 0) or 0, max_frame)
            except Exception:
                LOGGER.warning(f"Failed to read annotation file {ann_file} for sequence length inference")

    if getattr(args, "benchmark", None):
        save_dir = Path(args.project) / args.benchmark / args.name
    else:
        save_dir = Path(args.project) / args.name

    cfg = _load_benchmark_cfg(args)
    if not cfg:
        cfg_name = (
            getattr(args, "benchmark_id", None)
            or getattr(args, "dataset_id", None)
            or getattr(args, "benchmark", str(args.source.parent.name))
        )
        try:
            cfg = load_benchmark_cfg(cfg_name)
        except FileNotFoundError:
            found = False
            for config_file in BENCHMARK_CONFIGS.glob("*.yaml"):
                if config_file.stem in str(args.source):
                    cfg = load_benchmark_cfg(config_file.stem)
                    found = True
                    break
            if not found:
                LOGGER.warning(f"Could not find benchmark config for {cfg_name}. Class filtering might be incorrect.")
                cfg = {}

    if _resolve_eval_box_type(args, cfg) == "obb":
        trackeval_results = trackeval_obb(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)
    else:
        gt_folder = prepare_aabb_eval_gt(args, gt_folder, seq_info)
        trackeval_results = trackeval_aabb(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)

    parsed_results = parse_mot_results(
        trackeval_results,
        seq_names=set(seq_info.keys()),
        known_classes=_known_trackeval_class_names(args, cfg),
    )
    eval_box_type = _resolve_eval_box_type(args, cfg)

    single_class_mode = False
    if eval_box_type == "obb":
        parsed_results, single_class_mode = _filter_obb_trackeval_results(parsed_results, args, cfg.get("benchmark", {}))
    elif getattr(args, "remapped_class_names", None):
        remapped_lower = {name.lower() for name in args.remapped_class_names}
        parsed_results = {key: value for key, value in parsed_results.items() if key.lower() in remapped_lower}
        if len(args.remapped_class_names) == 1:
            single_class_mode = True
    elif "benchmark" in cfg:
        bench_cfg = cfg["benchmark"]
        bench_classes = _ordered_benchmark_eval_class_names(bench_cfg)
        if bench_classes:
            parsed_results = {key: value for key, value in parsed_results.items() if key in bench_classes}
            if len(bench_classes) == 1:
                single_class_mode = True
    elif hasattr(args, "classes") and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        user_classes = [COCO_CLASSES[int(index)] for index in class_indices]
        parsed_results = {key: value for key, value in parsed_results.items() if key in user_classes}
        if len(user_classes) == 1:
            single_class_mode = True

    final_results = list(parsed_results.values())[0] if single_class_mode and parsed_results else parsed_results

    if verbose:
        log_trackeval_report(
            render_trackeval_report(
                parsed_results,
                args,
                cfg,
                title="📊 RESULTS SUMMARY",
                include_sequences=True,
                colorize=False,
            )
        )

    if getattr(args, "ci", False):
        with open(args.tracker + "_output.json", "w") as outfile:
            outfile.write(json.dumps(final_results))

    return final_results


def eval_setup(args, workflow: ui.WorkflowProgress | None = None) -> None:
    """
    Common setup for eval and tune pipelines.
    """
    _ensure_eval_dependencies()
    status_fn = None
    if workflow is not None:
        status_fn = WorkflowDetailCallback(workflow, EVAL_GENERATE_STEP)
    eval_init(args, status_fn=status_fn)
    _, _, dataset_detector_cfg = _configure_benchmark_runtime(args)
    det_cfg = get_runtime_detector_cfg(args.detector[0], dataset_detector_cfg)
    apply_class_remap(args, det_cfg)


def apply_class_remap(args, det_cfg: dict) -> None:
    """
    Remap GT class IDs to match detector output.
    """
    bench_cfg: dict = {}
    benchmark_id = (
        getattr(args, "benchmark_id", None)
        or getattr(args, "dataset_id", None)
        or getattr(args, "benchmark", None)
    )
    if benchmark_id:
        try:
            bench_cfg = (load_benchmark_cfg(benchmark_id) or {}).get("benchmark", {})
        except Exception:
            pass

    if str(bench_cfg.get("box_type", "")).lower() == "obb":
        return

    remap_result = build_gt_class_remap(
        bench_cfg,
        det_cfg,
        benchmark_name=getattr(args, "benchmark", ""),
        model_stem=args.detector[0].stem,
    )
    if remap_result is not None:
        remap_dict, new_class_ids, new_class_names = remap_result
        distractor_ids = [int(key) for key in bench_cfg.get("distractor_classes", {}).keys()]
        args.gt_class_remap = remap_dict
        args.gt_class_distractor_ids = distractor_ids
        args.remapped_class_ids = new_class_ids
        args.remapped_class_names = [name.lower() for name in new_class_names]


def _normalize_eval_models(args: argparse.Namespace) -> None:
    args.detector = [resolve_model_path(model) for model in args.detector]
    args.reid = [resolve_model_path(model) for model in args.reid]


def _effective_eval_tracker_backend(args: argparse.Namespace) -> str | None:
    tracking_backend = str(getattr(args, "tracking_backend", "") or "").strip().lower()
    if tracking_backend == "cpp":
        return "cpp"

    raw_tracker_backend = getattr(args, "tracker_backend", None)
    if raw_tracker_backend in {None, ""}:
        return None

    return normalize_tracker_backend(raw_tracker_backend, default="python")


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

    pipeline_params = _build_eval_pipeline_parameter_fields(args, tracker_backend=tracker_backend, replay_backend=replay_backend)
    if pipeline_params:
        fields.append(("__panel__:Pipeline Parameters", pipeline_params))

    return fields


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
        (EVAL_GENERATE_STEP, "active"),
        (EVAL_TRACK_STEP, "todo"),
        (EVAL_EVALUATE_STEP, "todo"),
    )

    def fields(self) -> list[tuple[str, object]]:
        return _build_eval_workflow_fields(self.args)


def log_eval_pipeline_intro(args: argparse.Namespace) -> ui.WorkflowProgress:
    _normalize_eval_models(args)
    return EvalWorkflowReporter(args).create()


def run_eval(
    args: argparse.Namespace,
    *,
    evolve_config: dict | None = None,
    setup: bool = True,
    prepare_cache: bool = True,
    verbose: bool | None = None,
    show_progress: bool | None = None,
    workflow: ui.WorkflowProgress | None = None,
) -> ValidationResult:
    _ensure_eval_dependencies()
    _normalize_eval_models(args)
    if verbose is None:
        verbose = bool(getattr(args, "verbose", False))
    if show_progress is None:
        show_progress = bool(getattr(args, "show_progress", True))
    args.show_progress = bool(show_progress)

    timing_stats = TimingStats()
    if setup:
        eval_setup(args, workflow=workflow)
        _refresh_eval_pipeline_intro(workflow, args)
    if workflow is not None and prepare_cache:
        workflow.activate(EVAL_GENERATE_STEP)
    if workflow is not None and not prepare_cache:
        workflow.complete(EVAL_GENERATE_STEP, render=False)
        workflow.activate(EVAL_TRACK_STEP, render=False)
        workflow.set_detail(EVAL_TRACK_STEP, "Starting tracker...")
    if prepare_cache:
        generate_progress_callback = None
        if workflow is not None and bool(show_progress):
            generate_progress_callback = WorkflowDetailCallback(workflow, EVAL_GENERATE_STEP)
        with suppress_boxmot_logs((not verbose) or workflow is not None, level="WARNING"):
            if generate_progress_callback is None:
                run_generate_dets_embs(args, timing_stats=timing_stats)
            else:
                run_generate_dets_embs(
                    args,
                    timing_stats=timing_stats,
                    progress_callback=generate_progress_callback,
                )
        if workflow is not None:
            workflow.complete(EVAL_GENERATE_STEP, render=False)
            workflow.activate(EVAL_TRACK_STEP, render=False)
            workflow.set_detail(EVAL_TRACK_STEP, "Starting tracker...")
    progress_callback = None
    if workflow is not None and bool(show_progress):
        progress_callback = WorkflowDetailCallback(workflow, EVAL_TRACK_STEP)
    with suppress_boxmot_logs((not verbose) or workflow is not None, level="WARNING"):
        run_generate_mot_results(
            args,
            evolve_config=evolve_config,
            timing_stats=timing_stats,
            quiet=not bool(show_progress),
            progress_callback=progress_callback,
        )
    if workflow is not None:
        workflow.complete(EVAL_TRACK_STEP, render=False)
        workflow.activate(EVAL_EVALUATE_STEP, render=False)
        workflow.set_detail(EVAL_EVALUATE_STEP, "Computing metrics...")
    raw_results = run_trackeval(args, verbose=bool(verbose) and workflow is None)
    summary_label, summary = extract_summary(raw_results)
    result = ValidationResult(
        benchmark=str(getattr(args, "benchmark", getattr(args, "data", ""))),
        raw=raw_results,
        summary_label=summary_label,
        summary=summary,
        exp_dir=getattr(args, "exp_dir", None),
        timings=timing_summary_from_stats(timing_stats),
        args=args,
        workflow_rendered=workflow is not None,
    )
    if workflow is not None:
        workflow.complete(EVAL_EVALUATE_STEP, render=False)
        if hasattr(workflow, "set_detail_renderable"):
            workflow.set_detail_renderable(
                EVAL_EVALUATE_STEP,
                result.renderable(include_timings=bool(getattr(args, "show_timing", False))),
                render=False,
            )
        else:
            workflow.set_detail(
                EVAL_EVALUATE_STEP,
                result.render(include_timings=bool(getattr(args, "show_timing", False))),
                render=False,
            )

    return result


def main(args):
    workflow = log_eval_pipeline_intro(args)
    try:
        result = run_eval(args, verbose=False, workflow=workflow)
    except BaseException as exc:
        workflow.fail(error=exc)
        workflow.stop()
        raise
    else:
        workflow.stop()

    plot_class, metrics_data = _select_plot_metrics_data(result.raw)
    if metrics_data:
        plotter = MetricsPlotter(result.exp_dir)
        plot_metrics = ["HOTA", "MOTA", "IDF1"]
        plot_values = [metrics_data.get(metric, 0) for metric in plot_metrics]

        plotter.plot_radar_chart(
            {args.tracker: plot_values},
            plot_metrics,
            title=f"MOT metrics radar Chart ({plot_class})",
            ylim=(0, 100),
            yticks=[20, 40, 60, 80, 100],
            ytick_labels=["20", "40", "60", "80", "100"],
        )
    return result


if __name__ == "__main__":
    main()
