# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import boxmot.utils.rich.core.ui as ui
from boxmot.configs.benchmark import (
    ensure_benchmark_detector_model,
    ensure_benchmark_reid_model,
    load_benchmark_cfg,
    should_use_benchmark_detector,
    should_use_benchmark_reid,
)
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
from boxmot.detectors import get_runtime_detector_cfg
from boxmot.engine.workflows.reporting import extract_summary, timing_summary_from_stats
from boxmot.engine.workflows.results import ValidationResult
from boxmot.utils import (
    BENCHMARK_CONFIGS,
)
from boxmot.utils import (
    logger as LOGGER,
)
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.misc import resolve_model_path, suppress_boxmot_logs
from boxmot.utils.rich.reporters.eval import (
    EVAL_EVALUATE_STEP,
    EVAL_GENERATE_STEP,
    EVAL_SETUP_STEP,
    EVAL_TRACK_STEP,
    EvalWorkflowReporter,
    _build_eval_workflow_fields,
)
from boxmot.utils.rich.workflow.pipeline import PipelineTracker
from boxmot.utils.timing import TimingStats

if TYPE_CHECKING:
    from boxmot.data.cache import (
        AppendableNpyWriter,
        _existing_cache_path,
        _existing_embedding_cache_path,
        _load_embedding_cache_array,
        _load_numeric_cache_array,
        _max_frame_id,
        _saved_detection_column_count,
    )
    from boxmot.engine.eval.cache import generate_dets_embs_batched, run_generate_dets_embs
    from boxmot.engine.eval.motmetrics import _load_obb_gt_matrix
    from boxmot.engine.eval.replay import process_sequence, run_generate_mot_results
    from boxmot.engine.eval.results import (
        _select_plot_metrics_data,
        parse_mot_results,
    )

_EVAL_DEPENDENCIES_READY = False


def _has_eval_postprocessing(args: argparse.Namespace) -> bool:
    """Return True when eval will run at least one MOT postprocessing step."""
    pp_raw = getattr(args, "postprocessing", "none") or "none"
    return any(step.strip().lower() not in ("", "none") for step in str(pp_raw).split(","))


__all__ = [
    "AppendableNpyWriter",
    "_configure_benchmark_runtime",
    "_ensure_eval_dependencies",
    "_existing_cache_path",
    "_existing_embedding_cache_path",
    "EVAL_EVALUATE_STEP",
    "EVAL_GENERATE_STEP",
    "EVAL_SETUP_STEP",
    "EVAL_TRACK_STEP",
    "_load_benchmark_cfg",
    "_load_embedding_cache_array",
    "_load_numeric_cache_array",
    "_load_obb_gt_matrix",
    "_max_frame_id",
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
    "run_motmetrics",
]

_LAZY_EXPORTS = {
    "AppendableNpyWriter": ("boxmot.data.cache", "AppendableNpyWriter"),
    "_collect_seq_info": ("boxmot.data.cache", "_collect_seq_info"),
    "_existing_cache_path": ("boxmot.data.cache", "_existing_cache_path"),
    "_existing_embedding_cache_path": (
        "boxmot.data.cache",
        "_existing_embedding_cache_path",
    ),
    "filter_obb_mot_results": (
        "boxmot.engine.eval.results",
        "filter_obb_mot_results",
    ),
    "_load_embedding_cache_array": (
        "boxmot.data.cache",
        "_load_embedding_cache_array",
    ),
    "_load_numeric_cache_array": ("boxmot.data.cache", "_load_numeric_cache_array"),
    "_load_obb_gt_matrix": (
        "boxmot.engine.eval.motmetrics",
        "_load_obb_gt_matrix",
    ),
    "_max_frame_id": ("boxmot.data.cache", "_max_frame_id"),
    "_saved_detection_column_count": (
        "boxmot.data.cache",
        "_saved_detection_column_count",
    ),
    "_select_plot_metrics_data": ("boxmot.engine.eval.results", "_select_plot_metrics_data"),
    "generate_dets_embs_batched": (
        "boxmot.engine.eval.cache",
        "generate_dets_embs_batched",
    ),
    "log_mot_report": ("boxmot.engine.eval.results", "log_mot_report"),
    "MetricsPlotter": ("boxmot.engine.eval.plots", "MetricsPlotter"),
    "parse_mot_results": ("boxmot.engine.eval.results", "parse_mot_results"),
    "process_sequence": ("boxmot.engine.eval.replay", "process_sequence"),
    "render_mot_report": ("boxmot.engine.eval.results", "render_mot_report"),
    "run_generate_dets_embs": ("boxmot.engine.eval.cache", "run_generate_dets_embs"),
    "run_generate_mot_results": (
        "boxmot.engine.eval.replay",
        "run_generate_mot_results",
    ),
    "motmetrics_runner": ("boxmot.engine.eval.motmetrics", "run_motmetrics"),
}


def _get_lazy_export(name: str):
    if name in globals():
        return globals()[name]
    return __getattr__(name)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))


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


def run_motmetrics(args: argparse.Namespace, verbose: bool = True) -> dict:
    """
    Evaluate tracking results with BoxMOT's in-repo motmetrics implementation.
    """
    collect_seq_info = _get_lazy_export("_collect_seq_info")
    filter_obb_mot_results = _get_lazy_export("filter_obb_mot_results")
    log_mot_report_fn = _get_lazy_export("log_mot_report")
    render_mot_report_fn = _get_lazy_export("render_mot_report")
    motmetrics_runner = _get_lazy_export("motmetrics_runner")

    seq_paths, seq_info = collect_seq_info(args.source)
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
            except (ValueError, OSError) as exc:
                LOGGER.warning(f"Failed to read annotation file {ann_file} for sequence length inference: {exc}")

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
        parsed_results = motmetrics_runner(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)
    else:
        gt_folder = prepare_aabb_eval_gt(args, gt_folder, seq_info)
        parsed_results = motmetrics_runner(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)
    eval_box_type = _resolve_eval_box_type(args, cfg)

    single_class_mode = False
    if eval_box_type == "obb":
        parsed_results, single_class_mode = filter_obb_mot_results(parsed_results, args, cfg.get("benchmark", {}))
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
        log_mot_report_fn(
            render_mot_report_fn(
                parsed_results,
                args,
                cfg,
                title="📊 RESULTS SUMMARY",
                include_sequences=single_class_mode,
                colorize=False,
            )
        )

    if getattr(args, "ci", False):
        with open(args.tracker + "_output.json", "w") as outfile:
            outfile.write(json.dumps(final_results))

    return final_results


def eval_setup(args, pipeline: PipelineTracker | None = None) -> None:
    """
    Common setup for eval and tune pipelines.
    """
    _ensure_eval_dependencies()
    status_fn = pipeline.callback() if pipeline is not None else None
    eval_init(args, status_fn=status_fn)
    _, _, dataset_detector_cfg = _configure_benchmark_runtime(args)
    det_cfg = get_runtime_detector_cfg(args.detector[0], dataset_detector_cfg)
    apply_class_remap(args, det_cfg)


def apply_class_remap(args, det_cfg: dict) -> None:
    """
    Remap GT class IDs to match detector output.
    """
    if str(getattr(args, "eval_box_type", "")).lower() == "obb":
        return

    bench_cfg: dict = {}
    benchmark_id = (
        getattr(args, "benchmark_id", None)
        or getattr(args, "dataset_id", None)
        or getattr(args, "benchmark", None)
        or getattr(args, "data", None)
    )
    if benchmark_id:
        try:
            bench_cfg = (load_benchmark_cfg(benchmark_id) or {}).get("benchmark", {})
        except (FileNotFoundError, KeyError, ValueError) as exc:
            LOGGER.debug(f"Could not load benchmark config for class remap: {exc}")
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
        if "ignore_dataset_ids" in bench_cfg:
            distractor_ids = [int(class_id) for class_id in bench_cfg.get("ignore_dataset_ids") or []]
        else:
            distractor_ids = [int(key) for key in bench_cfg.get("distractor_classes", {}).keys()]
        args.gt_class_remap = remap_dict
        args.gt_class_distractor_ids = distractor_ids
        args.remapped_class_ids = new_class_ids
        args.remapped_class_names = [name.lower() for name in new_class_names]


def _normalize_eval_models(args: argparse.Namespace) -> None:
    args.detector = [resolve_model_path(model) for model in args.detector]
    args.reid = [resolve_model_path(model) for model in args.reid]


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
    pipeline: PipelineTracker | None = None,
) -> ValidationResult:
    _ensure_eval_dependencies()
    _normalize_eval_models(args)
    if verbose is None:
        verbose = bool(getattr(args, "verbose", False))
    if show_progress is None:
        show_progress = bool(getattr(args, "show_progress", True))
    args.show_progress = bool(show_progress)

    timing_stats = TimingStats()
    has_pipeline = pipeline is not None
    suppress = (not verbose) or has_pipeline
    has_postprocessing = _has_eval_postprocessing(args)
    tune_kf_step = bool(getattr(args, "tune_kf", False))

    # -- Setup --
    if setup:
        eval_setup(args, pipeline=pipeline)
        if pipeline is not None:
            pipeline.refresh_fields(_build_eval_workflow_fields(args))

    # -- Generate detections & embeddings --
    if prepare_cache:
        from boxmot.engine.workflows.support import REID_TRACKERS

        tracker_name = str(getattr(args, "tracker", "")).lower()
        if tracker_name not in REID_TRACKERS:
            args.reid = []
        if pipeline is not None:
            pipeline.advance("Generating detections & embeddings...")
        with suppress_boxmot_logs(suppress, level="WARNING"):
            _get_lazy_export("run_generate_dets_embs")(
                args,
                timing_stats=timing_stats,
                progress_callback=pipeline.callback() if pipeline and show_progress else None,
            )
    if pipeline is not None:
        pipeline.advance("Calibrating Kalman filter..." if tune_kf_step else "Starting tracker...")

    # -- KF calibration --
    if getattr(args, "tune_kf", False) and not getattr(args, "kf_tuning", None):
        from boxmot.motion.kalman_filters.calibration import run_kf_tuning, tracker_kf_type

        kf_type = tracker_kf_type(str(getattr(args, "tracker", "")))
        if kf_type:
            kf_result, _ = run_kf_tuning(args, kf_type, capture=True)
            if kf_result is not None:
                kf_result["kf_type"] = kf_type
                args.kf_tuning = kf_result
                LOGGER.info(
                    f"KF calibration ({kf_type}): "
                    f"std_weight_position={kf_result['std_weight_position']:.6f}, "
                    f"std_weight_velocity={kf_result['std_weight_velocity']:.6f}"
                )
            else:
                LOGGER.warning("KF calibration produced no result; using default noise weights.")
        else:
            LOGGER.debug(f"Tracker '{args.tracker}' has no KF parameterization; skipping --tune-kf.")

    if pipeline is not None and tune_kf_step:
        pipeline.advance("Starting tracker...")

    # -- Track --
    postprocess_started = False

    def _postprocess_progress(detail: str) -> None:
        nonlocal postprocess_started
        if pipeline is None:
            return
        if not postprocess_started:
            pipeline.advance(detail)
            postprocess_started = True
            return
        pipeline.callback()(detail)

    with suppress_boxmot_logs(suppress, level="WARNING"):
        _get_lazy_export("run_generate_mot_results")(
            args,
            evolve_config=evolve_config,
            timing_stats=timing_stats,
            quiet=not bool(show_progress),
            progress_callback=pipeline.callback() if pipeline and show_progress else None,
            postprocess_callback=_postprocess_progress if pipeline and has_postprocessing else None,
        )
    if pipeline is not None:
        if has_postprocessing and not postprocess_started:
            pipeline.advance("Postprocessing tracks...")
        pipeline.advance("Computing metrics...")

    # -- Evaluate --
    raw_results = run_motmetrics(args, verbose=verbose and not has_pipeline)
    summary_label, summary = extract_summary(raw_results)
    result = ValidationResult(
        benchmark=str(getattr(args, "benchmark", getattr(args, "data", ""))),
        raw=raw_results,
        summary_label=summary_label,
        summary=summary,
        exp_dir=getattr(args, "exp_dir", None),
        timings=timing_summary_from_stats(timing_stats),
        args=args,
        workflow_rendered=has_pipeline,
    )
    if pipeline is not None:
        include_timings = bool(getattr(args, "show_timing", False))
        pipeline.complete_step()
        pipeline.set_detail_renderable(
            pipeline.current_step,
            result.renderable(
                include_sequences=summary_label == "single_class",
                include_timings=include_timings,
            ),
        )

    return result


def main(args):
    _normalize_eval_models(args)
    pipeline = EvalWorkflowReporter(args).pipeline()
    with pipeline:
        result = run_eval(args, verbose=False, pipeline=pipeline)

    plot_class, metrics_data = _get_lazy_export("_select_plot_metrics_data")(result.raw)
    if metrics_data:
        plotter = _get_lazy_export("MetricsPlotter")(result.exp_dir)
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
