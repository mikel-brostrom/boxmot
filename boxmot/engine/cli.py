#!/usr/bin/env python3
from __future__ import annotations

"""
CLI for BoxMOT: multi-step multiple object tracking pipeline.
Provides commands to track, generate detections and embeddings, evaluate performance, tune models, or run all steps.
"""
import importlib

from pathlib import Path
from typing import Optional, Tuple

import click
from click.core import ParameterSource

from boxmot import __version__
# Shared CLI/Python API defaults and namespace normalization live under boxmot/configs.
from boxmot.configs import (
    BOXMOT_DEFAULTS,
    build_mode_namespace,
    ensure_model_extension,
)
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING
from boxmot.utils.benchmark_config import resolve_benchmark_cfg_path
from boxmot.utils.misc import parse_imgsz


RUNTIME_DEFAULTS = BOXMOT_DEFAULTS.eval
TRACK_DEFAULTS = BOXMOT_DEFAULTS.track
TUNE_DEFAULTS = BOXMOT_DEFAULTS.tune
EXPORT_DEFAULTS = BOXMOT_DEFAULTS.export
SHARED_DEFAULTS = BOXMOT_DEFAULTS.shared


def _click_imgsz_default(value):
    """Normalize configured image sizes into a Click-friendly default value."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(part) for part in value)
    return value


# Shared command options (excluding model, classes, and input selection)
def core_options(func):
    options = [
        click.option('--imgsz', callback=parse_imgsz, default=_click_imgsz_default(RUNTIME_DEFAULTS.imgsz), type=str,
                     help='Image size for model input as H,W (e.g. 800,1440) or single int for square. Default: read from the selected detector config, otherwise use detector-specific defaults.'),
        click.option('--fps', type=int, default=RUNTIME_DEFAULTS.fps,
                     help='video frame-rate'),
        click.option('--conf', type=float, default=RUNTIME_DEFAULTS.conf,
                     help='Min confidence threshold. Default: read from the selected detector config, fallback 0.01.'),
        click.option('--iou', type=float, default=RUNTIME_DEFAULTS.iou,
                     help='IoU threshold for NMS'),
        click.option('--device', default=RUNTIME_DEFAULTS.device,
                     help='cuda device(s), e.g. 0 or 0,1,2,3 or cpu'),
        click.option('--batch-size', type=int, default=RUNTIME_DEFAULTS.batch_size, show_default=True,
                 help='micro-batch size for batched detection/embedding'),
        click.option('--auto-batch/--no-auto-batch', default=RUNTIME_DEFAULTS.auto_batch, show_default=True,
                 help='probe GPU memory with a dummy pass to pick a safe batch size'),
        click.option('--resume/--no-resume', default=RUNTIME_DEFAULTS.resume, show_default=True,
             help='resume detection/embedding generation from progress checkpoints'),
        click.option('--n-threads', type=int, default=RUNTIME_DEFAULTS.n_threads,
                 help='CPU threads for image decoding; defaults to min(8, cpu_count)'),
        click.option('--project', type=Path, default=RUNTIME_DEFAULTS.project,
                     help='save results to project/name'),
        click.option('--name', default=RUNTIME_DEFAULTS.name, help='save results to project/name'),
        click.option('--exist-ok', is_flag=True, default=RUNTIME_DEFAULTS.exist_ok,
                     help='existing project/name ok, do not increment'),
        click.option('--half', is_flag=True, default=RUNTIME_DEFAULTS.half,
                     help='use FP16 half-precision inference'),
        click.option('--vid-stride', type=int, default=RUNTIME_DEFAULTS.vid_stride,
                     help='video frame-rate stride'),
        click.option('--ci', is_flag=True, default=RUNTIME_DEFAULTS.ci,
                     help='reuse existing runs in CI (no UI)'),
        click.option('--tracker', 'tracking_method', type=str, default=RUNTIME_DEFAULTS.tracker, show_default=True,
                     help='deepocsort, botsort, strongsort, ...'),
        click.option('--verbose', is_flag=True, default=RUNTIME_DEFAULTS.verbose,
                     help='print detailed logs'),
        click.option('--agnostic-nms', is_flag=True, default=RUNTIME_DEFAULTS.agnostic_nms,
                     help='class-agnostic NMS'),
        click.option(
            "--postprocessing", type=click.Choice(["none", "gsi", "gbrc"], case_sensitive=False), default=RUNTIME_DEFAULTS.postprocessing,
            help="Postprocess tracker output: none | gsi (Gaussian smoothed interpolation) | gbrc (gradient boosting smooth).",
        ),
        click.option('--show', is_flag=True, default=RUNTIME_DEFAULTS.show,
                     help='display tracking in a window'),
        click.option('--show-labels/--hide-labels', default=RUNTIME_DEFAULTS.show_labels,
                     help='show or hide detection labels'),
        click.option('--show-conf/--hide-conf', default=RUNTIME_DEFAULTS.show_conf,
                     help='show or hide detection confidences'),
        click.option('--show-trajectories', is_flag=True, default=RUNTIME_DEFAULTS.show_trajectories,
                     help='overlay past trajectories'),
           click.option('--show-kf-preds', 'show_kf_preds', is_flag=True, default=RUNTIME_DEFAULTS.show_kf_preds,
               help='show Kalman-filter predictions'),
        click.option('--save-txt', is_flag=True, default=RUNTIME_DEFAULTS.save_txt,
                     help='save results to a .txt file'),
        click.option('--save-crop', is_flag=True, default=RUNTIME_DEFAULTS.save_crop,
                     help='save cropped detections'),
        click.option('--save', is_flag=True, default=RUNTIME_DEFAULTS.save,
                     help='save annotated video'),
        click.option('--line-width', type=int, default=RUNTIME_DEFAULTS.line_width,
                     help='bounding box line width'),
        click.option('--per-class', is_flag=True, default=RUNTIME_DEFAULTS.per_class,
                     help='track each class separately'),
        click.option('--target-id', type=int, default=RUNTIME_DEFAULTS.target_id,
                     help='ID to highlight in green')
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def source_option(default='0', help_text='file/dir/URL/glob, 0 for webcam'):
    """Attach a ``--source`` option with command-specific defaults/help."""
    return click.option('--source', type=str, default=default, help=help_text)


def data_option(func):
    """Attach the benchmark-config option."""
    return click.option(
        '--benchmark',
        '--data',
        'data',
        type=str,
        default=None,
        help='benchmark config name or YAML file, e.g. mot17-ablation or boxmot/configs/benchmarks/mot17-ablation.yaml',
    )(func)

def _is_option_explicit(ctx: click.Context, option_name: str) -> bool:
    """Return True when a Click option came from the command line instead of defaults."""
    return ctx.get_parameter_source(option_name) != ParameterSource.DEFAULT


def _explicit_cli_keys(ctx: click.Context) -> set[str]:
    """Return the Click option names explicitly provided on the command line."""
    return {
        param.name
        for param in ctx.command.params
        if isinstance(param, click.Option) and _is_option_explicit(ctx, param.name)
    }


def _mark_explicit_positionals(explicit_keys: set[str], **positionals: Optional[str]) -> set[str]:
    """Augment explicit CLI keys with positional component arguments that were provided."""
    for name, value in positionals.items():
        if value is not None:
            explicit_keys.add(name)
    return explicit_keys


def _build_cli_namespace(
    ctx: click.Context,
    mode: str,
    payload: dict,
    **positionals: Optional[str],
):
    """Build the normalized mode namespace while preserving explicitly provided CLI values."""
    explicit_keys = _mark_explicit_positionals(_explicit_cli_keys(ctx), **positionals)
    return build_mode_namespace(mode, payload, explicit_keys=explicit_keys)


def _dispatch_cli_workflow(
    ctx: click.Context,
    mode: str,
    module_name: str,
    class_name: Optional[str],
    payload: dict,
    **positionals: Optional[str],
) -> None:
    """Build CLI args for a workflow and execute its engine class."""
    args = _build_cli_namespace(ctx, mode, payload, **positionals)
    _run_engine_workflow(module_name, class_name, args)


def _resolve_source_context(source: Optional[str]) -> Tuple[Optional[str], str, str]:
    """Return ``(source, benchmark, split)`` metadata for a concrete source path."""
    if source is None:
        return None, "", ""

    source_path = Path(source)
    return source, source_path.parent.name, source_path.name


def _require_generate_input(data: Optional[str], source: Optional[str], command_name: str) -> None:
    """Validate benchmark-vs-source selection for generate-like commands."""
    if data and source:
        raise click.UsageError(
            f"{command_name} accepts either --benchmark <benchmark.yaml> or --source <dataset-path>, not both."
        )
    if not data and not source:
        raise click.UsageError(
            f"{command_name} requires --benchmark <benchmark.yaml> for config-driven runs or --source <dataset-path> for direct datasets."
        )


def _normalize_generate_input(data: Optional[str], source: Optional[str], command_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Auto-promote benchmark names passed through ``--source`` to ``--benchmark``."""
    _require_generate_input(data, source, command_name)

    if data or source is None:
        return data, source

    if Path(source).exists():
        return data, source

    try:
        resolve_benchmark_cfg_path(source)
    except FileNotFoundError:
        return data, source

    click.echo(
        f"{command_name}: resolving benchmark config '{source}' from boxmot/configs/benchmarks; "
        f"use '--benchmark {source}' instead of '--source {source}'.",
        err=True,
    )
    return source, None


def _require_benchmark_input(data: Optional[str], command_name: str) -> str:
    """Require a benchmark config for benchmark-only commands such as eval/tune."""
    if not data:
        raise click.UsageError(
            f"{command_name} requires --benchmark <benchmark.yaml>. "
            f"Use 'generate --source <dataset-path>' to prepare direct datasets before running {command_name}."
        )
    return data


def _run_engine_workflow(module_name: str, class_name: Optional[str], args) -> None:
    """Instantiate an engine workflow class when present, otherwise call ``main(args)``."""
    module = importlib.import_module(module_name)
    workflow_cls = getattr(module, class_name, None) if class_name else None
    if workflow_cls is not None:
        workflow_cls(args).run()
        return

    main_fn = getattr(module, "main", None)
    if main_fn is None:
        raise AttributeError(f"{module_name} does not expose {class_name} or main")
    main_fn(args)


def _is_tracker_name(name: Optional[str]) -> bool:
    """Return True when ``name`` matches a registered tracker name."""
    return bool(name) and str(name).lower() in TRACKER_MAPPING


def _normalize_eval_positionals(
    detector: Optional[str],
    reid: Optional[str],
    tracker: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Interpret tracker-only positional benchmark calls while preserving detector/reid ordering."""
    if tracker is not None:
        return detector, reid, tracker

    if reid is not None and _is_tracker_name(reid):
        return detector, None, reid

    if reid is None and detector is not None and _is_tracker_name(detector):
        return None, None, detector

    return detector, reid, tracker


def singular_model_options(func):
    options = [
        click.option('--detector', 'yolo_model', type=Path,
                     default=SHARED_DEFAULTS.detector,
                     help='path to YOLO weights for detection'),
        click.option('--reid', 'reid_model', type=Path,
                     default=SHARED_DEFAULTS.reid,
                     help='path to ReID model weights'),
        click.option('--classes', type=str, default=None,
                     help='filter by class indices, e.g. 0 or "0,1"')
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def plural_model_options(func):
    options = [
        click.option('--detector', 'yolo_model', type=Path, multiple=True,
                     default=[SHARED_DEFAULTS.detector],
                     help='one or more YOLO weights for detection'),
        click.option('--reid', 'reid_model', type=Path, multiple=True,
                     default=[SHARED_DEFAULTS.reid],
                     help='one or more ReID model weights'),
        click.option('--classes', type=str, default=None,
                     help='filter by class indices, e.g. 0 or "0,1"')
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def export_options(func):
    """
    Decorator adding ReID export options (ported from argparse export script).
    """
    options = [
        click.option('--batch-size', type=int, default=EXPORT_DEFAULTS.batch_size,
                     help='Batch size for export'),
        click.option('--imgsz', '--img', '--img-size', callback=parse_imgsz, type=str,
                     default=_click_imgsz_default(EXPORT_DEFAULTS.imgsz), help='Image size as H,W (e.g. 256,128)'),
        click.option('--device', default=EXPORT_DEFAULTS.device,
                     help="CUDA device (e.g., '0', '0,1,2,3', or 'cpu')"),
        click.option('--optimize', is_flag=True, default=EXPORT_DEFAULTS.optimize,
                     help='Optimize TorchScript for mobile (CPU export only)'),
        click.option('--dynamic', is_flag=True, default=EXPORT_DEFAULTS.dynamic,
                     help='Enable dynamic axes for ONNX/TF/TensorRT export'),
        click.option('--simplify', is_flag=True, default=EXPORT_DEFAULTS.simplify,
                     help='Simplify ONNX model'),
        click.option('--opset', type=int, default=EXPORT_DEFAULTS.opset,
                     help='ONNX opset version'),
        click.option('--workspace', type=int, default=EXPORT_DEFAULTS.workspace,
                     help='TensorRT workspace size (GB)'),
        click.option('--verbose', is_flag=True,
                     help='Enable verbose logging for TensorRT'),
        click.option('--weights', type=Path,
                     default=EXPORT_DEFAULTS.weights,
                     help='Path to the model weights (.pt file)'),
        click.option('--half', is_flag=True,
                     help='Enable FP16 half-precision export (GPU only)'),
        click.option('--include', multiple=True, default=EXPORT_DEFAULTS.include,
                     help='Export formats to include. Options: torchscript, onnx, openvino, engine, tflite'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def tune_options(func):
    """
    Decorator adding ReID export options (ported from argparse export script).
    """
    options = [
        click.option('--n-trials', type=int, default=TUNE_DEFAULTS.n_trials,
                     help='number of trials for evolutionary tuning'),
        click.option('--objectives', type=str, multiple=True,
                     default=TUNE_DEFAULTS.objectives,
                     help='metrics to track and return from each trial'),
        click.option('--maximize', type=str, multiple=True, default=TUNE_DEFAULTS.maximize,
                     help='metrics to maximize; defaults to first --objectives value (e.g. HOTA)'),
        click.option('--minimize', type=str, multiple=True, default=TUNE_DEFAULTS.minimize,
                     help='metrics to minimize for Pareto search (e.g. IDSW_rate); '
                          'triggers multi-objective mode when set'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func



class CommandFirstGroup(click.Group):
    """Custom Click Group with improved help formatting - Ultralytics-style."""
    
    def format_help(self, ctx, formatter):
        """Override to show custom help with Ultralytics-style formatting."""
        
        # Main heading
        formatter.write_paragraph()
        formatter.write_text(
            "BoxMOT 'boxmot' commands use the following syntax:"
        )
        formatter.write_paragraph()
        
        # Command syntax
        with formatter.indentation():
            formatter.write_text("boxmot MODE [OPTIONS]")
        formatter.write_paragraph()
        
        # Argument descriptions
        formatter.width = 120  # Increase formatter width to prevent wrapping
        with formatter.indentation():
            formatter.write_text("Where  MODE (required) is one of [track, eval, tune, generate, export]")
            formatter.write_text("       --detector selects a YOLO model like yolov8n, yolov9c, yolo11m, yolox_x")
            formatter.write_text("       --reid selects a ReID model like osnet_x0_25_msmt17, mobilenetv2_x1_4")
            formatter.write_text("       --tracker selects one of [deepocsort, botsort, bytetrack, strongsort, ocsort, hybridsort, boosttrack, sfsort]")
            formatter.write_text("       OPTIONS (optional) flags like '--source 0' for tracking inputs or '--benchmark mot17-ablation' for benchmark-driven eval/tune runs.")
            formatter.write_text("       Benchmark configs select their dataset, detector, and ReID profiles.")
            formatter.write_text("          See all options at https://github.com/mikel-brostrom/boxmot or 'boxmot MODE --help'")
        formatter.write_paragraph()
        
        # Examples
        formatter.write_text("Examples:")
        with formatter.indentation():
            formatter.write_text("1. Track with webcam using defaults:")
            with formatter.indentation():
                formatter.write_text("boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker deepocsort --source 0 --show")
            formatter.write_paragraph()
            
            formatter.write_text("2. Track a video file:")
            with formatter.indentation():
                formatter.write_text("boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save")
            formatter.write_paragraph()
            
            formatter.write_text("3. Evaluate on MOT dataset:")
            with formatter.indentation():
                formatter.write_text("boxmot eval --benchmark mot17-ablation --tracker boosttrack")
            formatter.write_paragraph()
            
            formatter.write_text("4. Tune tracker hyperparameters:")
            with formatter.indentation():
                formatter.write_text("boxmot tune --benchmark mot17-ablation --tracker deepocsort --n-trials 10")
            formatter.write_paragraph()
            
            formatter.write_text("5. Export ReID model:")
            with formatter.indentation():
                formatter.write_text("boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine --dynamic")
        formatter.write_paragraph()
        
        # Available modes
        formatter.write_text("Modes:")
        with formatter.indentation():
            formatter.write_text("track      Track objects in video/webcam stream")
            formatter.write_text("eval       Evaluate tracker performance on MOT dataset")
            formatter.write_text("tune       Optimize tracker hyperparameters")
            formatter.write_text("generate   Generate detections and embeddings")
            formatter.write_text("export     Export ReID models to different formats")
        formatter.write_paragraph()
        
        # Resources
        formatter.write_text("Docs:      https://github.com/mikel-brostrom/boxmot")
        formatter.write_text("Community: https://github.com/mikel-brostrom/boxmot/discussions")


@click.group(cls=CommandFirstGroup)
@click.version_option(__version__, prog_name="BoxMOT")
@click.pass_context
def boxmot(ctx):
    """
    BoxMOT: Pluggable SOTA multi-object tracking modules for segmentation, object detection and pose estimation models
    """
    pass


@boxmot.command(help='Run tracking only')
@click.argument('detector', required=False)
@click.argument('reid', required=False)
@click.argument('tracker', required=False)
@source_option(default=TRACK_DEFAULTS.source, help_text='file/dir/URL/glob, 0 for webcam')
@core_options
@singular_model_options
@click.pass_context
def track(ctx, detector, reid, tracker, yolo_model, reid_model, classes, **kwargs):
    if tracker:
        kwargs.pop("tracking_method", None)
    src, bench, split = _resolve_source_context(kwargs.pop('source'))
    _dispatch_cli_workflow(
        ctx,
        "track",
        "boxmot.engine.tracker",
        "TrackingSession",
        {
            **kwargs,
            "yolo_model": detector or yolo_model,
            "reid_model": reid or reid_model,
            "classes": classes,
            "source": src,
            "benchmark": bench,
            "split": split,
            **({"tracker": tracker} if tracker is not None else {}),
        },
        detector=detector,
        reid=reid,
        tracker=tracker,
    )
    
@boxmot.command(help='Generate detections and embeddings')
@click.argument('detector', required=False)
@click.argument('reid', required=False)
@data_option
@source_option(default=BOXMOT_DEFAULTS.generate.source, help_text='direct dataset root to generate dets/embs for without a benchmark config')
@core_options
@plural_model_options
@click.pass_context
def generate(ctx, detector, reid, data, yolo_model, reid_model, classes, **kwargs):
    src = kwargs.pop('source')
    data, src = _normalize_generate_input(data, src, "generate")
    src, bench, split = _resolve_source_context(src)
    _dispatch_cli_workflow(
        ctx,
        "generate",
        "boxmot.engine.cache",
        "DetectionsEmbeddingsGenerator",
        {
            **kwargs,
            "yolo_model": detector or list(yolo_model),
            "reid_model": reid or list(reid_model),
            "classes": classes,
            "data": data,
            "source": src,
            "benchmark": bench,
            "split": split,
        },
        detector=detector,
        reid=reid,
    )


@boxmot.command(help='Evaluate tracking performance')
@click.argument('detector', required=False)
@click.argument('reid', required=False)
@click.argument('tracker', required=False)
@data_option
@core_options
@plural_model_options
@click.pass_context
def eval(ctx, detector, reid, tracker, data, yolo_model, reid_model, classes, **kwargs):
    # Allow benchmark/default-model runs to specify only the tracker positionally.
    detector, reid, tracker = _normalize_eval_positionals(detector, reid, tracker)
    if tracker:
        kwargs.pop("tracking_method", None)

    data = _require_benchmark_input(data, "eval")
    _dispatch_cli_workflow(
        ctx,
        "eval",
        "boxmot.engine.evaluator",
        None,
        {
            **kwargs,
            "yolo_model": detector or list(yolo_model),
            "reid_model": reid or list(reid_model),
            "classes": classes,
            "data": data,
            "source": None,
            "benchmark": "",
            "split": "",
            **({"tracker": tracker} if tracker is not None else {}),
        },
        detector=detector,
        reid=reid,
        tracker=tracker,
    )


@boxmot.command(help='Tune models via evolutionary algorithms')
@click.argument('detector', required=False)
@click.argument('reid', required=False)
@click.argument('tracker', required=False)
@data_option
@core_options
@tune_options
@plural_model_options
@click.pass_context
def tune(ctx, detector, reid, tracker, data, yolo_model, reid_model, classes, **kwargs):
    # Allow benchmark/default-model runs to specify only the tracker positionally.
    detector, reid, tracker = _normalize_eval_positionals(detector, reid, tracker)
    if tracker:
        kwargs.pop("tracking_method", None)

    data = _require_benchmark_input(data, "tune")
    _dispatch_cli_workflow(
        ctx,
        "tune",
        "boxmot.engine.tuner",
        "TrackerTuner",
        {
            **kwargs,
            "yolo_model": detector or list(yolo_model),
            "reid_model": reid or list(reid_model),
            "classes": classes,
            "data": data,
            "source": None,
            "benchmark": "",
            "split": "",
            **({"tracker": tracker} if tracker is not None else {}),
        },
        detector=detector,
        reid=reid,
        tracker=tracker,
    )


@boxmot.command(help='Export ReID models')
@export_options
@click.pass_context
def export(ctx, **kwargs):
    """
    Command 'export': export ReID model weights and configurations for deployment.
    Mirrors the standalone argparse-based export script.
    """
    args = _build_cli_namespace(ctx, "export", kwargs)
    _run_engine_workflow("boxmot.engine.export", None, args)


main = boxmot

if __name__ == "__main__":
    boxmot()
