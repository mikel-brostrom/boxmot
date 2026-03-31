#!/usr/bin/env python3
from __future__ import annotations

"""
CLI for BoxMOT: multi-step multiple object tracking pipeline.
Provides commands to track, generate detections and embeddings, evaluate performance, tune models, or run all steps.
"""
import multiprocessing as mp
mp.set_start_method("spawn", force=True)


from pathlib import Path
from typing import Optional, Tuple

import click
from click.core import ParameterSource

from boxmot.configs import (
    DEFAULT_DETECTOR,
    DEFAULT_REID,
    build_mode_namespace,
    get_mode_default,
)
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING
from boxmot.utils import WEIGHTS
from boxmot.utils.benchmark_config import resolve_benchmark_cfg_path
from boxmot.utils.misc import parse_imgsz


# Shared command options (excluding model, classes, and input selection)
def core_options(func):
    options = [
        click.option('--imgsz', callback=parse_imgsz, default=get_mode_default("eval", "imgsz"), type=str,
                     help='Image size for model input as H,W (e.g. 800,1440) or single int for square. Default: read from the selected detector config, otherwise use detector-specific defaults.'),
        click.option('--fps', type=int, default=get_mode_default("eval", "fps"),
                     help='video frame-rate'),
        click.option('--conf', type=float, default=get_mode_default("eval", "conf"),
                     help='Min confidence threshold. Default: read from the selected detector config, fallback 0.01.'),
        click.option('--iou', type=float, default=get_mode_default("eval", "iou"),
                     help='IoU threshold for NMS'),
        click.option('--device', default=get_mode_default("eval", "device"),
                     help='cuda device(s), e.g. 0 or 0,1,2,3 or cpu'),
        click.option('--batch-size', type=int, default=get_mode_default("eval", "batch_size"), show_default=True,
                 help='micro-batch size for batched detection/embedding'),
        click.option('--auto-batch/--no-auto-batch', default=get_mode_default("eval", "auto_batch"), show_default=True,
                 help='probe GPU memory with a dummy pass to pick a safe batch size'),
        click.option('--resume/--no-resume', default=get_mode_default("eval", "resume"), show_default=True,
             help='resume detection/embedding generation from progress checkpoints'),
        click.option('--n-threads', type=int, default=get_mode_default("eval", "n_threads"),
                 help='CPU threads for image decoding; defaults to min(8, cpu_count)'),
        click.option('--project', type=Path, default=get_mode_default("eval", "project"),
                     help='save results to project/name'),
        click.option('--name', default=get_mode_default("eval", "name"), help='save results to project/name'),
        click.option('--exist-ok', is_flag=True, default=get_mode_default("eval", "exist_ok"),
                     help='existing project/name ok, do not increment'),
        click.option('--half', is_flag=True, default=get_mode_default("eval", "half"),
                     help='use FP16 half-precision inference'),
        click.option('--vid-stride', type=int, default=get_mode_default("eval", "vid_stride"),
                     help='video frame-rate stride'),
        click.option('--ci', is_flag=True, default=get_mode_default("eval", "ci"),
                     help='reuse existing runs in CI (no UI)'),
        click.option('--tracker', 'tracking_method', type=str, default=get_mode_default("eval", "tracker"), show_default=True,
                     help='deepocsort, botsort, strongsort, ...'),
        click.option('--verbose', is_flag=True, default=get_mode_default("eval", "verbose"),
                     help='print detailed logs'),
        click.option('--agnostic-nms', is_flag=True, default=get_mode_default("eval", "agnostic_nms"),
                     help='class-agnostic NMS'),
        click.option(
            "--postprocessing", type=click.Choice(["none", "gsi", "gbrc"], case_sensitive=False), default=get_mode_default("eval", "postprocessing"),
            help="Postprocess tracker output: none | gsi (Gaussian smoothed interpolation) | gbrc (gradient boosting smooth).",
        ),
        click.option('--show', is_flag=True, default=get_mode_default("eval", "show"),
                     help='display tracking in a window'),
        click.option('--show-labels/--hide-labels', default=get_mode_default("eval", "show_labels"),
                     help='show or hide detection labels'),
        click.option('--show-conf/--hide-conf', default=get_mode_default("eval", "show_conf"),
                     help='show or hide detection confidences'),
        click.option('--show-trajectories', is_flag=True, default=get_mode_default("eval", "show_trajectories"),
                     help='overlay past trajectories'),
           click.option('--show-kf-preds', 'show_kf_preds', is_flag=True, default=get_mode_default("eval", "show_kf_preds"),
               help='show Kalman-filter predictions'),
        click.option('--save-txt', is_flag=True, default=get_mode_default("eval", "save_txt"),
                     help='save results to a .txt file'),
        click.option('--save-crop', is_flag=True, default=get_mode_default("eval", "save_crop"),
                     help='save cropped detections'),
        click.option('--save', is_flag=True, default=get_mode_default("eval", "save"),
                     help='save annotated video'),
        click.option('--line-width', type=int, default=get_mode_default("eval", "line_width"),
                     help='bounding box line width'),
        click.option('--per-class', is_flag=True, default=get_mode_default("eval", "per_class"),
                     help='track each class separately'),
        click.option('--target-id', type=int, default=get_mode_default("eval", "target_id"),
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
                     default=DEFAULT_DETECTOR,
                     help='path to YOLO weights for detection'),
        click.option('--reid', 'reid_model', type=Path,
                     default=DEFAULT_REID,
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
                     default=[DEFAULT_DETECTOR],
                     help='one or more YOLO weights for detection'),
        click.option('--reid', 'reid_model', type=Path, multiple=True,
                     default=[DEFAULT_REID],
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
        click.option('--batch-size', type=int, default=get_mode_default("export", "batch_size"),
                     help='Batch size for export'),
        click.option('--imgsz', '--img', '--img-size', callback=parse_imgsz, type=str,
                     default=get_mode_default("export", "imgsz"), help='Image size as H,W (e.g. 256,128)'),
        click.option('--device', default=get_mode_default("export", "device"),
                     help="CUDA device (e.g., '0', '0,1,2,3', or 'cpu')"),
        click.option('--optimize', is_flag=True, default=get_mode_default("export", "optimize"),
                     help='Optimize TorchScript for mobile (CPU export only)'),
        click.option('--dynamic', is_flag=True, default=get_mode_default("export", "dynamic"),
                     help='Enable dynamic axes for ONNX/TF/TensorRT export'),
        click.option('--simplify', is_flag=True, default=get_mode_default("export", "simplify"),
                     help='Simplify ONNX model'),
        click.option('--opset', type=int, default=get_mode_default("export", "opset"),
                     help='ONNX opset version'),
        click.option('--workspace', type=int, default=get_mode_default("export", "workspace"),
                     help='TensorRT workspace size (GB)'),
        click.option('--verbose', is_flag=True,
                     help='Enable verbose logging for TensorRT'),
        click.option('--weights', type=Path,
                     default=get_mode_default("export", "weights") or WEIGHTS / 'osnet_x0_25_msmt17.pt',
                     help='Path to the model weights (.pt file)'),
        click.option('--half', is_flag=True,
                     help='Enable FP16 half-precision export (GPU only)'),
        click.option('--include', multiple=True, default=tuple(get_mode_default("export", "include")),
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
        click.option('--n-trials', type=int, default=get_mode_default("tune", "n_trials"),
                     help='number of trials for evolutionary tuning'),
        click.option('--objectives', type=str, multiple=True,
                     default=tuple(get_mode_default("tune", "objectives")),
                     help='metrics to track and return from each trial'),
        click.option('--maximize', type=str, multiple=True, default=tuple(get_mode_default("tune", "maximize")),
                     help='metrics to maximize; defaults to first --objectives value (e.g. HOTA)'),
        click.option('--minimize', type=str, multiple=True, default=tuple(get_mode_default("tune", "minimize")),
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
@source_option(default='0', help_text='file/dir/URL/glob, 0 for webcam')
@core_options
@singular_model_options
@click.pass_context
def track(ctx, detector, reid, tracker, yolo_model, reid_model, classes, **kwargs):
    if tracker:
        kwargs.pop("tracking_method", None)
    src, bench, split = _resolve_source_context(kwargs.pop('source'))
    explicit_keys = _explicit_cli_keys(ctx)
    if detector:
        explicit_keys.add("detector")
    if reid:
        explicit_keys.add("reid")
    if tracker:
        explicit_keys.add("tracker")

    args = build_mode_namespace(
        "track",
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
        explicit_keys=explicit_keys,
    )

    from boxmot.engine.tracker import main as run_track
    run_track(args)
    
@boxmot.command(help='Generate detections and embeddings')
@click.argument('detector', required=False)
@click.argument('reid', required=False)
@data_option
@source_option(default=None, help_text='direct dataset root to generate dets/embs for without a benchmark config')
@core_options
@plural_model_options
@click.pass_context
def generate(ctx, detector, reid, data, yolo_model, reid_model, classes, **kwargs):
    src = kwargs.pop('source')
    data, src = _normalize_generate_input(data, src, "generate")
    src, bench, split = _resolve_source_context(src)
    explicit_keys = _explicit_cli_keys(ctx)
    if detector:
        explicit_keys.add("detector")
    if reid:
        explicit_keys.add("reid")

    args = build_mode_namespace(
        "generate",
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
        explicit_keys=explicit_keys,
    )
    from boxmot.engine.evaluator import run_generate_dets_embs
    run_generate_dets_embs(args)


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
    explicit_keys = _explicit_cli_keys(ctx)
    if detector:
        explicit_keys.add("detector")
    if reid:
        explicit_keys.add("reid")
    if tracker:
        explicit_keys.add("tracker")

    args = build_mode_namespace(
        "eval",
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
        explicit_keys=explicit_keys,
    )
    from boxmot.engine.evaluator import main as run_eval
    run_eval(args)


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
    explicit_keys = _explicit_cli_keys(ctx)
    if detector:
        explicit_keys.add("detector")
    if reid:
        explicit_keys.add("reid")
    if tracker:
        explicit_keys.add("tracker")

    args = build_mode_namespace(
        "tune",
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
        explicit_keys=explicit_keys,
    )
    from boxmot.engine.tuner import main as run_tuning
    run_tuning(args)


@boxmot.command(help='Export ReID models')
@export_options
@click.pass_context
def export(ctx, **kwargs):
    """
    Command 'export': export ReID model weights and configurations for deployment.
    Mirrors the standalone argparse-based export script.
    """
    args = build_mode_namespace("export", kwargs, explicit_keys=_explicit_cli_keys(ctx))
    from boxmot.engine.export import main as run_export
    run_export(args)


@boxmot.command(help='Show BoxMOT version')
def version():
    """Display the current BoxMOT version."""
    from boxmot import __version__
    click.echo(f"BoxMOT {__version__}")


@boxmot.command(help='Show help information')
@click.pass_context
def help(ctx):
    """Display help information."""
    # Get the parent context (main boxmot group)
    parent_ctx = ctx.parent
    click.echo(parent_ctx.get_help())


main = boxmot

if __name__ == "__main__":
    boxmot()
