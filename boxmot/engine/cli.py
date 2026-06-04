#!/usr/bin/env python3
from __future__ import annotations

"""
CLI for BoxMOT: multi-step multiple object tracking pipeline.
Provides commands to track, generate detections and embeddings, evaluate performance, tune models, research tracker changes, or run all steps.
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
    list_training_recipes,
)
from boxmot.configs.benchmark import resolve_benchmark_cfg_path
from boxmot.utils.misc import parse_imgsz

RUNTIME_DEFAULTS = BOXMOT_DEFAULTS.eval
TRACK_DEFAULTS = BOXMOT_DEFAULTS.track
TUNE_DEFAULTS = BOXMOT_DEFAULTS.tune
RESEARCH_DEFAULTS = BOXMOT_DEFAULTS.research
EXPORT_DEFAULTS = BOXMOT_DEFAULTS.export
TRAIN_DEFAULTS = BOXMOT_DEFAULTS.train
SHARED_DEFAULTS = BOXMOT_DEFAULTS.shared

_TUNE_METRIC_OPTIONS = {"--objectives", "--maximize", "--minimize"}


def _click_imgsz_default(value):
    """Normalize configured image sizes into a Click-friendly default value."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(part) for part in value)
    return value


def _normalize_tune_metric_cli_args(args: list[str]) -> list[str]:
    """Fold space-separated tune metric values into Click option values."""
    if "tune" not in args:
        return args

    tune_index = args.index("tune")
    prefix = args[: tune_index + 1]
    tokens = args[tune_index + 1 :]
    normalized: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        option = None
        inline_value = None

        if token in _TUNE_METRIC_OPTIONS:
            option = token
        else:
            for candidate in _TUNE_METRIC_OPTIONS:
                prefix_text = f"{candidate}="
                if token.startswith(prefix_text):
                    option = candidate
                    inline_value = token[len(prefix_text):]
                    break

        if option is None:
            normalized.append(token)
            index += 1
            continue

        values: list[str] = []
        if inline_value not in {None, ""}:
            values.append(inline_value)
        index += 1
        while index < len(tokens) and not tokens[index].startswith("-"):
            values.append(tokens[index])
            index += 1

        normalized.append(option)
        if values:
            normalized.append(",".join(values))

    return prefix + normalized


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
        click.option('--tracker', type=str, default=RUNTIME_DEFAULTS.tracker, show_default=True,
                     help='deepocsort, botsort, strongsort, ...'),
        click.option('--verbose', is_flag=True, default=RUNTIME_DEFAULTS.verbose,
                     help='print detailed logs'),
        click.option('--show-timing/--hide-timing', default=RUNTIME_DEFAULTS.show_timing, show_default=True,
                     help='print runtime timing summary after evaluation'),
        click.option('--agnostic-nms', is_flag=True, default=RUNTIME_DEFAULTS.agnostic_nms,
                     help='class-agnostic NMS'),
        click.option(
            "--postprocessing", type=str, default=RUNTIME_DEFAULTS.postprocessing,
            help="Postprocess tracker output (comma-separated, applied in order): none | gsi | gbrc | gta. E.g. 'gbrc,gta'.",
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
                     help='ID to highlight in green'),
        click.option('--masks-dir', type=str, default=None,
                     help='Override directory for cached segmentation masks (.npz files)'),
        click.option('--masks-model', type=click.Choice(['maskrcnn'], case_sensitive=False), default=None,
                     help='Mask model to use for generation (stored under cache tree automatically)'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def source_option(default='0', help_text='file/dir/URL/glob, 0 for webcam'):
    """Attach a ``--source`` option with command-specific defaults/help."""
    return click.option('--source', type=str, default=default, help=help_text)


def split_option(func):
    """Attach a ``--split`` option to override the dataset split (train/val/test)."""
    return click.option(
        '--split', type=str, default=None,
        help='Dataset split to use (e.g. train, val, test, ablation). Overrides auto-detection from source path.'
    )(func)


def detection_source_option(func):
    """Attach a ``--detection-source`` option to choose public or private detections."""
    return click.option(
        '--detection-source', type=click.Choice(['public', 'private']), default=None,
        help='Detection source: "public" reads det/det.txt from sequences, "private" (default) runs the configured detector model.'
    )(func)


def data_option(func):
    """Attach the benchmark-config option."""
    return click.option(
        '--benchmark',
        'data',
        type=str,
        default=None,
        help='benchmark config name or YAML file, e.g. mot17 or boxmot/configs/datasets/mot17.yaml',
    )(func)


def replay_backend_option(func):
    """Attach the cached-tracking backend option for eval-like workflows."""
    return click.option(
        '--tracking-backend',
        type=click.Choice(["process", "thread", "cpp"], case_sensitive=False),
        default="process",
        show_default=True,
        help=(
            "Cached replay executor for eval/tune/research. "
            "Use 'cpp' as a compatibility alias for '--tracker-backend cpp'."
        ),
    )(func)


def tracker_backend_option(func):
    """Attach the tracker implementation backend option."""
    return click.option(
        '--tracker-backend',
        type=click.Choice(["python", "cpp"], case_sensitive=False),
        default=RUNTIME_DEFAULTS.tracker_backend,
        show_default=True,
        help=(
            "Tracker implementation backend. Native 'cpp' is available for "
            "botsort, bytetrack, ocsort, and sfsort."
        ),
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


def _build_cli_namespace(
    ctx: click.Context,
    mode: str,
    payload: dict,
):
    """Build the normalized mode namespace while preserving explicitly provided CLI values."""
    return build_mode_namespace(mode, payload, explicit_keys=_explicit_cli_keys(ctx))


def _dispatch_cli_workflow(
    ctx: click.Context,
    mode: str,
    module_name: str,
    payload: dict,
) -> None:
    """Build CLI args for a workflow and execute its canonical ``main(args)`` entry point."""
    args = _build_cli_namespace(ctx, mode, payload)
    _run_engine_workflow(module_name, args)


def _resolve_source_context(source: Optional[str]) -> Tuple[Optional[str], str, str]:
    """Return ``(source, benchmark, split)`` metadata for a concrete source path."""
    if source is None:
        return None, "", ""

    source_path = Path(source)
    return source, source_path.parent.name, source_path.name


def _is_live_source_value(source: Optional[str]) -> bool:
    if source is None:
        return False
    return str(source).isdigit() or "://" in str(source)


def _apply_track_cli_defaults(ctx: click.Context, payload: dict) -> dict:
    resolved = dict(payload)
    source = resolved.get("source")
    has_explicit_output = any(
        _is_option_explicit(ctx, option_name)
        for option_name in ("show", "save", "save_txt")
    )
    if _is_live_source_value(source) and not has_explicit_output:
        resolved["show"] = True
    return resolved


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
    if source is None or Path(source).exists():
        return

    try:
        resolve_benchmark_cfg_path(source)
    except FileNotFoundError:
        return

    raise click.UsageError(
        f"{command_name} uses --benchmark <benchmark.yaml> for benchmark configs. "
        f"Pass '--benchmark {source}' instead of '--source {source}'."
    )


def _require_benchmark_input(data: Optional[str], command_name: str) -> str:
    """Require a benchmark config for benchmark-only commands such as eval/tune."""
    if not data:
        raise click.UsageError(
            f"{command_name} requires --benchmark <benchmark.yaml>. "
            f"Use 'generate --source <dataset-path>' to prepare direct datasets before running {command_name}."
        )
    return data


def _run_engine_workflow(module_name: str, args) -> None:
    """Run an engine module through its canonical ``main(args)`` entry point.

    Engine ``main`` functions render their own Rich workflow panels and capture
    failures into the panel's traceback view via ``WorkflowProgress.fail()``.
    When the panel has rendered the error (indicated by __exit__ setting
    ``_workflow_rendered_error`` on the exception), we convert the exception
    into a clean ``click.exceptions.Exit(1)`` to avoid a duplicate traceback.
    Otherwise the exception propagates normally so the user sees what went
    wrong.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise click.ClickException(
            f"Failed to import engine module '{module_name}': {exc}\n"
            f"Try running: uv sync --all-extras --all-groups"
        ) from exc
    main_fn = getattr(module, "main", None)
    if main_fn is None:
        raise AttributeError(f"{module_name} does not expose main")
    try:
        main_fn(args)
    except (KeyboardInterrupt, SystemExit, click.exceptions.Exit, click.ClickException):
        raise
    except BaseException as exc:
        if getattr(exc, "_workflow_rendered_error", False):
            raise click.exceptions.Exit(code=1)
        raise


def singular_model_options(func):
    options = [
        click.option('--detector', type=Path,
                     default=SHARED_DEFAULTS.detector,
                     help='path to YOLO weights for detection'),
        click.option('--reid', type=Path,
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
        click.option('--detector', type=Path, multiple=True,
                     default=[SHARED_DEFAULTS.detector],
                     help='one or more YOLO weights for detection'),
        click.option('--reid', type=Path, multiple=True,
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
                     help='Enable dynamic axes for ONNX/TensorRT export'),
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
        click.option('--max-concurrent-trials', type=int, default=0,
                     help='max concurrent trials (0 = auto, defaults to min(4, cpu_count)); '
                          'controls parallelism and improves Bayesian search effectiveness'),
        click.option('--time-budget-s', type=float, default=None,
                     help='optional time budget in seconds for the entire tuning run; '
                          'Tune stops launching new trials after this time'),
        click.option('--resume-tune', type=str, default=None,
                     help='resume a Ray Tune experiment; pass a folder name (e.g. deepocsort_tune_3) '
                          'or full path under runs/ray/. Retries errored trials and continues remaining ones.'),
        click.option('--objectives', type=str, multiple=True,
                     default=TUNE_DEFAULTS.objectives,
                     help='metrics to track and return from each trial; accepts repeated, comma-separated, or space-separated values'),
        click.option('--maximize', type=str, multiple=True, default=TUNE_DEFAULTS.maximize,
                     help='metrics to maximize; accepts repeated, comma-separated, or space-separated values; defaults to first --objectives value (e.g. HOTA)'),
        click.option('--minimize', type=str, multiple=True, default=TUNE_DEFAULTS.minimize,
                     help='metrics to minimize for Pareto search; accepts repeated, comma-separated, or space-separated values (e.g. IDSW_rate); '
                          'triggers multi-objective mode when set'),
        click.option('--search-alg', 'search_alg', type=click.Choice(['optuna', 'hyperopt', 'random'], case_sensitive=False),
                     default='optuna',
                     help='search algorithm backend for hyperparameter optimization; '
                          'optuna (default) uses TPE with conditional search spaces, '
                          'hyperopt uses Tree-structured Parzen Estimators via HyperOpt, '
                          'random uses uniform random sampling'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def research_options(func):
    """
    Decorator adding GEPA-backed research options.
    """
    options = [
        click.option('--proposal-model', type=str, default=RESEARCH_DEFAULTS.proposal_model, show_default=True,
                     help='proposal model identifier used by GEPA reflections, e.g. '
                          'openai/gpt-5.4, anthropic/claude-sonnet-4-20250514, '
                          'openrouter/openai/gpt-5.4'),
        click.option('--proposal-api-key', type=str, default=RESEARCH_DEFAULTS.proposal_api_key,
                     help='proposal model API key; prefer shell env vars in CI but this can inject the key at runtime'),
        click.option('--proposal-api-key-env', type=str, default=RESEARCH_DEFAULTS.proposal_api_key_env,
                     help='environment variable name for --proposal-api-key when the provider is not inferred, '
                          'e.g. OPENAI_API_KEY or ANTHROPIC_API_KEY'),
        click.option('--max-metric-calls', type=int, default=RESEARCH_DEFAULTS.max_metric_calls, show_default=True,
                     help='maximum number of benchmark evaluations during research'),
        click.option('--eval-timeout', type=float, default=RESEARCH_DEFAULTS.eval_timeout, show_default=True,
                     help='hard timeout in seconds for each benchmark evaluation'),
        click.option('--keep-workspace/--no-keep-workspace', default=RESEARCH_DEFAULTS.keep_workspace, show_default=True,
                     help='preserve the temporary research workspace after the run'),
        click.option('--hota-penalty', type=float, default=RESEARCH_DEFAULTS.hota_penalty, show_default=True,
                     help='penalty multiplier for combined HOTA regression versus baseline'),
        click.option('--idf1-penalty', type=float, default=RESEARCH_DEFAULTS.idf1_penalty, show_default=True,
                     help='penalty multiplier for combined IDF1 regression versus baseline'),
        click.option('--mota-penalty', type=float, default=RESEARCH_DEFAULTS.mota_penalty, show_default=True,
                     help='penalty multiplier for combined MOTA regression versus baseline'),
        click.option('--hota-tolerance', type=float, default=RESEARCH_DEFAULTS.hota_tolerance, show_default=True,
                     help='allowed combined HOTA drop before penalties apply'),
        click.option('--idf1-tolerance', type=float, default=RESEARCH_DEFAULTS.idf1_tolerance, show_default=True,
                     help='allowed combined IDF1 drop before penalties apply'),
        click.option('--mota-tolerance', type=float, default=RESEARCH_DEFAULTS.mota_tolerance, show_default=True,
                     help='allowed combined MOTA drop before penalties apply'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func



class CommandFirstGroup(click.Group):
    """Custom Click Group with improved help formatting - Ultralytics-style."""

    def parse_args(self, ctx, args):
        """Normalize tune metric lists before Click validates subcommand args."""
        return super().parse_args(ctx, _normalize_tune_metric_cli_args(list(args)))

    def format_help(self, _ctx, formatter):
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
            formatter.write_text("Where  MODE (required) is one of [track, eval, tune, research, generate, train, export]")
            formatter.write_text("       --detector selects a YOLO model like yolov8n, yolov9c, yolo11m, yolox_x")
            formatter.write_text("       --reid selects a ReID model like osnet_x0_25_msmt17, mobilenetv2_x1_4")
            formatter.write_text("       --tracker selects one of [deepocsort, botsort, bytetrack, strongsort, ocsort, hybridsort, boosttrack, sfsort]")
            formatter.write_text("       OPTIONS (optional) flags like '--source 0' for tracking inputs or '--benchmark mot17 --split ablation' for benchmark-driven eval/tune/research runs.")
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
                formatter.write_text("boxmot eval --benchmark mot17 --split ablation --tracker boosttrack")
            formatter.write_paragraph()

            formatter.write_text("4. Tune tracker hyperparameters:")
            with formatter.indentation():
                formatter.write_text("boxmot tune --benchmark mot17 --split ablation --tracker deepocsort --n-trials 10")
            formatter.write_paragraph()

            formatter.write_text("5. Research tracker code changes:")
            with formatter.indentation():
                formatter.write_text(
                    "boxmot research --benchmark mot17 --split ablation --tracker bytetrack "
                    "--proposal-model openai/gpt-5.4 --max-metric-calls 24"
                )
            formatter.write_paragraph()

            formatter.write_text("6. Train a ReID model:")
            with formatter.indentation():
                formatter.write_text("boxmot train --model osnet_x0_25 --dataset market1501 --data-dir /path/to/data --epochs 120 --device 0")
            formatter.write_paragraph()

            formatter.write_text("7. Train on all person datasets jointly:")
            with formatter.indentation():
                formatter.write_text("boxmot train --model vit_nano --dataset market1501,duke,cuhk03,msmt17 --data-dir /path/to/data --device 0")
            formatter.write_paragraph()

            formatter.write_text("8. Export ReID model:")
            with formatter.indentation():
                formatter.write_text("boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine --dynamic")
        formatter.write_paragraph()

        # Available modes
        formatter.write_text("Modes:")
        with formatter.indentation():
            formatter.write_text("track      Track objects in video/webcam stream")
            formatter.write_text("eval       Evaluate tracker performance on MOT dataset")
            formatter.write_text("tune       Optimize tracker hyperparameters")
            formatter.write_text("research   Evolve tracker code against benchmark metrics")
            formatter.write_text("generate   Generate detections and embeddings")
            formatter.write_text("train      Train a ReID model on a person/vehicle dataset")
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
@source_option(default=TRACK_DEFAULTS.source, help_text='file/dir/URL/glob, 0 for webcam')
@split_option
@tracker_backend_option
@core_options
@singular_model_options
@click.pass_context
def track(ctx, detector, reid, classes, split, **kwargs):
    src, bench, auto_split = _resolve_source_context(kwargs.pop('source'))
    _dispatch_cli_workflow(
        ctx,
        "track",
        "boxmot.engine.tracking.tracker",
        _apply_track_cli_defaults(ctx, {
            **kwargs,
            "detector": detector,
            "reid": reid,
            "classes": classes,
            "source": src,
            "benchmark": bench,
            "split": split if split else auto_split,
        }),
    )

@boxmot.command(help='Generate detections and embeddings')
@data_option
@source_option(default=BOXMOT_DEFAULTS.generate.source, help_text='direct dataset root to generate dets/embs for without a benchmark config')
@split_option
@detection_source_option
@core_options
@plural_model_options
@click.pass_context
def generate(ctx, data, detector, reid, classes, split, detection_source, **kwargs):
    src = kwargs.pop('source')
    _require_generate_input(data, src, "generate")
    src, bench, auto_split = _resolve_source_context(src)
    _dispatch_cli_workflow(
        ctx,
        "generate",
        "boxmot.engine.eval.cache",
        {
            **kwargs,
            "detector": list(detector),
            "reid": list(reid),
            "classes": classes,
            "data": data,
            "source": src,
            "benchmark": bench,
            "split": split if split else auto_split,
            "detection_source": detection_source,
        },
    )


@boxmot.command(help='Evaluate tracking performance')
@data_option
@split_option
@detection_source_option
@replay_backend_option
@tracker_backend_option
@core_options
@plural_model_options
@click.option('--tune-kf/--no-tune-kf', 'tune_kf', default=False,
              help='Run KF noise tuning (Q/R estimation) before tracking. '
              'Automatically selects parameterization based on the tracker. '
              'Requires cached dets and GT.')
@click.pass_context
def eval(ctx, data, detector, reid, classes, split, detection_source, tune_kf, **kwargs):
    data = _require_benchmark_input(data, "eval")
    _dispatch_cli_workflow(
        ctx,
        "eval",
        "boxmot.engine.eval.evaluator",
        {
            **kwargs,
            "detector": list(detector),
            "reid": list(reid),
            "classes": classes,
            "data": data,
            "source": None,
            "benchmark": "",
            "split": split or "",
            "detection_source": detection_source,
            "tune_kf": tune_kf,
        },
    )


@boxmot.command(help='Tune models via evolutionary algorithms')
@data_option
@split_option
@detection_source_option
@replay_backend_option
@tracker_backend_option
@core_options
@tune_options
@plural_model_options
@click.option('--tune-kf/--no-tune-kf', 'tune_kf', default=False,
              help='Run KF noise tuning (Q/R estimation) before tracker hyperparameter tuning. '
              'Applied once, then reused for all trials.')
@click.pass_context
def tune(ctx, data, detector, reid, classes, split, detection_source, tune_kf, **kwargs):
    data = _require_benchmark_input(data, "tune")
    _dispatch_cli_workflow(
        ctx,
        "tune",
        "boxmot.engine.tuning.tuner",
        {
            **kwargs,
            "detector": list(detector),
            "reid": list(reid),
            "classes": classes,
            "data": data,
            "source": None,
            "benchmark": "",
            "split": split or "",
            "detection_source": detection_source,
            "tune_kf": tune_kf,
        },
    )


@boxmot.command(help='Research tracker code changes with GEPA')
@data_option
@split_option
@detection_source_option
@replay_backend_option
@tracker_backend_option
@core_options
@research_options
@plural_model_options
@click.pass_context
def research(ctx, data, detector, reid, classes, split, detection_source, **kwargs):
    data = _require_benchmark_input(data, "research")
    _dispatch_cli_workflow(
        ctx,
        "research",
        "boxmot.engine.research",
        {
            **kwargs,
            "detector": list(detector),
            "reid": list(reid),
            "classes": classes,
            "data": data,
            "source": None,
            "benchmark": "",
            "split": split or "",
            "detection_source": detection_source,
        },
    )


def train_options(func):
    """Decorator adding ReID training options."""
    from boxmot.reid.core.config import MODEL_TYPES
    from boxmot.reid.core.preprocessing import PREPROCESS_REGISTRY
    from boxmot.reid.datasets import DATASET_REGISTRY

    options = [
        click.option('--recipe', type=click.Choice(list_training_recipes(), case_sensitive=False),
                     default=None,
                     help='Training recipe preset (overrides defaults; CLI flags still take priority). '
                          f'Available: {", ".join(list_training_recipes()) or "(none)"}'),
        click.option('--model', type=click.Choice(MODEL_TYPES, case_sensitive=False),
                     default=TRAIN_DEFAULTS.model, show_default=True,
                     help='ReID backbone architecture'),
        click.option('--dataset', type=str,
                     default=TRAIN_DEFAULTS.dataset, show_default=True,
                     help='Training dataset (comma-separated for joint training, '
                          f'e.g. market1501,duke,cuhk03,msmt17). '
                          f'Available: {", ".join(sorted(DATASET_REGISTRY.keys()))}'),
        click.option('--data-dir', type=click.Path(exists=True), required=False, default=None,
                     help='Root directory of the dataset (inferred from hparams.json when --resume is used)'),
        click.option('--loss', type=click.Choice(['softmax', 'triplet', 'ms'], case_sensitive=False),
                     default=TRAIN_DEFAULTS.loss, show_default=True,
                     help='Metric loss type (triplet=hard-mining triplet, ms=multi-similarity, softmax=CE only)'),
        click.option('--preprocess', type=click.Choice(sorted(PREPROCESS_REGISTRY.keys()), case_sensitive=False),
                     default=TRAIN_DEFAULTS.preprocess, show_default=True,
                     help='Crop preprocessing method; must match inference-time preprocessing'),
        click.option('--imgsz', callback=parse_imgsz, type=str,
                     default=_click_imgsz_default(TRAIN_DEFAULTS.imgsz),
                     help='Image size as H,W (e.g. 256,128)'),
        click.option('--batch-size', type=int, default=TRAIN_DEFAULTS.batch_size, show_default=True,
                     help='Training batch size'),
        click.option('--lr', type=float, default=TRAIN_DEFAULTS.lr, show_default=True,
                     help='Base learning rate'),
        click.option('--weight-decay', type=float, default=TRAIN_DEFAULTS.weight_decay, show_default=True,
                     help='Weight decay'),
        click.option('--epochs', type=int, default=TRAIN_DEFAULTS.epochs, show_default=True,
                     help='Number of training epochs'),
        click.option('--warmup-epochs', type=int, default=TRAIN_DEFAULTS.warmup_epochs, show_default=True,
                     help='Linear warmup epochs'),
        click.option('--eval-interval', type=int, default=TRAIN_DEFAULTS.eval_interval, show_default=True,
                     help='Validate every N epochs'),
        click.option('--p-ids', type=int, default=TRAIN_DEFAULTS.p_ids, show_default=True,
                     help='Number of identities per PK batch'),
        click.option('--k-instances', type=int, default=TRAIN_DEFAULTS.k_instances, show_default=True,
                     help='Number of instances per identity'),
        click.option('--margin', type=float, default=TRAIN_DEFAULTS.margin, show_default=True,
                     help='Triplet loss margin'),
        click.option('--label-smooth', type=float, default=TRAIN_DEFAULTS.label_smooth, show_default=True,
                     help='Label smoothing epsilon'),
        click.option('--center-loss-weight', type=float, default=TRAIN_DEFAULTS.center_loss_weight, show_default=True,
                     help='Center loss weight'),
        click.option('--eta-min', type=float, default=TRAIN_DEFAULTS.eta_min, show_default=True,
                     help='Minimum learning rate for cosine annealing schedule'),
        click.option('--pretrained/--no-pretrained', default=TRAIN_DEFAULTS.pretrained, show_default=True,
                     help='Use ImageNet-pretrained backbone'),
        click.option('--device', default=TRAIN_DEFAULTS.device,
                     help='cuda device, e.g. 0 or cpu or mps'),
        click.option('--project', type=click.Path(), default=TRAIN_DEFAULTS.project, show_default=True,
                     help='Save directory'),
        click.option('--name', default=TRAIN_DEFAULTS.name, show_default=True,
                     help='Experiment name'),
        click.option('--num-workers', type=int, default=TRAIN_DEFAULTS.num_workers, show_default=True,
                     help='Dataloader workers'),
        click.option('--seed', type=int, default=TRAIN_DEFAULTS.seed, show_default=True,
                     help='Random seed'),
        click.option('--eval-datasets', type=str, default=','.join(TRAIN_DEFAULTS.eval_datasets) if TRAIN_DEFAULTS.eval_datasets else '',
                     help='Comma-separated list of extra datasets for cross-domain evaluation '
                          '(e.g. duke,cuhk03,msmt17)'),
        click.option('--ema-decay', type=float, default=TRAIN_DEFAULTS.ema_decay,
                     help='EMA momentum decay for model averaging (e.g. 0.999). '
                          'Disabled by default. Inspired by DynaMix'),
        click.option('--gaussian-blur/--no-gaussian-blur', default=TRAIN_DEFAULTS.gaussian_blur, show_default=True,
                     help='Apply random Gaussian blur augmentation'),
        click.option('--color-jitter/--no-color-jitter', default=TRAIN_DEFAULTS.color_jitter, show_default=True,
                     help='Apply color jitter augmentation (auto-enabled for ViTs)'),
        click.option('--random-grayscale', type=float, default=TRAIN_DEFAULTS.random_grayscale, show_default=True,
                     help='Probability of random grayscale conversion (0 to disable)'),
        click.option('--random-erasing', type=float, default=TRAIN_DEFAULTS.random_erasing, show_default=True,
                     help='Probability of random erasing augmentation (0 to disable)'),
        click.option('--resume', type=click.Path(), default=None,
                     help='Resume training from a checkpoint dir or last.pt file'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


@boxmot.command(help='Train a ReID model')
@train_options
@click.pass_context
def train(ctx, **kwargs):
    # --data-dir is required unless --resume is provided
    if not kwargs.get('resume') and not kwargs.get('data_dir'):
        raise click.MissingParameter(param_hint="'--data-dir'", param_type='option')
    args = _build_cli_namespace(ctx, "train", kwargs)
    _run_engine_workflow("boxmot.engine.reid.trainer", args)


@boxmot.command(name='eval-reid', help='Evaluate a trained ReID model on query/gallery')
@click.option('--weights', type=click.Path(exists=True), required=True,
              help='Path to trained ReID checkpoint (.pt)')
@click.option('--model', type=str, default=None,
              help='Model architecture (auto-detected from checkpoint if omitted)')
@click.option('--dataset', type=str, required=True,
              help='Evaluation dataset (e.g. market1501, duke, msmt17)')
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Root directory of the dataset')
@click.option('--device', default='cpu', help='Device: cpu, mps, or cuda index')
@click.option('--batch-size', type=int, default=64, show_default=True,
              help='Batch size for feature extraction')
@click.option('--num-workers', type=int, default=4, show_default=True,
              help='Dataloader workers')
@click.option('--output', type=click.Path(), default=None,
              help='Directory to save eval JSON (default: next to weights)')
@click.pass_context
def eval_reid(ctx, **kwargs):
    args = _build_cli_namespace(ctx, "eval-reid", kwargs)
    _run_engine_workflow("boxmot.engine.reid.evaluator", args)


@boxmot.command(help='Export ReID models')
@export_options
@click.pass_context
def export(ctx, **kwargs):
    """
    Command 'export': export ReID model weights and configurations for deployment.
    Mirrors the standalone argparse-based export script.
    """
    args = _build_cli_namespace(ctx, "export", kwargs)
    _run_engine_workflow("boxmot.engine.reid.export", args)


@boxmot.command(help='Build native (C++) tracker shared libraries')
@click.option(
    '--tracker', 'trackers', multiple=True,
    type=click.Choice(['all', 'botsort', 'bytetrack', 'occluboost', 'ocsort', 'sfsort', 'reid'],
                      case_sensitive=False),
    default=('all',),
    help='Tracker(s) to build. Pass --tracker multiple times or use "all" (default).',
)
@click.option('--force', is_flag=True, default=False, help='Force rebuild even if libraries already exist.')
def build(trackers, force):
    """Compile the native C++ shared libraries shipped under ``boxmot/native/trackers``.

    Useful for editable installs (``pip install -e .``) where the wheel build
    step is skipped. Each tracker is built into ``build/native/<tracker>/`` and
    the resulting ``*_capi`` shared library is what the ctypes wrappers in
    ``boxmot.native`` load at runtime.
    """
    selected = {t.lower() for t in trackers}
    if 'all' in selected:
        selected = {'reid', 'botsort', 'bytetrack', 'occluboost', 'ocsort', 'sfsort'}

    # Sort so the shared ReID base is built first (other trackers depend on it
    # transitively at link time when configured standalone).
    order = ['reid', 'botsort', 'bytetrack', 'occluboost', 'ocsort', 'sfsort']
    selected = [name for name in order if name in selected]

    failures: list[tuple[str, str]] = []
    for name in selected:
        try:
            if name == 'reid':
                from boxmot.native.reid_capi import ensure_reid_capi_library
                lib = ensure_reid_capi_library(force_rebuild=force)
            else:
                module = __import__(f'boxmot.native.{name}_cpp', fromlist=['*'])
                ensure = getattr(module, f'ensure_{name}_cpp_library')
                lib = ensure(force_rebuild=force)
            click.echo(f"[boxmot build] {name}: built -> {lib}")
        except Exception as exc:  # noqa: BLE001 - surface CMake errors verbatim
            failures.append((name, str(exc)))
            click.echo(f"[boxmot build] {name}: FAILED\n{exc}", err=True)

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise click.ClickException(f"Native build failed for: {names}")


main = boxmot

if __name__ == "__main__":
    boxmot()
