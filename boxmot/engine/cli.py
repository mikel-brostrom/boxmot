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

# Shared CLI/Python API defaults and namespace normalization are engine concerns.
from boxmot.engine.config import (
    BOXMOT_DEFAULTS,
    build_mode_namespace,
    list_training_recipes,
)
from boxmot.engine.experiment import resolve_experiment_path
from boxmot.reid.backbones.head_registry import TRAIN_HEAD_TYPES
from boxmot.reid.backbones.option_registry import selector_choices
from boxmot.reid.core.preprocessing import PREPROCESS_REGISTRY
from boxmot.trackers.registry import TRACKER_MAPPING
from boxmot.utils.misc import parse_imgsz

RUNTIME_DEFAULTS = BOXMOT_DEFAULTS.eval
TRACK_DEFAULTS = BOXMOT_DEFAULTS.track
TUNE_DEFAULTS = BOXMOT_DEFAULTS.tune
RESEARCH_DEFAULTS = BOXMOT_DEFAULTS.research
EXPORT_DEFAULTS = BOXMOT_DEFAULTS.export
TRAIN_DEFAULTS = BOXMOT_DEFAULTS.train
SHARED_DEFAULTS = BOXMOT_DEFAULTS.shared

_TUNE_METRIC_OPTIONS = {"--objectives", "--maximize", "--minimize"}
_TRACKER_HELP = ", ".join(TRACKER_MAPPING)


def _click_imgsz_default(value):
    """Normalize configured image sizes into a Click-friendly default value."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(part) for part in value)
    return value


def _parse_head_parts(ctx, param, value):
    """Parse CSL-TinyViT head part granularities, e.g. ``1,2,4``."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = tuple(int(part) for part in value)
    else:
        tokens = [token for token in str(value).replace(";", ",").split(",") if token.strip()]
        parts = tuple(int(token) for token in tokens)
    if not parts:
        raise click.BadParameter("must contain at least one part granularity")
    if any(part < 1 for part in parts):
        raise click.BadParameter("part granularities must be positive integers")
    if 1 not in parts:
        raise click.BadParameter("must include 1 for the global branch")
    return tuple(dict.fromkeys(parts))


def _parse_int_tuple(ctx, param, value):
    """Parse optional comma-separated integer tuples."""
    del ctx, param
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        parts = tuple(int(part) for part in value)
    else:
        value = str(value).strip()
        if value.lower() in {"", "none", "off"}:
            return ()
        tokens = [token for token in value.replace(";", ",").split(",") if token.strip()]
        parts = tuple(int(token) for token in tokens)
    return tuple(dict.fromkeys(parts))


def _parse_coreml_buckets(ctx, param, value):
    """Parse safe, positive CoreML batch buckets capped at 32."""
    parts = _parse_int_tuple(ctx, param, value)
    if not parts:
        raise click.BadParameter("must contain at least one batch size")
    if any(part < 1 for part in parts):
        raise click.BadParameter("batch buckets must be positive integers")
    if max(parts) > 32:
        raise click.BadParameter("batch buckets are capped at 32; larger inputs are chunked")
    return tuple(sorted(set(parts)))


def _parse_int_pair(ctx, param, value):
    """Parse one or two comma-separated integers without deduplication."""
    del ctx, param
    if value is None:
        return None
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)):
        parts = tuple(int(part) for part in value)
    else:
        tokens = [token for token in str(value).replace(";", ",").split(",") if token.strip()]
        parts = tuple(int(token) for token in tokens)
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) != 2:
        raise click.BadParameter("must contain one integer or an H,W integer pair")
    if any(part <= 0 for part in parts):
        raise click.BadParameter("values must be positive integers")
    return parts


def _parse_tflite_static_activation_bits(ctx, param, value):
    """Parse TFLite static activation precision."""
    del ctx, param
    bits = int(value)
    if bits not in {8, 16}:
        raise click.BadParameter("must be 8 or 16")
    return bits


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
        click.option(
            '--fps',
            type=click.IntRange(min=1),
            default=RUNTIME_DEFAULTS.fps,
            help='frame-rate override: saved track video FPS or evaluation target FPS',
        ),
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
        click.option(
            '--n-threads', type=click.IntRange(min=1), default=RUNTIME_DEFAULTS.n_threads,
            help='Maximum CPU worker budget for image decoding and cached tracking',
        ),
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
                     help=f'one of: {_TRACKER_HELP}'),
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


def experiment_option(func):
    """Attach the experiment-config option."""
    return click.option(
        '--experiment',
        type=str,
        default=None,
        help=(
            'experiment id or YAML file, e.g. mot17-ablation-yolox-lmbn or '
            'boxmot/configs/experiments/mot17/ablation-yolox-lmbn.yaml'
        ),
    )(func)


def dataset_option(func):
    """Attach the model-free dataset-config option."""
    return click.option(
        '--dataset',
        type=str,
        default=RUNTIME_DEFAULTS.dataset,
        help=(
            'dataset id or YAML file, e.g. mot17 or '
            'boxmot/configs/datasets/mot17.yaml; uses the selected/default detector and ReID model'
        ),
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
            "botsort, bytetrack, occluboost, ocsort, and sfsort."
        ),
    )(func)


def association_function_option(func):
    """Attach the shared detection-track geometry selector."""
    return click.option(
        "--asso-func",
        type=click.Choice(
            ("iou", "giou", "diou", "ciou", "hmiou", "centroid"),
            case_sensitive=False,
        ),
        default=None,
        help=(
            "Association geometry override for AABB and OBB tracking. "
            "OBB ciou is a custom experimental adaptation; OBB hmiou is an "
            "experimental global-y height cue for scenes where image vertical is meaningful."
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


def _require_generate_input(experiment: Optional[str], source: Optional[str], command_name: str) -> None:
    """Validate experiment-vs-source selection for generate-like commands."""
    if experiment and source:
        raise click.UsageError(
            f"{command_name} accepts either --experiment <experiment-id-or-yaml> or "
            "--source <dataset-path>, not both."
        )
    if not experiment and not source:
        raise click.UsageError(
            f"{command_name} requires --experiment <experiment-id-or-yaml> for config-driven runs or "
            "--source <dataset-path> for direct datasets."
        )
    if source is None or Path(source).exists():
        return

    try:
        resolve_experiment_path(source)
    except FileNotFoundError:
        return

    raise click.UsageError(
        f"{command_name} uses --experiment <experiment-id-or-yaml> for experiment configs. "
        f"Pass '--experiment {source}' instead of '--source {source}'."
    )


def _require_experiment_input(experiment: Optional[str], command_name: str) -> str:
    """Require an experiment config for experiment-only commands such as eval/tune."""
    if not experiment:
        raise click.UsageError(
            f"{command_name} requires --experiment <experiment-id-or-yaml>. "
            f"Use 'generate --source <dataset-path>' to prepare direct datasets before running {command_name}."
        )
    return experiment


def _require_eval_input(
    experiment: Optional[str],
    dataset: Optional[str],
    command_name: str,
) -> tuple[Optional[str], Optional[str]]:
    """Require exactly one experiment config or model-free dataset."""
    if experiment and dataset:
        raise click.UsageError(
            f"{command_name} accepts either --dataset <dataset-id-or-yaml> or "
            "--experiment <experiment-id-or-yaml>, not both."
        )
    if not experiment and not dataset:
        raise click.UsageError(
            f"{command_name} requires either --dataset <dataset-id-or-yaml> or "
            "--experiment <experiment-id-or-yaml>."
        )
    return experiment, dataset


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
            "Install the required feature extra while repeating one PyTorch profile; "
            "for example: uv sync --extra cpu --extra yolo"
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
        click.option('--coreml-batch-buckets',
                     type=str,
                     callback=_parse_coreml_buckets,
                     default=",".join(str(value) for value in EXPORT_DEFAULTS.coreml_batch_buckets),
                     show_default=True,
                     help='Static MLProgram batch buckets; values above 32 are rejected'),
        click.option('--coreml-minimum-deployment-target',
                     type=click.Choice(['macOS12', 'macOS13', 'macOS14', 'macOS15', 'macOS26']),
                     default=EXPORT_DEFAULTS.coreml_minimum_deployment_target,
                     show_default=True,
                     help='Minimum macOS target; macOS15 enables native SDPA'),
        click.option('--coreml-compute-units',
                     type=click.Choice(['ALL', 'CPUAndGPU', 'CPUAndNeuralEngine', 'CPUOnly']),
                     default=EXPORT_DEFAULTS.coreml_compute_units,
                     show_default=True,
                     help='CoreML compute units used when compiling MLPrograms'),
        click.option('--coreml-timeout',
                     type=click.FloatRange(min=1.0),
                     default=EXPORT_DEFAULTS.coreml_timeout,
                     show_default=True,
                     help='Per-bucket CoreML conversion timeout in seconds'),
        click.option('--coreml-max-memory-gb',
                     type=click.FloatRange(min=1.0),
                     default=EXPORT_DEFAULTS.coreml_max_memory_gb,
                     show_default=True,
                     help='Per-bucket CoreML conversion process memory limit'),
        click.option('--tflite-quantize',
                     type=click.Choice(['none', 'weight', 'dynamic', 'static'], case_sensitive=False),
                     default=EXPORT_DEFAULTS.tflite_quantize,
                     show_default=True,
                     help=(
                         'Post-quantize TFLite export: weight=int8 weights with float compute, '
                         'dynamic=int8 dynamic range, static=int8 weights with calibrated activations'
                     )),
        click.option('--tflite-calibration-data',
                     type=Path,
                     default=EXPORT_DEFAULTS.tflite_calibration_data,
                     help='Image, image-list .txt, or directory of ReID crops for TFLite static calibration'),
        click.option('--tflite-calibration-samples',
                     type=int,
                     default=EXPORT_DEFAULTS.tflite_calibration_samples,
                     show_default=True,
                     help='Maximum number of calibration images for TFLite static export'),
        click.option('--tflite-calibration-preprocess',
                     type=click.Choice(sorted(PREPROCESS_REGISTRY.keys()), case_sensitive=False),
                     default=EXPORT_DEFAULTS.tflite_calibration_preprocess,
                     show_default=True,
                     help='Crop preprocessing for TFLite static calibration images'),
        click.option('--tflite-calibration-seed',
                     type=int,
                     default=EXPORT_DEFAULTS.tflite_calibration_seed,
                     show_default=True,
                     help='Seed for nested directory sampling in TFLite static calibration'),
        click.option('--tflite-calibration-update',
                     type=click.Choice(['minmax', 'moving_average'], case_sensitive=False),
                     default=EXPORT_DEFAULTS.tflite_calibration_update,
                     show_default=True,
                     help='Activation range update rule for TFLite static calibration'),
        click.option('--tflite-static-activation-bits',
                     type=int,
                     callback=_parse_tflite_static_activation_bits,
                     default=EXPORT_DEFAULTS.tflite_static_activation_bits,
                     show_default=True,
                     help='Activation precision for TFLite static quantization; weights remain int8'),
        click.option('--include', multiple=True, default=EXPORT_DEFAULTS.include,
                     help='Export formats to include. Options: torchscript, onnx, openvino, engine, coreml, tflite'),
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
            formatter.write_text(
                "Where  MODE (required) is one of "
                "[track, eval, tune, research, generate, train-reid, eval-reid, compare-reid, export, build]"
            )
            formatter.write_text("       --detector selects a YOLO model like yolov8n, yolov9c, yolo11m, yolox_x")
            formatter.write_text("       --reid selects a ReID model like osnet_x0_25_msmt17, mobilenetv2_x1_4")
            formatter.write_text(f"       --tracker selects one of [{_TRACKER_HELP}]")
            formatter.write_text(
                "       OPTIONS (optional) flags like '--source 0' for tracking inputs or "
                "'--dataset mot17 --split ablation' for model-free evaluation."
            )
            formatter.write_text(
                "       --experiment selects a config that fixes its dataset, detector, and ReID profiles."
            )
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
                formatter.write_text("boxmot eval --dataset mot17 --split ablation --tracker boosttrack")
            formatter.write_paragraph()

            formatter.write_text("4. Tune tracker hyperparameters:")
            with formatter.indentation():
                formatter.write_text(
                    "boxmot tune --experiment mot17-ablation-yolox-lmbn "
                    "--tracker deepocsort --n-trials 10"
                )
            formatter.write_paragraph()

            formatter.write_text("5. Research tracker code changes:")
            with formatter.indentation():
                formatter.write_text(
                    "boxmot research --experiment mot17-ablation-yolox-lmbn --tracker bytetrack "
                    "--proposal-model openai/gpt-5.4 --max-metric-calls 24"
                )
            formatter.write_paragraph()

            formatter.write_text("6. Train a ReID model:")
            with formatter.indentation():
                formatter.write_text("boxmot train-reid --model osnet_x0_25 --dataset market1501 --data-dir /path/to/data --epochs 120 --device 0")
            formatter.write_paragraph()

            formatter.write_text("7. Train on all person datasets jointly:")
            with formatter.indentation():
                formatter.write_text("boxmot train-reid --model csl_tinyvit_11m --dataset market1501,duke,cuhk03,msmt17 --data-dir /path/to/data --device 0")
            formatter.write_paragraph()

            formatter.write_text("8. Export ReID model:")
            with formatter.indentation():
                formatter.write_text("boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine --dynamic")
        formatter.write_paragraph()

        # Available modes
        formatter.write_text("Modes:")
        with formatter.indentation():
            formatter.write_text("track        Track objects in video/webcam stream")
            formatter.write_text("eval         Evaluate tracker performance on MOT dataset")
            formatter.write_text("tune         Optimize tracker hyperparameters")
            formatter.write_text("research     Evolve tracker code against benchmark metrics")
            formatter.write_text("generate     Generate detections and embeddings")
            formatter.write_text("train-reid   Train a ReID model on a person/vehicle dataset")
            formatter.write_text("eval-reid    Evaluate a trained ReID model on query/gallery data")
            formatter.write_text("compare-reid Compare ReID checkpoints across target datasets")
            formatter.write_text("export       Export ReID models to different formats")
            formatter.write_text("build        Build native tracker extensions")
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
@dataset_option
@source_option(default=TRACK_DEFAULTS.source, help_text='file/dir/URL/glob, 0 for webcam')
@split_option
@tracker_backend_option
@association_function_option
@core_options
@singular_model_options
@click.pass_context
def track(ctx, dataset, detector, reid, classes, split, **kwargs):
    source = kwargs.pop('source')
    if dataset and _is_option_explicit(ctx, "source"):
        raise click.UsageError(
            "track accepts either --dataset <dataset-id-or-yaml> or "
            "--source <input>, not both."
        )

    src, bench, auto_split = (
        (None, "", "")
        if dataset
        else _resolve_source_context(source)
    )
    _dispatch_cli_workflow(
        ctx,
        "track",
        "boxmot.engine.tracking.workflow",
        _apply_track_cli_defaults(ctx, {
            **kwargs,
            "detector": detector,
            "reid": reid,
            "classes": classes,
            "dataset": dataset,
            "source": src,
            "benchmark": bench,
            "split": split if split else auto_split,
            "workflow_mode": "track",
        }),
    )

@boxmot.command(help='Generate detections and embeddings')
@experiment_option
@source_option(
    default=BOXMOT_DEFAULTS.generate.source,
    help_text='direct dataset root to generate dets/embs for without an experiment config',
)
@split_option
@detection_source_option
@core_options
@plural_model_options
@click.pass_context
def generate(ctx, experiment, detector, reid, classes, split, detection_source, **kwargs):
    src = kwargs.pop('source')
    _require_generate_input(experiment, src, "generate")
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
            "experiment": experiment,
            "source": src,
            "benchmark": bench,
            "split": split if split else auto_split,
            "detection_source": detection_source,
        },
    )


@boxmot.command(help='Evaluate tracking performance')
@experiment_option
@dataset_option
@split_option
@detection_source_option
@replay_backend_option
@tracker_backend_option
@association_function_option
@core_options
@plural_model_options
@click.option('--tune-kf/--no-tune-kf', 'tune_kf', default=False,
              help='Run KF noise tuning (Q/R estimation) before tracking. '
              'Automatically selects parameterization based on the tracker. '
              'Requires cached dets and GT.')
@click.option(
    '--compare-trackeval/--no-compare-trackeval',
    default=False,
    help='Compare BoxMOT metrics against TrackEval for an AABB MOTChallenge benchmark.',
)
@click.pass_context
def eval(ctx, experiment, dataset, detector, reid, classes, split, detection_source, tune_kf, compare_trackeval, **kwargs):
    experiment, dataset = _require_eval_input(experiment, dataset, "eval")
    _dispatch_cli_workflow(
        ctx,
        "eval",
        "boxmot.engine.eval.evaluator",
        {
            **kwargs,
            "detector": list(detector),
            "reid": list(reid),
            "classes": classes,
            "experiment": experiment,
            "dataset": dataset,
            "source": None,
            "benchmark": "",
            "split": split or "",
            "detection_source": detection_source,
            "tune_kf": tune_kf,
            "compare_trackeval": compare_trackeval,
        },
    )


@boxmot.command(help='Tune models via evolutionary algorithms')
@experiment_option
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
def tune(ctx, experiment, detector, reid, classes, split, detection_source, tune_kf, **kwargs):
    experiment = _require_experiment_input(experiment, "tune")
    _dispatch_cli_workflow(
        ctx,
        "tune",
        "boxmot.engine.tuning.tuner",
        {
            **kwargs,
            "detector": list(detector),
            "reid": list(reid),
            "classes": classes,
            "experiment": experiment,
            "source": None,
            "benchmark": "",
            "split": split or "",
            "detection_source": detection_source,
            "tune_kf": tune_kf,
        },
    )


@boxmot.command(help='Research tracker code changes with GEPA')
@experiment_option
@split_option
@detection_source_option
@replay_backend_option
@tracker_backend_option
@core_options
@research_options
@plural_model_options
@click.pass_context
def research(ctx, experiment, detector, reid, classes, split, detection_source, **kwargs):
    experiment = _require_experiment_input(experiment, "research")
    _dispatch_cli_workflow(
        ctx,
        "research",
        "boxmot.engine.research",
        {
            **kwargs,
            "detector": list(detector),
            "reid": list(reid),
            "classes": classes,
            "experiment": experiment,
            "source": None,
            "benchmark": "",
            "split": split or "",
            "detection_source": detection_source,
        },
    )


def train_options(func):
    """Decorator adding ReID training options."""
    from boxmot.reid.backbones import registered_backbone_names
    from boxmot.reid.core.preprocessing import PREPROCESS_REGISTRY
    from boxmot.reid.datasets import DATASET_REGISTRY

    model_types = registered_backbone_names()

    options = [
        click.option('--cfg', type=click.Path(exists=True, dir_okay=False), default=None,
                     help='BoxMOT ReID YAML/JSON config or saved hparams.json. '
                          'Explicit CLI flags override config values.'),
        click.option('--recipe', type=click.Choice(list_training_recipes(), case_sensitive=False),
                     default=None,
                     help='Training recipe preset (overrides defaults; CLI flags still take priority). '
                          f'Available: {", ".join(list_training_recipes()) or "(none)"}'),
        click.option('--model', type=click.Choice(model_types, case_sensitive=False),
                     default=TRAIN_DEFAULTS.model, show_default=True,
                     help='ReID backbone architecture'),
        click.option('--data', type=str, multiple=True, default=(),
                     help='ReID dataset name or YAML data config. Repeat or comma-separate for multi-dataset '
                          'training, e.g. --data market1501.yaml --data duke.yaml. YAML supports '
                          'dataset/name, path, train, val, query, gallery, and download.'),
        click.option('--dataset', type=str,
                     default=TRAIN_DEFAULTS.dataset, show_default=True,
                     help='Training dataset (comma-separated for joint training, '
                          f'e.g. market1501,duke,cuhk03,msmt17). '
                          f'Available: {", ".join(sorted(DATASET_REGISTRY.keys()))}'),
        click.option('--data-dir', type=click.Path(exists=True), required=False, default=None,
                     help='Root directory of the dataset (inferred from hparams.json when --resume is used)'),
        click.option('--loss', type=click.Choice(['softmax', 'triplet', 'wrt', 'circle', 'ms'], case_sensitive=False),
                     default=TRAIN_DEFAULTS.loss, show_default=True,
                     help='Metric loss type (triplet=batch-hard triplet, wrt=weighted regularized triplet, '
                          'circle=Circle loss, ms=multi-similarity, softmax=classifier only)'),
        click.option('--classifier-loss', type=click.Choice(['ce', 'arcface', 'cosface'], case_sensitive=False),
                     default=TRAIN_DEFAULTS.classifier_loss, show_default=True,
                     help='ID classifier loss: ce, arcface, or cosface'),
        click.option('--preprocess', type=click.Choice(sorted(PREPROCESS_REGISTRY.keys()), case_sensitive=False),
                     default=TRAIN_DEFAULTS.preprocess, show_default=True,
                     help='Crop preprocessing method; must match inference-time preprocessing'),
        click.option('--imgsz', callback=parse_imgsz, type=str,
                     default=_click_imgsz_default(TRAIN_DEFAULTS.imgsz),
                     help='Image size as H,W (e.g. 256,128)'),
        click.option('--batch-size', type=int, default=TRAIN_DEFAULTS.batch_size, show_default=True,
                     help='Evaluation batch size; training uses --p-ids × --k-instances'),
        click.option('--lr', type=float, default=TRAIN_DEFAULTS.lr, show_default=True,
                     help='Base learning rate'),
        click.option('--weight-decay', type=float, default=TRAIN_DEFAULTS.weight_decay, show_default=True,
                     help='Weight decay'),
        click.option('--epochs', type=int, default=TRAIN_DEFAULTS.epochs, show_default=True,
                     help='Number of training epochs'),
        click.option('--warmup-epochs', type=int, default=TRAIN_DEFAULTS.warmup_epochs, show_default=True,
                     help='Linear warmup epochs'),
        click.option('--vit-lr-profile',
                     type=click.Choice(['layer_decay', 'reid_lrd'], case_sensitive=False),
                     default=TRAIN_DEFAULTS.vit_lr_profile, show_default=True,
                     help='Transformer LR grouping profile: geometric layer decay or ReID stage-wise decay'),
        click.option('--layer-decay', type=float,
                     default=TRAIN_DEFAULTS.layer_decay, show_default=True,
                     help='Geometric per-stage LR decay for hierarchical transformer backbones'),
        click.option('--backbone-lr-mult', type=float,
                     default=TRAIN_DEFAULTS.backbone_lr_mult, show_default=True,
                     help='Persistent pretrained-backbone LR multiplier for MobileNetV4'),
        click.option('--backbone-freeze-epochs', type=int,
                     default=TRAIN_DEFAULTS.backbone_freeze_epochs, show_default=True,
                     help='Freeze pretrained backbone layers for the first N epochs'),
        click.option('--gradual-unfreeze/--no-gradual-unfreeze',
                     default=TRAIN_DEFAULTS.gradual_unfreeze, show_default=True,
                     help='Use staged ReID unfreeze: head/neck, last backbone stage, then full model'),
        click.option('--gradual-unfreeze-head-epochs', type=int,
                     default=TRAIN_DEFAULTS.gradual_unfreeze_head_epochs, show_default=True,
                     help='Gradual unfreeze head/neck-only epoch boundary'),
        click.option('--gradual-unfreeze-stage-epochs', type=int,
                     default=TRAIN_DEFAULTS.gradual_unfreeze_stage_epochs, show_default=True,
                     help='Gradual unfreeze last-stage epoch boundary before full model training'),
        click.option('--gradual-unfreeze-backbone-lr-mult', type=float,
                     default=TRAIN_DEFAULTS.gradual_unfreeze_backbone_lr_mult, show_default=True,
                     help='Backbone LR multiplier for early full-model gradual-unfreeze epochs'),
        click.option('--gradual-unfreeze-backbone-lr-epochs', type=int,
                     default=TRAIN_DEFAULTS.gradual_unfreeze_backbone_lr_epochs, show_default=True,
                     help='Number of full-model epochs using the gradual-unfreeze backbone LR multiplier'),
        click.option('--eval-interval', type=int, default=TRAIN_DEFAULTS.eval_interval, show_default=True,
                     help='Validate every N epochs'),
        click.option('--p-ids', type=int, default=TRAIN_DEFAULTS.p_ids, show_default=True,
                     help='Number of identities per PK batch'),
        click.option('--k-instances', type=int, default=TRAIN_DEFAULTS.k_instances, show_default=True,
                     help='Number of instances per identity'),
        click.option('--source-balance', type=str, default=TRAIN_DEFAULTS.source_balance, show_default=True,
                     help='Source-balanced PK sampler spec, e.g. '
                          "'market1501:8,4;mot17_1501:8,4'. "
                          'Empty uses the global --p-ids x --k-instances sampler.'),
        click.option('--pk-steps-per-epoch', type=int,
                     default=TRAIN_DEFAULTS.pk_steps_per_epoch, show_default=True,
                     help='Fixed PK batches per epoch; zero uses one shuffled identity pass'),
        click.option('--camera-aware-sampler/--no-camera-aware-sampler',
                     default=TRAIN_DEFAULTS.camera_aware_sampler, show_default=True,
                     help='Prefer distinct-camera instances within each identity; cameras affect sampling only'),
        click.option('--margin', type=float, default=TRAIN_DEFAULTS.margin, show_default=True,
                     help='Triplet loss margin'),
        click.option('--triplet-soft-margin/--triplet-hard-margin',
                     default=TRAIN_DEFAULTS.triplet_soft_margin,
                     help='Use softplus batch-hard triplet instead of the hard margin. '
                          'Default: auto for transformer-family recipes, hard margin otherwise.'),
        click.option('--arcface-scale', type=float, default=TRAIN_DEFAULTS.arcface_scale, show_default=True,
                     help='ArcFace logit scale'),
        click.option('--arcface-margin', type=float, default=TRAIN_DEFAULTS.arcface_margin, show_default=True,
                     help='ArcFace angular margin'),
        click.option('--cosface-scale', type=float, default=TRAIN_DEFAULTS.cosface_scale, show_default=True,
                     help='CosFace logit scale'),
        click.option('--cosface-margin', type=float, default=TRAIN_DEFAULTS.cosface_margin, show_default=True,
                     help='CosFace cosine margin'),
        click.option('--label-smooth', type=float, default=TRAIN_DEFAULTS.label_smooth, show_default=True,
                     help='Label smoothing epsilon'),
        click.option('--center-loss-weight', type=float, default=TRAIN_DEFAULTS.center_loss_weight, show_default=True,
                     help='Center loss weight'),
        click.option('--id-loss-weight', type=float, default=TRAIN_DEFAULTS.id_loss_weight, show_default=True,
                     help='Weight applied to the ID classification loss term'),
        click.option('--metric-loss-weight', type=float, default=TRAIN_DEFAULTS.metric_loss_weight, show_default=True,
                     help='Weight applied to the metric loss term (triplet/wrt/circle/ms)'),
        click.option('--adasp-loss-weight', type=float, default=TRAIN_DEFAULTS.adasp_loss_weight, show_default=True,
                     help='Weight for AdaSP on the full normalized descriptor; 0 disables'),
        click.option('--adasp-temperature', type=float, default=TRAIN_DEFAULTS.adasp_temperature, show_default=True,
                     help='AdaSP similarity temperature'),
        click.option('--adasp-scale', type=float, default=TRAIN_DEFAULTS.adasp_scale, show_default=True,
                     help='AdaSP paper-scale multiplier applied before its ablation weight'),
        click.option('--coarse-branch-ce-weight', type=float,
                     default=TRAIN_DEFAULTS.coarse_branch_ce_weight, show_default=True,
                     help='Relative CE weight for two-stripe branches; 0 disables coarse CE'),
        click.option('--fine-branch-ce-weight', type=float,
                     default=TRAIN_DEFAULTS.fine_branch_ce_weight, show_default=True,
                     help='Relative CE weight for four-stripe branches; 0 disables fine CE'),
        click.option('--part-relation-weight', type=float,
                     default=TRAIN_DEFAULTS.part_relation_weight, show_default=True,
                     help='EMA cross-ID neighborhood loss weight for corresponding fine parts'),
        click.option('--part-to-global-weight', type=float,
                     default=TRAIN_DEFAULTS.part_to_global_weight, show_default=True,
                     help='Weight for distilling aggregate part neighborhoods into global features'),
        click.option('--part-relation-teacher-momentum', type=float,
                     default=TRAIN_DEFAULTS.part_relation_teacher_momentum, show_default=True,
                     help='EMA momentum for the training-only part-relation teacher'),
        click.option('--part-relation-temperature', type=float,
                     default=TRAIN_DEFAULTS.part_relation_temperature, show_default=True,
                     help='Temperature for cross-ID part-neighborhood distillation'),
        click.option('--compact-metric-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.compact_metric_loss_weight, show_default=True,
                     help='Triplet-loss weight for an enabled compact deployment descriptor'),
        click.option('--compact-cosine-distill-weight', type=float,
                     default=TRAIN_DEFAULTS.compact_cosine_distill_weight, show_default=True,
                     help='Cosine alignment weight from compact student to the full teacher descriptor'),
        click.option('--compact-pairwise-distill-weight', type=float,
                     default=TRAIN_DEFAULTS.compact_pairwise_distill_weight, show_default=True,
                     help='PK-batch pairwise-distance distillation weight for the compact student'),
        click.option('--csmm-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.csmm_loss_weight, show_default=True,
                     help='Cross-scale majority-margin auxiliary loss weight; 0 disables'),
        click.option('--csmm-margin', type=float,
                     default=TRAIN_DEFAULTS.csmm_margin, show_default=True,
                     help='Target cosine ranking margin for the median descriptor scale'),
        click.option('--csmm-temperature', type=float,
                     default=TRAIN_DEFAULTS.csmm_temperature, show_default=True,
                     help='Softplus temperature for the cross-scale majority-margin loss'),
        click.option('--csmm-topk-negatives', type=int,
                     default=TRAIN_DEFAULTS.csmm_topk_negatives, show_default=True,
                     help='Closest full-descriptor negatives evaluated per CSMM anchor'),
        click.option('--csmm-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.csmm_start_epoch, show_default=True,
                     help='Epoch through which the CSMM auxiliary weight remains zero'),
        click.option('--csmm-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.csmm_ramp_end_epoch, show_default=True,
                     help='Epoch where CSMM reaches --csmm-loss-weight'),
        click.option('--treeboost-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.treeboost_loss_weight, show_default=True,
                     help='TreeBoost-AP hierarchical retrieval auxiliary loss weight; 0 disables'),
        click.option('--treeboost-coarse-coefficient', type=float,
                     default=TRAIN_DEFAULTS.treeboost_coarse_coefficient, show_default=True,
                     help='Coefficient for coarse residual ranking supervision inside TreeBoost-AP'),
        click.option('--treeboost-fine-coefficient', type=float,
                     default=TRAIN_DEFAULTS.treeboost_fine_coefficient, show_default=True,
                     help='Coefficient for fine residual ranking supervision inside TreeBoost-AP'),
        click.option('--treeboost-node-coefficient', type=float,
                     default=TRAIN_DEFAULTS.treeboost_node_coefficient, show_default=True,
                     help='Coefficient for upper/lower parent-child refinement terms'),
        click.option('--treeboost-regression-coefficient', type=float,
                     default=TRAIN_DEFAULTS.treeboost_regression_coefficient, show_default=True,
                     help='Coefficient penalizing ranking regressions at finer hierarchy levels'),
        click.option('--treeboost-difficulty-floor', type=float,
                     default=TRAIN_DEFAULTS.treeboost_difficulty_floor, show_default=True,
                     help='Minimum supervision retained for hierarchy levels after easy parent rankings'),
        click.option('--treeboost-regression-tolerance', type=float,
                     default=TRAIN_DEFAULTS.treeboost_regression_tolerance, show_default=True,
                     help='Allowed SmoothAP loss increase when adding a finer hierarchy level'),
        click.option('--treeboost-temperature', type=float,
                     default=TRAIN_DEFAULTS.treeboost_temperature, show_default=True,
                     help='Pairwise sigmoid temperature for cross-camera-positive TreeBoost SmoothAP'),
        click.option('--treeboost-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.treeboost_start_epoch, show_default=True,
                     help='Epoch through which the TreeBoost-AP auxiliary weight remains zero'),
        click.option('--treeboost-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.treeboost_ramp_end_epoch, show_default=True,
                     help='Epoch where TreeBoost-AP reaches --treeboost-loss-weight'),
        click.option('--global-ap-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.global_ap_loss_weight, show_default=True,
                     help='Identity-defined dataset-memory SmoothAP weight on the deployed descriptor; 0 disables'),
        click.option('--global-ap-temperature', type=float,
                     default=TRAIN_DEFAULTS.global_ap_temperature, show_default=True,
                     help='Pairwise rank-relaxation temperature for GlobalAP'),
        click.option('--global-ap-topk', type=int,
                     default=TRAIN_DEFAULTS.global_ap_topk, show_default=True,
                     help='Different-identity hard negatives per GlobalAP query; '
                          'all non-self same-identity positives remain'),
        click.option('--global-ap-memory-size', type=int,
                     default=TRAIN_DEFAULTS.global_ap_memory_size, show_default=True,
                     help='Stable sample-index capacity of the GlobalAP memory'),
        click.option('--global-ap-momentum', type=float,
                     default=TRAIN_DEFAULTS.global_ap_momentum, show_default=True,
                     help='Descriptor momentum for repeated GlobalAP memory rows'),
        click.option('--global-ap-max-age', type=int,
                     default=TRAIN_DEFAULTS.global_ap_max_age, show_default=True,
                     help='Maximum memory age in optimizer steps; 0 keeps all populated rows'),
        click.option('--global-ap-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.global_ap_start_epoch, show_default=True,
                     help='Last epoch with GlobalAP disabled'),
        click.option('--global-ap-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.global_ap_ramp_end_epoch, show_default=True,
                     help='Epoch where GlobalAP reaches full weight'),
        click.option('--global-ap-decay-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.global_ap_decay_start_epoch, show_default=True,
                     help='Last epoch with GlobalAP at full weight'),
        click.option('--global-ap-decay-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.global_ap_decay_end_epoch, show_default=True,
                     help='Epoch where GlobalAP returns to zero'),
        click.option('--hpgrd-cache-dir', type=click.Path(exists=True), default=TRAIN_DEFAULTS.hpgrd_cache_dir,
                     help='Offline human-privileged teacher cache used only during training'),
        click.option('--hpgrd-global-weight', type=float,
                     default=TRAIN_DEFAULTS.hpgrd_global_weight, show_default=True,
                     help='External-teacher identity-relational distillation weight'),
        click.option('--hpgrd-part-weight', type=float,
                     default=TRAIN_DEFAULTS.hpgrd_part_weight, show_default=True,
                     help='Visibility-aware fixed-mask part relational distillation weight'),
        click.option('--hpgrd-background-weight', type=float,
                     default=TRAIN_DEFAULTS.hpgrd_background_weight, show_default=True,
                     help='Background-intervention descriptor consistency weight'),
        click.option('--hpgrd-part-drop-weight', type=float,
                     default=TRAIN_DEFAULTS.hpgrd_part_drop_weight, show_default=True,
                     help='Semantic part leave-out teacher consistency weight'),
        click.option('--hpgrd-part-drop-probability', type=float,
                     default=TRAIN_DEFAULTS.hpgrd_part_drop_probability, show_default=True,
                     help='Probability of masking one visible semantic part in a student view'),
        click.option('--hpgrd-gradient-fraction', type=float,
                     default=TRAIN_DEFAULTS.hpgrd_gradient_fraction, show_default=True,
                     help='Maximum shared-feature HP-GRD gradient norm as a fraction of the base objective'),
        click.option('--hpgrd-min-confidence', type=float,
                     default=TRAIN_DEFAULTS.hpgrd_min_confidence, show_default=True,
                     help='Minimum fused pose/parser teacher confidence'),
        click.option('--early-id-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.early_id_loss_weight, show_default=True,
                     help='Temporary ID loss weight for the first --early-id-loss-epochs epochs; 0 disables'),
        click.option('--early-id-loss-epochs', type=int,
                     default=TRAIN_DEFAULTS.early_id_loss_epochs, show_default=True,
                     help='Number of initial epochs using --early-id-loss-weight'),
        click.option('--center-loss-ramp-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.center_loss_ramp_start_epoch, show_default=True,
                     help='Epoch through which center loss weight stays at 0; 0 disables unless end is set'),
        click.option('--center-loss-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.center_loss_ramp_end_epoch, show_default=True,
                     help='Epoch where center loss reaches --center-loss-weight; 0 disables ramping'),
        click.option('--aux-ce-weight', type=float, default=TRAIN_DEFAULTS.aux_ce_weight, show_default=True,
                     help='Relative CE weight for auxiliary branch classifiers; 1.0 preserves equal branch averaging'),
        click.option('--aux-ce-drop-epoch', type=int, default=TRAIN_DEFAULTS.aux_ce_drop_epoch, show_default=True,
                     help='Set auxiliary CE weight to 0 after this epoch; 0 keeps it active for all epochs'),
        click.option('--branch-loss-agg', type=click.Choice(['mean', 'sum'], case_sensitive=False),
                     default=TRAIN_DEFAULTS.branch_loss_agg, show_default=True,
                     help='How to aggregate multi-branch losses before weighting'),
        click.option('--scale-balanced-branches/--no-scale-balanced-branches',
                     default=TRAIN_DEFAULTS.scale_balanced_branches, show_default=True,
                     help='Give global, two-stripe, and four-stripe scales equal CE and descriptor weight'),
        click.option('--multilevel-suppression/--no-multilevel-suppression',
                     default=TRAIN_DEFAULTS.multilevel_suppression, show_default=True,
                     help='Train coarse and fine classifiers on evidence suppressed by the preceding scale'),
        click.option('--multilevel-suppression-ratio', type=float,
                     default=TRAIN_DEFAULTS.multilevel_suppression_ratio, show_default=True,
                     help='Maximum top-saliency spatial fraction suppressed in the auxiliary path'),
        click.option('--multilevel-suppression-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.multilevel_suppression_loss_weight, show_default=True,
                     help='Peak weight of scale-balanced multilevel suppression CE'),
        click.option('--multilevel-suppression-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.multilevel_suppression_start_epoch, show_default=True,
                     help='Epoch through which multilevel suppression remains disabled'),
        click.option('--multilevel-suppression-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.multilevel_suppression_ramp_end_epoch, show_default=True,
                     help='Epoch where suppression ratio and auxiliary CE reach full strength'),
        click.option('--multilevel-suppression-decay-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.multilevel_suppression_decay_start_epoch, show_default=True,
                     help='Last epoch at full multilevel suppression strength'),
        click.option('--multilevel-suppression-decay-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.multilevel_suppression_decay_end_epoch, show_default=True,
                     help='Epoch where multilevel suppression and auxiliary CE return to zero'),
        click.option('--hierarchical-branch-attention/--no-hierarchical-branch-attention',
                     default=TRAIN_DEFAULTS.hierarchical_branch_attention, show_default=True,
                     help='Refine the 1-to-2-to-4 branch descriptors with tree-masked token attention'),
        click.option('--branch-attention-token-dim', type=int,
                     default=TRAIN_DEFAULTS.branch_attention_token_dim, show_default=True,
                     help='Token width used by hierarchical branch attention'),
        click.option('--branch-attention-num-heads', type=int,
                     default=TRAIN_DEFAULTS.branch_attention_num_heads, show_default=True,
                     help='Attention heads used by hierarchical branch attention'),
        click.option('--branch-attention-num-layers', type=int,
                     default=TRAIN_DEFAULTS.branch_attention_num_layers, show_default=True,
                     help='Transformer layers used by hierarchical branch attention'),
        click.option('--branch-attention-mlp-ratio', type=float,
                     default=TRAIN_DEFAULTS.branch_attention_mlp_ratio, show_default=True,
                     help='Transformer MLP expansion ratio for hierarchical branch attention'),
        click.option('--branch-attention-dropout', type=float,
                     default=TRAIN_DEFAULTS.branch_attention_dropout, show_default=True,
                     help='Dropout used by hierarchical branch attention'),
        click.option('--branch-set-attention/--no-branch-set-attention',
                     default=TRAIN_DEFAULTS.branch_set_attention, show_default=True,
                     help='Refine all seven pooled 512-D branches with shared unmasked attention'),
        click.option('--branch-set-attention-token-dim', type=int,
                     default=TRAIN_DEFAULTS.branch_set_attention_token_dim, show_default=True,
                     help='Shared token width used by branch-set attention'),
        click.option('--branch-set-attention-num-heads', type=int,
                     default=TRAIN_DEFAULTS.branch_set_attention_num_heads, show_default=True,
                     help='Attention heads used by branch-set attention'),
        click.option('--branch-set-attention-num-layers', type=int,
                     default=TRAIN_DEFAULTS.branch_set_attention_num_layers, show_default=True,
                     help='Transformer layers used by branch-set attention'),
        click.option('--branch-set-attention-mlp-ratio', type=float,
                     default=TRAIN_DEFAULTS.branch_set_attention_mlp_ratio, show_default=True,
                     help='Transformer MLP expansion ratio for branch-set attention'),
        click.option('--branch-set-attention-dropout', type=float,
                     default=TRAIN_DEFAULTS.branch_set_attention_dropout, show_default=True,
                     help='Dropout used by branch-set attention'),
        click.option('--multiscale-query-decoder/--no-multiscale-query-decoder',
                     default=TRAIN_DEFAULTS.multiscale_query_decoder, show_default=True,
                     help='Decode seven pooled queries against final, Stage-2, and Stage-0 spatial maps'),
        click.option('--query-decoder-dim', type=int,
                     default=TRAIN_DEFAULTS.query_decoder_dim, show_default=True,
                     help='Shared query and spatial-memory token width'),
        click.option('--query-decoder-num-heads', type=int,
                     default=TRAIN_DEFAULTS.query_decoder_num_heads, show_default=True,
                     help='Self- and cross-attention head count for the query decoder'),
        click.option('--query-decoder-num-layers', type=int,
                     default=TRAIN_DEFAULTS.query_decoder_num_layers, show_default=True,
                     help='Number of residual multi-scale query decoder layers'),
        click.option('--query-decoder-mlp-ratio', type=float,
                     default=TRAIN_DEFAULTS.query_decoder_mlp_ratio, show_default=True,
                     help='Query-decoder FFN expansion ratio'),
        click.option('--query-decoder-dropout', type=float,
                     default=TRAIN_DEFAULTS.query_decoder_dropout, show_default=True,
                     help='Dropout used by the query decoder'),
        click.option('--hierarchical-late-interaction/--no-hierarchical-late-interaction',
                     default=TRAIN_DEFAULTS.hierarchical_late_interaction, show_default=True,
                     help='Train the pair-conditioned hierarchical matcher and top-k reranker'),
        click.option('--late-interaction-dim', type=int,
                     default=TRAIN_DEFAULTS.late_interaction_dim, show_default=True,
                     help='Shared branch-token width for hierarchical late interaction'),
        click.option('--late-interaction-num-heads', type=int,
                     default=TRAIN_DEFAULTS.late_interaction_num_heads, show_default=True,
                     help='Cross-attention head count for hierarchical late interaction'),
        click.option('--late-interaction-num-layers', type=int,
                     default=TRAIN_DEFAULTS.late_interaction_num_layers, show_default=True,
                     help='Pair-conditioned cross-attention layer count'),
        click.option('--late-interaction-sinkhorn-iters', type=int,
                     default=TRAIN_DEFAULTS.late_interaction_sinkhorn_iters, show_default=True,
                     help='Sinkhorn normalization iterations for pair alignment'),
        click.option('--late-interaction-null-tokens', type=int,
                     default=TRAIN_DEFAULTS.late_interaction_null_tokens, show_default=True,
                     help='Learned null evidence tokens per image'),
        click.option('--late-interaction-negative-identities', type=int,
                     default=TRAIN_DEFAULTS.late_interaction_negative_identities, show_default=True,
                     help='Detached-base hard negative identities per anchor'),
        click.option('--late-interaction-rerank-topk', type=int,
                     default=TRAIN_DEFAULTS.late_interaction_rerank_topk, show_default=True,
                     help='Base-cosine candidates reranked by late interaction'),
        click.option('--late-interaction-base-score-init', type=float,
                     default=TRAIN_DEFAULTS.late_interaction_base_score_init, show_default=True,
                     help='Initial contribution of the proven base descriptor score'),
        click.option('--late-interaction-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.late_interaction_loss_weight, show_default=True,
                     help='Full-weight multi-positive matcher loss coefficient'),
        click.option('--late-interaction-distill-weight', type=float,
                     default=TRAIN_DEFAULTS.late_interaction_distill_weight, show_default=True,
                     help='Full-weight matcher-to-base ranking distillation coefficient'),
        click.option('--late-interaction-temperature', type=float,
                     default=TRAIN_DEFAULTS.late_interaction_temperature, show_default=True,
                     help='Listwise matcher and distillation temperature'),
        click.option('--late-interaction-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.late_interaction_start_epoch, show_default=True,
                     help='Epoch through which late-interaction auxiliary weights remain zero'),
        click.option('--late-interaction-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.late_interaction_ramp_end_epoch, show_default=True,
                     help='Epoch where matcher and distillation reach full weight'),
        click.option('--mcpt-mode',
                     type=click.Choice(
                         (
                             'none',
                             'dataset_boundaries',
                             'per_image_stage2',
                             'shared_multiscale',
                             'foreground_aware_shared_multiscale',
                         ),
                         case_sensitive=False,
                     ),
                     default=TRAIN_DEFAULTS.mcpt_mode, show_default=True,
                     help='Monotonic canonical part transport treatment'),
        click.option('--mcpt-hidden-dim', type=int,
                     default=TRAIN_DEFAULTS.mcpt_hidden_dim, show_default=True,
                     help='Hidden row-predictor width for RGB-conditioned MCPT'),
        click.option('--mcpt-max-displacement', type=float,
                     default=TRAIN_DEFAULTS.mcpt_max_displacement, show_default=True,
                     help='Maximum normalized vertical displacement'),
        click.option('--mcpt-smoothness-weight', type=float,
                     default=TRAIN_DEFAULTS.mcpt_smoothness_weight, show_default=True,
                     help='Second-difference MCPT regularization weight'),
        click.option('--mcpt-identity-weight', type=float,
                     default=TRAIN_DEFAULTS.mcpt_identity_weight, show_default=True,
                     help='Initial identity-warp regularization weight'),
        click.option('--mcpt-identity-decay-epoch', type=int,
                     default=TRAIN_DEFAULTS.mcpt_identity_decay_epoch, show_default=True,
                     help='Epoch where MCPT identity regularization reaches zero'),
        click.option('--mcpt-lr-multiplier', type=float,
                     default=TRAIN_DEFAULTS.mcpt_lr_multiplier, show_default=True,
                     help='MCPT learning-rate multiplier relative to the head'),
        click.option('--mcpt-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.mcpt_start_epoch, show_default=True,
                     help='Last epoch with transport forced exactly off'),
        click.option('--mcpt-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.mcpt_ramp_end_epoch, show_default=True,
                     help='Epoch where the MCPT gate schedule reaches full scale'),
        click.option('--mcpt-disabled-eval/--no-mcpt-disabled-eval',
                     default=TRAIN_DEFAULTS.mcpt_disabled_eval, show_default=True,
                     help='Also validate with MCPT forcibly disabled'),
        click.option('--jpm/--no-jpm',
                     default=TRAIN_DEFAULTS.jpm, show_default=True,
                     help='Enable training-only TransReID Jigsaw Patch Module'),
        click.option('--jpm-num-groups', type=int,
                     default=TRAIN_DEFAULTS.jpm_num_groups, show_default=True,
                     help='Number of JPM shuffled patch groups'),
        click.option('--jpm-shift', type=int,
                     default=TRAIN_DEFAULTS.jpm_shift, show_default=True,
                     help='Patch-token cyclic shift before JPM shuffle'),
        click.option('--jpm-token-dim', type=int,
                     default=TRAIN_DEFAULTS.jpm_token_dim, show_default=True,
                     help='JPM auxiliary transformer bottleneck width'),
        click.option('--jpm-num-heads', type=int,
                     default=TRAIN_DEFAULTS.jpm_num_heads, show_default=True,
                     help='JPM shared transformer attention heads'),
        click.option('--jpm-mlp-ratio', type=float,
                     default=TRAIN_DEFAULTS.jpm_mlp_ratio, show_default=True,
                     help='JPM shared transformer MLP expansion'),
        click.option('--jpm-dropout', type=float,
                     default=TRAIN_DEFAULTS.jpm_dropout, show_default=True,
                     help='JPM shared transformer dropout'),
        click.option('--jpm-id-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.jpm_id_loss_weight, show_default=True,
                     help='Mean JPM local identity-loss coefficient'),
        click.option('--jpm-metric-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.jpm_metric_loss_weight, show_default=True,
                     help='Mean JPM local triplet-loss coefficient'),
        click.option('--metric-feature',
                     type=click.Choice(
                         selector_choices("metric_feature"),
                         case_sensitive=False,
                     ),
                     default=TRAIN_DEFAULTS.metric_feature, show_default=True,
                     help='Feature representation used for metric losses when the model supports multiple branches'),
        click.option('--inference-feature',
                     type=click.Choice(
                         selector_choices("inference_feature"),
                         case_sensitive=False,
                     ),
                     default=TRAIN_DEFAULTS.inference_feature, show_default=True,
                     help='Feature representation emitted by CSL-TinyViT at validation/inference time'),
        click.option('--feature-fusion',
                     type=click.Choice(
                         selector_choices("feature_fusion"),
                         case_sensitive=False,
                     ),
                     default=TRAIN_DEFAULTS.feature_fusion, show_default=True,
                     help='CSL-TinyViT static or per-image dynamic spatial fusion before the ReID head'),
        click.option('--pyramid-resize-mode',
                     type=click.Choice(
                         selector_choices("pyramid_resize_mode"),
                         case_sensitive=False,
                     ),
                     default=TRAIN_DEFAULTS.pyramid_resize_mode, show_default=True,
                     help='Pyramid resizing: bilinear, average-pool down/nearest up, or '
                          'average-pool down/bilinear up'),
        click.option('--spatial-conv-mode',
                     type=click.Choice(
                         selector_choices("spatial_conv_mode"),
                         case_sensitive=False,
                     ),
                     default=TRAIN_DEFAULTS.spatial_conv_mode, show_default=True,
                     help='CSL-TinyViT neck/FPN 3x3 convolution implementation'),
        click.option('--post-fusion-mixer',
                     type=click.Choice(['none', 'dwconv'], case_sensitive=False),
                     default=TRAIN_DEFAULTS.post_fusion_mixer, show_default=True,
                     help='Optional zero-gated local mixer after CSL-TinyViT feature fusion'),
        click.option('--post-fusion-mixer-reduction', type=int,
                     default=TRAIN_DEFAULTS.post_fusion_mixer_reduction, show_default=True,
                     help='Channel reduction ratio for the post-fusion local mixer'),
        click.option('--post-fusion-mixer-kernel', callback=_parse_int_pair, type=str,
                     default=_click_imgsz_default(TRAIN_DEFAULTS.post_fusion_mixer_kernel), show_default=True,
                     help='Post-fusion depthwise mixer kernel as H,W, e.g. 5,3'),
        click.option('--post-fusion-mixer-gamma-init', type=float,
                     default=TRAIN_DEFAULTS.post_fusion_mixer_gamma_init, show_default=True,
                     help='Initial residual scale for the post-fusion local mixer'),
        click.option('--feat-dim', type=int, default=TRAIN_DEFAULTS.feat_dim, show_default=True,
                     help='Per-branch embedding dimension for ReID heads that support projection'),
        click.option('--neck-dim', type=int, default=TRAIN_DEFAULTS.neck_dim, show_default=True,
                     help='Neck channel dimension for ReID backbones that support a feature neck'),
        click.option('--drop-path-rate', type=float, default=TRAIN_DEFAULTS.drop_path_rate, show_default=True,
                     help='Maximum stochastic-depth probability for CSL-TinyViT'),
        click.option('--timm-model-name', type=str,
                     default=TRAIN_DEFAULTS.timm_model_name, show_default=True,
                     help='Optional exact timm pretrained model tag for MobileNetV4'),
        click.option('--timm-head-mode',
                     type=click.Choice(
                         ['pooled', 'spatial', 'spatial_adapt_norm', 'spatial_linear', 'off'],
                         case_sensitive=False,
                     ),
                     default=TRAIN_DEFAULTS.timm_head_mode, show_default=True,
                     help='MobileNetV4 C5 head path, including spatial normalization controls'),
        click.option('--mobilenetv4-last-stride',
                     type=click.IntRange(1, 2),
                     default=TRAIN_DEFAULTS.mobilenetv4_last_stride, show_default=True,
                     help='MobileNetV4 final spatial stride: 1 retains stride-16 C5; 2 keeps ImageNet topology'),
        click.option('--mobilenetv4-neck-mode',
                     type=click.Choice(['cnn', 'spatial_ln'], case_sensitive=False),
                     default=TRAIN_DEFAULTS.mobilenetv4_neck_mode, show_default=True,
                     help='MobileNetV4 ReID neck: CNN projection or TinyViT-matched spatial LayerNorm neck'),
        click.option('--attention-window-layout',
                     type=click.Choice(['legacy', 'rect'], case_sensitive=False),
                     default=TRAIN_DEFAULTS.attention_window_layout, show_default=True,
                     help='CSL-TinyViT attention windows: legacy square windows or ReID rectangular windows'),
        click.option('--attention-bias',
                     type=click.Choice(['absolute', 'signed_factorized'], case_sensitive=False),
                     default=TRAIN_DEFAULTS.attention_bias, show_default=True,
                     help='CSL-TinyViT relative attention bias parameterization'),
        click.option('--interpolate-pretrained-attention-bias/--no-interpolate-pretrained-attention-bias',
                     default=TRAIN_DEFAULTS.interpolate_pretrained_attention_bias, show_default=True,
                     help='Resize official absolute attention-bias tables for non-legacy attention windows'),
        click.option('--attention-mask/--no-attention-mask',
                     default=TRAIN_DEFAULTS.attention_mask, show_default=True,
                     help='Mask padded tokens in CSL-TinyViT window attention'),
        click.option('--attention-shift/--no-attention-shift',
                     default=TRAIN_DEFAULTS.attention_shift, show_default=True,
                     help='Alternate shifted CSL-TinyViT windows in attention stages 1 and 2'),
        click.option('--stage3-global/--no-stage3-global',
                     default=TRAIN_DEFAULTS.stage3_global, show_default=True,
                     help='Use full 24x8 attention in the final CSL-TinyViT block'),
        click.option('--stage3-downsample/--no-stage3-downsample',
                     default=TRAIN_DEFAULTS.stage3_downsample, show_default=True,
                     help='Downsample only the final/global transformer stage while retaining Stage-2 local tokens'),
        click.option('--stage2-width-merge-after', type=int,
                     default=TRAIN_DEFAULTS.stage2_width_merge_after, show_default=True,
                     help='Merge adjacent Stage-2 columns after this many blocks; 0 disables'),
        click.option('--stage2-mlp-ratio', type=float,
                     default=TRAIN_DEFAULTS.stage2_mlp_ratio, show_default=True,
                     help='MLP expansion ratio used only in CSL-TinyViT Stage 2'),
        click.option('--stage3-mlp-ratio', type=float,
                     default=TRAIN_DEFAULTS.stage3_mlp_ratio, show_default=True,
                     help='MLP expansion ratio used only in CSL-TinyViT Stage 3'),
        click.option('--stage2-depth', type=int,
                     default=TRAIN_DEFAULTS.stage2_depth, show_default=True,
                     help='Number of transformer blocks used only in CSL-TinyViT Stage 2'),
        click.option('--stage3-depth', type=int,
                     default=TRAIN_DEFAULTS.stage3_depth, show_default=True,
                     help='Number of transformer blocks used only in CSL-TinyViT Stage 3'),
        click.option('--width-first-hierarchy/--no-width-first-hierarchy',
                     default=TRAIN_DEFAULTS.width_first_hierarchy, show_default=True,
                     help='Preserve vertical detail via 48x16 -> 48x8 -> 24x8 CSL-TinyViT stages'),
        click.option('--identity-registers/--no-identity-registers',
                     default=TRAIN_DEFAULTS.identity_registers, show_default=True,
                     help='Exchange Stage-2/3 context through global identity-register tokens'),
        click.option('--identity-register-count', type=int,
                     default=TRAIN_DEFAULTS.identity_register_count, show_default=True,
                     help='Number of global identity-register tokens'),
        click.option('--identity-register-dim', type=int,
                     default=TRAIN_DEFAULTS.identity_register_dim, show_default=True,
                     help='Bottleneck width used for identity-register communication'),
        click.option('--identity-register-num-heads', type=int,
                     default=TRAIN_DEFAULTS.identity_register_num_heads, show_default=True,
                     help='Attention heads used by identity-register communication'),
        click.option('--identity-register-dropout', type=float,
                     default=TRAIN_DEFAULTS.identity_register_dropout, show_default=True,
                     help='Training-time probability of dropping each identity register'),
        click.option('--identity-register-gate-init', type=float,
                     default=TRAIN_DEFAULTS.identity_register_gate_init, show_default=True,
                     help='Initial residual broadcast gate for identity registers'),
        click.option('--identity-register-diversity-weight', type=float,
                     default=TRAIN_DEFAULTS.identity_register_diversity_weight, show_default=True,
                     help='Weight for the weak identity-register diversity loss'),
        click.option('--identity-register-diversity-margin', type=float,
                     default=TRAIN_DEFAULTS.identity_register_diversity_margin, show_default=True,
                     help='Maximum unpenalized cosine similarity between identity registers'),
        click.option('--native-branch-widths/--no-native-branch-widths',
                     default=TRAIN_DEFAULTS.native_branch_widths, show_default=True,
                     help='Keep global/local/fine fusion maps at descriptor-native 512/256/128 widths'),
        click.option('--fine-map-dim', type=int,
                     default=TRAIN_DEFAULTS.fine_map_dim, show_default=True,
                     help='Fine Stage-0 fusion-map channels; 0 keeps the full neck width'),
        click.option('--compact-deployment-head/--no-compact-deployment-head',
                     default=TRAIN_DEFAULTS.compact_deployment_head, show_default=True,
                     help='Train seven teacher branches but emit one distilled 512-D descriptor at inference'),
        click.option('--reid-adapter-stages', callback=_parse_int_tuple, type=str,
                     default=_click_imgsz_default(TRAIN_DEFAULTS.reid_adapter_stages), show_default=True,
                     help='CSL-TinyViT attention stages that receive zero-gated ReID residual adapters'),
        click.option('--reid-adapter-reduction', type=int,
                     default=TRAIN_DEFAULTS.reid_adapter_reduction, show_default=True,
                     help='Channel reduction ratio for CSL-TinyViT ReID residual adapters'),
        click.option('--reid-adapter-suppression-tau', type=float,
                     default=TRAIN_DEFAULTS.reid_adapter_suppression_tau, show_default=True,
                     help='RMS-saliency suppression threshold for ReID adapters; 0 disables'),
        click.option('--head-pool',
                     type=click.Choice(
                         selector_choices("head_pool"),
                         case_sensitive=False,
                     ),
                     default=TRAIN_DEFAULTS.head_pool, show_default=True,
                     help='Pooling layer used by CSL-TinyViT multi-branch heads'),
        click.option('--head-parts', callback=_parse_head_parts, type=str,
                     default=_click_imgsz_default(TRAIN_DEFAULTS.head_parts), show_default=True,
                     help='CSL-TinyViT head granularities, e.g. 1,2 for global+2 parts or 1,2,4 for MGN'),
        click.option('--head-type',
                     type=click.Choice(
                         TRAIN_HEAD_TYPES,
                         case_sensitive=False,
                     ),
                     default=TRAIN_DEFAULTS.head_type, show_default=True,
                     help='CSL-TinyViT branch head, including optional channel and G/P/C specialists'),
        click.option('--multiscale-channel-alpha', type=float,
                     default=TRAIN_DEFAULTS.multiscale_channel_alpha, show_default=True,
                     help='Channel power amplitude mixed inside each global/coarse/fine scale'),
        click.option('--body-slot-mode',
                     type=click.Choice(
                         ('recurrent_read', 'recurrent_read_write'),
                         case_sensitive=False,
                     ),
                     default=TRAIN_DEFAULTS.body_slot_mode, show_default=True,
                     help='Persistent body-slot communication: read-only Tier B or zero-gated read/write Tier C'),
        click.option('--body-slot-alpha', type=float,
                     default=TRAIN_DEFAULTS.body_slot_alpha, show_default=True,
                     help='Descriptor power allocated to the 512-D global stream'),
        click.option('--body-slot-visibility-floor', type=float,
                     default=TRAIN_DEFAULTS.body_slot_visibility_floor, show_default=True,
                     help='Minimum retrieval power retained for every body slot'),
        click.option('--part-pooling',
                     type=click.Choice(selector_choices("part_pooling"), case_sensitive=False),
                     default=TRAIN_DEFAULTS.part_pooling, show_default=True,
                     help='CSL-TinyViT local pooling: fixed, overlapping, learned-token, or semantic-visibility parts'),
        click.option('--num-part-tokens', type=int,
                     default=TRAIN_DEFAULTS.num_part_tokens, show_default=True,
                     help='Number of learned local/evidence queries for token or semantic-part pooling'),
        click.option('--evidence-num-roles', type=int,
                     default=TRAIN_DEFAULTS.evidence_num_roles, show_default=True,
                     help='Number of latent semantic role bins for CSL-TinyViT evidence tokens'),
        click.option('--decouple-patterns/--no-decouple-patterns',
                     default=TRAIN_DEFAULTS.decouple_patterns, show_default=True,
                     help='Use separate zero-initialized residual adapters for global and local features'),
        click.option('--pattern-adapter-dim', type=int,
                     default=TRAIN_DEFAULTS.pattern_adapter_dim, show_default=True,
                     help='Bottleneck width of each global/local pattern adapter'),
        click.option('--stripe-visibility/--no-stripe-visibility',
                     default=TRAIN_DEFAULTS.stripe_visibility, show_default=True,
                     help='Learn a per-image confidence for each fixed local stripe'),
        click.option('--drop-global-aux/--no-drop-global-aux',
                     default=TRAIN_DEFAULTS.drop_global_aux, show_default=True,
                     help='Add a training-only dropped-global CE auxiliary classifier to the standard CSL-TinyViT head'),
        click.option('--drop-global-aux-ratio', type=float,
                     default=TRAIN_DEFAULTS.drop_global_aux_ratio, show_default=True,
                     help='Horizontal activation-band ratio suppressed by --drop-global-aux'),
        click.option('--branch-aware-metric/--no-branch-aware-metric',
                     default=TRAIN_DEFAULTS.branch_aware_metric, show_default=True,
                     help='Apply metric loss separately to CSL-TinyViT global and part branches'),
        click.option('--branch-metric-part-weight', type=float,
                     default=TRAIN_DEFAULTS.branch_metric_part_weight, show_default=True,
                     help='Weight for each part branch metric loss when branch-aware metric is enabled'),
        click.option('--evidence-alignment-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.evidence_alignment_loss_weight, show_default=True,
                     help='Weight for batch Sinkhorn evidence alignment loss; 0 disables'),
        click.option('--evidence-alignment-margin', type=float,
                     default=TRAIN_DEFAULTS.evidence_alignment_margin, show_default=True,
                     help='Negative-pair margin for evidence alignment loss'),
        click.option('--evidence-sinkhorn-iters', type=int,
                     default=TRAIN_DEFAULTS.evidence_sinkhorn_iters, show_default=True,
                     help='Sinkhorn iterations for evidence alignment and reranking'),
        click.option('--evidence-sinkhorn-temperature', type=float,
                     default=TRAIN_DEFAULTS.evidence_sinkhorn_temperature, show_default=True,
                     help='Sinkhorn temperature for evidence alignment and reranking'),
        click.option('--evidence-rerank-topk', type=int,
                     default=TRAIN_DEFAULTS.evidence_rerank_topk, show_default=True,
                     help='Gallery top-K reranked with evidence Sinkhorn distance; 0 reranks all'),
        click.option('--evidence-null-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.evidence_null_loss_weight, show_default=True,
                     help='Weight for explicit final-token null/background supervision; 0 disables'),
        click.option('--evidence-diversity-loss-weight', type=float,
                     default=TRAIN_DEFAULTS.evidence_diversity_loss_weight, show_default=True,
                     help='Weight for role/descriptor diversity regularization across evidence tokens'),
        click.option('--head-warmup-epochs', type=int,
                     default=TRAIN_DEFAULTS.head_warmup_epochs, show_default=True,
                     help='Train only CSL-TinyViT neck/head for the first N epochs'),
        click.option('--head-warmup-lr-mult', type=float,
                     default=TRAIN_DEFAULTS.head_warmup_lr_mult, show_default=True,
                     help='LR multiplier for neck/head parameter groups during head warmup'),
        click.option('--eta-min', type=float, default=TRAIN_DEFAULTS.eta_min, show_default=True,
                     help='Minimum learning rate for cosine annealing schedule'),
        click.option('--pretrained/--no-pretrained', default=TRAIN_DEFAULTS.pretrained, show_default=True,
                     help='Use ImageNet-pretrained backbone'),
        click.option('--pretrained-weights', type=click.Path(exists=True, dir_okay=False),
                     default=TRAIN_DEFAULTS.pretrained_weights,
                     help='Local exact-backbone checkpoint from human pretraining; overrides model-zoo init'),
        click.option('--device', default=TRAIN_DEFAULTS.device,
                     help='cuda device, e.g. 0 or cpu or mps'),
        click.option('--project', type=click.Path(), default=TRAIN_DEFAULTS.project, show_default=True,
                     help='Save directory'),
        click.option('--name', default=TRAIN_DEFAULTS.name, show_default=True,
                     help='Experiment name'),
        click.option('--num-workers', type=int, default=TRAIN_DEFAULTS.num_workers, show_default=True,
                     help='Dataloader workers'),
        click.option('--seed', type=int, default=TRAIN_DEFAULTS.seed, show_default=True,
                     help='Global random seed for Python, NumPy, PyTorch, samplers, and dataloader workers'),
        click.option('--deterministic/--no-deterministic',
                     default=TRAIN_DEFAULTS.deterministic, show_default=True,
                     help='Require deterministic PyTorch algorithms and backend behavior'),
        click.option('--eval-datasets', type=str, default=','.join(TRAIN_DEFAULTS.eval_datasets) if TRAIN_DEFAULTS.eval_datasets else '',
                     help='Comma-separated list of extra datasets for cross-domain evaluation '
                          '(e.g. duke,cuhk03,msmt17)'),
        click.option('--flip-tta/--no-flip-tta', default=None,
                     help='Use horizontal flip augmentation during validation (default: recipe value)'),
        click.option('--ema-decay', type=float, default=TRAIN_DEFAULTS.ema_decay,
                     help='EMA momentum decay for model averaging (e.g. 0.999). '
                          'Disabled by default. Inspired by DynaMix'),
        click.option('--gaussian-blur/--no-gaussian-blur', default=TRAIN_DEFAULTS.gaussian_blur, show_default=True,
                     help='Apply random Gaussian blur augmentation'),
        click.option('--color-jitter/--no-color-jitter', default=TRAIN_DEFAULTS.color_jitter, show_default=True,
                     help='Apply color jitter augmentation (auto-enabled for transformer-family recipes)'),
        click.option('--random-grayscale', type=float, default=TRAIN_DEFAULTS.random_grayscale, show_default=True,
                     help='Probability of random grayscale conversion (0 to disable)'),
        click.option('--random-erasing', type=float, default=TRAIN_DEFAULTS.random_erasing, show_default=True,
                     help='Probability of random erasing augmentation (0 to disable)'),
        click.option('--random-patch/--no-random-patch', default=TRAIN_DEFAULTS.random_patch, show_default=True,
                     help='Apply random patch augmentation'),
        click.option('--random-crop-scale', type=float, default=TRAIN_DEFAULTS.random_crop_scale, show_default=True,
                     help='Random2DTranslation resize factor before crop; 1.05 matches LMBN implementation'),
        click.option('--color-augmentation/--no-color-augmentation',
                     default=TRAIN_DEFAULTS.color_augmentation, show_default=True,
                     help='Enable additional color augmentation mix used by LMBN-style recipes'),
        click.option('--background-mosaic/--no-background-mosaic',
                     default=TRAIN_DEFAULTS.background_mosaic, show_default=True,
                     help='Replace only the anchor background with a four-source donor mosaic'),
        click.option('--background-mosaic-mask-dir', type=click.Path(),
                     default=TRAIN_DEFAULTS.background_mosaic_mask_dir,
                     help='Mask root containing primary/ anchor and all_people/ donor trees'),
        click.option('--background-mosaic-probability', type=float,
                     default=TRAIN_DEFAULTS.background_mosaic_probability, show_default=True,
                     help='Maximum probability of identity-preserving background mosaic'),
        click.option('--background-mosaic-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.background_mosaic_start_epoch, show_default=True,
                     help='Keep background mosaic disabled through this epoch'),
        click.option('--background-mosaic-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.background_mosaic_ramp_end_epoch, show_default=True,
                     help='Epoch at which background mosaic reaches its maximum probability'),
        click.option('--background-mosaic-min-foreground-ratio', type=float,
                     default=TRAIN_DEFAULTS.background_mosaic_min_foreground_ratio,
                     show_default=True,
                     help='Reject anchor masks retaining less than this image fraction'),
        click.option('--background-mosaic-max-foreground-ratio', type=float,
                     default=TRAIN_DEFAULTS.background_mosaic_max_foreground_ratio,
                     show_default=True,
                     help='Reject anchor masks retaining more than this image fraction'),
        click.option('--background-mosaic-feather', type=float,
                     default=TRAIN_DEFAULTS.background_mosaic_feather, show_default=True,
                     help='Gaussian mask-edge feather radius in source-image pixels'),
        click.option('--background-mosaic-dilation', type=int,
                     default=TRAIN_DEFAULTS.background_mosaic_dilation, show_default=True,
                     help='Foreground-mask dilation radius before background compositing'),
        click.option('--background-mosaic-occluder-probability', type=float,
                     default=TRAIN_DEFAULTS.background_mosaic_occluder_probability,
                     show_default=True,
                     help='Probability of adding a boundary-entering person occluder'),
        click.option('--background-mosaic-occluder-min-area', type=float,
                     default=TRAIN_DEFAULTS.background_mosaic_occluder_min_area,
                     show_default=True,
                     help='Minimum image fraction covered by a context occluder'),
        click.option('--background-mosaic-occluder-max-area', type=float,
                     default=TRAIN_DEFAULTS.background_mosaic_occluder_max_area,
                     show_default=True,
                     help='Maximum image fraction covered by a context occluder'),
        click.option('--same-id-part-mosaic/--no-same-id-part-mosaic',
                     default=TRAIN_DEFAULTS.same_id_part_mosaic, show_default=True,
                     help='Copy body-aligned regions from same-ID batch donors'),
        click.option('--same-id-part-mosaic-probability', type=float,
                     default=TRAIN_DEFAULTS.same_id_part_mosaic_probability, show_default=True,
                     help='Probability of selecting each sample for same-ID part mosaic'),
        click.option('--same-id-part-mosaic-max-regions', type=int,
                     default=TRAIN_DEFAULTS.same_id_part_mosaic_max_regions, show_default=True,
                     help='Maximum number of body regions copied per augmented sample'),
        click.option('--same-id-part-mosaic-min-area', type=float,
                     default=TRAIN_DEFAULTS.same_id_part_mosaic_min_area, show_default=True,
                     help='Minimum total image fraction replaced by same-ID regions'),
        click.option('--same-id-part-mosaic-max-area', type=float,
                     default=TRAIN_DEFAULTS.same_id_part_mosaic_max_area, show_default=True,
                     help='Maximum total image fraction replaced by same-ID regions'),
        click.option('--same-id-part-mosaic-boundary-jitter', type=float,
                     default=TRAIN_DEFAULTS.same_id_part_mosaic_boundary_jitter, show_default=True,
                     help='Body-region boundary jitter as a fraction of image height'),
        click.option('--same-id-part-mosaic-cross-camera-rate', type=float,
                     default=TRAIN_DEFAULTS.same_id_part_mosaic_cross_camera_rate, show_default=True,
                     help='Rate of preferring different-camera same-ID donors when available'),
        click.option('--same-id-part-mosaic-min-unaltered', type=float,
                     default=TRAIN_DEFAULTS.same_id_part_mosaic_min_unaltered, show_default=True,
                     help='Minimum fraction of each training batch left unaltered'),
        click.option('--pav-mosaic/--no-pav-mosaic',
                     default=TRAIN_DEFAULTS.pav_mosaic, show_default=True,
                     help='Warp semantic body parts from pose-aligned same-ID donors'),
        click.option('--pav-metadata-dir', type=click.Path(),
                     default=TRAIN_DEFAULTS.pav_metadata_dir,
                     help='PAV metadata root generated by tools.create_market1501_pav_metadata'),
        click.option('--pav-mosaic-probability', type=float,
                     default=TRAIN_DEFAULTS.pav_mosaic_probability, show_default=True,
                     help='Maximum scheduled probability of PAV-Mosaic'),
        click.option('--pav-mosaic-max-parts', type=int,
                     default=TRAIN_DEFAULTS.pav_mosaic_max_parts, show_default=True,
                     help='Maximum semantic body parts replaced per PAV sample'),
        click.option('--pav-mosaic-max-foreground-replacement', type=float,
                     default=TRAIN_DEFAULTS.pav_mosaic_max_foreground_replacement,
                     show_default=True,
                     help='Maximum anchor-foreground fraction replaced by PAV'),
        click.option('--pav-mosaic-cross-camera-rate', type=float,
                     default=TRAIN_DEFAULTS.pav_mosaic_cross_camera_rate, show_default=True,
                     help='Rate of preferring cross-camera same-ID PAV donors'),
        click.option('--pav-mosaic-different-pose-rate', type=float,
                     default=TRAIN_DEFAULTS.pav_mosaic_different_pose_rate, show_default=True,
                     help='Rate of favoring pose-diverse PAV donors'),
        click.option('--pav-mosaic-min-keypoint-confidence', type=float,
                     default=TRAIN_DEFAULTS.pav_mosaic_min_keypoint_confidence,
                     show_default=True,
                     help='Minimum pose-keypoint confidence for a semantic part'),
        click.option('--pav-mosaic-min-unaltered', type=float,
                     default=TRAIN_DEFAULTS.pav_mosaic_min_unaltered, show_default=True,
                     help='Minimum fraction of each batch reverted to a clean view'),
        click.option('--pav-mosaic-warmup-epochs', type=int,
                     default=TRAIN_DEFAULTS.pav_mosaic_warmup_epochs, show_default=True,
                     help='Epochs used to ramp PAV probability from zero'),
        click.option('--pav-mosaic-decay-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.pav_mosaic_decay_start_epoch, show_default=True,
                     help='Epoch at which final PAV probability decay begins'),
        click.option('--pav-mosaic-final-probability-scale', type=float,
                     default=TRAIN_DEFAULTS.pav_mosaic_final_probability_scale,
                     show_default=True,
                     help='Fraction of maximum PAV probability retained at the final epoch'),
        click.option('--pav-consistency-weight', type=float,
                     default=TRAIN_DEFAULTS.pav_consistency_weight, show_default=True,
                     help='Clean-versus-PAV cosine embedding consistency weight'),
        click.option('--clean-student-consistency-weight', type=float,
                     default=TRAIN_DEFAULTS.clean_student_consistency_weight,
                     show_default=True,
                     help='Weight for clean-teacher query and descriptor consistency on augmented RGB views'),
        click.option('--anatomical-auxiliary/--no-anatomical-auxiliary',
                     default=TRAIN_DEFAULTS.anatomical_auxiliary, show_default=True,
                     help='Train RGB anatomical tokens from privileged pose/mask targets'),
        click.option('--anatomical-metadata-dir', type=click.Path(),
                     default=TRAIN_DEFAULTS.anatomical_metadata_dir,
                     help='Pose/person-mask metadata root used only during training'),
        click.option('--anatomical-person-mask-dir', type=click.Path(),
                     default=TRAIN_DEFAULTS.anatomical_person_mask_dir,
                     help='External high-confidence person-mask directory used only during training'),
        click.option('--anatomical-min-keypoint-confidence', type=float,
                     default=TRAIN_DEFAULTS.anatomical_min_keypoint_confidence,
                     show_default=True,
                     help='Minimum pose confidence used to rasterize anatomical targets'),
        click.option('--anatomical-token-dim', type=int,
                     default=TRAIN_DEFAULTS.anatomical_token_dim, show_default=True,
                     help='Width of the six grid-aligned anatomical tokens (minimum 16)'),
        click.option('--anatomical-distill-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_distill_weight, show_default=True,
                     help='Weight for same-scale mask-routed token consistency'),
        click.option('--anatomical-attention-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_attention_weight, show_default=True,
                     help='Weight for scale-aware anatomical cell-routing KL supervision'),
        click.option('--anatomical-foreground-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_foreground_weight, show_default=True,
                     help='Weight for RGB foreground mask supervision'),
        click.option('--anatomical-semantic-part-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_semantic_part_weight,
                     show_default=True,
                     help='Weight for training-only six-part semantic BCE/Dice supervision'),
        click.option('--anatomical-visibility-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_visibility_weight, show_default=True,
                     help='Weight for per-part visibility supervision'),
        click.option('--anatomical-contrastive-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_contrastive_weight, show_default=True,
                     help='Weight for visible same-part cross-camera contrastive learning'),
        click.option('--anatomical-descriptor-distill-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_descriptor_distill_weight,
                     show_default=True,
                     help='Weight for local semantic anatomy distillation into the final descriptor'),
        click.option('--anatomical-branch-distill-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_branch_distill_weight,
                     show_default=True,
                     help='Weight for EMA anatomy relations distilled into deployed 1/2/4-stripe branches'),
        click.option('--anatomical-branch-global-coefficient', type=float,
                     default=TRAIN_DEFAULTS.anatomical_branch_global_coefficient,
                     show_default=True,
                     help='Global-level share of anatomical branch distillation'),
        click.option('--anatomical-branch-coarse-coefficient', type=float,
                     default=TRAIN_DEFAULTS.anatomical_branch_coarse_coefficient,
                     show_default=True,
                     help='Two-stripe-level share of anatomical branch distillation'),
        click.option('--anatomical-branch-fine-coefficient', type=float,
                     default=TRAIN_DEFAULTS.anatomical_branch_fine_coefficient,
                     show_default=True,
                     help='Four-stripe-level share of anatomical branch distillation'),
        click.option('--anatomical-pose-teacher-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_pose_teacher_weight,
                     show_default=True,
                     help='Weight for the selected privileged pose-teacher objective'),
        click.option('--anatomical-query-distill-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_query_distill_weight,
                     show_default=True,
                     help='Weight for masked-teacher to unrestricted-RGB query distillation'),
        click.option('--anatomical-query-relational-distill-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_query_relational_distill_weight,
                     show_default=True,
                     help='Weight for visibility-weighted teacher/student query relation matching'),
        click.option('--anatomical-query-diversity-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_query_diversity_weight,
                     show_default=True,
                     help='Weight discouraging collapse among RGB anatomical queries'),
        click.option('--anatomical-query-diversity-margin', type=float,
                     default=TRAIN_DEFAULTS.anatomical_query_diversity_margin,
                     show_default=True,
                     help='Maximum allowed cosine similarity between distinct queries'),
        click.option('--anatomical-part-triplet-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_part_triplet_weight,
                     show_default=True,
                     help='Weight for visible same-part cross-camera hard triplets'),
        click.option('--anatomical-target-type',
                     type=click.Choice([
                         'deterministic_scale_aware_geometry',
                         'learned_pose_concat_ema',
                         'learned_pose_semantic_ema',
                         'learned_pose_semantic_fused_ema',
                         'privileged_mask_pose_attention',
                         'decoupled_pose_parsing_teacher',
                         'body_slot_privileged_ema',
                     ]),
                     default=TRAIN_DEFAULTS.anatomical_target_type,
                     show_default=True,
                     help='Anatomical teacher implementation used for training'),
        click.option('--anatomical-teacher-momentum', type=float,
                     default=TRAIN_DEFAULTS.anatomical_teacher_momentum,
                     show_default=True,
                     help='EMA momentum for learned pose-teacher targets'),
        click.option('--anatomical-multiscale/--no-anatomical-multiscale',
                     default=TRAIN_DEFAULTS.anatomical_multiscale,
                     show_default=True,
                     help='Supervise matched anatomical roles on Stage-2 local and Stage-0 fine maps'),
        click.option('--anatomical-accessory-query/--no-anatomical-accessory-query',
                     default=TRAIN_DEFAULTS.anatomical_accessory_query,
                     show_default=True,
                     help='Add a training-only seventh mask-supervised bag/accessory query'),
        click.option('--anatomical-deployment/--no-anatomical-deployment',
                     default=TRAIN_DEFAULTS.anatomical_deployment,
                     show_default=True,
                     help='Append six pose-supervised RGB semantic-part tokens to the retrieval descriptor'),
        click.option('--anatomical-deployment-dim', type=int,
                     default=TRAIN_DEFAULTS.anatomical_deployment_dim,
                     show_default=True,
                     help='Deployed width of each RGB anatomical part token'),
        click.option('--anatomical-deployment-alpha', type=float,
                     default=TRAIN_DEFAULTS.anatomical_deployment_alpha,
                     show_default=True,
                     help='Relative retrieval energy assigned to the deployed anatomical descriptor'),
        click.option('--anatomical-deployment-id-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_deployment_id_weight,
                     show_default=True,
                     help='Persistent visibility-weighted ID loss for deployed anatomical parts'),
        click.option('--anatomical-deployment-metric-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_deployment_metric_weight,
                     show_default=True,
                     help='Persistent cross-camera contrastive loss for deployed anatomical parts'),
        click.option('--anatomical-local-scale-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_local_scale_weight,
                     show_default=True,
                     help='Balanced contribution of the Stage-2 anatomical student'),
        click.option('--anatomical-fine-scale-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_fine_scale_weight,
                     show_default=True,
                     help='Balanced contribution of the Stage-0 anatomical student'),
        click.option('--anatomical-cross-scale-weight', type=float,
                     default=TRAIN_DEFAULTS.anatomical_cross_scale_weight,
                     show_default=True,
                     help='Weight aligning within-image anatomical role relations across scales'),
        click.option('--anatomical-pose-only-reliability', type=float,
                     default=TRAIN_DEFAULTS.anatomical_pose_only_reliability,
                     show_default=True,
                     help='Reliability multiplier for pose targets without a person mask'),
        click.option('--anatomical-min-effective-coverage', type=float,
                     default=TRAIN_DEFAULTS.anatomical_min_effective_coverage,
                     show_default=True,
                     help='Minimum fraction of training samples with usable anatomical targets'),
        click.option('--anatomical-student-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.anatomical_student_start_epoch,
                     show_default=True,
                     help='Last epoch before shared anatomical supervision starts'),
        click.option('--anatomical-student-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.anatomical_student_ramp_end_epoch,
                     show_default=True,
                     help='Epoch at which anatomy distillation reaches full weight'),
        click.option('--anatomical-query-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.anatomical_query_start_epoch,
                     show_default=True,
                     help='Last epoch before decoupled query distillation and triplets start'),
        click.option('--anatomical-query-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.anatomical_query_ramp_end_epoch,
                     show_default=True,
                     help='Epoch at which decoupled query losses reach full weight'),
        click.option('--anatomical-fine-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.anatomical_fine_start_epoch,
                     show_default=True,
                     help='Last epoch before fine-map anatomy starts; 0/0 follows the shared student schedule'),
        click.option('--anatomical-fine-ramp-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.anatomical_fine_ramp_end_epoch,
                     show_default=True,
                     help='Epoch at which fine-map and cross-scale anatomy reach full weight'),
        click.option('--anatomical-decay-start-epoch', type=int,
                     default=TRAIN_DEFAULTS.anatomical_decay_start_epoch,
                     show_default=True,
                     help='Epoch at which all anatomical losses begin decaying'),
        click.option('--anatomical-decay-end-epoch', type=int,
                     default=TRAIN_DEFAULTS.anatomical_decay_end_epoch,
                     show_default=True,
                     help='Epoch at which all anatomy losses become zero'),
        click.option('--anatomical-temperature', type=float,
                     default=TRAIN_DEFAULTS.anatomical_temperature, show_default=True,
                     help='Temperature for anatomical supervised contrastive learning'),
        click.option('--resume', type=click.Path(), default=None,
                     help='Resume training from a checkpoint dir or last.pt file'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


@boxmot.command(name='train-reid', help='Train a ReID model')
@train_options
@click.pass_context
def train_reid(ctx, **kwargs):
    args = _build_cli_namespace(ctx, "train", kwargs)
    from boxmot.engine.reid.data import resolve_reid_train_data

    try:
        args = resolve_reid_train_data(args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    # --data-dir is required unless --resume is provided, but recipes may
    # supply it and YAML --data specs may infer it, so validate after resolution.
    if (
        not getattr(args, "resume", None)
        and not getattr(args, "data_dir", None)
        and not getattr(args, "data_specs", ())
    ):
        raise click.MissingParameter(param_hint="'--data-dir'", param_type='option')
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
@click.option('--preprocess', type=click.Choice(sorted(PREPROCESS_REGISTRY.keys()), case_sensitive=False),
              default=None,
              help='Crop preprocessing method (default: checkpoint/hparams value)')
@click.option('--imgsz', callback=parse_imgsz, type=str, default=None,
              help='Image size as H,W (default: hparams value, fallback 256,128)')
@click.option('--inference-feature',
              type=click.Choice(
                  [
                      'concat_bn',
                      'norm_concat_bn',
                      'global',
                      'raw_mean',
                      'raw_concat',
                      'visibility_weighted_parts',
                      'evidence_sinkhorn',
                      'dse_weighted',
                      'dse_mix',
                  ],
                  case_sensitive=False,
              ),
              default=None,
              help='Override CSL-TinyViT eval embedding without retraining')
@click.option('--flip-tta/--no-flip-tta', default=None,
              help='Use horizontal flip test-time augmentation (default: hparams value)')
@click.option('--device', default='cpu', help='Device: cpu, mps, or cuda index')
@click.option('--batch-size', type=int, default=64, show_default=True,
              help='Batch size for feature extraction')
@click.option('--num-workers', type=int, default=4, show_default=True,
              help='Dataloader workers')
@click.option('--latency-warmup', type=int, default=5, show_default=True,
              help='Warmup forward passes before measuring ReID inference latency')
@click.option('--latency-iters', type=int, default=30, show_default=True,
              help='Timed forward passes for ReID inference latency; 0 disables latency measurement')
@click.option('--output', type=click.Path(), default=None,
              help='Directory to save eval JSON (default: next to weights)')
@click.pass_context
def eval_reid(ctx, **kwargs):
    args = _build_cli_namespace(ctx, "eval-reid", kwargs)
    _run_engine_workflow("boxmot.engine.reid.evaluator", args)


@boxmot.command(name='compare-reid', help='Compare ReID checkpoints across target datasets')
@click.option('--weights', type=click.Path(exists=True), multiple=True, required=True,
              help='Path to a trained ReID checkpoint. Repeat for multiple models.')
@click.option('--target', multiple=True, required=True,
              help='Evaluation target as DATASET=DATA_DIR. Repeat for multiple target datasets.')
@click.option('--label', multiple=True,
              help='Optional display/output label for each --weights entry.')
@click.option('--model', multiple=True,
              help='Model architecture override. Pass once for all weights or once per checkpoint.')
@click.option('--include-same-dataset/--cross-domain-only', default=False, show_default=True,
              help='Also evaluate models on their training dataset when checkpoint metadata is available.')
@click.option('--preprocess', type=click.Choice(sorted(PREPROCESS_REGISTRY.keys()), case_sensitive=False),
              default=None,
              help='Crop preprocessing method (default: checkpoint/hparams value)')
@click.option('--imgsz', callback=parse_imgsz, type=str, default=None,
              help='Image size as H,W (default: hparams value, fallback 256,128)')
@click.option('--inference-feature',
              type=click.Choice(
                  [
                      'concat_bn',
                      'norm_concat_bn',
                      'global',
                      'raw_mean',
                      'raw_concat',
                      'visibility_weighted_parts',
                      'evidence_sinkhorn',
                      'dse_weighted',
                      'dse_mix',
                  ],
                  case_sensitive=False,
              ),
              default=None,
              help='Override CSL-TinyViT eval embedding without retraining')
@click.option('--flip-tta/--no-flip-tta', default=None,
              help='Use horizontal flip test-time augmentation (default: hparams value)')
@click.option('--device', default='cpu', help='Device: cpu, mps, or cuda index')
@click.option('--batch-size', type=int, default=64, show_default=True,
              help='Batch size for feature extraction')
@click.option('--num-workers', type=int, default=4, show_default=True,
              help='Dataloader workers')
@click.option('--latency-warmup', type=int, default=5, show_default=True,
              help='Warmup forward passes before measuring ReID inference latency')
@click.option('--latency-iters', type=int, default=30, show_default=True,
              help='Timed forward passes for ReID inference latency; 0 disables the mAP/latency plot')
@click.option('--continue-on-error/--fail-fast', default=False, show_default=True,
              help='Record failed pairs and continue instead of stopping at the first failure.')
@click.option('--output', type=click.Path(), default='runs/reid_cross_domain', show_default=True,
              help='Directory to save aggregate comparison files and per-model eval JSONs')
@click.pass_context
def compare_reid(ctx, **kwargs):
    args = _build_cli_namespace(ctx, "compare-reid", kwargs)
    _run_engine_workflow("boxmot.engine.reid.comparison", args)


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
    """Compile the native C++ shared libraries shipped under ``boxmot/native/cpp/trackers``.

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
                from boxmot.native.reid import ensure_reid_capi_library
                lib = ensure_reid_capi_library(force_rebuild=force)
            else:
                module = importlib.import_module(f"boxmot.native.trackers.{name}")
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
