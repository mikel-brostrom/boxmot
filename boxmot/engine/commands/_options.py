"""Reusable Click options for engine tracking workflows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import click

_TRACK_CORE_OPTION_NAMES = (
    "imgsz",
    "fps",
    "conf",
    "iou",
    "device",
    "project",
    "half",
    "tracker",
    "variable_dt",
    "verbose",
    "show",
    "show_trajectories",
    "show_kf_preds",
    "save_txt",
    "save",
    "per_class",
)

_REPLAY_CORE_OPTION_NAMES = (
    "project",
    "name",
    "exist_ok",
    "ci",
    "tracker",
    "variable_dt",
    "verbose",
    "show_timing",
    "per_class",
)


def _click_imgsz_default(value: Any) -> Any:
    """Normalize configured image sizes into a Click-friendly default value."""

    if isinstance(value, (list, tuple)):
        return ",".join(str(part) for part in value)
    return value


def _parse_imgsz(_ctx: click.Context, _param: click.Parameter, value: Any) -> int | tuple[int, int] | None:
    """Parse one integer or an ``H,W`` pair for Click image-size options."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return int(value[0])
        if len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise click.BadParameter(f"Invalid --imgsz: {value}")

    parts = str(value).replace(",", " ").split()
    try:
        if len(parts) == 1:
            return int(parts[0])
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        pass
    raise click.BadParameter(f"Invalid --imgsz: {value}")


def _parse_int_tuple(_ctx: click.Context, _param: click.Parameter, value: Any) -> tuple[int, ...]:
    """Parse an optional comma-separated integer tuple."""

    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        parts = tuple(int(part) for part in value)
    else:
        normalized = str(value).strip()
        if normalized.lower() in {"", "none", "off"}:
            return ()
        tokens = [token for token in normalized.replace(";", ",").split(",") if token.strip()]
        parts = tuple(int(token) for token in tokens)
    return tuple(dict.fromkeys(parts))


def _parse_device(_ctx: click.Context, _param: click.Parameter, value: str | None) -> str | None:
    """Validate one device selector without loading accelerator runtimes for help."""

    if value is None:
        return None
    from boxmot.utils.devices import normalize_device

    try:
        return normalize_device(value)
    except (TypeError, ValueError) as exc:
        raise click.BadParameter(str(exc)) from exc


def _core_option_decorators(defaults: Any, *, half_help: str) -> dict[str, Callable]:
    """Build the ordered runtime decorators used by tracking and replay commands."""

    from boxmot.trackers.common.registry import TRACKER_MAPPING

    tracker_help = ", ".join(TRACKER_MAPPING)
    return {
        "imgsz": click.option(
            "--imgsz",
            callback=_parse_imgsz,
            default=_click_imgsz_default(defaults.imgsz),
            type=str,
            help=(
                "Image size for model input as H,W (e.g. 800,1440) or single int for square. "
                "Default: read from the selected detector config, otherwise use detector-specific defaults."
            ),
        ),
        "fps": click.option(
            "--fps",
            type=click.IntRange(min=1),
            default=defaults.fps,
            help="Frame rate of saved tracking video; input frames and capture timestamps are unchanged.",
        ),
        "conf": click.option(
            "--conf",
            type=float,
            default=defaults.conf,
            help="Min confidence threshold. Default: read from the selected detector config, fallback 0.01.",
        ),
        "iou": click.option("--iou", type=float, default=defaults.iou, help="IoU threshold for NMS"),
        "device": click.option(
            "--device",
            default=defaults.device,
            callback=_parse_device,
            help="One device: cpu, mps, cuda:N, or N (e.g. 0). GPU lists are not supported.",
        ),
        "sequence_workers": click.option(
            "--sequence-workers",
            type=click.IntRange(min=1),
            default=defaults.sequence_workers,
            help=(
                "Maximum sequence worker processes. Default: min(sequences, CPU cores - 2), at least 1. "
                "During tuning, this limit applies per trial."
            ),
        ),
        "cache_inputs": click.option(
            "--cache-inputs/--no-cache-inputs",
            default=defaults.cache_inputs,
            show_default=True,
            help=(
                "Build and reuse mapped inputs for repeated eval/tune runs, including images, detections, "
                "embeddings, masks, sensors, and annotations. Uses extra disk space."
            ),
        ),
        "project": click.option(
            "--project",
            type=Path,
            default=defaults.project,
            help="save results to project/name",
        ),
        "name": click.option("--name", default=defaults.name, help="save results to project/name"),
        "exist_ok": click.option(
            "--exist-ok",
            is_flag=True,
            default=defaults.exist_ok,
            help="existing project/name ok, do not increment",
        ),
        "half": click.option("--half", is_flag=True, default=defaults.half, help=half_help),
        "ci": click.option(
            "--ci",
            is_flag=True,
            default=defaults.ci,
            help="reuse existing runs in CI (no UI)",
        ),
        "tracker": click.option(
            "--tracker",
            type=click.Choice(tuple(TRACKER_MAPPING)),
            default=defaults.tracker,
            show_default=True,
            help=f"one of: {tracker_help}",
        ),
        "variable_dt": click.option(
            "--variable-dt/--fixed-dt",
            default=None,
            help=(
                "Use capture timestamps for Kalman prediction; otherwise keep the tracker YAML setting "
                "(default: fixed dt)."
            ),
        ),
        "verbose": click.option(
            "--verbose",
            is_flag=True,
            default=defaults.verbose,
            help="print detailed logs",
        ),
        "show_timing": click.option(
            "--show-timing/--hide-timing",
            default=defaults.show_timing,
            show_default=True,
            help="print runtime timing summary after evaluation",
        ),
        "show": click.option(
            "--show",
            is_flag=True,
            default=defaults.show,
            help="display tracking in a window",
        ),
        "show_trajectories": click.option(
            "--show-trajectories",
            is_flag=True,
            default=defaults.show_trajectories,
            help="overlay past trajectories",
        ),
        "show_kf_preds": click.option(
            "--show-kf-preds",
            "show_kf_preds",
            is_flag=True,
            default=defaults.show_kf_preds,
            help="show Kalman-filter predictions",
        ),
        "save_txt": click.option(
            "--save-txt",
            is_flag=True,
            default=defaults.save_txt,
            help="save results to a .txt file",
        ),
        "save": click.option(
            "--save",
            is_flag=True,
            default=defaults.save,
            help="save annotated video",
        ),
        "per_class": click.option(
            "--per-class",
            is_flag=True,
            default=defaults.per_class,
            help="track each class separately",
        ),
    }


def _apply_core_options(
    func: Callable,
    defaults: Any,
    option_names: tuple[str, ...],
    *,
    half_help: str = "use FP16 half-precision inference",
) -> Callable:
    options = _core_option_decorators(defaults, half_help=half_help)
    unknown_names = set(option_names).difference(options)
    if unknown_names:
        unknown = ", ".join(sorted(unknown_names))
        raise ValueError(f"Unknown core CLI option(s): {unknown}")
    for option_name in reversed(option_names):
        func = options[option_name](func)
    return func


def track_options(func: Callable) -> Callable:
    """Attach only runtime options consumed by direct tracking."""

    from boxmot.engine.config.runtime import BOXMOT_DEFAULTS

    return _apply_core_options(
        func,
        BOXMOT_DEFAULTS.track,
        _TRACK_CORE_OPTION_NAMES,
        half_help="use FP16 for supported ReID inference paths; detector precision is unchanged",
    )


def replay_options(*, mode: str, parallel: bool = False) -> Callable:
    """Attach tracker/replay controls for a cached workflow mode."""

    from boxmot.engine.config.runtime import BOXMOT_DEFAULTS

    defaults = getattr(BOXMOT_DEFAULTS, mode)
    option_names = _REPLAY_CORE_OPTION_NAMES
    if mode in {"eval", "tune"}:
        option_names = ("cache_inputs", *option_names)
    if parallel:
        option_names = ("sequence_workers", *option_names)

    def decorator(func: Callable) -> Callable:
        return _apply_core_options(func, defaults, option_names)

    return decorator


def source_option(*, default: str | None = "0", help_text: str = "file/dir/URL/glob, 0 for webcam") -> Callable:
    """Attach a ``--source`` option with command-specific defaults and help."""

    return click.option("--source", type=str, default=default, help=help_text)


def split_option(func: Callable) -> Callable:
    """Attach a dataset-split override."""

    return click.option(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use (e.g. train, val, test, ablation). Overrides auto-detection from source path.",
    )(func)


def sequence_option(func: Callable) -> Callable:
    """Attach a repeatable sequence selection shared by evaluation and tuning."""
    return click.option(
        "--sequence",
        "sequence_names",
        type=str,
        multiple=True,
        metavar="NAME",
        help="Limit the selected split to one sequence. Repeat to select multiple sequences.",
    )(func)


def experiment_option(func: Callable | None = None, *, required: bool = False) -> Callable:
    """Attach the experiment-config option, optionally making it mandatory."""

    decorator = click.option(
        "--experiment",
        type=str,
        required=required,
        help=(
            "experiment YAML filename or path, e.g. mot17/ablation-yolox-lmbn.yaml or "
            "boxmot/configs/experiments/mot17/ablation-yolox-lmbn.yaml"
        ),
    )
    return decorator if func is None else decorator(func)


def dataset_option(*, default: str | None = None) -> Callable:
    """Attach the dataset profile or saved-sensor bundle selection."""

    return click.option(
        "--dataset",
        type=str,
        default=default,
        help="Dataset id, YAML file, or sensor dataset folder containing dataset.yaml (eval and tune).",
    )


def _parse_dataset_fps(_ctx: click.Context, _param: click.Parameter, value: float | None) -> float | None:
    """Require a finite, positive target rate for dataset frame selection."""

    if value is not None and (not math.isfinite(value) or value <= 0):
        raise click.BadParameter("must be a finite number greater than zero")
    return value


def dataset_fps_option(func: Callable) -> Callable:
    """Attach synchronized dataset frame selection for materialization and replay."""

    return click.option(
        "--fps",
        type=float,
        callback=_parse_dataset_fps,
        default=None,
        help=(
            "Target dataset FPS: sample frames before perception and align ground truth. "
            "Default: original frames, or the selected build's FPS when replaying."
        ),
    )(func)


def build_selection_options(func: Callable | None = None, *, required: bool = True) -> Callable:
    """Attach immutable-build selection, optionally allowing workflow preparation."""

    build_help = "Materialized build ID or explicit build directory."
    build_root_help = (
        "Materialized-dataset root for build IDs and shared detector cache; "
        "overrides BOXMOT_BUILDS_DIR and defaults to ./runs/materializations."
    )
    if not required:
        build_help += (
            " Omit with --experiment, or with --dataset and --detector, to materialize and reuse "
            "a compatible canonical build automatically."
        )
        build_root_help = (
            "Root for build IDs, automatic materialization, and shared detector cache; "
            "overrides BOXMOT_BUILDS_DIR and defaults to ./runs/materializations."
        )

    def decorator(command: Callable) -> Callable:
        decorators = (
            click.option(
                "--build",
                "build_ref",
                type=str,
                required=required,
                help=build_help,
            ),
            click.option(
                "--build-root",
                type=click.Path(path_type=Path),
                default=None,
                help=build_root_help,
            ),
        )
        for option in reversed(decorators):
            command = option(command)
        return command

    return decorator if func is None else decorator(func)


def data_root_option(func: Callable) -> Callable:
    """Attach the raw-dataset root used to verify build source identity."""

    return click.option(
        "--data-root",
        type=click.Path(path_type=Path),
        default=None,
        help="Tracking-dataset root; defaults to ./datasets/mot.",
    )(func)


def replay_build_options(*, dataset_default: str | None = None) -> Callable:
    """Attach shared eval/tune inputs for build reuse or automatic materialization."""

    from boxmot.engine.config.runtime import BOXMOT_DEFAULTS

    def decorator(func: Callable) -> Callable:
        options = (
            experiment_option,
            dataset_option(default=dataset_default),
            click.option(
                "--detector",
                type=str,
                default=None,
                help=(
                    "Detector profile ID or YAML config used to resolve an authored experiment with --dataset; "
                    "append /CHECKPOINT to disambiguate."
                ),
            ),
            click.option(
                "--reid",
                type=str,
                default=None,
                help="ReID profile used to resolve an authored experiment; omit only for experiments without ReID.",
            ),
            build_selection_options(required=False),
            click.option(
                "--device",
                default=BOXMOT_DEFAULTS.materialize.device,
                callback=_parse_device,
                help="One device for uncached perception: cpu, mps, cuda:N, or N (e.g. 0). Matching builds are reused.",
            ),
        )
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


def tracker_backend_option(*, default: str) -> Callable:
    """Attach the tracker implementation backend option."""

    return click.option(
        "--tracker-backend",
        type=click.Choice(["python", "cpp"]),
        default=default,
        show_default=True,
        help=(
            "Tracker implementation backend. Native 'cpp' is available for "
            "botsort, bytetrack, occluboost, ocsort, and sfsort."
        ),
    )


def tracker_config_option(func: Callable) -> Callable:
    """Attach a reusable tracker runtime config or built-in preset selector."""

    return click.option(
        "--tracker-config",
        type=str,
        default=None,
        help="Tracker runtime YAML path or built-in preset; overrides must preserve calibrated time units.",
    )(func)


def eval_masks_option(func: Callable) -> Callable:
    """Select segmentation scoring for KITTI-MOTS evaluation and tuning."""

    return click.option(
        "--eval-masks",
        is_flag=True,
        default=False,
        help="Evaluate KITTI-MOTS with mask IoU; requires published masks. Default: evaluate bounding boxes.",
    )(func)


def kalman_calibration_option(*, mode: str) -> Callable:
    """Expose direct covariance calibration before evaluation or tracker tuning."""
    outcome = {
        "eval": "then evaluate once",
        "tune": "then tune tracker parameters with calibrated KF settings fixed",
    }[mode]
    return click.option(
        "--calibrate-kf",
        is_flag=True,
        default=False,
        help=(
            f"Calibrate Kalman noise from detections and ground truth, {outcome}; "
            "Python Kalman trackers only. EagerMOT requires ground_truth_3d with track IDs."
        ),
    )


def association_function_option(func: Callable) -> Callable:
    """Attach the shared detection-track geometry selector."""

    return click.option(
        "--asso-func",
        type=click.Choice(("iou", "giou", "diou", "ciou", "hmiou", "centroid")),
        default=None,
        help=(
            "Association geometry override for AABB and OBB tracking. "
            "OBB ciou is a custom experimental adaptation; OBB hmiou is an "
            "experimental global-y height cue for scenes where image vertical is meaningful."
        ),
    )(func)


__all__ = (
    "_click_imgsz_default",
    "_parse_imgsz",
    "_parse_int_tuple",
    "association_function_option",
    "build_selection_options",
    "data_root_option",
    "dataset_fps_option",
    "dataset_option",
    "eval_masks_option",
    "experiment_option",
    "kalman_calibration_option",
    "replay_build_options",
    "replay_options",
    "source_option",
    "split_option",
    "track_options",
    "tracker_backend_option",
    "tracker_config_option",
)
