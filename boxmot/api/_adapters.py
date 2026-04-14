from __future__ import annotations

from pathlib import Path
from typing import Any

from boxmot.configs import BOXMOT_DEFAULTS, build_mode_namespace
from boxmot.engine import workflow_support as support


class _DefaultArg:
    def __repr__(self) -> str:
        return "DEFAULT"


_UNSET = _DefaultArg()


def _explicit_api_keys(
    api,
    *,
    device: str,
    half: bool,
    defaults,
) -> set[str]:
    return {
        *({"detector"} if api._detector_explicit else set()),
        *({"reid"} if api._reid_explicit else set()),
        *({"tracker"} if api._tracker_explicit else set()),
        *({"device"} if device != defaults.device else set()),
        *({"half"} if bool(half) != bool(defaults.half) else set()),
    }


def build_track_args(
    api,
    *,
    source: Any,
    imgsz=None,
    conf=None,
    iou: float = BOXMOT_DEFAULTS.track.iou,
    device: str = BOXMOT_DEFAULTS.track.device,
    half: bool = BOXMOT_DEFAULTS.track.half,
    save: bool = BOXMOT_DEFAULTS.track.save,
    save_txt: bool = BOXMOT_DEFAULTS.track.save_txt,
    show: bool = BOXMOT_DEFAULTS.track.show,
    verbose: bool = BOXMOT_DEFAULTS.track.verbose,
):
    return build_mode_namespace(
        "track",
        {
            "source": source,
            "benchmark": "",
            "split": "",
            "detector": api._detector_path(required=False) or BOXMOT_DEFAULTS.shared.detector,
            "reid": api._reid_path(required=False) or BOXMOT_DEFAULTS.shared.reid,
            "tracker": api._tracker_name(required=False) or BOXMOT_DEFAULTS.track.tracker,
            "imgsz": imgsz,
            "conf": conf,
            "iou": float(iou),
            "device": device,
            "half": bool(half),
            "classes": api.classes,
            "project": api.project,
            "show": bool(show),
            "save": bool(save),
            "save_txt": bool(save_txt),
            "verbose": bool(verbose),
        },
        explicit_keys=_explicit_api_keys(api, device=device, half=half, defaults=BOXMOT_DEFAULTS.track),
    )


def build_eval_args(
    api,
    benchmark: str | Path,
    *,
    imgsz=None,
    conf=None,
    iou: float = BOXMOT_DEFAULTS.eval.iou,
    device: str = BOXMOT_DEFAULTS.eval.device,
    half: bool = BOXMOT_DEFAULTS.eval.half,
    project: str | Path | None = None,
    verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
    show_progress: bool = True,
    postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
    mode: str = "eval",
    extra: dict[str, Any] | None = None,
):
    reid_path = api._reid_path(required=False) or BOXMOT_DEFAULTS.shared.reid
    tracker_spec = api.tracker
    per_class = bool(getattr(tracker_spec, "per_class", False)) if not isinstance(tracker_spec, str) else False

    args = build_mode_namespace(
        mode,
        {
            "data": str(benchmark),
            "benchmark": str(benchmark),
            "source": None,
            "split": "",
            "detector": [api._detector_path(required=True)],
            "reid": [reid_path],
            "device": device,
            "half": bool(half),
            "imgsz": imgsz,
            "conf": conf,
            "iou": float(iou),
            "classes": api.classes,
            "project": Path(project or api.project),
            "name": "python_api",
            "exist_ok": True,
            "ci": True,
            "tracker": api._tracker_name(required=True),
            "verbose": bool(verbose),
            "show_progress": bool(show_progress),
            "postprocessing": postprocessing,
            "fps": None,
            "show": False,
            "show_trajectories": False,
            "show_kf_preds": False,
            "save": False,
            "save_txt": False,
            "save_crop": False,
            "per_class": per_class,
            "target_id": None,
            "vid_stride": BOXMOT_DEFAULTS.eval.vid_stride,
            "tracking_backend": "thread",
            **(extra or {}),
        },
        explicit_keys=_explicit_api_keys(api, device=device, half=half, defaults=BOXMOT_DEFAULTS.eval),
    )
    args.reid_device = device
    args.reid_half = bool(half)
    args.dataset_detector_cfg = None
    args.eval_box_type = None
    args.gt_class_remap = None
    args.gt_class_distractor_ids = None
    args.remapped_class_ids = None
    args.remapped_class_names = None
    args.translated_benchmark_class_names = None
    return args


def build_tune_args(
    api,
    benchmark: str | Path,
    *,
    n_trials: int = BOXMOT_DEFAULTS.tune.n_trials,
    imgsz=None,
    conf=None,
    iou: float = BOXMOT_DEFAULTS.eval.iou,
    device: str = BOXMOT_DEFAULTS.eval.device,
    half: bool = BOXMOT_DEFAULTS.eval.half,
    project: str | Path | None = None,
    maximize=BOXMOT_DEFAULTS.tune.maximize,
    minimize=BOXMOT_DEFAULTS.tune.minimize,
    verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
    seed: int | None = None,
):
    return build_eval_args(
        api,
        benchmark,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        project=project,
        verbose=verbose,
        show_progress=False,
        postprocessing=BOXMOT_DEFAULTS.eval.postprocessing,
        mode="tune",
        extra={
            "n_trials": int(n_trials),
            "objectives": tuple(BOXMOT_DEFAULTS.tune.objectives),
            "maximize": tuple(maximize),
            "minimize": tuple(minimize),
            "seed": seed,
        },
    )


def build_export_args(
    api,
    *,
    include,
    device: str = BOXMOT_DEFAULTS.export.device,
    half: bool = BOXMOT_DEFAULTS.export.half,
    optimize: bool = BOXMOT_DEFAULTS.export.optimize,
    dynamic: bool = BOXMOT_DEFAULTS.export.dynamic,
    simplify: bool = BOXMOT_DEFAULTS.export.simplify,
    opset: int = BOXMOT_DEFAULTS.export.opset,
    workspace: int = BOXMOT_DEFAULTS.export.workspace,
    verbose: bool = False,
    batch_size: int = BOXMOT_DEFAULTS.export.batch_size,
    imgsz=None,
):
    return build_mode_namespace(
        "export",
        {
            "weights": api._reid_path(required=True),
            "include": tuple(include),
            "device": device,
            "half": bool(half),
            "optimize": bool(optimize),
            "dynamic": bool(dynamic),
            "simplify": bool(simplify),
            "opset": int(opset),
            "workspace": int(workspace),
            "verbose": bool(verbose),
            "batch_size": int(batch_size),
            "imgsz": imgsz,
        },
        explicit_keys={
            "weights",
            "device",
            "half",
            "optimize",
            "dynamic",
            "simplify",
            "opset",
            "workspace",
            "batch_size",
            "imgsz",
            "include",
        },
    )


__all__ = (
    "_UNSET",
    "build_eval_args",
    "build_export_args",
    "build_track_args",
    "build_tune_args",
    "support",
)
