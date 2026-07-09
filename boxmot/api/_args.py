from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from boxmot.configs import BOXMOT_DEFAULTS, build_mode_namespace

_MAXIMIZE_TUNE_METRICS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe")
_MINIMIZE_TUNE_METRICS = ("IDSW", "IDs", "IDSW_rate")
_TUNE_METRIC_ALIASES = {
    "hota": "HOTA",
    "mota": "MOTA",
    "idf1": "IDF1",
    "assa": "AssA",
    "assre": "AssRe",
    "idsw": "IDSW",
    "id_switch": "IDSW",
    "id_switches": "IDSW",
    "idswitch": "IDSW",
    "idswitches": "IDSW",
    "ids": "IDs",
    "idsw_rate": "IDSW_rate",
    "id_switch_rate": "IDSW_rate",
    "id_switches_rate": "IDSW_rate",
    "idswitch_rate": "IDSW_rate",
    "idswitches_rate": "IDSW_rate",
}


def _metric_values(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    raw_values = (values,) if isinstance(values, str) else tuple(values)
    metrics: list[str] = []
    for value in raw_values:
        for part in str(value).split(","):
            metric = part.strip()
            if metric:
                metrics.append(_TUNE_METRIC_ALIASES.get(metric.lower(), metric))
    return tuple(metrics)


def _split_objectives_by_direction(objectives: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    maximize: list[str] = []
    minimize: list[str] = []
    for metric in objectives:
        if metric in _MINIMIZE_TUNE_METRICS:
            minimize.append(metric)
        else:
            maximize.append(metric)
    return tuple(maximize), tuple(minimize)


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
    tracker_backend: str | None = None,
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
            "tracker_backend": (
                tracker_backend
                if tracker_backend is not None
                else (api._tracker_backend(required=False) or BOXMOT_DEFAULTS.track.tracker_backend)
            ),
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
    split: str | None = None,
    imgsz=None,
    conf=None,
    iou: float = BOXMOT_DEFAULTS.eval.iou,
    device: str = BOXMOT_DEFAULTS.eval.device,
    half: bool = BOXMOT_DEFAULTS.eval.half,
    project: str | Path | None = None,
    verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
    show_progress: bool = True,
    postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
    tracker_backend: str | None = None,
    tracking_backend: str = "thread",
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
            "split": "" if split is None else str(split),
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
            "tracker_backend": tracker_backend if tracker_backend is not None else api._tracker_backend(required=False),
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
            "tracking_backend": str(tracking_backend),
            **(extra or {}),
        },
        explicit_keys={
            *_explicit_api_keys(api, device=device, half=half, defaults=BOXMOT_DEFAULTS.eval),
            *({"split"} if split is not None else set()),
        },
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
    split: str | None = None,
    n_trials: int = BOXMOT_DEFAULTS.tune.n_trials,
    objectives: Sequence[str] | str | None = None,
    imgsz=None,
    conf=None,
    iou: float = BOXMOT_DEFAULTS.eval.iou,
    device: str = BOXMOT_DEFAULTS.eval.device,
    half: bool = BOXMOT_DEFAULTS.eval.half,
    project: str | Path | None = None,
    maximize: Sequence[str] | str | None = None,
    minimize: Sequence[str] | str | None = None,
    verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
    tracker_backend: str | None = None,
    tracking_backend: str = "thread",
    seed: int | None = None,
):
    tune_objectives = _metric_values(objectives) or tuple(BOXMOT_DEFAULTS.tune.objectives)
    if objectives is not None and maximize is None and minimize is None:
        tune_maximize, tune_minimize = _split_objectives_by_direction(tune_objectives)
    else:
        tune_maximize = _metric_values(maximize) or tuple(BOXMOT_DEFAULTS.tune.maximize)
        tune_minimize = _metric_values(minimize) or tuple(BOXMOT_DEFAULTS.tune.minimize)

    return build_eval_args(
        api,
        benchmark,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        project=project,
        verbose=verbose,
        show_progress=False,
        postprocessing=BOXMOT_DEFAULTS.eval.postprocessing,
        tracker_backend=tracker_backend,
        tracking_backend=tracking_backend,
        mode="tune",
        extra={
            "n_trials": int(n_trials),
            "objectives": tune_objectives,
            "maximize": tune_maximize,
            "minimize": tune_minimize,
            "seed": seed,
        },
    )


def build_generate_args(
    api,
    *,
    benchmark: str | Path | None = None,
    source: str | Path | None = None,
    imgsz=None,
    conf=None,
    iou: float = BOXMOT_DEFAULTS.generate.iou,
    device: str = BOXMOT_DEFAULTS.generate.device,
    half: bool = BOXMOT_DEFAULTS.generate.half,
    project: str | Path | None = None,
    verbose: bool = BOXMOT_DEFAULTS.generate.verbose,
    batch_size: int = BOXMOT_DEFAULTS.generate.batch_size,
    auto_batch: bool = BOXMOT_DEFAULTS.generate.auto_batch,
    resume: bool = BOXMOT_DEFAULTS.generate.resume,
    n_threads: int = BOXMOT_DEFAULTS.generate.n_threads,
):
    data = None if benchmark is None else str(benchmark)
    return build_mode_namespace(
        "generate",
        {
            "data": data,
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
            "project": Path(project or api.project),
            "name": "python_api",
            "exist_ok": True,
            "ci": True,
            "verbose": bool(verbose),
            "batch_size": int(batch_size),
            "auto_batch": bool(auto_batch),
            "resume": bool(resume),
            "n_threads": int(n_threads),
        },
        explicit_keys=_explicit_api_keys(api, device=device, half=half, defaults=BOXMOT_DEFAULTS.generate),
    )


def build_research_args(
    api,
    benchmark: str | Path,
    *,
    project: str | Path | None = None,
    verbose: bool = BOXMOT_DEFAULTS.research.verbose,
    proposal_model: str = BOXMOT_DEFAULTS.research.proposal_model,
    proposal_api_key: str | None = BOXMOT_DEFAULTS.research.proposal_api_key,
    proposal_api_key_env: str | None = BOXMOT_DEFAULTS.research.proposal_api_key_env,
    max_metric_calls: int = BOXMOT_DEFAULTS.research.max_metric_calls,
    eval_timeout: float = BOXMOT_DEFAULTS.research.eval_timeout,
    keep_workspace: bool = BOXMOT_DEFAULTS.research.keep_workspace,
    hota_penalty: float = BOXMOT_DEFAULTS.research.hota_penalty,
    idf1_penalty: float = BOXMOT_DEFAULTS.research.idf1_penalty,
    mota_penalty: float = BOXMOT_DEFAULTS.research.mota_penalty,
    hota_tolerance: float = BOXMOT_DEFAULTS.research.hota_tolerance,
    idf1_tolerance: float = BOXMOT_DEFAULTS.research.idf1_tolerance,
    mota_tolerance: float = BOXMOT_DEFAULTS.research.mota_tolerance,
    tracker_backend: str | None = None,
    tracking_backend: str = "thread",
):
    return build_mode_namespace(
        "research",
        {
            "data": str(benchmark),
            "benchmark": "",
            "source": None,
            "split": "",
            "detector": [api._detector_path(required=False) or BOXMOT_DEFAULTS.shared.detector],
            "reid": [api._reid_path(required=False) or BOXMOT_DEFAULTS.shared.reid],
            "tracker": api._tracker_name(required=True),
            "tracker_backend": tracker_backend if tracker_backend is not None else api._tracker_backend(required=False),
            "classes": api.classes,
            "project": Path(project or api.project),
            "name": "python_api",
            "exist_ok": True,
            "ci": True,
            "verbose": bool(verbose),
            "proposal_model": str(proposal_model),
            "proposal_api_key": proposal_api_key,
            "proposal_api_key_env": proposal_api_key_env,
            "max_metric_calls": int(max_metric_calls),
            "eval_timeout": float(eval_timeout),
            "keep_workspace": bool(keep_workspace),
            "hota_penalty": float(hota_penalty),
            "idf1_penalty": float(idf1_penalty),
            "mota_penalty": float(mota_penalty),
            "hota_tolerance": float(hota_tolerance),
            "idf1_tolerance": float(idf1_tolerance),
            "mota_tolerance": float(mota_tolerance),
            "tracking_backend": str(tracking_backend),
        },
        explicit_keys=_explicit_api_keys(
            api,
            device=BOXMOT_DEFAULTS.research.device,
            half=BOXMOT_DEFAULTS.research.half,
            defaults=BOXMOT_DEFAULTS.research,
        ),
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
    tflite_quantize: str = BOXMOT_DEFAULTS.export.tflite_quantize,
    tflite_calibration_data=None,
    tflite_calibration_samples: int = BOXMOT_DEFAULTS.export.tflite_calibration_samples,
    tflite_calibration_preprocess: str = BOXMOT_DEFAULTS.export.tflite_calibration_preprocess,
    tflite_calibration_seed: int = BOXMOT_DEFAULTS.export.tflite_calibration_seed,
    tflite_calibration_update: str = BOXMOT_DEFAULTS.export.tflite_calibration_update,
    tflite_static_activation_bits: int = BOXMOT_DEFAULTS.export.tflite_static_activation_bits,
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
            "tflite_quantize": tflite_quantize,
            "tflite_calibration_data": tflite_calibration_data,
            "tflite_calibration_samples": int(tflite_calibration_samples),
            "tflite_calibration_preprocess": tflite_calibration_preprocess,
            "tflite_calibration_seed": int(tflite_calibration_seed),
            "tflite_calibration_update": tflite_calibration_update,
            "tflite_static_activation_bits": int(tflite_static_activation_bits),
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
            "tflite_quantize",
            "tflite_calibration_data",
            "tflite_calibration_samples",
            "tflite_calibration_preprocess",
            "tflite_calibration_seed",
            "tflite_calibration_update",
            "tflite_static_activation_bits",
        },
    )


__all__ = (
    "build_eval_args",
    "build_export_args",
    "build_generate_args",
    "build_research_args",
    "build_track_args",
    "build_tune_args",
)
