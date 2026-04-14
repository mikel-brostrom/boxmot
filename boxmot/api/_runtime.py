from __future__ import annotations

import math
import random
import re
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urlparse

import cv2
import yaml

from boxmot.configs import BOXMOT_DEFAULTS, build_mode_namespace
from boxmot.data import IMAGE_EXTS, VIDEO_EXTS
from boxmot.engine.results import Results
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING, create_tracker, get_tracker_config
from boxmot.utils import configure_logging as _configure_boxmot_logging, logger as LOGGER
from boxmot.utils.misc import increment_path, resolve_model_path
from boxmot.utils.timing import TimingStats
from boxmot.utils.torch_utils import select_device

from ._reporting import extract_summary, timing_summary_from_stats
from ._results import ExportResult, ValidationResult

REID_TRACKERS = {"strongsort", "botsort", "deepocsort", "hybridsort", "boosttrack"}
TRACKER_CLASS_TO_NAME = {
    class_path.rsplit(".", 1)[-1].lower(): tracker_name
    for tracker_name, class_path in TRACKER_MAPPING.items()
}


class _DefaultArg:
    def __repr__(self) -> str:
        return "DEFAULT"


_UNSET = _DefaultArg()


def normalize_classes(classes: Any) -> list[int] | None:
    if classes is None:
        return None
    if isinstance(classes, str):
        parts = [part for part in re.split(r"[\s,]+", classes.strip()) if part]
        return [int(part) for part in parts]
    if isinstance(classes, int):
        return [int(classes)]
    return [int(value) for value in classes]


def is_leaf_source(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    if not path.is_dir():
        return False
    img_dir = path / "img1" if (path / "img1").is_dir() else path
    return any(child.is_file() and child.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS for child in img_dir.iterdir())


def expand_sources(source: Any) -> list[Any]:
    if isinstance(source, (list, tuple)):
        return list(source)

    if not isinstance(source, (str, Path)):
        return [source]

    path = Path(source)
    if not path.is_dir() or is_leaf_source(path):
        return [source]

    children = [child for child in sorted(path.iterdir()) if is_leaf_source(child)]
    return children or [source]


def coerce_results(
    data: Any,
    *,
    detector=None,
    reid=None,
    tracker=None,
    verbose: bool = False,
    track_fn=None,
) -> list[Results]:
    if isinstance(data, Results):
        return [data]

    if isinstance(data, (list, tuple)) and all(isinstance(item, Results) for item in data):
        return list(data)

    if detector is None or tracker is None:
        raise ValueError("Detector and tracker are required when evaluating raw sources.")
    if track_fn is None:
        raise ValueError("A tracking function is required when evaluating raw sources.")

    return [track_fn(source, detector, reid, tracker, verbose=verbose) for source in expand_sources(data)]


def ensure_model_path(model_ref: str | Path | None) -> Path | None:
    if model_ref is None:
        return None
    path = Path(model_ref)
    if not path.suffix:
        path = path.with_suffix(".pt")
    return resolve_model_path(path)


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned or "run"


def resolve_output_stem(source: Any) -> str:
    source_str = str(source)
    if source_str.isdigit():
        return f"camera_{source_str}"

    if "://" in source_str:
        parsed = urlparse(source_str)
        pieces = [parsed.scheme, parsed.netloc, parsed.path.strip("/")]
        return sanitize_name("_".join(piece for piece in pieces if piece))

    path = Path(source_str)
    if path.name == "img1" and path.parent.name:
        return sanitize_name(path.parent.name)
    if path.suffix:
        return sanitize_name(path.stem)
    return sanitize_name(path.name)


def compare_scores(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return left > right


@contextmanager
def suppress_boxmot_logs(enabled: bool, *, level: str = "WARNING"):
    if not enabled:
        yield
        return

    LOGGER.remove()
    LOGGER.add(
        sys.stderr,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,
        format="<level>{level: <8}</level> | <level>{message}</level>",
    )
    try:
        yield
    finally:
        _configure_boxmot_logging(main_only=True)


class TrackerReIDAdapter:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __call__(self, inputs, boxes=None, **_kwargs):
        if boxes is None:
            raise TypeError("boxes are required when reusing a tracker ReID backend")
        return self.backend.get_features(boxes, inputs)


def detector_path_from_spec(spec: Any, *, required: bool = True) -> Path | None:
    if spec is None:
        if required:
            raise ValueError("A detector model path is required for this operation.")
        return None
    if isinstance(spec, (str, Path)):
        return ensure_model_path(spec)
    path = getattr(spec, "path", None)
    if path is not None:
        return ensure_model_path(path)
    if required:
        raise ValueError("Detector benchmark workflows require a detector with a resolvable .path.")
    return None


def reid_path_from_spec(spec: Any, *, required: bool = True) -> Path | None:
    if spec is None:
        if required:
            raise ValueError("A ReID model path is required for this operation.")
        return None
    if isinstance(spec, (str, Path)):
        return ensure_model_path(spec)
    path = getattr(spec, "path", None) or getattr(spec, "weights", None)
    if path is not None:
        return ensure_model_path(path)
    if required:
        raise ValueError("This operation requires a ReID model with a resolvable .path or .weights.")
    return None


def tracker_name_from_spec(spec: Any, *, required: bool = True) -> str | None:
    if spec is None:
        if required:
            raise ValueError("A tracker is required.")
        return None
    if isinstance(spec, str):
        name = spec.lower()
        if name in TRACKER_MAPPING:
            return name
    class_name = spec.__class__.__name__.lower() if spec is not None else ""
    if class_name in TRACKER_CLASS_TO_NAME:
        return TRACKER_CLASS_TO_NAME[class_name]
    if required:
        raise ValueError("Could not infer a registered tracker name from the provided tracker spec.")
    return None


def tracker_config_from_spec(spec: Any) -> dict[str, Any] | None:
    if isinstance(spec, str) or spec is None:
        return None

    tracker_name = tracker_name_from_spec(spec, required=False)
    if tracker_name is None:
        return None

    with open(get_tracker_config(tracker_name), "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    resolved: dict[str, Any] = {}
    for key, details in config.items():
        if hasattr(spec, key):
            resolved[key] = getattr(spec, key)
        else:
            resolved[key] = details.get("default")
    return resolved


def build_detector_from_spec(
    spec: Any,
    *,
    classes: list[int] | None = None,
    device: str = BOXMOT_DEFAULTS.track.device,
    imgsz=None,
    conf=None,
    iou: float = BOXMOT_DEFAULTS.track.iou,
):
    from boxmot.detectors import Detector as PublicDetector

    if isinstance(spec, (str, Path)):
        return PublicDetector(
            path=ensure_model_path(spec),
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            classes=classes,
        )

    current_device = getattr(spec, "device", None)
    if current_device is not None and str(current_device) != str(device):
        raise ValueError(
            f"Detector instance is already bound to device '{current_device}'. "
            f"Create it on '{device}' or pass a path/string detector spec instead."
        )

    if imgsz is not None and hasattr(spec, "imgsz"):
        spec.imgsz = imgsz
    if conf is not None and hasattr(spec, "conf"):
        spec.conf = float(conf)
    if iou is not None and hasattr(spec, "iou"):
        spec.iou = float(iou)
    if classes is not None and hasattr(spec, "classes"):
        spec.classes = classes
    return spec


def build_reid_from_spec(
    spec: Any,
    *,
    device: str = BOXMOT_DEFAULTS.track.device,
    half: bool = BOXMOT_DEFAULTS.track.half,
):
    from boxmot.reid import ReID as PublicReID

    if spec is None:
        return None

    if isinstance(spec, (str, Path)):
        return PublicReID(ensure_model_path(spec), device=device, half=half)

    current_device = getattr(spec, "device", None)
    if current_device is not None and str(current_device) != str(device):
        raise ValueError(
            f"ReID instance is already bound to device '{current_device}'. "
            f"Create it on '{device}' or pass a path/string ReID spec instead."
        )
    return spec


def build_tracker_from_spec(
    spec: Any,
    *,
    device: str = BOXMOT_DEFAULTS.track.device,
    half: bool = BOXMOT_DEFAULTS.track.half,
    reid_weights=None,
):
    if not isinstance(spec, str):
        return spec

    tracker_name = tracker_name_from_spec(spec, required=True)
    return create_tracker(
        tracker_type=tracker_name,
        tracker_config=get_tracker_config(tracker_name),
        reid_weights=reid_weights,
        device=select_device(device),
        half=half,
        per_class=False,
    )


def build_tracker_with_reid_spec(
    tracker_spec: Any,
    tracker: Any,
    *,
    device: str = BOXMOT_DEFAULTS.track.device,
    half: bool = BOXMOT_DEFAULTS.track.half,
    build_reid_fn,
):
    if isinstance(tracker_spec, str):
        tracker_name = tracker_name_from_spec(tracker_spec, required=False)
        if tracker_name in REID_TRACKERS:
            if hasattr(tracker, "with_reid") and not bool(getattr(tracker, "with_reid")):
                return None

            tracker_backend = getattr(tracker, "reid_model", None) or getattr(tracker, "model", None)
            if tracker_backend is not None:
                return TrackerReIDAdapter(tracker_backend)

    return build_reid_fn(device=device, half=half)


def base_eval_args(
    api: Any,
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
):
    reid_path = api._reid_path(required=False) or BOXMOT_DEFAULTS.shared.reid
    tracker_spec = api.tracker
    per_class = bool(getattr(tracker_spec, "per_class", False)) if not isinstance(tracker_spec, str) else False

    args = build_mode_namespace(
        "eval",
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
        },
        explicit_keys={
            *({"detector"} if api._detector_explicit else set()),
            *({"reid"} if api._reid_explicit else set()),
            *({"tracker"} if api._tracker_explicit else set()),
            *({"device"} if device != BOXMOT_DEFAULTS.eval.device else set()),
            *({"half"} if bool(half) != bool(BOXMOT_DEFAULTS.eval.half) else set()),
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


def run_validation_pipeline(
    api: Any,
    *,
    benchmark: str | Path,
    imgsz=None,
    conf=None,
    iou: float = BOXMOT_DEFAULTS.eval.iou,
    device: str = BOXMOT_DEFAULTS.eval.device,
    half: bool = BOXMOT_DEFAULTS.eval.half,
    project: str | Path | None = None,
    verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
    show_progress: bool = True,
    postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
    evolve_config: dict[str, Any] | None = None,
) -> ValidationResult:
    evaluator = import_module("boxmot.engine.evaluator")
    replay = import_module("boxmot.engine.replay")
    args = api._base_eval_args(
        benchmark,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        project=project,
        verbose=verbose,
        show_progress=show_progress,
        postprocessing=postprocessing,
    )

    timing_stats = TimingStats()
    evaluator.eval_setup(args)
    evaluator.run_generate_dets_embs(args, timing_stats=timing_stats)
    tracker_config = evolve_config if evolve_config is not None else api._tracker_config_from_spec()
    replay.run_generate_mot_results(
        args,
        evolve_config=tracker_config,
        timing_stats=timing_stats,
        quiet=not show_progress,
    )
    raw_results = evaluator.run_trackeval(args, verbose=verbose)
    summary_label, summary = extract_summary(raw_results)

    return ValidationResult(
        benchmark=str(benchmark),
        raw=raw_results,
        summary_label=summary_label,
        summary=summary,
        exp_dir=getattr(args, "exp_dir", None),
        timings=timing_summary_from_stats(timing_stats),
        args=args,
    )


def load_tracker_search_space(api: Any) -> dict[str, Any]:
    tracker_name = api._tracker_name(required=True)
    with open(get_tracker_config(tracker_name), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def default_tracker_config(api: Any) -> dict[str, Any]:
    existing = api._tracker_config_from_spec()
    if existing is not None:
        return existing

    search_space = api._load_tracker_search_space()
    return {
        key: details.get("default")
        for key, details in search_space.items()
    }


def sample_param(spec: dict[str, Any], rng: random.Random):
    param_type = str(spec.get("type", "choice")).lower()

    if param_type == "uniform":
        low, high = spec["range"]
        return float(rng.uniform(float(low), float(high)))

    if param_type == "loguniform":
        low, high = spec["range"]
        return float(math.exp(rng.uniform(math.log(float(low)), math.log(float(high)))))

    if param_type == "randint":
        low, high = spec["range"]
        return int(rng.randint(int(low), int(high)))

    if param_type == "qrandint":
        low, high, step = spec["range"]
        choices = list(range(int(low), int(high), int(step)))
        return int(rng.choice(choices))

    if param_type in {"choice", "grid_search"}:
        options = spec.get("options") or spec.get("values") or []
        if not options:
            return spec.get("default")
        return rng.choice(list(options))

    return spec.get("default")


def iter_tune_configs(api: Any, n_trials: int, rng: random.Random) -> Iterator[dict[str, Any]]:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    search_space = api._load_tracker_search_space()
    yield api._default_tracker_config()

    for _ in range(n_trials - 1):
        yield {
            key: api._sample_param(details, rng)
            for key, details in search_space.items()
        }


def score_summary(
    summary: dict[str, Any],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> tuple[float, ...]:
    score: list[float] = []
    for metric in maximize:
        score.append(float(summary.get(metric, float("-inf"))))
    for metric in minimize:
        score.append(-float(summary.get(metric, float("inf"))))
    return tuple(score)


def resolve_track_output_dir(project: Path, source: Any) -> Path:
    base = project / "track" / resolve_output_stem(source)
    return increment_path(base, mkdir=True)


def resolve_output_fps(source: Any, *, fallback: float = 30.0, cv2_module=cv2) -> float:
    if isinstance(source, (str, Path)):
        source_str = str(source)
        if source_str.isdigit() or "://" in source_str:
            return fallback
        path = Path(source_str)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            capture = cv2_module.VideoCapture(str(path))
            try:
                fps = capture.get(cv2_module.CAP_PROP_FPS)
            finally:
                capture.release()
            if fps and fps > 0:
                return float(fps)
    return fallback


def save_video(results: Results, video_path: Path, fps: float, *, cv2_module=cv2) -> Path:
    frames = results.materialize()
    if not frames:
        return video_path

    height, width = frames[0].frame.shape[:2]
    writer = cv2_module.VideoWriter(
        str(video_path),
        cv2_module.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for track_result in frames:
            writer.write(track_result.render())
    finally:
        writer.release()
    return video_path


def run_export_pipeline(
    api: Any,
    *,
    include: Sequence[str],
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
) -> ExportResult:
    export_module = import_module("boxmot.engine.export")
    weights = api._reid_path(required=True)
    args = build_mode_namespace(
        "export",
        {
            "weights": weights,
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
    model, dummy_input = export_module.setup_model(args)
    export_tasks = export_module.create_export_tasks(args, model, dummy_input)
    files = export_module.perform_exports(export_tasks)
    return ExportResult(weights=args.weights, files=files)
