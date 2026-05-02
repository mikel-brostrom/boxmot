from __future__ import annotations

import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import cv2
import yaml

from boxmot.configs import BOXMOT_DEFAULTS
from boxmot.data import VIDEO_EXTS
from boxmot.detectors import Detector as PublicDetector
from boxmot.engine.results import Results
from boxmot.native import get_native_live_backend
from boxmot.reid import ReID as PublicReID
from boxmot.trackers.specs import normalize_tracker_backend, parse_tracker_spec
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING, create_tracker, get_tracker_config
from boxmot.utils import configure_logging as _configure_boxmot_logging, logger as LOGGER
from boxmot.utils.misc import increment_path, resolve_model_path
from boxmot.utils.torch_utils import select_device

REID_TRACKERS = {"strongsort", "botsort", "deepocsort", "hybridsort", "boosttrack"}
TRACKER_CLASS_TO_NAME = {
    class_path.rsplit(".", 1)[-1].lower(): tracker_name
    for tracker_name, class_path in TRACKER_MAPPING.items()
}


def normalize_classes(classes: Any) -> list[int] | None:
    if classes is None:
        return None
    if isinstance(classes, str):
        parts = [part for part in re.split(r"[\s,]+", classes.strip()) if part]
        return [int(part) for part in parts]
    if isinstance(classes, int):
        return [int(classes)]
    return [int(value) for value in classes]


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


def resolve_track_output_dir(project: Path, source: Any) -> Path:
    base = project / "track" / resolve_output_stem(source)
    return increment_path(base, mkdir=True)


def compare_scores(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return left > right


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


def TrackerReIDAdapter(backend: Any):
    """Reuse a tracker-owned ReID backend through the standard ReID stage hooks.

    Returns a :class:`boxmot.reid.ReID` runtime that wraps ``backend`` directly
    without reloading weights, so timing breakdowns can attribute work to
    ``preprocess`` / ``process`` / ``postprocess``.
    """
    return PublicReID.from_backend(backend)


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

    try:
        parsed = parse_tracker_spec(spec, class_to_name=TRACKER_CLASS_TO_NAME)
    except ValueError:
        parsed = None
    if parsed is not None and parsed.name in TRACKER_MAPPING:
        return parsed.name

    if required:
        raise ValueError("Could not infer a registered tracker name from the provided tracker spec.")
    return None


def tracker_backend_from_spec(spec: Any, *, required: bool = True) -> str | None:
    if spec is None:
        if required:
            raise ValueError("A tracker is required.")
        return None

    try:
        parsed = parse_tracker_spec(spec, class_to_name=TRACKER_CLASS_TO_NAME)
    except ValueError:
        if required:
            raise
        return None
    return parsed.backend


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


def load_tracker_search_space(tracker_spec: Any) -> dict[str, Any]:
    tracker_name = tracker_name_from_spec(tracker_spec, required=True)
    with open(get_tracker_config(tracker_name), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def default_tracker_config(tracker_spec: Any) -> dict[str, Any]:
    existing = tracker_config_from_spec(tracker_spec)
    if existing is not None:
        return existing
    search_space = load_tracker_search_space(tracker_spec)
    return {
        key: details.get("default")
        for key, details in search_space.items()
    }


def sample_param(spec: dict[str, Any], rng) -> Any:
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


def build_detector_from_spec(
    spec: Any,
    *,
    classes: list[int] | None = None,
    device: str = BOXMOT_DEFAULTS.track.device,
    imgsz=None,
    conf=None,
    iou: float = BOXMOT_DEFAULTS.track.iou,
):
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
    tracker_backend: str | None = None,
    reid_weights=None,
    reid_preprocess: str | None = None,
):
    if not isinstance(spec, str):
        return spec

    tracker_name = tracker_name_from_spec(spec, required=True)
    resolved_backend = tracker_backend_from_spec(spec, required=False)
    if tracker_backend is not None:
        resolved_backend = normalize_tracker_backend(
            tracker_backend,
            default=resolved_backend or "python",
        )
    if resolved_backend == "cpp":
        native_backend = get_native_live_backend(tracker_name)
        return native_backend.create_tracker(
            default_tracker_config(spec),
            reid_weights=reid_weights,
            reid_preprocess=reid_preprocess,
        )

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
    reid_spec: Any,
    *,
    device: str = BOXMOT_DEFAULTS.track.device,
    half: bool = BOXMOT_DEFAULTS.track.half,
):
    tracker_name = tracker_name_from_spec(tracker_spec, required=False)
    if tracker_name not in REID_TRACKERS:
        return None

    if tracker_name in REID_TRACKERS:
        if hasattr(tracker, "with_reid") and not bool(getattr(tracker, "with_reid")):
            return None

        if bool(getattr(tracker, "provides_reid", False)):
            return None

        tracker_backend = getattr(tracker, "reid_model", None) or getattr(tracker, "model", None)
        if tracker_backend is not None:
            return TrackerReIDAdapter(tracker_backend)

    return build_reid_from_spec(reid_spec, device=device, half=half)


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


__all__ = (
    "REID_TRACKERS",
    "TRACKER_CLASS_TO_NAME",
    "TrackerReIDAdapter",
    "build_detector_from_spec",
    "build_reid_from_spec",
    "build_tracker_from_spec",
    "build_tracker_with_reid_spec",
    "compare_scores",
    "default_tracker_config",
    "detector_path_from_spec",
    "ensure_model_path",
    "load_tracker_search_space",
    "normalize_classes",
    "reid_path_from_spec",
    "resolve_output_fps",
    "resolve_output_stem",
    "resolve_track_output_dir",
    "sample_param",
    "save_video",
    "score_summary",
    "suppress_boxmot_logs",
    "tracker_backend_from_spec",
    "tracker_config_from_spec",
    "tracker_name_from_spec",
)
