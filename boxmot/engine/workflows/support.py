from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import yaml

from boxmot.configs import BOXMOT_DEFAULTS
from boxmot.data import VIDEO_EXTS
from boxmot.detectors import Detector as PublicDetector
from boxmot.engine.tuning.search_space import flatten_yaml_config
from boxmot.native import get_native_live_backend
from boxmot.reid import ReID as PublicReID
from boxmot.trackers.specs import normalize_tracker_backend, parse_tracker_spec
from boxmot.trackers.registry import (
    REID_TRACKERS as REGISTERED_REID_TRACKERS,
    TRACKER_CLASS_TO_NAME,
    TRACKER_MAPPING,
    create_tracker,
    get_tracker_config,
)
from boxmot.utils.misc import increment_path, resolve_model_path
from boxmot.utils.torch_utils import select_device

REID_TRACKERS = set(REGISTERED_REID_TRACKERS)


def normalize_classes(classes: Any) -> list[int] | None:
    if classes is None:
        return None
    if isinstance(classes, str):
        parts = [part for part in re.split(r"[\s,]+", classes.strip()) if part]
        return [int(part) for part in parts]
    if isinstance(classes, int):
        return [int(classes)]
    return [int(value) for value in classes]


def normalize_class_names(class_names: Any) -> dict[int, str]:
    if class_names is None:
        return {}
    if isinstance(class_names, Mapping):
        return {int(class_id): str(name) for class_id, name in class_names.items()}
    if isinstance(class_names, Sequence) and not isinstance(class_names, (str, bytes)):
        return {int(class_id): str(name) for class_id, name in enumerate(class_names)}
    return {}


def _component_kwargs(kwargs: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if kwargs is None:
        return {}
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"{name} must be a mapping of keyword arguments.")
    return dict(kwargs)


def tracker_reid_model_from_spec(spec: Any) -> Any:
    """Return a tracker-compatible ReID model from an initialized ReID spec."""

    if spec is None or isinstance(spec, (str, Path)):
        return None
    if hasattr(spec, "get_features"):
        return spec
    model = getattr(spec, "model", None)
    if model is not None and hasattr(model, "get_features"):
        return model
    return None


def tracker_class_metadata_from_detector_config(
    detector_cfg: dict | None,
) -> tuple[tuple[int, ...] | None, dict[int, str] | None]:
    if not detector_cfg:
        return None, None

    class_names = normalize_class_names(detector_cfg.get("classes"))
    if not class_names:
        return None, None

    class_ids = tuple(sorted(class_names))
    return class_ids, class_names


def tracker_class_metadata_from_detector(detector: Any) -> tuple[tuple[int, ...] | None, dict[int, str] | None]:
    class_names = normalize_class_names(getattr(detector, "names", None))
    if not class_names:
        backend = getattr(detector, "backend", None)
        class_names = normalize_class_names(getattr(backend, "names", None))
    if not class_names:
        return None, None
    return tuple(sorted(class_names)), class_names


def resolve_tracker_class_metadata(
    args: Any,
    detector: Any = None,
) -> tuple[tuple[int, ...] | None, dict[int, str] | None]:
    class_ids, class_names = tracker_class_metadata_from_detector_config(getattr(args, "dataset_detector_cfg", None))
    if class_names is None and detector is not None:
        class_ids, class_names = tracker_class_metadata_from_detector(detector)

    selected_classes = normalize_classes(getattr(args, "classes", None))
    if selected_classes is not None:
        class_ids = tuple(sorted(set(selected_classes)))

    return class_ids, class_names


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
    flat_config = flatten_yaml_config(config)

    resolved: dict[str, Any] = {}
    for key, details in flat_config.items():
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
    flat_search_space = flatten_yaml_config(search_space)
    return {
        key: details.get("default")
        for key, details in flat_search_space.items()
    }



def build_detector_from_spec(
    spec: Any,
    *,
    classes: list[int] | None = None,
    device: str = BOXMOT_DEFAULTS.track.device,
    imgsz=None,
    conf=None,
    iou: float = BOXMOT_DEFAULTS.track.iou,
    detector_kwargs: Mapping[str, Any] | None = None,
):
    runtime_kwargs = _component_kwargs(detector_kwargs, "detector_kwargs")
    if isinstance(spec, (str, Path)):
        legacy_kwargs = sorted(set(runtime_kwargs) & {"conf", "imgsz"})
        if legacy_kwargs:
            names = ", ".join(legacy_kwargs)
            raise ValueError(
                f"Unsupported detector_kwargs: {names}. Use confidence=... and image_size=... instead."
            )
        resolved_device = runtime_kwargs.pop("device", device)
        resolved_imgsz = runtime_kwargs.pop("image_size", imgsz)
        resolved_conf = runtime_kwargs.pop("confidence", conf)
        resolved_iou = runtime_kwargs.pop("iou", iou)
        resolved_classes = runtime_kwargs.pop("classes", classes)
        half = runtime_kwargs.pop("half", None)
        detector = PublicDetector(
            path=ensure_model_path(spec),
            device=resolved_device,
            imgsz=resolved_imgsz,
            conf=resolved_conf,
            iou=resolved_iou,
            classes=resolved_classes,
            **runtime_kwargs,
        )
        if half is not None:
            detector.half = bool(half)
        return detector

    if runtime_kwargs:
        raise ValueError("detector_kwargs cannot be supplied with an initialized detector.")

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
    reid_kwargs: Mapping[str, Any] | None = None,
):
    runtime_kwargs = _component_kwargs(reid_kwargs, "reid_kwargs")
    if spec is None:
        if runtime_kwargs:
            raise ValueError("reid_kwargs cannot be supplied when ReID is disabled.")
        return None

    if isinstance(spec, (str, Path)):
        if "preprocess_name" in runtime_kwargs:
            raise ValueError("Unsupported reid_kwargs: preprocess_name. Use preprocess=... instead.")
        resolved_device = runtime_kwargs.pop("device", device)
        resolved_half = runtime_kwargs.pop("half", half)
        preprocess_name = runtime_kwargs.pop("preprocess", None)
        if runtime_kwargs:
            unknown = ", ".join(sorted(runtime_kwargs))
            raise ValueError(f"Unknown reid_kwargs: {unknown}")
        return PublicReID(
            ensure_model_path(spec),
            device=resolved_device,
            half=resolved_half,
            preprocess_name=preprocess_name,
        )

    if runtime_kwargs:
        raise ValueError("reid_kwargs cannot be supplied with an initialized ReID model.")

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
    reid_model=None,
    reid_preprocess: str | None = None,
    class_ids: tuple[int, ...] | None = None,
    class_names: dict[int, str] | None = None,
    tracker_kwargs: Mapping[str, Any] | None = None,
):
    runtime_kwargs = _component_kwargs(tracker_kwargs, "tracker_kwargs")
    if not isinstance(spec, (str, type)):
        if runtime_kwargs:
            raise ValueError("tracker_kwargs cannot be supplied with an initialized tracker.")
        if hasattr(spec, "configure_class_catalog") and (class_ids is not None or class_names is not None):
            spec.configure_class_catalog(class_ids=class_ids, class_names=class_names)
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
        native_kwargs: dict[str, Any] = {
            "reid_weights": reid_weights,
            "reid_preprocess": reid_preprocess,
        }
        # Forward device to native backends that support ReID device selection.
        import inspect
        sig = inspect.signature(native_backend.create_tracker)
        if "reid_device" in sig.parameters:
            native_kwargs["reid_device"] = str(device) if device else None
        native_config = default_tracker_config(spec)
        native_config.update(runtime_kwargs)
        return native_backend.create_tracker(
            native_config,
            **native_kwargs,
        )

    return create_tracker(
        tracker_type=tracker_name,
        tracker_config=get_tracker_config(tracker_name),
        reid_weights=reid_weights,
        device=select_device(device),
        half=half,
        per_class=False,
        class_ids=class_ids,
        class_names=class_names,
        tracker_kwargs=runtime_kwargs,
        reid_model=reid_model,
    )


def build_tracker_with_reid_spec(
    tracker_spec: Any,
    tracker: Any,
    reid_spec: Any,
    *,
    device: str = BOXMOT_DEFAULTS.track.device,
    half: bool = BOXMOT_DEFAULTS.track.half,
    reid_kwargs: Mapping[str, Any] | None = None,
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

    return build_reid_from_spec(reid_spec, device=device, half=half, reid_kwargs=reid_kwargs)


def resolve_output_fps(source: Any, *, fallback: float = 30.0, cv2_module=cv2) -> float:
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap_id = int(source) if isinstance(source, str) else source
        capture = cv2_module.VideoCapture(cap_id)
        try:
            fps = capture.get(cv2_module.CAP_PROP_FPS)
        finally:
            capture.release()
        if fps and fps > 0:
            return float(fps)
        return fallback
    if isinstance(source, (str, Path)):
        source_str = str(source)
        if "://" in source_str:
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


__all__ = (
    "REID_TRACKERS",
    "TRACKER_CLASS_TO_NAME",
    "TrackerReIDAdapter",
    "build_detector_from_spec",
    "build_reid_from_spec",
    "build_tracker_from_spec",
    "build_tracker_with_reid_spec",
    "default_tracker_config",
    "detector_path_from_spec",
    "ensure_model_path",
    "load_tracker_search_space",
    "normalize_classes",
    "normalize_class_names",
    "reid_path_from_spec",
    "resolve_tracker_class_metadata",
    "resolve_output_fps",
    "resolve_output_stem",
    "resolve_track_output_dir",
    "tracker_reid_model_from_spec",
    "tracker_backend_from_spec",
    "tracker_config_from_spec",
    "tracker_name_from_spec",
)
