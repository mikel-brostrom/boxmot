"""Shared configuration policy for engine-owned tracking runtime modes."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import yaml

from boxmot.configs import CONFIG_ROOT
from boxmot.trackers.specs import parse_tracker_spec

RUNTIME_MODES = frozenset({"track", "materialize", "time-variant", "eval", "tune", "research"})
RUNTIME_DEFAULTS_PATH = CONFIG_ROOT / "runtime.yaml"


def _load_mode_defaults() -> dict[str, Any]:
    with open(RUNTIME_DEFAULTS_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _merged_mode_defaults(mode: str) -> dict[str, Any]:
    normalized_mode = str(mode).lower()
    if normalized_mode not in RUNTIME_MODES:
        available = ", ".join(sorted(RUNTIME_MODES))
        raise ValueError(f"Unknown runtime mode {mode!r}; expected one of: {available}")
    raw_defaults = _load_mode_defaults()

    defaults = deepcopy(raw_defaults.get("shared", {}))
    defaults.update(deepcopy(raw_defaults.get("runtime", {})))
    defaults.update(deepcopy(raw_defaults.get(normalized_mode, {})))
    return defaults


def _resolve_default_value(key: str, value: Any) -> Any:
    if key == "sequence_workers" and str(value).lower() == "auto":
        return min(8, max(1, os.cpu_count() or 1))

    if key == "project" and value is not None:
        return Path(value)

    return value


def _normalize_classes(classes: Any) -> list[int] | None:
    if classes is None:
        return None
    if isinstance(classes, str):
        parts = [part for part in classes.replace(",", " ").split() if part]
        return [int(part) for part in parts]
    if isinstance(classes, int):
        return [int(classes)]
    return [int(value) for value in classes]


def get_mode_defaults(mode: str) -> dict[str, Any]:
    """Return normalized merged defaults for a CLI/Python API mode."""
    return {key: _resolve_default_value(key, value) for key, value in _merged_mode_defaults(mode).items()}


def get_mode_default(mode: str, key: str, default: Any = None) -> Any:
    """Return a normalized default value for a CLI/Python API mode."""
    defaults = get_mode_defaults(mode)
    if key not in defaults:
        return default
    return defaults[key]


def build_mode_namespace(
    mode: str,
    payload: Mapping[str, Any],
    *,
    explicit_keys: Iterable[str] | None = None,
):
    """Build a normalized SimpleNamespace for CLI and Python API workflows."""
    normalized_mode = str(mode).lower()
    explicit = set(explicit_keys or ())

    values = get_mode_defaults(normalized_mode)
    values.update(dict(payload))

    if normalized_mode == "materialize":
        allowed_keys = frozenset(
            {
                "build_root",
                "data_root",
                "device",
                "experiment",
                "fps",
                "plan_overrides",
                "plan_path",
                "publish_embeddings",
                "publish_image_refs",
                "publish_masks",
                "resume",
            }
        )
        values = {key: value for key, value in values.items() if key in allowed_keys}
        for path_key in ("data_root", "build_root", "plan_path"):
            if values.get(path_key) is not None:
                values[path_key] = Path(values[path_key])
        values["materialize_explicit_keys"] = tuple(sorted(explicit & allowed_keys))
    elif normalized_mode == "time-variant":
        allowed_keys = frozenset({"dataset", "split", "sequence", "build", "build_root", "data_root", "name", "seed"})
        values = {key: value for key, value in values.items() if key in allowed_keys}
        for path_key in ("data_root", "build_root"):
            if values.get(path_key) is not None:
                values[path_key] = Path(values[path_key])
    elif normalized_mode in RUNTIME_MODES:
        if normalized_mode == "track":
            values["detector"] = values.get("detector", DEFAULT_DETECTOR)
            values["reid"] = values.get("reid", DEFAULT_REID)
        else:
            values.pop("detector", None)
            values.pop("reid", None)
        tracker_spec = parse_tracker_spec(
            values.get("tracker") or get_mode_default(normalized_mode, "tracker"),
            default_backend=str(values.get("tracker_backend", "python")),
        )
        values["tracker"] = tracker_spec.name
        values["tracker_backend"] = tracker_spec.backend
        if normalized_mode == "track":
            values["classes"] = _normalize_classes(values.get("classes"))
        else:
            values.pop("classes", None)
        values["project"] = Path(values.get("project") or "runs")
        if normalized_mode == "track":
            values.setdefault("detector_explicit", "detector" in explicit)
            values.setdefault("reid_explicit", "reid" in explicit)
        values.setdefault("tracker_explicit", "tracker" in explicit)
        values.setdefault("tracker_backend_explicit", "tracker_backend" in explicit)
        values.setdefault("device_explicit", "device" in explicit)
        values.setdefault("half_explicit", "half" in explicit)
        values.setdefault("split_explicit", "split" in explicit)

    return SimpleNamespace(**values)


DEFAULT_DETECTOR = get_mode_default("track", "detector")
DEFAULT_REID = get_mode_default("track", "reid")


def _runtime_mode_kwargs(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "imgsz": values.get("imgsz"),
        "fps": values.get("fps"),
        "conf": values.get("conf"),
        "iou": float(values.get("iou", 0.7)),
        "device": str(values.get("device", "cpu")),
        "batch_size": int(values.get("batch_size", 1)),
        "auto_batch": bool(values.get("auto_batch", True)),
        "resume": bool(values.get("resume", True)),
        "sequence_workers": int(values.get("sequence_workers", 1)),
        "project": Path(values.get("project") or "runs"),
        "name": str(values.get("name", "exp")),
        "exist_ok": bool(values.get("exist_ok", False)),
        "half": bool(values.get("half", False)),
        "vid_stride": int(values.get("vid_stride", 1)),
        "ci": bool(values.get("ci", False)),
        "tracker": str(values.get("tracker", "bytetrack")),
        "tracker_backend": str(values.get("tracker_backend", "python")),
        "verbose": bool(values.get("verbose", False)),
        "show_timing": bool(values.get("show_timing", False)),
        "agnostic_nms": bool(values.get("agnostic_nms", False)),
        "show": bool(values.get("show", False)),
        "show_labels": bool(values.get("show_labels", True)),
        "show_conf": bool(values.get("show_conf", True)),
        "show_trajectories": bool(values.get("show_trajectories", False)),
        "show_kf_preds": bool(values.get("show_kf_preds", False)),
        "save_txt": bool(values.get("save_txt", False)),
        "save_crop": bool(values.get("save_crop", False)),
        "save": bool(values.get("save", False)),
        "line_width": values.get("line_width"),
        "per_class": bool(values.get("per_class", False)),
        "target_id": values.get("target_id"),
    }


@dataclass(frozen=True, slots=True)
class SharedModeDefaults:
    detector: str | Path
    reid: str | Path


@dataclass(frozen=True, slots=True)
class RuntimeModeDefaults:
    imgsz: Any
    fps: float | None
    conf: float | None
    iou: float
    device: str
    batch_size: int
    auto_batch: bool
    resume: bool
    sequence_workers: int
    project: Path
    name: str
    exist_ok: bool
    half: bool
    vid_stride: int
    ci: bool
    tracker: str
    tracker_backend: str
    verbose: bool
    show_timing: bool
    agnostic_nms: bool
    show: bool
    show_labels: bool
    show_conf: bool
    show_trajectories: bool
    show_kf_preds: bool
    save_txt: bool
    save_crop: bool
    save: bool
    line_width: int | None
    per_class: bool
    target_id: int | None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RuntimeModeDefaults":
        return cls(**_runtime_mode_kwargs(values))


@dataclass(frozen=True, slots=True)
class TrackModeDefaults(RuntimeModeDefaults):
    source: str
    benchmark: str
    split: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrackModeDefaults":
        return cls(
            **_runtime_mode_kwargs(values),
            source=str(values.get("source", "0")),
            benchmark=str(values.get("benchmark", "")),
            split=str(values.get("split", "")),
        )


@dataclass(frozen=True, slots=True)
class MaterializeModeDefaults(RuntimeModeDefaults):
    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MaterializeModeDefaults":
        return cls(**_runtime_mode_kwargs(values))


@dataclass(frozen=True, slots=True)
class EvalModeDefaults(RuntimeModeDefaults):
    experiment: str | None
    dataset: str | None
    source: str | None
    benchmark: str
    split: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EvalModeDefaults":
        experiment = values.get("experiment")
        dataset = values.get("dataset")
        source = values.get("source")
        return cls(
            **_runtime_mode_kwargs(values),
            experiment=None if experiment is None else str(experiment),
            dataset=None if dataset is None else str(dataset),
            source=None if source is None else str(source),
            benchmark=str(values.get("benchmark", "")),
            split=str(values.get("split", "")),
        )


@dataclass(frozen=True, slots=True)
class TuneModeDefaults(RuntimeModeDefaults):
    experiment: str | None
    source: str | None
    benchmark: str
    split: str
    n_trials: int
    objectives: tuple[str, ...]
    maximize: tuple[str, ...]
    minimize: tuple[str, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TuneModeDefaults":
        experiment = values.get("experiment")
        source = values.get("source")
        objectives = tuple(values.get("objectives") or ())
        return cls(
            **_runtime_mode_kwargs(values),
            experiment=None if experiment is None else str(experiment),
            source=None if source is None else str(source),
            benchmark=str(values.get("benchmark", "")),
            split=str(values.get("split", "")),
            n_trials=int(values.get("n_trials", 10)),
            objectives=objectives,
            maximize=tuple(values.get("maximize") or objectives or ("HOTA",)),
            minimize=tuple(values.get("minimize") or ()),
        )


@dataclass(frozen=True, slots=True)
class ResearchModeDefaults(RuntimeModeDefaults):
    experiment: str | None
    source: str | None
    benchmark: str
    split: str
    proposal_model: str
    proposal_api_key: str | None
    proposal_api_key_env: str | None
    max_metric_calls: int
    eval_timeout: float
    keep_workspace: bool
    hota_penalty: float
    idf1_penalty: float
    mota_penalty: float
    hota_tolerance: float
    idf1_tolerance: float
    mota_tolerance: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ResearchModeDefaults":
        experiment = values.get("experiment")
        source = values.get("source")
        return cls(
            **_runtime_mode_kwargs(values),
            experiment=None if experiment is None else str(experiment),
            source=None if source is None else str(source),
            benchmark=str(values.get("benchmark", "")),
            split=str(values.get("split", "")),
            proposal_model=str(values.get("proposal_model", "openai/gpt-5.4")),
            proposal_api_key=(
                None if values.get("proposal_api_key") in {None, ""} else str(values.get("proposal_api_key"))
            ),
            proposal_api_key_env=(
                None if values.get("proposal_api_key_env") in {None, ""} else str(values.get("proposal_api_key_env"))
            ),
            max_metric_calls=int(values.get("max_metric_calls", 24)),
            eval_timeout=float(values.get("eval_timeout", 900.0)),
            keep_workspace=bool(values.get("keep_workspace", False)),
            hota_penalty=float(values.get("hota_penalty", 0.0)),
            idf1_penalty=float(values.get("idf1_penalty", 1.0)),
            mota_penalty=float(values.get("mota_penalty", 1.0)),
            hota_tolerance=float(values.get("hota_tolerance", 0.0)),
            idf1_tolerance=float(values.get("idf1_tolerance", 0.0)),
            mota_tolerance=float(values.get("mota_tolerance", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class BoxMOTDefaults:
    shared: SharedModeDefaults
    track: TrackModeDefaults
    materialize: MaterializeModeDefaults
    eval: EvalModeDefaults
    tune: TuneModeDefaults
    research: ResearchModeDefaults


BOXMOT_DEFAULTS = BoxMOTDefaults(
    shared=SharedModeDefaults(detector=DEFAULT_DETECTOR, reid=DEFAULT_REID),
    track=TrackModeDefaults.from_mapping(get_mode_defaults("track")),
    materialize=MaterializeModeDefaults.from_mapping(get_mode_defaults("materialize")),
    eval=EvalModeDefaults.from_mapping(get_mode_defaults("eval")),
    tune=TuneModeDefaults.from_mapping(get_mode_defaults("tune")),
    research=ResearchModeDefaults.from_mapping(get_mode_defaults("research")),
)

__all__ = (
    "BOXMOT_DEFAULTS",
    "BoxMOTDefaults",
    "DEFAULT_DETECTOR",
    "DEFAULT_REID",
    "EvalModeDefaults",
    "MaterializeModeDefaults",
    "RUNTIME_DEFAULTS_PATH",
    "ResearchModeDefaults",
    "RuntimeModeDefaults",
    "SharedModeDefaults",
    "TrackModeDefaults",
    "TuneModeDefaults",
    "build_mode_namespace",
    "get_mode_default",
    "get_mode_defaults",
)
