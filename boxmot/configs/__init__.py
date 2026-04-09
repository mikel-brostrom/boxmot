from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import yaml

from boxmot.utils import WEIGHTS
from boxmot.utils.compat import dataclass_slots_kwargs
from boxmot.utils.misc import resolve_model_path

RUNTIME_MODES = frozenset({"track", "generate", "eval", "tune", "research"})
MODE_DEFAULTS_PATH = Path(__file__).resolve().parent / "modes.yaml"


def _load_mode_defaults() -> dict[str, Any]:
    with open(MODE_DEFAULTS_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _merged_mode_defaults(mode: str) -> dict[str, Any]:
    normalized_mode = str(mode).lower()
    raw_defaults = _load_mode_defaults()

    defaults = deepcopy(raw_defaults.get("shared", {}))
    if normalized_mode in RUNTIME_MODES:
        defaults.update(deepcopy(raw_defaults.get("runtime", {})))
    defaults.update(deepcopy(raw_defaults.get(normalized_mode, {})))
    return defaults


def _resolve_default_value(key: str, value: Any) -> Any:
    if key == "n_threads" and str(value).lower() == "auto":
        return min(8, max(1, os.cpu_count() or 1))

    if key in {"detector", "reid", "weights"} and value is not None:
        return ensure_model_extension(value)

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


def _normalize_model_list(values: Any, *, multiple: bool) -> Any:
    if multiple:
        if values is None:
            return []
        if isinstance(values, (list, tuple)):
            return [ensure_model_extension(value) for value in values]
        return [ensure_model_extension(values)]

    if values is None:
        return None
    return ensure_model_extension(values)


def ensure_model_extension(model_path: str | Path, default_dir: Path = WEIGHTS) -> Path:
    """Preserve explicit paths and resolve bare model names into the shared weights directory."""
    path = Path(model_path)
    if not path.suffix:
        path = path.with_suffix(".pt")

    if not path.is_absolute() and path.parent == Path("."):
        return default_dir / path.name

    return resolve_model_path(path, default_dir=default_dir)


def get_mode_defaults(mode: str) -> dict[str, Any]:
    """Return normalized merged defaults for a CLI/Python API mode."""
    return {
        key: _resolve_default_value(key, value)
        for key, value in _merged_mode_defaults(mode).items()
    }


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

    if normalized_mode in RUNTIME_MODES:
        multiple_models = normalized_mode in {"generate", "eval", "tune", "research"}
        values["detector"] = _normalize_model_list(
            values.get("detector", [DEFAULT_DETECTOR] if multiple_models else DEFAULT_DETECTOR),
            multiple=multiple_models,
        )
        values["reid"] = _normalize_model_list(
            values.get("reid", [DEFAULT_REID] if multiple_models else DEFAULT_REID),
            multiple=multiple_models,
        )
        values["tracker"] = str(values.get("tracker") or get_mode_default(normalized_mode, "tracker"))
        values["classes"] = _normalize_classes(values.get("classes"))
        values["project"] = Path(values.get("project") or "runs")
        values.setdefault("detector_explicit", "detector" in explicit)
        values.setdefault("reid_explicit", "reid" in explicit)
        values.setdefault("tracker_explicit", "tracker" in explicit)
        values.setdefault("device_explicit", "device" in explicit)
        values.setdefault("half_explicit", "half" in explicit)
    elif normalized_mode == "export":
        values["weights"] = ensure_model_extension(values.get("weights") or get_mode_default("export", "weights"))
        include = values.get("include") or ()
        values["include"] = tuple(include)
        project = values.get("project")
        if project is not None:
            values["project"] = Path(project)

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
        "n_threads": int(values.get("n_threads", 1)),
        "project": Path(values.get("project") or "runs"),
        "name": str(values.get("name", "exp")),
        "exist_ok": bool(values.get("exist_ok", False)),
        "half": bool(values.get("half", False)),
        "vid_stride": int(values.get("vid_stride", 1)),
        "ci": bool(values.get("ci", False)),
        "tracker": str(values.get("tracker", "bytetrack")),
        "verbose": bool(values.get("verbose", False)),
        "agnostic_nms": bool(values.get("agnostic_nms", False)),
        "postprocessing": str(values.get("postprocessing", "none")),
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


@dataclass(frozen=True, **dataclass_slots_kwargs())
class SharedModeDefaults:
    detector: Path
    reid: Path


@dataclass(frozen=True, **dataclass_slots_kwargs())
class RuntimeModeDefaults:
    imgsz: Any
    fps: int | None
    conf: float | None
    iou: float
    device: str
    batch_size: int
    auto_batch: bool
    resume: bool
    n_threads: int
    project: Path
    name: str
    exist_ok: bool
    half: bool
    vid_stride: int
    ci: bool
    tracker: str
    verbose: bool
    agnostic_nms: bool
    postprocessing: str
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


@dataclass(frozen=True, **dataclass_slots_kwargs())
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


@dataclass(frozen=True, **dataclass_slots_kwargs())
class GenerateModeDefaults(RuntimeModeDefaults):
    data: str | None
    source: str | None
    benchmark: str
    split: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GenerateModeDefaults":
        data = values.get("data")
        source = values.get("source")
        return cls(
            **_runtime_mode_kwargs(values),
            data=None if data is None else str(data),
            source=None if source is None else str(source),
            benchmark=str(values.get("benchmark", "")),
            split=str(values.get("split", "")),
        )


@dataclass(frozen=True, **dataclass_slots_kwargs())
class EvalModeDefaults(RuntimeModeDefaults):
    data: str | None
    source: str | None
    benchmark: str
    split: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EvalModeDefaults":
        data = values.get("data")
        source = values.get("source")
        return cls(
            **_runtime_mode_kwargs(values),
            data=None if data is None else str(data),
            source=None if source is None else str(source),
            benchmark=str(values.get("benchmark", "")),
            split=str(values.get("split", "")),
        )


@dataclass(frozen=True, **dataclass_slots_kwargs())
class TuneModeDefaults(RuntimeModeDefaults):
    data: str | None
    source: str | None
    benchmark: str
    split: str
    n_trials: int
    objectives: tuple[str, ...]
    maximize: tuple[str, ...]
    minimize: tuple[str, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TuneModeDefaults":
        data = values.get("data")
        source = values.get("source")
        objectives = tuple(values.get("objectives") or ())
        return cls(
            **_runtime_mode_kwargs(values),
            data=None if data is None else str(data),
            source=None if source is None else str(source),
            benchmark=str(values.get("benchmark", "")),
            split=str(values.get("split", "")),
            n_trials=int(values.get("n_trials", 10)),
            objectives=objectives,
            maximize=tuple(values.get("maximize") or objectives or ("HOTA",)),
            minimize=tuple(values.get("minimize") or ()),
        )


@dataclass(frozen=True, **dataclass_slots_kwargs())
class ResearchModeDefaults(RuntimeModeDefaults):
    data: str | None
    source: str | None
    benchmark: str
    split: str
    proposal_model: str
    proposal_api_key: str | None
    proposal_api_key_env: str | None
    max_metric_calls: int
    eval_timeout: float
    keep_workspace: bool
    idf1_penalty: float
    mota_penalty: float
    idf1_tolerance: float
    mota_tolerance: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ResearchModeDefaults":
        data = values.get("data")
        source = values.get("source")
        return cls(
            **_runtime_mode_kwargs(values),
            data=None if data is None else str(data),
            source=None if source is None else str(source),
            benchmark=str(values.get("benchmark", "")),
            split=str(values.get("split", "")),
            proposal_model=str(values.get("proposal_model", "openai/gpt-5.4")),
            proposal_api_key=None if values.get("proposal_api_key") in {None, ""} else str(values.get("proposal_api_key")),
            proposal_api_key_env=(
                None if values.get("proposal_api_key_env") in {None, ""} else str(values.get("proposal_api_key_env"))
            ),
            max_metric_calls=int(values.get("max_metric_calls", 24)),
            eval_timeout=float(values.get("eval_timeout", 900.0)),
            keep_workspace=bool(values.get("keep_workspace", False)),
            idf1_penalty=float(values.get("idf1_penalty", 1.0)),
            mota_penalty=float(values.get("mota_penalty", 1.0)),
            idf1_tolerance=float(values.get("idf1_tolerance", 0.0)),
            mota_tolerance=float(values.get("mota_tolerance", 0.0)),
        )


@dataclass(frozen=True, **dataclass_slots_kwargs())
class ExportModeDefaults:
    batch_size: int
    imgsz: Any
    device: str
    optimize: bool
    dynamic: bool
    simplify: bool
    opset: int
    workspace: int
    weights: Path
    half: bool
    include: tuple[str, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExportModeDefaults":
        return cls(
            batch_size=int(values.get("batch_size", 1)),
            imgsz=values.get("imgsz"),
            device=str(values.get("device", "cpu")),
            optimize=bool(values.get("optimize", False)),
            dynamic=bool(values.get("dynamic", False)),
            simplify=bool(values.get("simplify", False)),
            opset=int(values.get("opset", 17)),
            workspace=int(values.get("workspace", 4)),
            weights=ensure_model_extension(values.get("weights") or DEFAULT_REID),
            half=bool(values.get("half", False)),
            include=tuple(values.get("include") or ()),
        )


@dataclass(frozen=True, **dataclass_slots_kwargs())
class BoxMOTDefaults:
    shared: SharedModeDefaults
    track: TrackModeDefaults
    generate: GenerateModeDefaults
    eval: EvalModeDefaults
    tune: TuneModeDefaults
    research: ResearchModeDefaults
    export: ExportModeDefaults


BOXMOT_DEFAULTS = BoxMOTDefaults(
    shared=SharedModeDefaults(detector=DEFAULT_DETECTOR, reid=DEFAULT_REID),
    track=TrackModeDefaults.from_mapping(get_mode_defaults("track")),
    generate=GenerateModeDefaults.from_mapping(get_mode_defaults("generate")),
    eval=EvalModeDefaults.from_mapping(get_mode_defaults("eval")),
    tune=TuneModeDefaults.from_mapping(get_mode_defaults("tune")),
    research=ResearchModeDefaults.from_mapping(get_mode_defaults("research")),
    export=ExportModeDefaults.from_mapping(get_mode_defaults("export")),
)

__all__ = (
    "BOXMOT_DEFAULTS",
    "BoxMOTDefaults",
    "DEFAULT_DETECTOR",
    "DEFAULT_REID",
    "EvalModeDefaults",
    "ExportModeDefaults",
    "GenerateModeDefaults",
    "MODE_DEFAULTS_PATH",
    "ResearchModeDefaults",
    "RuntimeModeDefaults",
    "SharedModeDefaults",
    "TrackModeDefaults",
    "TuneModeDefaults",
    "build_mode_namespace",
    "ensure_model_extension",
    "get_mode_default",
    "get_mode_defaults",
)
