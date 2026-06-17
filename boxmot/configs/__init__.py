from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import yaml

from boxmot.trackers.specs import parse_tracker_spec
from boxmot.utils import WEIGHTS
from boxmot.utils.misc import resolve_model_path

RUNTIME_MODES = frozenset({"track", "generate", "eval", "tune", "research"})
MODE_DEFAULTS_PATH = Path(__file__).resolve().parent / "modes.yaml"
TRAINING_RECIPES_DIR = Path(__file__).resolve().parent / "training"


def _load_mode_defaults() -> dict[str, Any]:
    with open(MODE_DEFAULTS_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_training_recipe(name: str) -> dict[str, Any]:
    """Load a training recipe YAML by name (e.g. ``'lmbn_n'``)."""
    recipe_path = TRAINING_RECIPES_DIR / f"{name}.yaml"
    if not recipe_path.exists():
        available = list_training_recipes()
        raise FileNotFoundError(
            f"Training recipe '{name}' not found at {recipe_path}. "
            f"Available recipes: {', '.join(available) or '(none)'}"
        )
    with open(recipe_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def list_training_recipes() -> list[str]:
    """Return sorted names of available training recipes."""
    if not TRAINING_RECIPES_DIR.is_dir():
        return []
    return sorted(p.stem for p in TRAINING_RECIPES_DIR.glob("*.yaml"))


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


def _normalize_int_tuple(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        parts = [part for part in values.replace(";", ",").split(",") if part.strip()]
        return tuple(int(part) for part in parts)
    if isinstance(values, int):
        return (int(values),)
    return tuple(int(value) for value in values)


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
        tracker_spec = parse_tracker_spec(
            values.get("tracker") or get_mode_default(normalized_mode, "tracker"),
            default_backend=str(values.get("tracker_backend", "python")),
        )
        values["tracker"] = tracker_spec.name
        values["tracker_backend"] = tracker_spec.backend
        values["classes"] = _normalize_classes(values.get("classes"))
        values["project"] = Path(values.get("project") or "runs")
        values.setdefault("detector_explicit", "detector" in explicit)
        values.setdefault("reid_explicit", "reid" in explicit)
        values.setdefault("tracker_explicit", "tracker" in explicit)
        values.setdefault("device_explicit", "device" in explicit)
        values.setdefault("half_explicit", "half" in explicit)
        values.setdefault("split_explicit", "split" in explicit)
    elif normalized_mode == "export":
        values["weights"] = ensure_model_extension(values.get("weights") or get_mode_default("export", "weights"))
        calibration_data = values.get("tflite_calibration_data")
        values["tflite_calibration_data"] = Path(calibration_data) if calibration_data else None
        include = values.get("include") or ()
        values["include"] = tuple(include)
        project = values.get("project")
        if project is not None:
            values["project"] = Path(project)
    elif normalized_mode == "train":
        # Apply training recipe if specified (between defaults and CLI overrides)
        recipe_name = values.pop("recipe", None)
        if recipe_name is not None:
            recipe_values = load_training_recipe(recipe_name)
            for key, val in recipe_values.items():
                if key not in explicit:
                    values[key] = val
        project = values.get("project")
        if project is not None:
            values["project"] = Path(project)
        imgsz = values.get("imgsz")
        if isinstance(imgsz, (list, tuple)):
            values["imgsz"] = tuple(imgsz)
        elif isinstance(imgsz, int):
            values["imgsz"] = (imgsz, imgsz // 2)
        values["head_parts"] = _normalize_int_tuple(values.get("head_parts", (1, 2)))
        # Parse eval_datasets: accept comma-separated string or list
        ed = values.get("eval_datasets", ())
        if isinstance(ed, str):
            ed = [s.strip() for s in ed.split(",") if s.strip()]
        values["eval_datasets"] = list(ed)
        values.setdefault("train_explicit_keys", tuple(sorted(explicit)))

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
        "tracker_backend": str(values.get("tracker_backend", "python")),
        "verbose": bool(values.get("verbose", False)),
        "show_timing": bool(values.get("show_timing", False)),
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


@dataclass(frozen=True, slots=True)
class SharedModeDefaults:
    detector: Path
    reid: Path


@dataclass(frozen=True, slots=True)
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
    tracker_backend: str
    verbose: bool
    show_timing: bool
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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
    hota_penalty: float
    idf1_penalty: float
    mota_penalty: float
    hota_tolerance: float
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
    tflite_quantize: str
    tflite_calibration_data: Path | None
    tflite_calibration_samples: int
    tflite_calibration_preprocess: str
    tflite_calibration_seed: int
    tflite_calibration_update: str
    tflite_static_activation_bits: int
    include: tuple[str, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExportModeDefaults":
        calibration_data = values.get("tflite_calibration_data")
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
            tflite_quantize=str(values.get("tflite_quantize", "none")),
            tflite_calibration_data=Path(calibration_data) if calibration_data else None,
            tflite_calibration_samples=int(values.get("tflite_calibration_samples", 256)),
            tflite_calibration_preprocess=str(values.get("tflite_calibration_preprocess", "resize")),
            tflite_calibration_seed=int(values.get("tflite_calibration_seed", 0)),
            tflite_calibration_update=str(values.get("tflite_calibration_update", "minmax")),
            tflite_static_activation_bits=int(values.get("tflite_static_activation_bits", 16)),
            include=tuple(values.get("include") or ()),
        )


@dataclass(frozen=True, slots=True)
class TrainModeDefaults:
    model: str
    dataset: str
    data_dir: str | None
    loss: str
    preprocess: str
    imgsz: Any
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    warmup_epochs: int
    eval_interval: int
    p_ids: int
    k_instances: int
    margin: float
    label_smooth: float
    classifier_loss: str
    triplet_soft_margin: bool | None
    arcface_scale: float
    arcface_margin: float
    cosface_scale: float
    cosface_margin: float
    center_loss_weight: float
    id_loss_weight: float
    metric_loss_weight: float
    branch_loss_agg: str
    metric_feature: str
    inference_feature: str
    feature_fusion: str
    feat_dim: int
    neck_dim: int
    head_pool: str
    head_parts: tuple[int, ...]
    branch_aware_metric: bool
    branch_metric_part_weight: float
    head_warmup_epochs: int
    head_warmup_lr_mult: float
    eta_min: float
    pretrained: bool
    device: str
    project: str
    name: str
    num_workers: int
    seed: int
    eval_datasets: tuple
    ema_decay: float | None
    gaussian_blur: bool
    color_jitter: bool
    random_grayscale: float
    random_erasing: float
    random_patch: bool
    color_augmentation: bool
    flip_tta: bool | None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainModeDefaults":
        imgsz = values.get("imgsz")
        if isinstance(imgsz, (list, tuple)):
            imgsz = tuple(imgsz)
        elif isinstance(imgsz, int):
            imgsz = (imgsz, imgsz // 2)
        return cls(
            model=str(values.get("model", "osnet_x0_25")),
            dataset=str(values.get("dataset", "market1501")),
            data_dir=None if values.get("data_dir") is None else str(values["data_dir"]),
            loss=str(values.get("loss", "triplet")),
            preprocess=str(values.get("preprocess", "resize")),
            imgsz=imgsz,
            batch_size=int(values.get("batch_size", 64)),
            lr=float(values.get("lr", 3.5e-4)),
            weight_decay=float(values.get("weight_decay", 5e-4)),
            epochs=int(values.get("epochs", 120)),
            warmup_epochs=int(values.get("warmup_epochs", 10)),
            eval_interval=int(values.get("eval_interval", 10)),
            p_ids=int(values.get("p_ids", 16)),
            k_instances=int(values.get("k_instances", 4)),
            margin=float(values.get("margin", 0.3)),
            label_smooth=float(values.get("label_smooth", 0.1)),
            classifier_loss=str(values.get("classifier_loss", "ce")),
            triplet_soft_margin=values.get("triplet_soft_margin"),
            arcface_scale=float(values.get("arcface_scale", 30.0)),
            arcface_margin=float(values.get("arcface_margin", 0.5)),
            cosface_scale=float(values.get("cosface_scale", 30.0)),
            cosface_margin=float(values.get("cosface_margin", 0.35)),
            center_loss_weight=float(values.get("center_loss_weight", 5e-4)),
            id_loss_weight=float(values.get("id_loss_weight", 1.0)),
            metric_loss_weight=float(values.get("metric_loss_weight", 1.0)),
            branch_loss_agg=str(values.get("branch_loss_agg", "mean")),
            metric_feature=str(values.get("metric_feature", "auto")),
            inference_feature=str(values.get("inference_feature", "concat_bn")),
            feature_fusion=str(values.get("feature_fusion", "last3")),
            feat_dim=int(values.get("feat_dim", 512)),
            neck_dim=int(values.get("neck_dim", 512)),
            head_pool=str(values.get("head_pool", "avg")),
            head_parts=_normalize_int_tuple(values.get("head_parts", (1, 2))),
            branch_aware_metric=bool(values.get("branch_aware_metric", False)),
            branch_metric_part_weight=float(values.get("branch_metric_part_weight", 0.5)),
            head_warmup_epochs=int(values.get("head_warmup_epochs", 0)),
            head_warmup_lr_mult=float(values.get("head_warmup_lr_mult", 2.0)),
            eta_min=float(values.get("eta_min", 1e-7)),
            pretrained=bool(values.get("pretrained", True)),
            device=str(values.get("device", "cpu")),
            project=str(values.get("project", "runs/reid_train")),
            name=str(values.get("name", "exp")),
            num_workers=int(values.get("num_workers", 4)),
            seed=int(values.get("seed", 42)),
            eval_datasets=tuple(values.get("eval_datasets", ())),
            ema_decay=values.get("ema_decay"),
            gaussian_blur=bool(values.get("gaussian_blur", False)),
            color_jitter=bool(values.get("color_jitter", False)),
            random_grayscale=float(values.get("random_grayscale", 0.0)),
            random_erasing=float(values.get("random_erasing", 0.5)),
            random_patch=bool(values.get("random_patch", True)),
            color_augmentation=bool(values.get("color_augmentation", True)),
            flip_tta=values.get("flip_tta"),
        )


@dataclass(frozen=True, slots=True)
class BoxMOTDefaults:
    shared: SharedModeDefaults
    track: TrackModeDefaults
    generate: GenerateModeDefaults
    eval: EvalModeDefaults
    tune: TuneModeDefaults
    research: ResearchModeDefaults
    export: ExportModeDefaults
    train: TrainModeDefaults


BOXMOT_DEFAULTS = BoxMOTDefaults(
    shared=SharedModeDefaults(detector=DEFAULT_DETECTOR, reid=DEFAULT_REID),
    track=TrackModeDefaults.from_mapping(get_mode_defaults("track")),
    generate=GenerateModeDefaults.from_mapping(get_mode_defaults("generate")),
    eval=EvalModeDefaults.from_mapping(get_mode_defaults("eval")),
    tune=TuneModeDefaults.from_mapping(get_mode_defaults("tune")),
    research=ResearchModeDefaults.from_mapping(get_mode_defaults("research")),
    export=ExportModeDefaults.from_mapping(get_mode_defaults("export")),
    train=TrainModeDefaults.from_mapping(get_mode_defaults("train")),
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
    "TrainModeDefaults",
    "TuneModeDefaults",
    "build_mode_namespace",
    "ensure_model_extension",
    "get_mode_default",
    "get_mode_defaults",
    "list_training_recipes",
    "load_training_recipe",
)
