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


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _flatten_training_recipe_values(recipe_values: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize nested recipe layout into train-arg keys.

    Supports both legacy flat recipes and nested hparams-like recipes.
    """
    sections = {
        "run",
        "data",
        "model",
        "optimization",
        "losses",
        "augmentation",
        "evaluation",
        "system",
        "derived",
    }
    if not any(section in recipe_values for section in sections):
        return dict(recipe_values)

    flattened: dict[str, Any] = {
        key: value
        for key, value in recipe_values.items()
        if key not in sections
    }

    mappings: dict[str, tuple[str, ...]] = {
        "model": ("run", "model_name"),
        "seed": ("run", "seed"),
        "deterministic": ("run", "deterministic"),
        "pretrained": ("run", "pretrained"),
        "dataset": ("data", "dataset"),
        "data_dir": ("data", "data_dir"),
        "data_specs": ("data", "data_specs"),
        "imgsz": ("data", "img_size"),
        "preprocess": ("data", "preprocess"),
        "batch_size": ("data", "batch_size"),
        "p_ids": ("data", "sampler", "p"),
        "k_instances": ("data", "sampler", "k"),
        "source_balance": ("data", "sampler", "source_balance"),
        "num_workers": ("data", "num_workers"),
        "feature_fusion": ("model", "feature_fusion"),
        "post_fusion_mixer": ("model", "post_fusion_mixer", "mode"),
        "post_fusion_mixer_reduction": ("model", "post_fusion_mixer", "reduction"),
        "post_fusion_mixer_kernel": ("model", "post_fusion_mixer", "kernel"),
        "post_fusion_mixer_gamma_init": ("model", "post_fusion_mixer", "gamma_init"),
        "feat_dim": ("model", "feat_dim"),
        "neck_dim": ("model", "neck_dim"),
        "drop_path_rate": ("model", "regularization", "drop_path_rate"),
        "attention_window_layout": ("model", "attention", "window_layout"),
        "attention_bias": ("model", "attention", "bias"),
        "attention_mask": ("model", "attention", "mask"),
        "attention_shift": ("model", "attention", "shift"),
        "stage3_global": ("model", "attention", "stage3_global"),
        "reid_adapter_stages": ("model", "reid_adapters", "stages"),
        "reid_adapter_reduction": ("model", "reid_adapters", "reduction"),
        "head_pool": ("model", "head", "pool"),
        "head_parts": ("model", "head", "parts"),
        "head_type": ("model", "head", "head_type"),
        "part_pooling": ("model", "head", "part_pooling"),
        "num_part_tokens": ("model", "head", "num_part_tokens"),
        "evidence_num_roles": ("model", "head", "evidence_num_roles"),
        "decouple_patterns": ("model", "head", "decouple_patterns"),
        "pattern_adapter_dim": ("model", "head", "pattern_adapter_dim"),
        "stripe_visibility": ("model", "head", "stripe_visibility"),
        "drop_global_aux": ("model", "head", "drop_global_aux"),
        "drop_global_aux_ratio": ("model", "head", "drop_global_aux_ratio"),
        "head_warmup_epochs": ("model", "head", "warmup_epochs"),
        "head_warmup_lr_mult": ("model", "head", "warmup_lr_mult"),
        "metric_feature": ("model", "feature_selection", "metric_feature"),
        "inference_feature": ("model", "feature_selection", "inference_feature"),
        "branch_aware_metric": ("model", "branch", "aware_metric"),
        "branch_metric_part_weight": ("model", "branch", "metric_part_weight"),
        "branch_loss_agg": ("model", "branch", "loss_agg"),
        "evidence_alignment_loss_weight": ("model", "evidence", "alignment_loss_weight"),
        "evidence_alignment_margin": ("model", "evidence", "alignment_margin"),
        "evidence_sinkhorn_iters": ("model", "evidence", "sinkhorn_iters"),
        "evidence_sinkhorn_temperature": ("model", "evidence", "sinkhorn_temperature"),
        "evidence_rerank_topk": ("model", "evidence", "rerank_topk"),
        "evidence_null_loss_weight": ("model", "evidence", "null_loss_weight"),
        "evidence_diversity_loss_weight": ("model", "evidence", "diversity_loss_weight"),
        "epochs": ("optimization", "epochs"),
        "lr": ("optimization", "lr"),
        "weight_decay": ("optimization", "weight_decay"),
        "eta_min": ("optimization", "scheduler", "eta_min"),
        "warmup_epochs": ("optimization", "scheduler", "warmup_epochs"),
        "vit_lr_profile": ("optimization", "vit_lr_profile"),
        "backbone_freeze_epochs": ("optimization", "backbone_freeze_epochs"),
        "gradual_unfreeze": ("optimization", "gradual_unfreeze", "enabled"),
        "gradual_unfreeze_head_epochs": ("optimization", "gradual_unfreeze", "head_epochs"),
        "gradual_unfreeze_stage_epochs": ("optimization", "gradual_unfreeze", "stage_epochs"),
        "gradual_unfreeze_backbone_lr_mult": ("optimization", "gradual_unfreeze", "backbone_lr_mult"),
        "gradual_unfreeze_backbone_lr_epochs": ("optimization", "gradual_unfreeze", "backbone_lr_epochs"),
        "ema_decay": ("optimization", "ema_decay"),
        "loss": ("losses", "loss_type"),
        "classifier_loss": ("losses", "classifier_loss"),
        "label_smooth": ("losses", "label_smooth"),
        "margin": ("losses", "triplet", "margin"),
        "triplet_soft_margin": ("losses", "triplet", "soft_margin"),
        "id_loss_weight": ("losses", "weights", "id_loss_weight"),
        "metric_loss_weight": ("losses", "weights", "metric_loss_weight"),
        "center_loss_weight": ("losses", "weights", "center_loss_weight"),
        "early_id_loss_weight": ("losses", "schedules", "early_id_loss", "weight"),
        "early_id_loss_epochs": ("losses", "schedules", "early_id_loss", "epochs"),
        "center_loss_ramp_start_epoch": ("losses", "schedules", "center_loss_ramp", "start_epoch"),
        "center_loss_ramp_end_epoch": ("losses", "schedules", "center_loss_ramp", "end_epoch"),
        "aux_ce_weight": ("losses", "weights", "aux_ce_weight"),
        "aux_ce_drop_epoch": ("losses", "aux_ce_drop_epoch"),
        "color_jitter": ("augmentation", "color_jitter"),
        "gaussian_blur": ("augmentation", "gaussian_blur"),
        "random_grayscale": ("augmentation", "random_grayscale"),
        "random_erasing": ("augmentation", "random_erasing"),
        "random_patch": ("augmentation", "random_patch"),
        "random_crop_scale": ("augmentation", "random_crop_scale"),
        "color_augmentation": ("augmentation", "color_augmentation"),
        "eval_interval": ("evaluation", "eval_interval"),
        "eval_datasets": ("evaluation", "eval_datasets"),
        "flip_tta": ("evaluation", "flip_tta"),
        "device": ("system", "device"),
    }

    for target_key, path in mappings.items():
        value = _nested_get(recipe_values, *path)
        if value is not None:
            flattened[target_key] = value

    return flattened


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
        recipe_values = yaml.safe_load(handle) or {}
    return _flatten_training_recipe_values(recipe_values)


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load a BoxMOT ReID training config YAML."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg_values = yaml.safe_load(handle) or {}
    if not isinstance(cfg_values, Mapping):
        raise ValueError(f"Training config must contain a mapping: {cfg_path}")
    return _flatten_training_recipe_values(cfg_values)


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


def _normalize_int_pair(value: Any, default: tuple[int, int] = (5, 3)) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, str):
        parts = [part for part in value.replace(";", ",").split(",") if part.strip()]
        if len(parts) == 1:
            parts = parts * 2
        if len(parts) != 2:
            raise ValueError(f"Expected one or two comma-separated integers, got {value!r}")
        return (int(parts[0]), int(parts[1]))
    values = tuple(int(part) for part in value)
    if len(values) == 1:
        return (values[0], values[0])
    if len(values) != 2:
        raise ValueError(f"Expected one or two integers, got {value!r}")
    return values


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
        cfg_values: dict[str, Any] | None = None
        cfg_path = values.pop("cfg", None)
        if cfg_path is not None:
            cfg_values = load_training_config(cfg_path)
        # Apply training recipe if specified (between defaults/config and CLI overrides)
        recipe_name = values.pop("recipe", None)
        if cfg_values is not None and "recipe" not in explicit and cfg_values.get("recipe") is not None:
            recipe_name = cfg_values["recipe"]
        if recipe_name is not None:
            recipe_values = load_training_recipe(recipe_name)
            for key, val in recipe_values.items():
                if key not in explicit:
                    values[key] = val
        if cfg_values is not None:
            for key, val in cfg_values.items():
                if key != "recipe" and key not in explicit:
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
        values["reid_adapter_stages"] = _normalize_int_tuple(values.get("reid_adapter_stages", ()))
        values["post_fusion_mixer_kernel"] = _normalize_int_pair(
            values.get("post_fusion_mixer_kernel", (5, 3))
        )
        # Parse eval_datasets: accept comma-separated string or list
        ed = values.get("eval_datasets", ())
        if isinstance(ed, str):
            ed = [s.strip() for s in ed.split(",") if s.strip()]
        values["eval_datasets"] = list(ed)
        if "backbone_freeze_epochs" not in explicit:
            epochs = int(values.get("epochs", 0) or 0)
            freeze_epochs = int(values.get("backbone_freeze_epochs", 0) or 0)
            if epochs >= 0 and freeze_epochs > epochs:
                values["backbone_freeze_epochs"] = epochs
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
    source_balance: str
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
    early_id_loss_weight: float
    early_id_loss_epochs: int
    center_loss_ramp_start_epoch: int
    center_loss_ramp_end_epoch: int
    aux_ce_weight: float
    aux_ce_drop_epoch: int
    branch_loss_agg: str
    metric_feature: str
    inference_feature: str
    feature_fusion: str
    post_fusion_mixer: str
    post_fusion_mixer_reduction: int
    post_fusion_mixer_kernel: tuple[int, int]
    post_fusion_mixer_gamma_init: float
    feat_dim: int
    neck_dim: int
    drop_path_rate: float
    attention_window_layout: str
    attention_bias: str
    attention_mask: bool
    attention_shift: bool
    stage3_global: bool
    reid_adapter_stages: tuple[int, ...]
    reid_adapter_reduction: int
    head_pool: str
    head_parts: tuple[int, ...]
    head_type: str
    part_pooling: str
    num_part_tokens: int
    evidence_num_roles: int
    decouple_patterns: bool
    pattern_adapter_dim: int
    stripe_visibility: bool
    drop_global_aux: bool
    drop_global_aux_ratio: float
    branch_aware_metric: bool
    branch_metric_part_weight: float
    evidence_alignment_loss_weight: float
    evidence_alignment_margin: float
    evidence_sinkhorn_iters: int
    evidence_sinkhorn_temperature: float
    evidence_rerank_topk: int
    evidence_null_loss_weight: float
    evidence_diversity_loss_weight: float
    head_warmup_epochs: int
    head_warmup_lr_mult: float
    vit_lr_profile: str
    backbone_freeze_epochs: int
    gradual_unfreeze: bool
    gradual_unfreeze_head_epochs: int
    gradual_unfreeze_stage_epochs: int
    gradual_unfreeze_backbone_lr_mult: float
    gradual_unfreeze_backbone_lr_epochs: int
    eta_min: float
    pretrained: bool
    device: str
    project: str
    name: str
    num_workers: int
    seed: int
    deterministic: bool
    eval_datasets: tuple
    ema_decay: float | None
    gaussian_blur: bool
    color_jitter: bool
    random_grayscale: float
    random_erasing: float
    random_patch: bool
    random_crop_scale: float
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
            source_balance=str(values.get("source_balance", "")),
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
            early_id_loss_weight=float(values.get("early_id_loss_weight", 0.0)),
            early_id_loss_epochs=int(values.get("early_id_loss_epochs", 0)),
            center_loss_ramp_start_epoch=int(values.get("center_loss_ramp_start_epoch", 0)),
            center_loss_ramp_end_epoch=int(values.get("center_loss_ramp_end_epoch", 0)),
            aux_ce_weight=float(values.get("aux_ce_weight", 1.0)),
            aux_ce_drop_epoch=int(values.get("aux_ce_drop_epoch", 0)),
            branch_loss_agg=str(values.get("branch_loss_agg", "mean")),
            metric_feature=str(values.get("metric_feature", "auto")),
            inference_feature=str(values.get("inference_feature", "concat_bn")),
            feature_fusion=str(values.get("feature_fusion", "last3")),
            post_fusion_mixer=str(values.get("post_fusion_mixer", "none")),
            post_fusion_mixer_reduction=int(values.get("post_fusion_mixer_reduction", 4)),
            post_fusion_mixer_kernel=_normalize_int_pair(values.get("post_fusion_mixer_kernel", (5, 3))),
            post_fusion_mixer_gamma_init=float(values.get("post_fusion_mixer_gamma_init", 0.0)),
            feat_dim=int(values.get("feat_dim", 512)),
            neck_dim=int(values.get("neck_dim", 512)),
            drop_path_rate=float(values.get("drop_path_rate", 0.1)),
            attention_window_layout=str(values.get("attention_window_layout", "legacy")),
            attention_bias=str(values.get("attention_bias", "absolute")),
            attention_mask=bool(values.get("attention_mask", False)),
            attention_shift=bool(values.get("attention_shift", False)),
            stage3_global=bool(values.get("stage3_global", False)),
            reid_adapter_stages=_normalize_int_tuple(values.get("reid_adapter_stages", ())),
            reid_adapter_reduction=int(values.get("reid_adapter_reduction", 4)),
            head_pool=str(values.get("head_pool", "avg")),
            head_parts=_normalize_int_tuple(values.get("head_parts", (1, 2))),
            head_type=str(values.get("head_type", "standard")),
            part_pooling=str(values.get("part_pooling", "stripes")),
            num_part_tokens=int(values.get("num_part_tokens", 4)),
            evidence_num_roles=int(values.get("evidence_num_roles", 8)),
            decouple_patterns=bool(values.get("decouple_patterns", False)),
            pattern_adapter_dim=int(values.get("pattern_adapter_dim", 128)),
            stripe_visibility=bool(values.get("stripe_visibility", False)),
            drop_global_aux=bool(values.get("drop_global_aux", False)),
            drop_global_aux_ratio=float(values.get("drop_global_aux_ratio", 0.25)),
            branch_aware_metric=bool(values.get("branch_aware_metric", False)),
            branch_metric_part_weight=float(values.get("branch_metric_part_weight", 0.5)),
            evidence_alignment_loss_weight=float(values.get("evidence_alignment_loss_weight", 0.0)),
            evidence_alignment_margin=float(values.get("evidence_alignment_margin", 0.2)),
            evidence_sinkhorn_iters=int(values.get("evidence_sinkhorn_iters", 20)),
            evidence_sinkhorn_temperature=float(values.get("evidence_sinkhorn_temperature", 0.1)),
            evidence_rerank_topk=int(values.get("evidence_rerank_topk", 100)),
            evidence_null_loss_weight=float(values.get("evidence_null_loss_weight", 0.0)),
            evidence_diversity_loss_weight=float(values.get("evidence_diversity_loss_weight", 0.0)),
            head_warmup_epochs=int(values.get("head_warmup_epochs", 0)),
            head_warmup_lr_mult=float(values.get("head_warmup_lr_mult", 2.0)),
            vit_lr_profile=str(values.get("vit_lr_profile", "layer_decay")),
            backbone_freeze_epochs=int(values.get("backbone_freeze_epochs", 0)),
            gradual_unfreeze=bool(values.get("gradual_unfreeze", False)),
            gradual_unfreeze_head_epochs=int(values.get("gradual_unfreeze_head_epochs", 5)),
            gradual_unfreeze_stage_epochs=int(values.get("gradual_unfreeze_stage_epochs", 10)),
            gradual_unfreeze_backbone_lr_mult=float(values.get("gradual_unfreeze_backbone_lr_mult", 0.1)),
            gradual_unfreeze_backbone_lr_epochs=int(values.get("gradual_unfreeze_backbone_lr_epochs", 5)),
            eta_min=float(values.get("eta_min", 1e-7)),
            pretrained=bool(values.get("pretrained", True)),
            device=str(values.get("device", "cpu")),
            project=str(values.get("project", "runs/reid_train")),
            name=str(values.get("name", "exp")),
            num_workers=int(values.get("num_workers", 4)),
            seed=int(values.get("seed", 0)),
            deterministic=bool(values.get("deterministic", True)),
            eval_datasets=tuple(values.get("eval_datasets", ())),
            ema_decay=values.get("ema_decay"),
            gaussian_blur=bool(values.get("gaussian_blur", False)),
            color_jitter=bool(values.get("color_jitter", False)),
            random_grayscale=float(values.get("random_grayscale", 0.0)),
            random_erasing=float(values.get("random_erasing", 0.5)),
            random_patch=bool(values.get("random_patch", True)),
            random_crop_scale=float(values.get("random_crop_scale", 1.05)),
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
    "load_training_config",
    "load_training_recipe",
)
