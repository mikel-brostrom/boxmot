from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from boxmot.data.config import (
    DATASET_CONFIGS_DIR,
    ConfigurationError,
    load_dataset_config,
    resolve_dataset_config_path,
)
from boxmot.detectors import default_conf, default_imgsz, get_runtime_detector_cfg
from boxmot.detectors.config import load_detector_profile, resolve_detector_config_path
from boxmot.engine.experiment import (
    load_experiment_runtime_config,
    resolve_experiment_path,
)
from boxmot.reid.config import load_reid_profile, load_runtime_reid_config, resolve_reid_config_path
from boxmot.utils import BENCHMARK_DATA
from boxmot.utils import logger as LOGGER
from boxmot.utils.download import download_eval_data, download_file
from boxmot.utils.misc import resolve_model_path


def _normalize_download_value(value: Any) -> str | dict[str, Any]:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if item in (None, ""):
                continue
            if isinstance(item, dict):
                normalized[str(key)] = {
                    str(nested_key): str(nested_value)
                    for nested_key, nested_value in item.items()
                    if nested_value not in (None, "")
                }
            else:
                normalized[str(key)] = str(item)
        return normalized
    return str(value) if value not in (None, "") else ""


def _resolve_split_download_value(value: Any, split_name: str | None) -> str:
    normalized = _normalize_download_value(value)
    if isinstance(normalized, dict):
        split_key = str(split_name or "").strip()
        entry = normalized.get(split_key) if split_key else None
        if entry is None:
            entry = normalized.get("default") or normalized.get("*")
        if isinstance(entry, dict):
            return str(entry.get("url") or "")
        return str(entry or "")
    return normalized


def _resolve_split_download_entry(value: Any, split_name: str | None) -> str | dict[str, str]:
    normalized = _normalize_download_value(value)
    if not isinstance(normalized, dict):
        return str(normalized or "")

    split_key = str(split_name or "").strip()
    entry = normalized.get(split_key) if split_key else None
    if entry is None:
        entry = normalized.get("default") or normalized.get("*")

    if isinstance(entry, dict):
        return {str(k): str(v) for k, v in entry.items()}
    return str(entry or "")


def _scope_hf_url_to_split(url: str, cfg: dict[str, Any], split_name: str) -> str:
    """Append the split's directory name to a bare HF repo URL.

    When a benchmark config specifies ``download.dataset: hf://owner/repo``
    without a subfolder, we scope the download to only the active split's
    subfolder (e.g. ``hf://owner/repo/ablation``) to avoid downloading the
    entire repository.
    """
    if not url or not url.startswith("hf://"):
        return url
    parts = url[len("hf://"):].split("/")
    # Only modify bare repo URLs (exactly 2 path parts: owner/repo)
    if len(parts) != 2:
        return url
    # Look up the split's directory name from the splits mapping
    splits = cfg.get("splits") or {}
    split_dir = splits.get(split_name)
    if isinstance(split_dir, dict):
        split_dir = split_dir.get("path") or split_name
    if not split_dir:
        split_dir = split_name
    return f"{url}/{split_dir}"


def _normalize_profile_key(value: Any) -> str:
    if value in (None, ""):
        return ""
    return Path(str(value)).stem.lower().replace("-", "").replace("_", "")


def _profile_selector_matches(selector: str | None, candidates: list[Any]) -> bool:
    if not selector:
        return True
    selector_key = _normalize_profile_key(selector)
    if not selector_key:
        return True
    for candidate in candidates:
        if _normalize_profile_key(candidate) == selector_key:
            return True
    return False


def _primary_arg_model(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _resolve_runs_download_url(args: Any, cfg: dict[str, Any], split_name: str | None) -> str:
    runs_entry = _resolve_split_download_entry((cfg.get("download") or {}).get("runs"), split_name)
    if isinstance(runs_entry, str):
        return runs_entry

    url = str(runs_entry.get("url") or "")
    if not url:
        return ""

    detector_selector = runs_entry.get("detector")
    reid_selector = runs_entry.get("reid")

    detector_explicit = bool(getattr(args, "detector_explicit", False))
    if detector_explicit:
        detector_candidates = [_primary_arg_model(getattr(args, "detector", None))]
    else:
        detector_cfg = get_benchmark_detector_cfg(cfg)
        detector_candidates = [
            detector_cfg.get("id"),
            detector_cfg.get("model"),
            detector_cfg.get("default_model"),
        ]

    reid_explicit = bool(getattr(args, "reid_explicit", False))
    if reid_explicit:
        reid_candidates = [_primary_arg_model(getattr(args, "reid", None))]
    else:
        reid_cfg = get_benchmark_reid_cfg(cfg)
        reid_candidates = [
            reid_cfg.get("id"),
            reid_cfg.get("model"),
            reid_cfg.get("default_model"),
        ]

    if not _profile_selector_matches(detector_selector, detector_candidates):
        return ""
    if not _profile_selector_matches(reid_selector, reid_candidates):
        return ""

    return url






def resolve_dataset_cfg_path(name: str | Path) -> Path:
    """Resolve a dataset profile by id, filename, or path."""
    return resolve_dataset_config_path(name)


def resolve_experiment_cfg_path(name: str | Path) -> Path:
    """Resolve an experiment manifest by id, filename, or path."""
    return resolve_experiment_path(name)


def resolve_detector_cfg_path(name: str | Path) -> Path:
    """Resolve a detector profile by id, filename, or path."""
    return resolve_detector_config_path(name)


def resolve_reid_cfg_path(name: str | Path) -> Path:
    """Resolve a ReID profile by id, filename, or path."""
    return resolve_reid_config_path(name)




def _metric_backend_for_box_type(box_type: str) -> str:
    """Map the configured box type to the MOT metric backend used at runtime."""
    normalized = str(box_type or "aabb").lower()
    return "mot_challenge_obb" if normalized == "obb" else "mot_challenge"










def _build_filtered_split(
    base_dir: Path,
    split_name: str,
    seq_pattern: str,
    dataset_root: Path,
    frame_split: str | None = None,
) -> Path:
    """Build a split directory with sequences matching *seq_pattern*.

    When *frame_split* is ``None``, creates symlinks to full sequences.
    When *frame_split* is ``"val-half"``, creates physical copies trimmed to
    the second half of frames (the standard ByteTrack ablation protocol).

    The directory is reused if it already exists and is populated.
    """
    from fnmatch import fnmatch

    split_dir = dataset_root / split_name

    # Determine which sequences should be included
    wanted = sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and fnmatch(p.name, seq_pattern)
    )
    wanted_names = {p.name for p in wanted}

    if frame_split == "val-half":
        # Physical copy with frame halving — only build once
        if split_dir.is_dir() and any(split_dir.iterdir()):
            return split_dir
        split_dir.mkdir(parents=True, exist_ok=True)
        from boxmot.engine.tracking.mot import _build_val_half_split
        _build_val_half_split(wanted, split_dir)
    else:
        # Symlink mode — lightweight, no frame trimming
        split_dir.mkdir(parents=True, exist_ok=True)
        # Remove stale symlinks that no longer match
        for existing in split_dir.iterdir():
            if existing.is_symlink() and existing.name not in wanted_names:
                existing.unlink()
        # Create missing symlinks
        for seq_dir in wanted:
            link = split_dir / seq_dir.name
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(seq_dir.resolve())

    return split_dir












def load_dataset_cfg(
    name: str | Path,
    *,
    split: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Load a model-free dataset profile in evaluator-runtime form."""
    dataset = load_dataset_config(name)
    split_name = str(split or dataset["default_split"])
    if split_name not in dataset["splits"]:
        available = ", ".join(sorted(dataset["splits"]))
        raise ConfigurationError(
            f'Dataset "{dataset["id"]}" has no split "{split_name}". Available splits: {available}.'
        )
    if str(mode or "").lower() in {"eval", "evaluation", "tune", "research"}:
        if not dataset["splits"][split_name]["has_ground_truth"]:
            valid = [
                name
                for name, metadata in dataset["splits"].items()
                if metadata["has_ground_truth"]
            ]
            choices = ", ".join(valid) or "none"
            raise ConfigurationError(
                f'Dataset "{dataset["id"]}" split "{split_name}" has no ground truth and cannot be '
                f"evaluated. Available evaluation splits: {choices}."
            )
    split_paths = {key: dict(value) for key, value in dataset["splits"].items()}
    ignored = {
        int(metadata["id"]): class_name
        for class_name, metadata in dataset["classes"].items()
        if metadata["evaluation"] == "ignore"
    }
    targets = {
        int(metadata["id"]): class_name
        for class_name, metadata in dataset["classes"].items()
        if metadata["evaluation"] == "target"
    }
    dataset_resource = dataset["resources"].get("dataset") or {}
    dataset_download: str | dict[str, str] = ""
    if isinstance(dataset_resource, dict):
        if isinstance(dataset_resource.get("uris"), dict):
            dataset_download = {str(key): str(value) for key, value in dataset_resource["uris"].items()}
        elif dataset_resource.get("uri"):
            dataset_download = str(dataset_resource["uri"])
    download: dict[str, Any] = {"dataset": dataset_download, "runs": ""}
    if isinstance(dataset_resource, dict) and dataset_resource.get("backend") == "mot17_parquet":
        download.update({"source": "parquet", "parquet_repo": str(dataset_resource.get("repository") or "")})
    metric_backend = _metric_backend_for_box_type(dataset["box_type"])
    ignore_dataset_ids = sorted(ignored)
    benchmark = {
        "source": dataset["root"],
        "split": split_paths[split_name]["path"],
        "box_type": dataset["box_type"],
        "layout": dataset["layout"],
        "metric_eval": metric_backend,
        "eval_classes": targets,
        "distractor_classes": ignored,
        "ignore_dataset_ids": ignore_dataset_ids,
        "has_ground_truth": split_paths[split_name]["has_ground_truth"],
    }
    return {
        "id": dataset["id"],
        "dataset_config": dataset["id"],
        "path": dataset["root"],
        "split": split_name,
        "splits": split_paths,
        "train": split_paths.get("train"),
        "val": split_paths.get("val"),
        "test": split_paths.get("test"),
        "layout": dataset["layout"],
        "box_type": dataset["box_type"],
        "metric_backend": metric_backend,
        "names": targets,
        "distractors": ignored,
        "benchmark": benchmark,
        "evaluation": {
            "box_type": dataset["box_type"],
            "layout": dataset["layout"],
            "metric_eval": metric_backend,
            "classes": {
                "eval": targets,
                "distractor": ignored,
                "mapping": {},
                "bridge": [],
                "ignore_dataset_ids": ignore_dataset_ids,
            },
        },
        "download": download,
        "storage": {"root": dataset["root"], "split": dataset["splits"][split_name]["path"]},
        "resources": dataset["resources"],
        "artifacts": dataset["artifacts"],
        "dataset_source_path": dataset["config_path"],
    }


def load_experiment_only_cfg(name: str | Path) -> dict[str, Any]:
    """Load and fully resolve an experiment manifest."""
    return load_experiment_runtime_config(name)


def load_detector_component_cfg(name: str | Path) -> dict[str, Any]:
    """Load a detector profile in runtime-friendly form."""
    return load_detector_profile(name)


def load_reid_component_cfg(name: str | Path) -> dict[str, Any]:
    """Load a ReID profile in runtime-friendly form."""
    return load_reid_profile(name)


def load_runtime_reid_component_cfg(name: str | Path | None) -> dict[str, Any]:
    """Load a ReID component config by model/config reference, returning ``{}`` when unmatched."""
    return load_runtime_reid_config(name)




def load_experiment_cfg(
    name: str | Path,
    *,
    split: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Load an experiment into the evaluator runtime structure."""
    return load_experiment_runtime_config(name, split=split, mode=mode)


def get_benchmark_detector_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return detector settings from a combined dataset+model config."""
    detector_cfg = cfg.get("detector", {})
    return dict(detector_cfg) if isinstance(detector_cfg, dict) else {}


def get_benchmark_reid_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return ReID settings from a combined dataset+model config."""
    reid_cfg = cfg.get("reid", {})
    return dict(reid_cfg) if isinstance(reid_cfg, dict) else {}


def _normalize_google_drive_url(url: str) -> str:
    """Normalize common Google Drive share URLs to the canonical ``uc?id=...`` form."""
    if url.startswith("gdrive://"):
        file_id = url[len("gdrive://") :].strip("/")
        return f"https://drive.google.com/uc?id={file_id}"
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return url

    query = parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return f"https://drive.google.com/uc?id={query['id'][0]}"

    match = re.search(r"/file/d/([^/]+)", parsed.path)
    if match:
        return f"https://drive.google.com/uc?id={match.group(1)}"

    return url


def get_benchmark_detector_url(cfg: dict[str, Any]) -> str | None:
    """Return the configured detector download URL, if present."""
    detector_cfg = get_benchmark_detector_cfg(cfg)
    model_url = detector_cfg.get("model_url") or detector_cfg.get("url")
    return _normalize_google_drive_url(str(model_url)) if model_url else None


def get_benchmark_reid_url(cfg: dict[str, Any]) -> str | None:
    """Return the configured ReID download URL, if present."""
    reid_cfg = get_benchmark_reid_cfg(cfg)
    model_url = reid_cfg.get("model_url") or reid_cfg.get("url")
    return _normalize_google_drive_url(str(model_url)) if model_url else None


def resolve_required_yolo_model(cfg: dict[str, Any]) -> Path | None:
    """Return the detector model path configured for the active dataset/model bundle."""
    detector_cfg = get_benchmark_detector_cfg(cfg)
    model = detector_cfg.get("default_model") or detector_cfg.get("model")
    if model:
        return Path(model)
    return None


def resolve_required_reid_model(cfg: dict[str, Any]) -> Path | None:
    """Return the ReID model path configured for the active dataset/model bundle."""
    reid_cfg = get_benchmark_reid_cfg(cfg)
    model = reid_cfg.get("default_model") or reid_cfg.get("model")
    return Path(model) if model else None


def resolve_required_reid_device(cfg: dict[str, Any]) -> str | None:
    """Return the ReID device configured for the active dataset/model bundle."""
    reid_cfg = get_benchmark_reid_cfg(cfg)
    device = reid_cfg.get("device")
    if device is None:
        return None
    device = str(device).strip()
    return device or None


def resolve_required_reid_half(cfg: dict[str, Any]) -> bool | None:
    """Return the ReID half-precision preference configured for the active dataset/model bundle."""
    reid_cfg = get_benchmark_reid_cfg(cfg)
    if "half" not in reid_cfg:
        return None
    return bool(reid_cfg["half"])


def resolve_required_reid_preprocess(cfg: dict[str, Any]) -> str | None:
    """Return the ReID preprocess method configured for the active dataset/model bundle."""
    reid_cfg = get_benchmark_reid_cfg(cfg)
    preprocess = reid_cfg.get("preprocess")
    if preprocess is None:
        return None
    return str(preprocess).strip() or None


def apply_reid_runtime_defaults(args: Any, cfg: dict[str, Any], use_config: bool = True) -> None:
    """Populate ReID runtime args from config when the CLI did not override them."""
    fallback_device = getattr(args, "device", "")
    fallback_half = bool(getattr(args, "half", False))

    reid_device = fallback_device
    if use_config and not getattr(args, "device_explicit", False):
        configured_device = resolve_required_reid_device(cfg)
        if configured_device is not None:
            reid_device = configured_device

    reid_half = fallback_half
    if use_config and not getattr(args, "half_explicit", False):
        configured_half = resolve_required_reid_half(cfg)
        if configured_half is not None:
            reid_half = configured_half

    args.reid_device = reid_device
    args.reid_half = reid_half

    if not getattr(args, "reid_preprocess", None):
        from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
        reid_preprocess = DEFAULT_PREPROCESS
        if use_config:
            configured_preprocess = resolve_required_reid_preprocess(cfg)
            if configured_preprocess is not None:
                reid_preprocess = configured_preprocess
        args.reid_preprocess = reid_preprocess


def ensure_benchmark_detector_model(cfg: dict[str, Any], overwrite: bool = False) -> Path | None:
    """Ensure the configured detector model exists locally and return its path."""
    model_path = resolve_required_yolo_model(cfg)
    if model_path is None:
        return None

    resolved_path = resolve_model_path(model_path)
    if resolved_path.exists() or overwrite:
        if overwrite and get_benchmark_detector_url(cfg):
            download_file(get_benchmark_detector_url(cfg), resolved_path, overwrite=True)
        return resolved_path

    model_url = get_benchmark_detector_url(cfg)
    if model_url:
        download_file(model_url, resolved_path, overwrite=False)
        return resolved_path
    return resolved_path


def ensure_benchmark_reid_model(cfg: dict[str, Any], overwrite: bool = False) -> Path | None:
    """Ensure the configured ReID model exists locally and return its path."""
    model_path = resolve_required_reid_model(cfg)
    if model_path is None:
        return None

    resolved_path = resolve_model_path(model_path)
    if resolved_path.exists() or overwrite:
        if overwrite and get_benchmark_reid_url(cfg):
            download_file(get_benchmark_reid_url(cfg), resolved_path, overwrite=True)
        return resolved_path

    model_url = get_benchmark_reid_url(cfg)
    if model_url:
        download_file(model_url, resolved_path, overwrite=False)
        return resolved_path
    return resolved_path


def should_use_benchmark_detector(args: Any, cfg: dict[str, Any]) -> bool:
    """Return True when the model config should provide the active detector."""
    benchmark_model = resolve_required_yolo_model(cfg)
    if benchmark_model is None:
        return False

    current_model = getattr(args, "detector", None)
    if current_model is None:
        return False

    if isinstance(current_model, (list, tuple)):
        if not current_model:
            return False
        current_model = current_model[0]

    resolved_current = resolve_model_path(current_model)
    resolved_benchmark = resolve_model_path(benchmark_model)
    if resolved_current == resolved_benchmark:
        return True
    if Path(current_model).name.lower() == Path(benchmark_model).name.lower():
        return True
    if (
        Path(current_model).stem.lower().replace("-", "").replace("_", "")
        == Path(benchmark_model).stem.lower().replace("-", "").replace("_", "")
    ):
        return True

    if getattr(args, "detector_explicit", None) is True:
        return False

    default_stem = Path("yolov8n").stem.lower()
    return Path(current_model).stem.lower() == default_stem


def should_use_benchmark_reid(args: Any, cfg: dict[str, Any]) -> bool:
    """Return True when the model config should provide the active ReID model."""
    benchmark_model = resolve_required_reid_model(cfg)
    if benchmark_model is None:
        return False

    current_model = getattr(args, "reid", None)
    if current_model is None:
        return False

    if isinstance(current_model, (list, tuple)):
        if not current_model:
            return False
        current_model = current_model[0]

    resolved_current = resolve_model_path(current_model)
    resolved_benchmark = resolve_model_path(benchmark_model)
    if resolved_current == resolved_benchmark:
        return True
    if Path(current_model).name.lower() == Path(benchmark_model).name.lower():
        return True

    if getattr(args, "reid_explicit", None) is True:
        return False

    default_stem = Path("osnet_x0_25_msmt17").stem.lower()
    return Path(current_model).stem.lower() == default_stem


def _resolve_benchmark_dest(cfg: dict[str, Any], benchmark_name: str, source_root: Path | None) -> Path:
    download_cfg = dict(cfg.get("download") or {})
    dataset_dest = download_cfg.get("dataset_dest")
    if dataset_dest:
        return Path(dataset_dest)

    dataset_url = _resolve_split_download_value(download_cfg.get("dataset"), cfg.get("split"))
    if source_root is not None:
        if str(dataset_url).startswith("hf://"):
            return source_root
        if dataset_url:
            return source_root.parent / f"{source_root.name}.zip"
        return source_root

    if str(dataset_url).startswith("hf://"):
        return BENCHMARK_DATA / benchmark_name
    if dataset_url:
        return BENCHMARK_DATA / f"{benchmark_name}.zip"
    return Path(f"assets/{benchmark_name}")


def _resolve_runtime_benchmark_name(cfg: dict[str, Any], source_root: Path | None, cfg_path: Path) -> str:
    """Resolve a stable runtime benchmark name for cache and results paths."""
    del source_root
    return str(cfg.get("dataset_config") or cfg.get("id") or cfg_path.stem)


def _normalize_path_match_key(path_like: str | Path) -> str:
    return Path(str(path_like)).as_posix().lower().rstrip("/")


def _resolve_active_split_path(cfg: dict[str, Any]) -> str:
    split = str(cfg.get("split") or "train")
    split_path = cfg.get(split)
    if split_path is None:
        split_path = split
    return str(split_path)


def _apply_evaluation_config_ref(
    args: Any,
    config_ref: str | Path | None,
    overwrite: bool = False,
    status_fn: Callable[[str], None] | None = None,
    *,
    config_kind: str = "experiment",
) -> dict[str, Any] | None:
    """Apply an experiment or dataset config to the current args namespace."""
    if not config_ref:
        return None

    cli_split = getattr(args, "split", None)
    if cli_split and getattr(args, "split_explicit", False):
        cfg_split = cli_split
    else:
        cfg_split = None
    workflow_mode = getattr(args, "workflow_mode", None)
    is_dataset = config_kind == "dataset"
    if is_dataset:
        cfg_path = resolve_dataset_cfg_path(config_ref)
        cfg = load_dataset_cfg(cfg_path, split=cfg_split, mode=workflow_mode)
    else:
        try:
            cfg_path = resolve_experiment_cfg_path(config_ref)
        except FileNotFoundError as exc:
            try:
                resolve_dataset_cfg_path(config_ref)
            except FileNotFoundError:
                raise exc
            raise ConfigurationError(
                f'"{config_ref}" is a dataset profile, not an experiment. '
                f'Use --dataset {config_ref} instead of --experiment {config_ref}.'
            ) from None
        cfg = load_experiment_cfg(cfg_path, split=cfg_split, mode=workflow_mode)
    cfg_split = str(cfg.get("split") or "train")

    download_cfg = dict(cfg.get("download") or {})
    path_str = str(cfg.get("path") or "")
    source_root = Path(path_str) if path_str else None
    benchmark_name = _resolve_runtime_benchmark_name(cfg, source_root, cfg_path)

    benchmark_dest = _resolve_benchmark_dest(cfg, benchmark_name, source_root)

    # Resolve source path using the active split (check splits dict first).
    all_splits = cfg.get("splits") or {}
    split_entry = all_splits.get(cfg_split) or cfg.get(cfg_split) or cfg_split
    # Split entries can be a string (path) or a dict with path + seq_pattern + detection_source.
    if isinstance(split_entry, dict):
        active_split_path = str(split_entry.get("path") or cfg_split)
        seq_pattern = split_entry.get("seq_pattern")
        detection_source = split_entry.get("detection_source")
        frame_split = split_entry.get("frame_split")
    else:
        active_split_path = str(split_entry)
        seq_pattern = None
        detection_source = None
        frame_split = None
    # Allow top-level seq_pattern as fallback (e.g. from benchmark config).
    if seq_pattern is None:
        seq_pattern = cfg.get("seq_pattern")
    base_source = (source_root / active_split_path) if source_root is not None else (benchmark_dest / active_split_path)
    dataset_ready = base_source.is_dir() and any(base_source.iterdir())

    # Resolve the dataset download URL, scoping bare HF repo URLs to the
    # active split's subfolder so we don't download the entire repository.
    dataset_url = _resolve_split_download_value(download_cfg.get("dataset"), cfg_split)
    dataset_url = _scope_hf_url_to_split(dataset_url, cfg, cfg_split)
    if dataset_ready and not overwrite:
        dataset_url = ""

    runs_check_path = Path("runs") / "dets_n_embs" / benchmark_name / cfg_split

    # Parquet-based dataset setup (e.g. MOT17 with deduplicated images)
    download_source = download_cfg.get("source", "").lower()
    if download_source == "parquet":
        from boxmot.data.mot17_parquet import setup_mot17_from_parquet

        # Determine public detector: CLI --detection-source overrides config
        cli_det_source = getattr(args, "detection_source", None)
        public_det = (
            cli_det_source.upper()
            if cli_det_source and cli_det_source.upper() in ("DPM", "FRCNN", "SDP")
            else download_cfg.get("public_detector", "FRCNN")
        )

        setup_mot17_from_parquet(
            dest=benchmark_dest,
            split=cfg_split,
            detector=public_det,
            overwrite=overwrite,
            status_fn=status_fn,
        )

        # Still download pre-computed YOLOX dets/embs if available (for default eval)
        runs_url = _resolve_runs_download_url(args, cfg, cfg_split)
        if runs_url:
            download_eval_data(
                runs_url=runs_url,
                dataset_url="",
                dataset_dest=benchmark_dest,
                overwrite=overwrite,
                runs_check_path=runs_check_path,
                status_fn=status_fn,
            )
    else:
        download_eval_data(
            runs_url=_resolve_runs_download_url(args, cfg, cfg_split),
            dataset_url=dataset_url,
            dataset_dest=benchmark_dest,
            overwrite=overwrite,
            runs_check_path=runs_check_path,
            status_fn=status_fn,
        )

    args.experiment_id = None if is_dataset else cfg.get("id", benchmark_name)
    args.dataset_id = cfg.get("id", benchmark_name) if is_dataset else cfg.get("dataset_config")
    args.benchmark = benchmark_name
    args.runtime_evaluation_config = cfg

    args.split = cfg_split

    # Build filtered split directory at runtime when seq_pattern is specified
    if seq_pattern and base_source.is_dir():
        args.source = _build_filtered_split(
            base_source, cfg_split, seq_pattern,
            source_root or benchmark_dest, frame_split=frame_split,
        )
    else:
        args.source = base_source
    if seq_pattern:
        args.seq_pattern = seq_pattern
    # CLI --detection-source overrides the experiment's selected source.
    cli_detection_source = getattr(args, "detection_source", None)
    if cli_detection_source:
        pass  # keep the explicit CLI value
    else:
        configured_detections = cfg.get("detections") or {}
        configured_source = configured_detections.get("source")
        if configured_source == "public":
            args.detection_source = configured_detections.get("name")
        elif configured_source in {"model", "precomputed"}:
            args.detection_source = "private"
        elif detection_source:
            args.detection_source = detection_source

    box_type = cfg.get("box_type")
    if box_type:
        args.eval_box_type = str(box_type).lower()

    detector_cfg = get_benchmark_detector_cfg(cfg)
    if detector_cfg:
        args.dataset_detector_cfg = detector_cfg

    required_yolo_model = resolve_required_yolo_model(cfg)
    if required_yolo_model:
        args.required_yolo_model = required_yolo_model

    required_reid_model = resolve_required_reid_model(cfg)
    if required_reid_model:
        args.required_reid_model = required_reid_model

    args.dataset_source_path = cfg.get("dataset_source_path")
    args.experiment_source_path = None if is_dataset else cfg.get("source_path")
    args.resolved_experiment_config = None if is_dataset else cfg.get("resolved")
    args.config_paths = cfg.get("config_paths") or {}

    return cfg


def find_dataset_cfg_for_source(source: str | Path | None) -> dict[str, Any] | None:
    """Return the dataset config whose configured root best matches ``source``."""
    if not source:
        return None

    source_key = _normalize_path_match_key(source)
    best_match = None
    best_len = -1

    config_candidates = [(cfg_path, load_dataset_cfg) for cfg_path in sorted(DATASET_CONFIGS_DIR.glob("**/*.yaml"))]

    for cfg_path, loader in config_candidates:
        try:
            cfg = loader(cfg_path)
        except Exception:
            continue

        root = cfg.get("path") or ""
        if not root:
            continue

        root_key = _normalize_path_match_key(root)
        if source_key == root_key or source_key.startswith(root_key + "/"):
            if len(root_key) > best_len:
                best_match = cfg
                best_len = len(root_key)

    return best_match


def ensure_dataset_source_available(
    args: Any,
    overwrite: bool = False,
    status_fn: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Download a configured dataset when ``args.source`` targets a missing dataset path."""
    source = getattr(args, "source", None)
    if not source:
        return None

    source_path = Path(source)
    if source_path.exists():
        return None

    cfg = find_dataset_cfg_for_source(source)
    if cfg is None:
        return None

    download_cfg = dict(cfg.get("download") or {})
    source_root = Path(str(cfg.get("path") or "")) if cfg.get("path") else None
    dataset_name = str(cfg.get("id") or (source_root.name if source_root is not None else "dataset"))
    dataset_dest = _resolve_benchmark_dest(cfg, dataset_name, source_root)
    split_name = getattr(args, "split", None) or cfg.get("split")

    dataset_url = _resolve_split_download_value(download_cfg.get("dataset"), split_name)
    dataset_url = _scope_hf_url_to_split(dataset_url, cfg, split_name) if split_name else dataset_url

    download_eval_data(
        runs_url=_resolve_runs_download_url(args, cfg, split_name),
        dataset_url=dataset_url,
        dataset_dest=dataset_dest,
        overwrite=overwrite,
        runs_check_path=None,
        status_fn=status_fn,
    )

    args.dataset_id = cfg.get("id", dataset_name)
    box_type = cfg.get("box_type")
    if box_type:
        args.eval_box_type = str(box_type).lower()

    return cfg


def apply_evaluation_config(
    args: Any,
    overwrite: bool = False,
    status_fn: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Apply the selected experiment or model-free dataset config."""
    experiment_ref = getattr(args, "experiment", None)
    dataset_ref = getattr(args, "dataset", None)
    if experiment_ref and dataset_ref:
        raise ConfigurationError("Select either an experiment or a dataset, not both.")
    return _apply_evaluation_config_ref(
        args,
        dataset_ref or experiment_ref,
        overwrite=overwrite,
        status_fn=status_fn,
        config_kind="dataset" if dataset_ref else "experiment",
    )


def load_evaluation_config_from_args(args: argparse.Namespace) -> dict:
    runtime_cfg = getattr(args, "runtime_evaluation_config", None)
    if isinstance(runtime_cfg, dict):
        return runtime_cfg

    for benchmark in (
        getattr(args, "experiment_id", None),
        getattr(args, "experiment", None),
    ):
        if not benchmark:
            continue
        try:
            return load_experiment_cfg(benchmark) or {}
        except FileNotFoundError:
            continue

    dataset_ref = getattr(args, "dataset", None) or getattr(args, "dataset_id", None)
    if dataset_ref:
        return load_dataset_cfg(
            dataset_ref,
            split=getattr(args, "split", None) or None,
            mode=getattr(args, "workflow_mode", None),
        )

    return {}


def _matches_benchmark_model_reference(
    current_model: str | Path | None,
    benchmark_model: str | Path | None,
    *,
    normalize_stem: bool = False,
) -> bool:
    """Return True when the current runtime model points at the benchmark-selected artifact."""
    if current_model in (None, "") or benchmark_model in (None, ""):
        return False

    current_path = resolve_model_path(current_model)
    benchmark_path = Path(benchmark_model)

    if current_path.name.lower() == benchmark_path.name.lower():
        return True

    if normalize_stem:
        current_stem = current_path.stem.lower().replace("-", "").replace("_", "")
        benchmark_stem = benchmark_path.stem.lower().replace("-", "").replace("_", "")
        if current_stem == benchmark_stem:
            return True

    return False


def configure_benchmark_runtime(
    args: argparse.Namespace,
    *,
    load_evaluation_cfg_fn: Callable[[argparse.Namespace], dict] = load_evaluation_config_from_args,
    should_use_benchmark_detector_fn: Callable[[argparse.Namespace, dict], bool] = should_use_benchmark_detector,
    should_use_benchmark_reid_fn: Callable[[argparse.Namespace, dict], bool] = should_use_benchmark_reid,
    ensure_benchmark_detector_model_fn: Callable[[dict], Optional[Path]] = ensure_benchmark_detector_model,
    ensure_benchmark_reid_model_fn: Callable[[dict], Optional[Path]] = ensure_benchmark_reid_model,
) -> tuple[dict, dict, dict]:
    """Apply benchmark-driven detector and ReID defaults to the current args namespace."""
    benchmark_bundle = load_evaluation_cfg_fn(args)
    benchmark_cfg = benchmark_bundle.get("benchmark", {})
    verbose = bool(getattr(args, "verbose", False))

    use_benchmark_detector = should_use_benchmark_detector_fn(args, benchmark_bundle)
    use_benchmark_reid = should_use_benchmark_reid_fn(args, benchmark_bundle)
    benchmark_detector_cfg = get_benchmark_detector_cfg(benchmark_bundle) if use_benchmark_detector else {}

    required_yolo_model = resolve_required_yolo_model(benchmark_bundle)
    required_reid_model = resolve_required_reid_model(benchmark_bundle)

    # Resolve which artefacts (if any) need to be downloaded so the two
    # ensure_* calls can run concurrently when both downloads are pending.
    detector_needs_download = False
    detector_current: Path | None = None
    if required_yolo_model and use_benchmark_detector:
        detector_current = resolve_model_path(args.detector[0]) if getattr(args, "detector", None) else None
        if not (
            detector_current is not None
            and detector_current.exists()
            and _matches_benchmark_model_reference(
                detector_current, required_yolo_model, normalize_stem=True
            )
        ):
            detector_needs_download = True

    reid_needs_download = False
    reid_current: Path | None = None
    if required_reid_model and use_benchmark_reid:
        reid_current = resolve_model_path(args.reid[0]) if getattr(args, "reid", None) else None
        if not (
            reid_current is not None
            and reid_current.exists()
            and _matches_benchmark_model_reference(reid_current, required_reid_model)
        ):
            reid_needs_download = True

    detector_resolved: Path | None = None
    reid_resolved: Path | None = None

    if detector_needs_download and reid_needs_download:
        import concurrent.futures

        from boxmot.utils.download import (
            get_download_status_fn,
            set_download_status_fn,
        )

        parent_status_fn = get_download_status_fn()
        descriptions = [
            f"Downloading {Path(required_yolo_model).name}",
            f"Downloading {Path(required_reid_model).name}",
        ]

        def _worker(ensure_fn, per_task_cb):
            # Worker threads have their own thread-local: install the
            # per-task callback so download_file routes its progress into
            # the shared parallel-bars panel instead of falling back to
            # tqdm (which would corrupt the Rich Live region).
            if per_task_cb is not None:
                set_download_status_fn(per_task_cb)
            try:
                return ensure_fn(benchmark_bundle)
            finally:
                set_download_status_fn(None)

        if parent_status_fn is not None and callable(
            getattr(parent_status_fn, "parallel_bars", None)
        ):
            with parent_status_fn.parallel_bars(descriptions, unit="B") as task_callbacks:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    det_future = ex.submit(
                        _worker, ensure_benchmark_detector_model_fn, task_callbacks[0]
                    )
                    reid_future = ex.submit(
                        _worker, ensure_benchmark_reid_model_fn, task_callbacks[1]
                    )
                    detector_resolved = det_future.result() or resolve_model_path(required_yolo_model)
                    reid_resolved = reid_future.result() or resolve_model_path(required_reid_model)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                det_future = ex.submit(_worker, ensure_benchmark_detector_model_fn, None)
                reid_future = ex.submit(_worker, ensure_benchmark_reid_model_fn, None)
                detector_resolved = det_future.result() or resolve_model_path(required_yolo_model)
                reid_resolved = reid_future.result() or resolve_model_path(required_reid_model)
    else:
        if detector_needs_download:
            detector_resolved = (
                ensure_benchmark_detector_model_fn(benchmark_bundle)
                or resolve_model_path(required_yolo_model)
            )
        if reid_needs_download:
            reid_resolved = (
                ensure_benchmark_reid_model_fn(benchmark_bundle)
                or resolve_model_path(required_reid_model)
            )

    if required_yolo_model and use_benchmark_detector:
        required_model = detector_resolved if detector_resolved is not None else detector_current
        if verbose and args.detector[0] != required_model:
            LOGGER.info(f"Using benchmark-default detector: {required_model}")
        args.detector = [required_model]

    if required_reid_model and use_benchmark_reid:
        required_model = reid_resolved if reid_resolved is not None else reid_current
        if verbose and args.reid[0] != required_model:
            LOGGER.info(f"Using benchmark-default ReID: {required_model}")
        args.reid = [required_model]

    runtime_reid_cfg = (
        get_benchmark_reid_cfg(benchmark_bundle)
        if use_benchmark_reid
        else (load_runtime_reid_component_cfg(args.reid[0]) if args.reid else {})
    )
    apply_reid_runtime_defaults(args, {"reid": runtime_reid_cfg}, use_config=bool(runtime_reid_cfg))

    dataset_detector_cfg = get_runtime_detector_cfg(args.detector[0], benchmark_detector_cfg)
    args.dataset_detector_cfg = dataset_detector_cfg or None

    if not getattr(args, "eval_box_type", None):
        box_type = benchmark_cfg.get("box_type") or dataset_detector_cfg.get("box_type")
        if box_type:
            args.eval_box_type = str(box_type).lower()

    if args.imgsz is None:
        args.imgsz = (
            list(dataset_detector_cfg["imgsz"])
            if "imgsz" in dataset_detector_cfg
            else default_imgsz(args.detector[0])
        )

    if args.conf is None:
        args.conf = (
            float(dataset_detector_cfg["conf"])
            if "conf" in dataset_detector_cfg
            else default_conf(args.detector[0])
        )

    return benchmark_bundle, benchmark_cfg, dataset_detector_cfg
def eval_init(
    args: argparse.Namespace,
    overwrite: bool = False,
    status_fn: Callable[[str], None] | None = None,
) -> None:
    """Common initialization: apply benchmark data config, then canonicalize paths."""
    apply_evaluation_config(args, overwrite=overwrite, status_fn=status_fn)

    if getattr(args, "source", None) is None:
        raise ConfigurationError(
            "Evaluation setup did not resolve a dataset source. "
            "Select a dataset with --dataset or a composed experiment with --experiment."
        )
    args.source = Path(args.source).resolve()
    args.project = Path(args.project).resolve()
    args.project.mkdir(parents=True, exist_ok=True)
__all__ = [
    "apply_evaluation_config",
    "ensure_dataset_source_available",
    "ensure_benchmark_detector_model",
    "ensure_benchmark_reid_model",
    "find_dataset_cfg_for_source",
    "apply_reid_runtime_defaults",
    "get_benchmark_detector_cfg",
    "get_benchmark_detector_url",
    "get_benchmark_reid_cfg",
    "get_benchmark_reid_url",
    "load_experiment_only_cfg",
    "load_experiment_cfg",
    "load_evaluation_config_from_args",
    "load_detector_component_cfg",
    "load_dataset_cfg",
    "load_reid_component_cfg",
    "load_runtime_reid_component_cfg",
    "resolve_experiment_cfg_path",
    "resolve_detector_cfg_path",
    "resolve_dataset_cfg_path",
    "resolve_reid_cfg_path",
    "resolve_required_reid_device",
    "resolve_required_reid_half",
    "resolve_required_reid_model",
    "resolve_required_yolo_model",
    "should_use_benchmark_detector",
    "should_use_benchmark_reid",
    "configure_benchmark_runtime",
    "eval_init",
]
