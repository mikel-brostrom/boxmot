from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import yaml

from boxmot.utils import (
    BENCHMARK_CONFIGS,
    BENCHMARK_DATA,
)
from boxmot.utils.download import download_eval_data, download_file
from boxmot.utils.misc import resolve_model_path


def _resolve_yaml_path(config_dir: Path, name: str | Path) -> Path:
    path = Path(name)
    if path.suffix.lower() in {".yaml", ".yml"} and path.exists():
        return path.resolve()

    file_name = path.name if path.suffix else f"{path.name}.yaml"
    # Exact path (including subdirectory, e.g. "ultralytics/default.yaml")
    exact = (config_dir / file_name).resolve()
    if exact.exists():
        return exact

    # Recursive search by stem
    stem = Path(file_name).stem.lower()
    matches = sorted(p.resolve() for p in config_dir.glob("**/*.yaml") if p.stem.lower() == stem)
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Config not found for '{name}' in {config_dir}")


def _load_yaml_cfg(cfg_path: Path) -> dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


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


def _apply_split_component_overrides(cfg: dict[str, Any], split_name: str, cfg_path: Path) -> None:
    """Apply split-specific detector/ReID overrides from ``by_split`` blocks."""
    for component_name in ("detector", "reid"):
        component_cfg = cfg.get(component_name)
        if not isinstance(component_cfg, dict):
            continue

        split_overrides = component_cfg.get("by_split")
        if not isinstance(split_overrides, dict):
            continue

        override = split_overrides.get(split_name)
        if override is None:
            override = split_overrides.get("default") or split_overrides.get("*")
        if not isinstance(override, dict):
            continue

        base_cfg = {k: v for k, v in component_cfg.items() if k != "by_split"}
        merged_cfg = {**base_cfg, **override}
        # If the override updates one side of a mirrored key pair, drop the
        # stale counterpart so normalization can rebuild it from the override.
        if "model" in override and "default_model" not in override:
            merged_cfg.pop("default_model", None)
        if "default_model" in override and "model" not in override:
            merged_cfg.pop("model", None)
        if "url" in override and "model_url" not in override:
            merged_cfg.pop("model_url", None)
        if "model_url" in override and "url" not in override:
            merged_cfg.pop("url", None)

        fallback_id = str(merged_cfg.get("id") or f"{cfg.get('id', cfg_path.stem)}_{component_name}")

        cfg[component_name] = _normalize_inline_component_cfg(merged_cfg, cfg_path, fallback_id=fallback_id)
        cfg[f"{component_name}_config_id"] = cfg[component_name].get("id")


def _component_ref_name(value: Any) -> str | None:
    if isinstance(value, dict):
        ref = value.get("id") or value.get("name")
        return str(ref) if ref else None
    if value in (None, ""):
        return None
    return str(value)


def resolve_dataset_cfg_path(name: str | Path) -> Path:
    """Resolve a dataset config by benchmark name or dataset id in benchmark YAMLs."""
    path = Path(name)
    if path.suffix.lower() in {".yaml", ".yml"} and path.exists():
        return path.resolve()

    benchmark_path = _resolve_yaml_path(BENCHMARK_CONFIGS, name)
    raw_cfg = _load_yaml_cfg(benchmark_path)
    dataset_cfg = raw_cfg.get("dataset")
    if isinstance(dataset_cfg, dict):
        return benchmark_path
    raise FileNotFoundError(f"Dataset config not found for '{name}' in {BENCHMARK_CONFIGS}")


def resolve_benchmark_cfg_path(name: str | Path) -> Path:
    """Resolve a benchmark config by stem or YAML filename."""
    path = Path(name)
    if path.suffix.lower() in {".yaml", ".yml"} and path.exists():
        return path.resolve()
    return _resolve_yaml_path(BENCHMARK_CONFIGS, name)


def resolve_detector_cfg_path(name: str | Path, *, benchmark: str | None = None) -> Path:
    """Resolve a detector profile id from inline benchmark detector blocks."""
    target = str(name)
    # Prefer the current benchmark's YAML when multiple benchmarks share the same id
    if benchmark:
        preferred = BENCHMARK_CONFIGS / f"{benchmark}.yaml"
        if preferred.exists():
            raw = _load_yaml_cfg(preferred)
            det = raw.get("detector")
            if isinstance(det, dict) and str(det.get("id") or "") == target:
                return preferred
    for cfg_path in sorted(BENCHMARK_CONFIGS.glob("*.yaml")):
        raw_cfg = _load_yaml_cfg(cfg_path)
        detector_cfg = raw_cfg.get("detector")
        if isinstance(detector_cfg, dict) and str(detector_cfg.get("id") or "") == target:
            return cfg_path
    raise FileNotFoundError(f"Detector config not found for '{name}' in {BENCHMARK_CONFIGS}")


def resolve_reid_cfg_path(name: str | Path, *, benchmark: str | None = None) -> Path:
    """Resolve a ReID profile id from inline benchmark ReID blocks."""
    target = str(name)
    # Prefer the current benchmark's YAML when multiple benchmarks share the same id
    if benchmark:
        preferred = BENCHMARK_CONFIGS / f"{benchmark}.yaml"
        if preferred.exists():
            raw = _load_yaml_cfg(preferred)
            reid = raw.get("reid")
            if isinstance(reid, dict) and str(reid.get("id") or "") == target:
                return preferred
    for cfg_path in sorted(BENCHMARK_CONFIGS.glob("*.yaml")):
        raw_cfg = _load_yaml_cfg(cfg_path)
        reid_cfg = raw_cfg.get("reid")
        if isinstance(reid_cfg, dict) and str(reid_cfg.get("id") or "") == target:
            return cfg_path
    raise FileNotFoundError(f"ReID config not found for '{name}' in {BENCHMARK_CONFIGS}")


def _normalize_dataset_download(cfg: dict[str, Any]) -> dict[str, Any]:
    download_cfg = dict(cfg.get("download") or {})
    dataset_url = _normalize_download_value(download_cfg.get("dataset"))
    runs_url = _normalize_download_value(download_cfg.get("runs"))
    normalized = {
        "dataset": dataset_url,
        "runs": runs_url,
    }
    dataset_dest = download_cfg.get("dataset_dest")
    if dataset_dest:
        normalized["dataset_dest"] = dataset_dest
    # Preserve additional download options (source, parquet_repo, public_detector, etc.)
    for key in ("source", "parquet_repo", "public_detector"):
        if key in download_cfg:
            normalized[key] = download_cfg[key]
    return normalized


def _metric_backend_for_box_type(box_type: str) -> str:
    """Map the configured box type to the MOT metric backend used at runtime."""
    normalized = str(box_type or "aabb").lower()
    return "mot_challenge_obb" if normalized == "obb" else "mot_challenge"


def _normalize_class_bridge(
    evaluation_cfg: dict[str, Any],
    default_classes: dict,
) -> list[dict[str, Any]]:
    """Normalize benchmark dataset classes to detector classes.

    ``dataset_id`` is the annotation/evaluation class id. ``detector_id`` is
    the class id emitted by the detector and preserved by trackers.
    """
    raw_classes = evaluation_cfg.get("classes") if isinstance(evaluation_cfg, dict) else None
    if not raw_classes:
        return []

    bridge: list[dict[str, Any]] = []
    if isinstance(raw_classes, dict):
        iterable = []
        for class_id, entry in raw_classes.items():
            if isinstance(entry, dict):
                keyed_entry = {"dataset_id": class_id, **entry}
            else:
                keyed_entry = {"dataset_id": class_id, "name": entry}
            iterable.append(keyed_entry)
    else:
        iterable = raw_classes

    for entry in iterable:
        if not isinstance(entry, dict):
            continue
        dataset_id = entry.get("dataset_id")
        name = entry.get("name")
        if dataset_id is None:
            continue
        if name is None:
            name = default_classes.get(int(dataset_id), f"class_{dataset_id}")

        normalized = {
            "name": str(name),
            "dataset_id": int(dataset_id),
        }
        if entry.get("detector_id") is not None:
            normalized["detector_id"] = int(entry["detector_id"])
        if entry.get("detector_name") is not None:
            normalized["detector_name"] = str(entry["detector_name"])
        bridge.append(normalized)

    return sorted(bridge, key=lambda item: int(item["dataset_id"]))


def _class_bridge_eval_names(class_bridge: list[dict[str, Any]], fallback: dict) -> dict:
    if not class_bridge:
        return fallback
    return {
        int(entry["dataset_id"]): str(entry["name"])
        for entry in class_bridge
    }


def _class_bridge_name_mapping(class_bridge: list[dict[str, Any]], fallback: dict) -> dict:
    if not class_bridge:
        return fallback
    return {
        str(entry["name"]): str(entry.get("detector_name") or entry["name"])
        for entry in class_bridge
    }


def _class_bridge_ignore_ids(evaluation_cfg: dict[str, Any], distractors: dict) -> list[int]:
    if isinstance(evaluation_cfg, dict) and "ignore_dataset_ids" in evaluation_cfg:
        return sorted(set(int(value) for value in evaluation_cfg.get("ignore_dataset_ids") or []))
    return sorted(int(class_id) for class_id in distractors)


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


def _normalize_benchmark_cfg(raw_cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    """Normalize dataset-like configs while preserving the current runtime accessors."""
    cfg = dict(raw_cfg or {})
    cfg.setdefault("id", cfg_path.stem.lower())

    path_value = cfg.get("root") or cfg.get("path") or ""

    split_name = str(cfg.get("default_split") or cfg.get("split") or "")

    train_value = cfg.get("train")
    val_value = cfg.get("val")
    test_value = cfg.get("test")

    if not split_name:
        if val_value:
            split_name = "val"
        elif train_value:
            split_name = "train"
        elif test_value:
            split_name = "test"
        else:
            split_name = "train"

    split_paths = {
        "train": train_value,
        "val": val_value,
        "test": test_value,
    }
    # Merge additional named splits (string path or dict with path + seq_pattern + detection_source)
    # Entries from the ``splits:`` block override None top-level values.
    for name, entry in (cfg.get("splits") or {}).items():
        if split_paths.get(name) is None:
            split_paths[name] = entry
    # Pick up extra top-level split keys (e.g. ablation: "ablation")
    _KNOWN_KEYS = {
        "id", "root", "path", "split", "default_split", "train", "val", "test", "splits",
        "layout", "box_type", "detector", "reid", "names", "classes",
        "distractors", "class_map", "download", "dataset_config",
        "detector_config", "reid_config", "seq_pattern",
        "metric_backend", "storage", "evaluation", "benchmark", "defaults",
        "no_gt_splits",
    }
    for key, value in cfg.items():
        if key not in _KNOWN_KEYS and isinstance(value, (str, dict)) and value:
            split_paths.setdefault(key, value)
    if split_paths.get(split_name) is None:
        split_paths[split_name] = split_name

    box_type = cfg.get("box_type") or "aabb"
    layout = cfg.get("layout") or "mot"
    metric_backend = _metric_backend_for_box_type(str(box_type).lower())

    names = dict(cfg.get("classes") or cfg.get("names") or {})
    distractors = dict(cfg.get("distractors") or {})
    class_map = dict(cfg.get("class_map") or {})
    evaluation_cfg = cfg.get("evaluation") if isinstance(cfg.get("evaluation"), dict) else {}
    class_bridge = _normalize_class_bridge(evaluation_cfg, names)
    eval_names = _class_bridge_eval_names(class_bridge, names)
    class_map = _class_bridge_name_mapping(class_bridge, class_map)
    ignore_dataset_ids = _class_bridge_ignore_ids(evaluation_cfg, distractors)

    download_cfg = _normalize_dataset_download(cfg)

    # Support new schema: ``defaults.detector`` / ``defaults.reid``
    defaults_block = cfg.get("defaults") or {}
    detector_ref = _component_ref_name(cfg.get("detector")) or _component_ref_name(defaults_block.get("detector"))
    reid_ref = _component_ref_name(cfg.get("reid")) or _component_ref_name(defaults_block.get("reid"))

    normalized = {
        "id": cfg["id"],
        "path": str(path_value),
        "split": split_name,
        "train": split_paths.get("train"),
        "val": split_paths.get("val"),
        "test": split_paths.get("test"),
        "splits": dict(split_paths),
        "layout": str(layout),
        "box_type": str(box_type).lower(),
        "metric_backend": str(metric_backend),
        "names": names,
        "distractors": distractors,
        "class_map": class_map,
        "download": download_cfg,
        "dataset_config": cfg["id"],
        "detector_config": str(detector_ref) if detector_ref else None,
        "reid_config": str(reid_ref) if reid_ref else None,
        "seq_pattern": cfg.get("seq_pattern"),
        "no_gt_splits": list(cfg.get("no_gt_splits") or []),
    }

    normalized["storage"] = {
        "root": normalized["path"],
        "split": str(split_paths.get(split_name) or split_name),
    }
    normalized["evaluation"] = {
        "box_type": normalized["box_type"],
        "layout": normalized["layout"],
        "metric_eval": normalized["metric_backend"],
        "classes": {
            "eval": eval_names,
            "distractor": distractors,
            "mapping": class_map,
            "bridge": class_bridge,
            "ignore_dataset_ids": ignore_dataset_ids,
        },
    }
    normalized["benchmark"] = {
        "source": normalized["path"],
        "split": str(split_paths.get(split_name) or split_name),
        "box_type": normalized["box_type"],
        "layout": normalized["layout"],
        "metric_eval": normalized["metric_backend"],
        "eval_classes": eval_names,
        "distractor_classes": distractors,
        "class_mapping": class_map,
        "class_bridge": class_bridge,
        "ignore_dataset_ids": ignore_dataset_ids,
    }
    return normalized


def _merge_download_cfg(base_download: dict[str, Any], overlay_download: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_download or {})
    base_dataset = _normalize_download_value(merged.get("dataset"))
    overlay_dataset = _normalize_download_value(overlay_download.get("dataset"))
    if isinstance(base_dataset, dict) and isinstance(overlay_dataset, dict):
        merged["dataset"] = {**base_dataset, **overlay_dataset}
    elif overlay_dataset:
        merged["dataset"] = overlay_dataset
    elif "dataset" in merged:
        merged["dataset"] = base_dataset

    base_runs = _normalize_download_value(merged.get("runs"))
    overlay_runs = _normalize_download_value(overlay_download.get("runs"))
    if isinstance(base_runs, dict) and isinstance(overlay_runs, dict):
        merged["runs"] = {**base_runs, **overlay_runs}
    elif overlay_runs:
        merged["runs"] = overlay_runs
    elif "runs" in merged:
        merged["runs"] = base_runs

    if overlay_download.get("dataset_dest"):
        merged["dataset_dest"] = overlay_download["dataset_dest"]
    # Preserve additional download options from the overlay
    for key in ("source", "parquet_repo", "public_detector"):
        if key in overlay_download:
            merged[key] = overlay_download[key]
    return merged


def _merge_benchmark_bundle_cfg(
    raw_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    cfg_path: Path,
) -> dict[str, Any]:
    """Merge a thin benchmark bundle onto its referenced dataset config."""
    cfg = dict(raw_cfg or {})
    payload = dict(dataset_cfg)

    payload["id"] = str(cfg.get("id") or cfg_path.stem.lower())
    dataset_ref = cfg.get("dataset") or dataset_cfg.get("id")
    if isinstance(dataset_ref, dict):
        dataset_ref = dataset_ref.get("id") or dataset_cfg.get("id")
    if dataset_ref:
        payload["dataset_config"] = str(dataset_ref)

    split_ref = cfg.get("split")
    if split_ref:
        payload["split"] = str(split_ref)

    seq_pattern_ref = cfg.get("seq_pattern")
    if seq_pattern_ref:
        payload["seq_pattern"] = str(seq_pattern_ref)

    if isinstance(cfg.get("evaluation"), dict):
        payload["evaluation"] = cfg["evaluation"]

    detector_ref = _component_ref_name(cfg.get("detector")) or dataset_cfg.get("detector_config")
    reid_ref = _component_ref_name(cfg.get("reid")) or dataset_cfg.get("reid_config")
    if detector_ref:
        payload["detector"] = str(detector_ref)
    if reid_ref:
        payload["reid"] = str(reid_ref)

    payload["download"] = _merge_download_cfg(
        dict(dataset_cfg.get("download") or {}),
        _normalize_dataset_download(cfg),
    )

    normalized = _normalize_benchmark_cfg(payload, cfg_path)
    normalized["dataset_config"] = str(dataset_ref) if dataset_ref else None
    return normalized


def _normalize_component_cfg(raw_cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    """Normalize standalone detector/ReID component configs."""
    cfg = dict(raw_cfg or {})
    cfg.setdefault("id", cfg_path.stem.lower())
    if "default_model" not in cfg and "model" in cfg:
        cfg["default_model"] = cfg["model"]
    if "model" not in cfg and "default_model" in cfg:
        cfg["model"] = cfg["default_model"]
    if "model_url" not in cfg and "url" in cfg:
        cfg["model_url"] = cfg["url"]
    if "url" not in cfg and "model_url" in cfg:
        cfg["url"] = cfg["model_url"]
    return cfg


def _normalize_inline_component_cfg(raw_cfg: dict[str, Any], cfg_path: Path, fallback_id: str) -> dict[str, Any]:
    cfg = dict(raw_cfg or {})
    cfg.setdefault("id", fallback_id)
    inline_path = cfg_path.with_name(f"{cfg['id']}.yaml")
    return _normalize_component_cfg(cfg, inline_path)


def load_dataset_cfg(name: str | Path) -> dict[str, Any]:
    """Load the inline dataset block from a benchmark YAML."""
    cfg_path = resolve_dataset_cfg_path(name)
    raw_cfg = _load_yaml_cfg(cfg_path)
    dataset_cfg = raw_cfg.get("dataset")
    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"Benchmark config '{cfg_path}' must define an inline 'dataset' mapping.")
    dataset_payload = dict(dataset_cfg)
    if isinstance(raw_cfg.get("evaluation"), dict):
        dataset_payload["evaluation"] = raw_cfg["evaluation"]
    normalized_dataset = _normalize_benchmark_cfg(dataset_payload, cfg_path)
    return _merge_benchmark_bundle_cfg(raw_cfg, normalized_dataset, cfg_path)


def load_benchmark_only_cfg(name: str | Path) -> dict[str, Any]:
    """Load a canonical benchmark bundle with an inline dataset mapping."""
    cfg_path = resolve_benchmark_cfg_path(name)
    raw_cfg = _load_yaml_cfg(cfg_path)
    dataset_ref = raw_cfg.get("dataset")
    if not isinstance(dataset_ref, dict):
        raise ValueError(f"Benchmark config '{cfg_path}' must define an inline 'dataset' mapping.")
    dataset_payload = dict(dataset_ref)
    if isinstance(raw_cfg.get("evaluation"), dict):
        dataset_payload["evaluation"] = raw_cfg["evaluation"]
    dataset_cfg = _normalize_benchmark_cfg(dataset_payload, cfg_path)
    return _merge_benchmark_bundle_cfg(raw_cfg, dataset_cfg, cfg_path)


def load_detector_component_cfg(name: str | Path) -> dict[str, Any]:
    """Load a detector component from inline benchmark detector blocks."""
    target = str(name)
    for cfg_path in sorted(BENCHMARK_CONFIGS.glob("*.yaml")):
        raw_cfg = _load_yaml_cfg(cfg_path)
        detector_cfg = raw_cfg.get("detector")
        if not isinstance(detector_cfg, dict):
            continue
        if str(detector_cfg.get("id") or "") != target:
            continue
        return _normalize_inline_component_cfg(detector_cfg, cfg_path, fallback_id=target)
    raise FileNotFoundError(f"Detector config not found for '{name}' in {BENCHMARK_CONFIGS}")


def load_reid_component_cfg(name: str | Path) -> dict[str, Any]:
    """Load a ReID component from inline benchmark ReID blocks."""
    target = str(name)
    for cfg_path in sorted(BENCHMARK_CONFIGS.glob("*.yaml")):
        raw_cfg = _load_yaml_cfg(cfg_path)
        reid_cfg = raw_cfg.get("reid")
        if not isinstance(reid_cfg, dict):
            continue
        if str(reid_cfg.get("id") or "") != target:
            continue
        return _normalize_inline_component_cfg(reid_cfg, cfg_path, fallback_id=target)
    raise FileNotFoundError(f"ReID config not found for '{name}' in {BENCHMARK_CONFIGS}")


def load_runtime_reid_component_cfg(name: str | Path | None) -> dict[str, Any]:
    """Load a ReID component config by model/config reference, returning ``{}`` when unmatched."""
    if name in (None, ""):
        return {}
    target = str(name)
    # First match by explicit config id.
    try:
        return load_reid_component_cfg(target)
    except FileNotFoundError:
        pass

    # Then match by model filename across canonical benchmark files.
    target_name = Path(target).name.lower()
    for cfg_path in sorted(BENCHMARK_CONFIGS.glob("*.yaml")):
        raw_cfg = _load_yaml_cfg(cfg_path)
        reid_cfg = raw_cfg.get("reid")
        if not isinstance(reid_cfg, dict):
            continue
        model_ref = reid_cfg.get("model") or reid_cfg.get("default_model")
        if model_ref and Path(str(model_ref)).name.lower() == target_name:
            fallback_id = str(reid_cfg.get("id") or cfg_path.stem)
            return _normalize_inline_component_cfg(reid_cfg, cfg_path, fallback_id=fallback_id)

    return {}


def _combine_benchmark_and_component_cfg(
    benchmark_cfg: dict[str, Any],
    detector_cfg: dict[str, Any] | None = None,
    reid_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(benchmark_cfg)
    cfg["detector"] = dict(detector_cfg or {})
    cfg["reid"] = dict(reid_cfg or {})
    if detector_cfg and detector_cfg.get("id"):
        cfg["detector_config_id"] = detector_cfg["id"]
    if reid_cfg and reid_cfg.get("id"):
        cfg["reid_config_id"] = reid_cfg["id"]
    return cfg


def load_benchmark_cfg(name: str | Path) -> dict[str, Any]:
    """Load a benchmark config and merge in its detector/ReID defaults."""
    cfg_path = resolve_benchmark_cfg_path(name)
    raw_cfg = _load_yaml_cfg(cfg_path)
    benchmark_cfg = load_benchmark_only_cfg(name)

    detector_value = raw_cfg.get("detector")
    reid_value = raw_cfg.get("reid")
    detector_ref = benchmark_cfg.get("detector_config")
    reid_ref = benchmark_cfg.get("reid_config")

    if not isinstance(detector_value, dict):
        raise ValueError(f"Benchmark config '{cfg_path}' must define an inline 'detector' mapping.")
    if not isinstance(reid_value, dict):
        raise ValueError(f"Benchmark config '{cfg_path}' must define an inline 'reid' mapping.")

    detector_cfg = _normalize_inline_component_cfg(
        detector_value,
        cfg_path,
        fallback_id=str(detector_ref or f"{benchmark_cfg['id']}_detector"),
    )
    reid_cfg = _normalize_inline_component_cfg(
        reid_value,
        cfg_path,
        fallback_id=str(reid_ref or f"{benchmark_cfg['id']}_reid"),
    )
    return _combine_benchmark_and_component_cfg(benchmark_cfg, detector_cfg=detector_cfg, reid_cfg=reid_cfg)


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
    return str(cfg.get("id") or cfg_path.stem)


def _normalize_path_match_key(path_like: str | Path) -> str:
    return Path(str(path_like)).as_posix().lower().rstrip("/")


def _resolve_active_split_path(cfg: dict[str, Any]) -> str:
    split = str(cfg.get("split") or "train")
    split_path = cfg.get(split)
    if split_path is None:
        split_path = split
    return str(split_path)


def _apply_benchmark_config_ref(
    args: Any,
    benchmark_ref: str | Path | None,
    overwrite: bool = False,
    status_fn: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Apply a benchmark YAML referenced by ``benchmark_ref`` to the current args namespace."""
    if not benchmark_ref:
        return None

    try:
        cfg_path = resolve_benchmark_cfg_path(benchmark_ref)
    except FileNotFoundError:
        return None

    cfg = load_benchmark_cfg(cfg_path)

    download_cfg = dict(cfg.get("download") or {})
    path_str = str(cfg.get("path") or "")
    source_root = Path(path_str) if path_str else None
    benchmark_name = _resolve_runtime_benchmark_name(cfg, source_root, cfg_path)
    # Respect explicit --split from CLI; otherwise use config default
    cli_split = getattr(args, "split", None)
    if cli_split and getattr(args, "split_explicit", False):
        cfg_split = cli_split
    else:
        cfg_split = str(cfg.get("split") or "train")

    _apply_split_component_overrides(cfg, cfg_split, cfg_path)

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

    args.benchmark_id = cfg.get("id", benchmark_name)
    args.dataset_id = args.benchmark_id
    args.benchmark = benchmark_name

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
    # CLI --detection-source overrides config; config value is the fallback
    cli_detection_source = getattr(args, "detection_source", None)
    if cli_detection_source:
        pass  # keep the explicit CLI value
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

    return cfg


def find_dataset_cfg_for_source(source: str | Path | None) -> dict[str, Any] | None:
    """Return the dataset config whose configured root best matches ``source``."""
    if not source:
        return None

    source_key = _normalize_path_match_key(source)
    best_match = None
    best_len = -1

    config_candidates = [
        (cfg_path, load_benchmark_only_cfg) for cfg_path in sorted(BENCHMARK_CONFIGS.glob("*.yaml"))
    ]

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


def apply_benchmark_config(
    args: Any,
    overwrite: bool = False,
    status_fn: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Apply a benchmark YAML referenced via ``args.data`` to the current args namespace."""
    return _apply_benchmark_config_ref(
        args,
        getattr(args, "data", None),
        overwrite=overwrite,
        status_fn=status_fn,
    )


__all__ = [
    "apply_benchmark_config",
    "ensure_dataset_source_available",
    "ensure_benchmark_detector_model",
    "ensure_benchmark_reid_model",
    "find_dataset_cfg_for_source",
    "apply_reid_runtime_defaults",
    "get_benchmark_detector_cfg",
    "get_benchmark_detector_url",
    "get_benchmark_reid_cfg",
    "get_benchmark_reid_url",
    "load_benchmark_only_cfg",
    "load_benchmark_cfg",
    "load_detector_component_cfg",
    "load_dataset_cfg",
    "load_reid_component_cfg",
    "load_runtime_reid_component_cfg",
    "resolve_benchmark_cfg_path",
    "resolve_detector_cfg_path",
    "resolve_dataset_cfg_path",
    "resolve_reid_cfg_path",
    "resolve_required_reid_device",
    "resolve_required_reid_half",
    "resolve_required_reid_model",
    "resolve_required_yolo_model",
    "should_use_benchmark_detector",
    "should_use_benchmark_reid",
]
