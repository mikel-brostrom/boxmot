"""Resolution and validation for dataset, artifact, model, and experiment configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from boxmot.configs import CONFIG_ROOT
from boxmot.data.config import load_dataset_config
from boxmot.detectors.config import load_detector_config
from boxmot.reid.config import load_reid_config
from boxmot.utils.config import ConfigurationError, load_yaml_mapping, resolve_config_path, validate_config_id

EXPERIMENT_CONFIGS_DIR = CONFIG_ROOT / "experiments"

_load_yaml = load_yaml_mapping


def resolve_experiment_path(reference: str | Path) -> Path:
    """Resolve an experiment reference by id, filename, or path."""
    return resolve_config_path(EXPERIMENT_CONFIGS_DIR, reference, "experiment")


def _required_mapping(payload: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f'{context} must define a "{key}" mapping.')
    return dict(value)


def _required_text(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if value in (None, ""):
        raise ConfigurationError(f'{context} must define "{key}".')
    return str(value)


def _normalize_named_classes(raw_classes: Any, context: str) -> dict[str, int]:
    if not isinstance(raw_classes, dict) or not raw_classes:
        raise ConfigurationError(f"{context} must define at least one class.")

    classes: dict[str, int] = {}
    for key, value in raw_classes.items():
        if isinstance(value, bool):
            raise ConfigurationError(f"{context} class ids must be integers.")
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            name, class_id = str(value), int(key)
        else:
            name, class_id = str(key), value
            if isinstance(value, dict):
                class_id = value.get("id")
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise ConfigurationError(f'{context} class "{name}" must define an integer id.')
        if name in classes:
            raise ConfigurationError(f'{context} defines duplicate class name "{name}".')
        if int(class_id) in classes.values():
            raise ConfigurationError(f"{context} defines duplicate class id {class_id}.")
        classes[name] = int(class_id)
    return classes


def _split_detector_producer(value: Any, context: str) -> tuple[str, str]:
    if isinstance(value, dict):
        return _required_text(value, "ref", context), str(value.get("checkpoint") or "default")
    parts = str(value or "").rsplit("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ConfigurationError(f'{context} must identify a detector and checkpoint as "detector/checkpoint".')
    return parts[0], parts[1]


def _resolve_detector_checkpoint(
    detector_ref: str,
    checkpoint_name: str,
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    detector = load_detector_config(detector_ref)
    if detector["box_type"] != dataset["box_type"]:
        raise ConfigurationError(
            f'Detector "{detector["id"]}" uses {detector["box_type"]} boxes, but dataset '
            f'"{dataset["id"]}" uses {dataset["box_type"]} boxes.'
        )
    checkpoint = detector["checkpoints"].get(checkpoint_name)
    if checkpoint is None:
        available = ", ".join(sorted(detector["checkpoints"]))
        raise ConfigurationError(
            f'Detector "{detector["id"]}" has no checkpoint "{checkpoint_name}". Available checkpoints: {available}.'
        )
    return {
        "id": detector["id"],
        "checkpoint": checkpoint_name,
        "model": checkpoint["path"],
        "uri": checkpoint["uri"],
        "box_type": detector["box_type"],
        "image_size": detector["image_size"],
        "confidence_threshold": detector["confidence_threshold"],
        "classes": detector["classes"],
        "classes_by_name": detector["classes_by_name"],
    }


def _resolve_detection_source(
    experiment: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    context = f'Experiment "{experiment.get("id", "<unknown>")}"'
    detections = _required_mapping(experiment, "detections", context)
    source = str(detections.get("source") or "").lower()
    if source not in {"model", "public", "precomputed"}:
        raise ConfigurationError(f"{context} detections.source must be model, public, or precomputed.")

    artifacts = dataset.get("artifacts") or {}
    if source == "model":
        model = _required_mapping(detections, "model", context)
        detector_ref = _required_text(model, "ref", context)
        checkpoint = _required_text(model, "checkpoint", context)
        detector = _resolve_detector_checkpoint(detector_ref, checkpoint, dataset)
        return {"source": source, "model": {"ref": detector_ref, "checkpoint": checkpoint}}, detector

    if source == "public":
        name = _required_text(detections, "name", context)
        public_sources = artifacts.get("public_detections") or {}
        if not isinstance(public_sources, dict) or name not in public_sources:
            available = ", ".join(sorted(public_sources)) or "none"
            raise ConfigurationError(
                f'Public detection source "{name}" is not available for dataset "{dataset["id"]}". '
                f"Available sources: {available}."
            )
        public_cfg = public_sources[name]
        if not isinstance(public_cfg, dict):
            raise ConfigurationError(f'Public detection source "{name}" must be a mapping.')
        classes_by_name = _normalize_named_classes(public_cfg.get("classes"), f'Public detection source "{name}"')
        detector = {
            "id": f"{dataset['id']}-public-{name}",
            "checkpoint": None,
            "model": None,
            "uri": str(public_cfg.get("uri") or ""),
            "box_type": dataset["box_type"],
            "image_size": None,
            "confidence_threshold": None,
            "classes": {class_id: class_name for class_name, class_id in classes_by_name.items()},
            "classes_by_name": classes_by_name,
        }
        return {"source": source, "name": name}, detector

    artifact_name = _required_text(detections, "artifact", context)
    precomputed = artifacts.get("precomputed") or {}
    if not isinstance(precomputed, dict) or artifact_name not in precomputed:
        available = ", ".join(sorted(precomputed)) or "none"
        raise ConfigurationError(
            f'Precomputed artifact "{artifact_name}" is not available for dataset "{dataset["id"]}". '
            f"Available artifacts: {available}."
        )
    artifact = precomputed[artifact_name]
    if not isinstance(artifact, dict):
        raise ConfigurationError(f'Precomputed artifact "{artifact_name}" must be a mapping.')
    contains = {str(value) for value in artifact.get("contains") or []}
    required = {"detections"}
    if experiment.get("reid") or (artifact.get("produced_by") or {}).get("reid"):
        required.add("embeddings")
    missing = sorted(required - contains)
    if missing:
        raise ConfigurationError(
            f'Precomputed artifact "{artifact_name}" is missing required content: {", ".join(missing)}.'
        )
    produced_by = artifact.get("produced_by") or {}
    if not isinstance(produced_by, dict) or not produced_by.get("detector"):
        raise ConfigurationError(f'Precomputed artifact "{artifact_name}" must define produced_by.detector.')
    detector_ref, checkpoint = _split_detector_producer(
        produced_by["detector"], f'Precomputed artifact "{artifact_name}" produced_by.detector'
    )
    detector = _resolve_detector_checkpoint(detector_ref, checkpoint, dataset)
    return {
        "source": source,
        "artifact": artifact_name,
        "uri": str(artifact.get("uri") or ""),
        "contains": sorted(contains),
    }, detector


def _resolve_reid(
    experiment: Mapping[str, Any],
    dataset: Mapping[str, Any],
    detections: Mapping[str, Any],
) -> dict[str, Any] | None:
    reid_cfg = experiment.get("reid")
    reid_ref: str | None = None
    if isinstance(reid_cfg, dict):
        reid_ref = _required_text(reid_cfg, "ref", f'Experiment "{experiment.get("id")}"')
    elif reid_cfg not in (None, ""):
        raise ConfigurationError(f'Experiment "{experiment.get("id")}" reid must be a mapping.')

    if reid_ref is None and detections.get("source") == "precomputed":
        artifact = (dataset.get("artifacts") or {}).get("precomputed", {}).get(detections["artifact"], {})
        produced_reid = (artifact.get("produced_by") or {}).get("reid")
        reid_ref = str(produced_reid) if produced_reid else None
    return load_reid_config(reid_ref) if reid_ref else None


def _resolve_class_bridge(
    experiment: Mapping[str, Any],
    dataset: Mapping[str, Any],
    detector: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    if detector is None:
        raise ConfigurationError(f'Experiment "{experiment.get("id")}" has no detector class metadata.')
    evaluation = _required_mapping(experiment, "evaluation", f'Experiment "{experiment.get("id")}"')
    class_map = evaluation.get("class_map")
    dataset_classes = dataset["classes"]
    detector_classes = detector["classes_by_name"]

    if class_map == "auto":
        class_map = {
            name: name
            for name, metadata in dataset_classes.items()
            if metadata["evaluation"] == "target" and name in detector_classes
        }
        missing_targets = [
            name
            for name, metadata in dataset_classes.items()
            if metadata["evaluation"] == "target" and name not in detector_classes
        ]
        if missing_targets:
            raise ConfigurationError(
                "Automatic class mapping failed; detector classes are missing: " + ", ".join(missing_targets) + "."
            )
    elif not isinstance(class_map, dict) or not class_map:
        raise ConfigurationError(
            f'Experiment "{experiment.get("id")}" evaluation.class_map must be "auto" or a mapping.'
        )

    bridge: list[dict[str, Any]] = []
    for dataset_name, detector_name in class_map.items():
        dataset_name = str(dataset_name)
        detector_name = str(detector_name)
        if dataset_name not in dataset_classes:
            raise ConfigurationError(
                f'Dataset class "{dataset_name}" in class_map does not exist in dataset "{dataset["id"]}".'
            )
        if dataset_classes[dataset_name]["evaluation"] != "target":
            raise ConfigurationError(f'Dataset class "{dataset_name}" is marked as ignored and cannot be evaluated.')
        if detector_name not in detector_classes:
            raise ConfigurationError(
                f'Detector class "{detector_name}" in class_map does not exist in detector "{detector["id"]}".'
            )
        bridge.append(
            {
                "name": dataset_name,
                "dataset_id": int(dataset_classes[dataset_name]["id"]),
                "detector_name": detector_name,
                "detector_id": int(detector_classes[detector_name]),
            }
        )
    bridge.sort(key=lambda entry: entry["dataset_id"])
    ignore_ids = sorted(
        int(metadata["id"]) for metadata in dataset_classes.values() if metadata["evaluation"] == "ignore"
    )
    return bridge, ignore_ids


def _validate_evaluation_split(dataset: Mapping[str, Any], split: str, mode: str | None) -> None:
    if str(mode or "").lower() not in {"eval", "evaluation", "tune", "research"}:
        return
    if dataset["splits"][split]["has_ground_truth"]:
        return
    valid = [name for name, metadata in dataset["splits"].items() if metadata["has_ground_truth"]]
    choices = "\n".join(f"  - split: {name}" for name in valid) or "  (none)"
    raise ConfigurationError(
        "Configuration error:\n"
        f'Dataset "{dataset["id"]}" split "{split}" has no ground truth and cannot be evaluated.\n\n'
        f"Use one of:\n{choices}\n\nOr run with mode: inference."
    )


def resolve_experiment_config(
    reference: str | Path,
    *,
    split: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Resolve an experiment into a complete, validated semantic configuration."""
    source_path = resolve_experiment_path(reference)
    experiment = _load_yaml(source_path)
    context = f'Experiment config "{source_path}"'
    experiment_id = validate_config_id(
        _required_text(experiment, "id", context),
        path=source_path,
        label="experiment",
    )
    dataset_selection = _required_mapping(experiment, "dataset", context)
    dataset_ref = _required_text(dataset_selection, "ref", context)
    dataset = load_dataset_config(dataset_ref)
    split_name = str(split or dataset_selection.get("split") or dataset["default_split"])
    if split_name not in dataset["splits"]:
        available = ", ".join(sorted(dataset["splits"]))
        raise ConfigurationError(
            f'Dataset "{dataset["id"]}" has no split "{split_name}". Available splits: {available}.'
        )
    effective_mode = mode or experiment.get("mode")
    _validate_evaluation_split(dataset, split_name, effective_mode)

    detections, detector = _resolve_detection_source(experiment, dataset)
    reid = _resolve_reid(experiment, dataset, detections)
    bridge, ignore_ids = _resolve_class_bridge(experiment, dataset, detector)
    split_cfg = dataset["splits"][split_name]

    return {
        "id": experiment_id,
        "mode": str(experiment.get("mode") or "evaluation"),
        "source_path": source_path,
        "dataset": {
            "id": dataset["id"],
            "root": dataset["root"],
            "split": split_name,
            "split_path": split_cfg["path"],
            "layout": dataset["layout"],
            "box_type": dataset["box_type"],
            "has_ground_truth": split_cfg["has_ground_truth"],
            "splits": deepcopy(dataset["splits"]),
            "classes": deepcopy(dataset["classes"]),
            "resources": deepcopy(dataset["resources"]),
            "artifacts": deepcopy(dataset["artifacts"]),
        },
        "detections": detections,
        "detector": detector,
        "reid": None if reid is None else {key: deepcopy(value) for key, value in reid.items() if key != "config_path"},
        "evaluation": {
            "classes": bridge,
            "ignore_dataset_ids": ignore_ids,
        },
    }


def experiment_to_runtime_config(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a resolved semantic config to the evaluator's runtime structure."""
    dataset = resolved["dataset"]
    detector = resolved.get("detector") or {}
    reid = resolved.get("reid") or {}
    bridge = list((resolved.get("evaluation") or {}).get("classes") or [])
    ignore_ids = list((resolved.get("evaluation") or {}).get("ignore_dataset_ids") or [])
    split_name = str(dataset["split"])
    split_paths = {name: dict(metadata) for name, metadata in (dataset.get("splits") or {}).items()}
    eval_names = {int(entry["dataset_id"]): str(entry["name"]) for entry in bridge}
    ignored_names = {
        int(metadata["id"]): name
        for name, metadata in (dataset.get("classes") or {}).items()
        if metadata.get("evaluation") == "ignore"
    }
    class_mapping = {str(entry["name"]): str(entry["detector_name"]) for entry in bridge}

    resources = dataset.get("resources") or {}
    artifacts = dataset.get("artifacts") or {}
    dataset_resource = resources.get("dataset") or {}
    dataset_download: str | dict[str, str] = ""
    if isinstance(dataset_resource, dict):
        uris = dataset_resource.get("uris")
        if isinstance(uris, dict):
            dataset_download = {str(name): str(uri) for name, uri in uris.items()}
        elif dataset_resource.get("uri"):
            dataset_download = str(dataset_resource["uri"])

    download: dict[str, Any] = {"dataset": dataset_download, "runs": ""}
    if isinstance(dataset_resource, dict) and dataset_resource.get("backend") == "mot17_parquet":
        download.update(
            {
                "source": "parquet",
                "parquet_repo": str(dataset_resource.get("repository") or ""),
            }
        )
    detections = resolved.get("detections") or {}
    if detections.get("source") == "precomputed":
        download["runs"] = {split_name: str(detections.get("uri") or "")}
    if detections.get("source") == "public":
        download["public_detector"] = str(detections["name"]).upper()

    detector_runtime: dict[str, Any] = {}
    if detector:
        detector_runtime = {
            "id": detector.get("id"),
            "checkpoint": detector.get("checkpoint"),
            "model": detector.get("model"),
            "default_model": detector.get("model"),
            "uri": detector.get("uri") or "",
            "url": detector.get("uri") or "",
            "model_url": detector.get("uri") or "",
            "box_type": detector.get("box_type"),
            "imgsz": detector.get("image_size"),
            "conf": detector.get("confidence_threshold"),
            "classes": dict(detector.get("classes") or {}),
        }

    reid_runtime: dict[str, Any] = {}
    if reid:
        reid_runtime = {
            "id": reid.get("id"),
            "model": reid.get("model"),
            "default_model": reid.get("model"),
            "uri": reid.get("uri") or "",
            "url": reid.get("uri") or "",
            "model_url": reid.get("uri") or "",
            "device": "" if reid.get("device") == "auto" else reid.get("device"),
            "half": reid.get("precision") == "fp16",
            "precision": reid.get("precision"),
            "preprocess": reid.get("preprocess"),
            "imgsz": reid.get("image_size"),
        }

    metric_backend = "mot_challenge_obb" if dataset["box_type"] == "obb" else "mot_challenge"
    benchmark = {
        "source": dataset["root"],
        "split": dataset["split_path"],
        "box_type": dataset["box_type"],
        "layout": dataset["layout"],
        "metric_eval": metric_backend,
        "eval_classes": eval_names,
        "distractor_classes": ignored_names,
        "class_mapping": class_mapping,
        "class_bridge": bridge,
        "ignore_dataset_ids": ignore_ids,
        "has_ground_truth": dataset["has_ground_truth"],
    }
    evaluation = {
        "box_type": dataset["box_type"],
        "layout": dataset["layout"],
        "metric_eval": metric_backend,
        "classes": {
            "eval": eval_names,
            "distractor": ignored_names,
            "mapping": class_mapping,
            "bridge": bridge,
            "ignore_dataset_ids": ignore_ids,
        },
    }
    return {
        "id": resolved["id"],
        "dataset_config": dataset["id"],
        "detector_config": detector.get("id"),
        "reid_config": reid.get("id"),
        "detector_config_id": detector.get("id"),
        "reid_config_id": reid.get("id"),
        "path": dataset["root"],
        "split": split_name,
        "splits": split_paths,
        "train": split_paths.get("train"),
        "val": split_paths.get("val"),
        "test": split_paths.get("test"),
        "layout": dataset["layout"],
        "box_type": dataset["box_type"],
        "metric_backend": metric_backend,
        "names": eval_names,
        "distractors": ignored_names,
        "class_map": class_mapping,
        "download": download,
        "storage": {"root": dataset["root"], "split": dataset["split_path"]},
        "evaluation": evaluation,
        "benchmark": benchmark,
        "detector": detector_runtime,
        "reid": reid_runtime,
        "detections": deepcopy(detections),
        "resolved": deepcopy(dict(resolved)),
        "source_path": resolved["source_path"],
        "public_detectors": deepcopy(artifacts.get("public_detections") or {}),
    }


def load_experiment_runtime_config(
    reference: str | Path,
    *,
    split: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Resolve an experiment and return the evaluator-compatible configuration."""
    return experiment_to_runtime_config(resolve_experiment_config(reference, split=split, mode=mode))


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    return value


def _effective_tracker_config(args: Any, overrides: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Resolve the tracker parameters that the runtime factory will receive."""
    tracker = getattr(args, "tracker", None)
    if not isinstance(tracker, str) or not tracker:
        return None

    from boxmot.trackers.config import load_tracker_config

    config_reference = getattr(args, "tracker_config", None)
    tracker_kwargs = getattr(args, "tracker_kwargs", None)
    return load_tracker_config(
        tracker,
        config_reference,
        overrides,
        tracker_kwargs if isinstance(tracker_kwargs, Mapping) else None,
    )


def write_experiment_snapshots(
    args: Any,
    run_dir: str | Path,
    *,
    tracker_config: Mapping[str, Any] | None = None,
) -> tuple[Path, Path] | None:
    """Write authored and resolved configs, including effective tracker parameters."""
    source_path = getattr(args, "experiment_source_path", None)
    resolved = getattr(args, "resolved_experiment_config", None)
    if not source_path or not isinstance(resolved, dict):
        return None

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    source_target = run_path / "config.source.yaml"
    resolved_target = run_path / "config.resolved.yaml"
    source_payload = _load_yaml(Path(source_path))
    source_target.write_text(
        yaml.safe_dump(source_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    payload = deepcopy(resolved)
    runtime = {
        "tracker": getattr(args, "tracker", None),
        "tracker_backend": getattr(args, "tracker_backend", None),
        "tracker_config": _effective_tracker_config(args, tracker_config),
        "detection_source": getattr(args, "detection_source", None),
        "detector_models": getattr(args, "detector", None),
        "reid_models": getattr(args, "reid", None),
        "device": getattr(args, "device", None),
        "image_size": getattr(args, "imgsz", None),
        "confidence_threshold": getattr(args, "conf", None),
    }
    payload["runtime"] = {key: value for key, value in runtime.items() if value is not None}
    resolved_target.write_text(
        yaml.safe_dump(_yaml_safe(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return source_target, resolved_target


__all__ = [
    "ConfigurationError",
    "EXPERIMENT_CONFIGS_DIR",
    "experiment_to_runtime_config",
    "load_experiment_runtime_config",
    "resolve_experiment_config",
    "resolve_experiment_path",
    "write_experiment_snapshots",
]
