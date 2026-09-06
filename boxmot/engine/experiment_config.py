"""Resolution and validation for authored experiment configurations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from boxmot.configs import CONFIG_ROOT
from boxmot.datasets.config import load_dataset_config
from boxmot.detectors.config import load_detector_config
from boxmot.reid.config import load_reid_config
from boxmot.utils.config import ConfigurationError, load_yaml_mapping, resolve_config_path, validate_config_id

EXPERIMENT_CONFIGS_DIR = CONFIG_ROOT / "experiments"
_REID_CROP_STRATEGIES = frozenset({"aabb", "mask_aware", "perspective", "rotated"})


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
        "sha256": checkpoint["sha256"],
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
    if source != "model":
        raise ConfigurationError(
            f'{context} detections.source must be "model"; public and positional perception caches are unsupported.'
        )
    model = _required_mapping(detections, "model", context)
    detector_ref = _required_text(model, "ref", context)
    checkpoint = _required_text(model, "checkpoint", context)
    detector = _resolve_detector_checkpoint(detector_ref, checkpoint, dataset)
    return {"source": source, "model": {"ref": detector_ref, "checkpoint": checkpoint}}, detector


def _resolve_reid(
    experiment: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> dict[str, Any] | None:
    reid_cfg = experiment.get("reid")
    reid_ref: str | None = None
    if isinstance(reid_cfg, dict):
        unknown = set(reid_cfg).difference({"ref", "crop_strategy"})
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ConfigurationError(f'Experiment "{experiment.get("id")}" reid has unknown keys: {names}.')
        reid_ref = _required_text(reid_cfg, "ref", f'Experiment "{experiment.get("id")}"')
    elif reid_cfg not in (None, ""):
        raise ConfigurationError(f'Experiment "{experiment.get("id")}" reid must be a mapping.')

    if not reid_ref:
        return None

    resolved = load_reid_config(reid_ref)
    crop_strategy_value = reid_cfg.get("crop_strategy", "aabb")
    if (
        not isinstance(crop_strategy_value, str)
        or not crop_strategy_value
        or crop_strategy_value != crop_strategy_value.strip()
    ):
        raise ConfigurationError(f'Experiment "{experiment.get("id")}" reid.crop_strategy must be a non-empty string.')
    crop_strategy = crop_strategy_value.lower()
    if crop_strategy not in _REID_CROP_STRATEGIES:
        available = ", ".join(sorted(_REID_CROP_STRATEGIES))
        raise ConfigurationError(f'Experiment "{experiment.get("id")}" reid.crop_strategy must be one of: {available}.')
    if crop_strategy in {"perspective", "rotated"} and dataset["box_type"] != "obb":
        raise ConfigurationError(
            f'Experiment "{experiment.get("id")}" reid.crop_strategy={crop_strategy!r} requires an OBB dataset.'
        )
    resolved["crop_strategy"] = crop_strategy
    return resolved


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
    experiment = load_yaml_mapping(source_path)
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
    reid = _resolve_reid(experiment, dataset)
    segmentor = experiment.get("segmentor")
    if segmentor is not None and not isinstance(segmentor, (str, dict)):
        raise ConfigurationError(f'Experiment "{experiment_id}" segmentor must be a config reference or mapping.')
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
        },
        "detections": detections,
        "detector": detector,
        "segmentor": deepcopy(segmentor),
        "reid": None if reid is None else {key: deepcopy(value) for key, value in reid.items() if key != "config_path"},
        "evaluation": {
            "classes": bridge,
            "ignore_dataset_ids": ignore_ids,
        },
    }


__all__ = [
    "ConfigurationError",
    "EXPERIMENT_CONFIGS_DIR",
    "resolve_experiment_config",
    "resolve_experiment_path",
]
