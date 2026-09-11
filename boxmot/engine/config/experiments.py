"""Resolution and validation for authored experiment configurations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from boxmot.configs import CONFIG_ROOT
from boxmot.datasets.config import load_dataset_config
from boxmot.detectors.config import load_detector_config
from boxmot.engine.config.datasets import validate_perception_dataset_inputs
from boxmot.reid.config import load_reid_config
from boxmot.utils.config import CONFIG_ID_PATTERN, ConfigurationError, iter_config_paths, load_yaml_mapping

EXPERIMENT_CONFIGS_DIR = CONFIG_ROOT / "experiments"
_EXPERIMENT_KEYS = frozenset({"mode", "dataset", "detector", "segmentor", "reid", "evaluation"})


@dataclass(frozen=True, slots=True)
class _ResolvedExperiment:
    """Validated semantics and the source identities needed for selector matching."""

    config: dict[str, Any]
    dataset_path: Path
    detector_path: Path
    reid_path: Path | None


def resolve_experiment_path(reference: str | Path) -> Path:
    """Resolve an experiment by explicit path or catalog filename."""

    reference_text = str(reference).strip()
    if not reference_text:
        raise FileNotFoundError("Experiment config reference must not be empty.")

    path = Path(reference_text)
    yaml_suffixes = {".yaml", ".yml"}
    if path.suffix.lower() in yaml_suffixes and path.is_file():
        return path.resolve()

    if path.suffix.lower() in yaml_suffixes:
        relative_candidates = (path,)
    elif path.suffix:
        relative_candidates = ()
    else:
        relative_candidates = (path.with_suffix(".yaml"), path.with_suffix(".yml"))
    catalog_root = EXPERIMENT_CONFIGS_DIR.resolve()
    exact_matches: list[Path] = []
    if not path.is_absolute():
        for relative in relative_candidates:
            candidate = (catalog_root / relative).resolve()
            try:
                candidate.relative_to(catalog_root)
            except ValueError:
                continue
            if candidate.is_file():
                exact_matches.append(candidate)
    exact_matches = list(dict.fromkeys(exact_matches))
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        choices = "\n  - ".join(str(candidate.relative_to(catalog_root)) for candidate in exact_matches)
        raise ConfigurationError(f'Ambiguous experiment reference "{reference}":\n  - {choices}')

    if path.is_absolute() or path.parent != Path("."):
        raise FileNotFoundError(f'Experiment config path does not exist: "{path}"')

    if path.suffix.lower() in yaml_suffixes:
        matches = [candidate.resolve() for candidate in iter_config_paths(catalog_root) if candidate.name == path.name]
    elif path.suffix:
        matches = []
    else:
        matches = [candidate.resolve() for candidate in iter_config_paths(catalog_root) if candidate.stem == path.name]
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = "\n  - ".join(str(candidate.relative_to(catalog_root)) for candidate in matches)
        raise ConfigurationError(
            f'Ambiguous experiment filename "{reference}". Use a catalog-relative path:\n  - {choices}'
        )
    raise FileNotFoundError(f'Experiment config not found for filename "{reference}" in {catalog_root}')


def _experiment_identity(source_path: Path) -> str:
    """Derive stable experiment identity from its filename or catalog path."""

    catalog_root = EXPERIMENT_CONFIGS_DIR.resolve()
    try:
        relative = source_path.resolve().relative_to(catalog_root).with_suffix("")
        identity = "-".join(relative.parts)
        requirement = "catalog directory and filename components"
    except ValueError:
        identity = source_path.stem
        requirement = "filename stem"
    if CONFIG_ID_PATTERN.fullmatch(identity) is None:
        raise ConfigurationError(
            f'Experiment config "{source_path}" has invalid {requirement}; use lowercase kebab-case names.'
        )
    return identity


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


def _resolve_experiment_dataset(reference: str, source_path: Path) -> dict[str, Any]:
    """Resolve authored dataset paths beside the experiment, then use the catalog."""
    path = Path(reference).expanduser()
    if path.is_absolute():
        return load_dataset_config(path)
    local = source_path.parent / path
    if local.exists() or local.is_symlink() or "/" in reference or "\\" in reference or reference in {".", ".."}:
        return load_dataset_config(local)
    # A catalog filename must not be replaced by a same-named file in cwd.
    if path.suffix.lower() in {".yaml", ".yml"}:
        catalog_path = CONFIG_ROOT / "datasets" / path
        if catalog_path.is_file():
            return load_dataset_config(catalog_path)
    return load_dataset_config(reference)


def _resolve_detector_checkpoint(
    detector_ref: str,
    checkpoint_name: str,
    dataset: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Resolve checkpoint semantics while retaining the authored detector's identity."""

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
        "ref": detector_ref,
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
    }, Path(detector["config_path"]).resolve()


def _resolve_detector(
    experiment: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Validate detector selection and resolve its checkpoint and profile path."""

    context = f'Experiment "{experiment.get("id", "<unknown>")}"'
    detector_cfg = _required_mapping(experiment, "detector", context)
    unknown = set(detector_cfg).difference({"ref", "checkpoint"})
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ConfigurationError(f'Experiment "{experiment.get("id")}" detector has unknown keys: {names}.')
    detector_ref = _required_text(detector_cfg, "ref", context)
    checkpoint = _required_text(detector_cfg, "checkpoint", context)
    return _resolve_detector_checkpoint(detector_ref, checkpoint, dataset)


def _resolve_reid(
    experiment: Mapping[str, Any],
) -> dict[str, Any] | None:
    reid_cfg = experiment.get("reid")
    reid_ref: str | None = None
    if isinstance(reid_cfg, dict):
        unknown = set(reid_cfg).difference({"ref"})
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ConfigurationError(f'Experiment "{experiment.get("id")}" reid has unknown keys: {names}.')
        reid_ref = _required_text(reid_cfg, "ref", f'Experiment "{experiment.get("id")}"')
    elif reid_cfg not in (None, ""):
        raise ConfigurationError(f'Experiment "{experiment.get("id")}" reid must be a mapping.')

    if not reid_ref:
        return None

    return load_reid_config(reid_ref)


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

    return _resolve_experiment(reference, split=split, mode=mode).config


def _resolve_experiment(
    reference: str | Path,
    *,
    split: str | None = None,
    mode: str | None = None,
) -> _ResolvedExperiment:
    """Load and validate one experiment, retaining component paths without reloading."""

    source_path = resolve_experiment_path(reference)
    experiment = load_yaml_mapping(source_path)
    context = f'Experiment config "{source_path}"'
    if "id" in experiment:
        raise ConfigurationError(
            f'{context} must not define "id"; experiment identity is derived from its YAML filename.'
        )
    unknown = set(experiment).difference(_EXPERIMENT_KEYS)
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ConfigurationError(f"{context} has unknown keys: {names}.")
    experiment_id = _experiment_identity(source_path)
    experiment["id"] = experiment_id
    dataset_selection = _required_mapping(experiment, "dataset", context)
    dataset_ref = _required_text(dataset_selection, "ref", context)
    dataset = _resolve_experiment_dataset(dataset_ref, source_path)
    split_name = str(split or dataset_selection.get("split") or dataset["default_split"])
    if split_name not in dataset["splits"]:
        available = ", ".join(sorted(dataset["splits"]))
        raise ConfigurationError(
            f'Dataset "{dataset["id"]}" has no split "{split_name}". Available splits: {available}.'
        )
    effective_mode = mode or experiment.get("mode")
    _validate_evaluation_split(dataset, split_name, effective_mode)
    try:
        validate_perception_dataset_inputs(dataset, split_name, mode=effective_mode or "perception experiments")
    except ValueError as error:
        raise ConfigurationError(str(error)) from error

    detector, detector_path = _resolve_detector(experiment, dataset)
    reid = _resolve_reid(experiment)
    segmentor = experiment.get("segmentor")
    if segmentor is not None and not isinstance(segmentor, (str, dict)):
        raise ConfigurationError(f'Experiment "{experiment_id}" segmentor must be a config reference or mapping.')
    bridge, ignore_ids = _resolve_class_bridge(experiment, dataset, detector)
    split_cfg = dataset["splits"][split_name]

    config = {
        "id": experiment_id,
        "mode": str(experiment.get("mode") or "evaluation"),
        "source_path": source_path,
        "dataset": {
            "id": dataset["id"],
            "root": dataset["root"],
            "split": split_name,
            **({"split_path": split_cfg["path"]} if "path" in split_cfg else {}),
            **(
                {"config_path": dataset["config_path"]}
                if not Path(dataset["config_path"]).is_relative_to(CONFIG_ROOT / "datasets")
                else {}
            ),
            "fps": dataset["fps"],
            "modalities": deepcopy(dataset["modalities"]),
            "default_split": dataset["default_split"],
            "layout": dataset["layout"],
            "box_type": dataset["box_type"],
            "has_ground_truth": split_cfg["has_ground_truth"],
            "splits": deepcopy(dataset["splits"]),
            "classes": deepcopy(dataset["classes"]),
            "resources": deepcopy(dataset["resources"]),
        },
        "detector": detector,
        "segmentor": deepcopy(segmentor),
        "reid": None if reid is None else {key: deepcopy(value) for key, value in reid.items() if key != "config_path"},
        "evaluation": {
            "classes": bridge,
            "ignore_dataset_ids": ignore_ids,
        },
    }
    return _ResolvedExperiment(
        config=config,
        dataset_path=Path(dataset["config_path"]).resolve(),
        detector_path=detector_path,
        reid_path=None if reid is None else Path(reid["config_path"]).resolve(),
    )


def _direct_detector_selection(reference: str | Path) -> tuple[dict[str, Any], str | None]:
    """Resolve a direct detector profile and an optional explicit checkpoint."""

    detector_ref = str(reference).strip()
    if not detector_ref:
        raise ConfigurationError("--detector must not be empty.")
    explicit_checkpoint: str | None = None
    try:
        detector = load_detector_config(detector_ref)
    except FileNotFoundError as original_error:
        if "/" not in detector_ref:
            raise
        profile_ref, checkpoint = detector_ref.rsplit("/", 1)
        if not profile_ref or not checkpoint:
            raise original_error
        try:
            detector = load_detector_config(profile_ref)
        except FileNotFoundError:
            raise original_error from None
        detector_ref = profile_ref
        explicit_checkpoint = checkpoint

    checkpoints = detector["checkpoints"]
    if explicit_checkpoint is not None and explicit_checkpoint not in checkpoints:
        available = ", ".join(sorted(checkpoints))
        raise ConfigurationError(
            f'Detector "{detector["id"]}" has no checkpoint "{explicit_checkpoint}". '
            f"Available checkpoints: {available}."
        )
    return detector, explicit_checkpoint


def resolve_matching_experiment_path(
    *,
    dataset: str | Path,
    detector: str | Path,
    reid: str | Path | None = None,
    split: str | None = None,
    mode: str = "eval",
) -> Path:
    """Find the one authored experiment matching direct component selectors."""

    dataset_config = load_dataset_config(dataset)
    split_name = str(split or dataset_config["default_split"])
    if split_name not in dataset_config["splits"]:
        available = ", ".join(sorted(dataset_config["splits"]))
        raise ConfigurationError(
            f'Dataset "{dataset_config["id"]}" has no split "{split_name}". Available splits: {available}.'
        )
    _validate_evaluation_split(dataset_config, split_name, mode)
    try:
        validate_perception_dataset_inputs(dataset_config, split_name, mode=mode)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    detector_config, explicit_checkpoint = _direct_detector_selection(detector)
    reid_config = None if reid is None else load_reid_config(reid)
    reid_id = None if reid_config is None else str(reid_config["id"])
    dataset_path = Path(dataset_config["config_path"]).resolve()
    detector_path = Path(detector_config["config_path"]).resolve()
    reid_path = None if reid_config is None else Path(reid_config["config_path"]).resolve()

    matches: list[Path] = []
    for candidate in iter_config_paths(EXPERIMENT_CONFIGS_DIR):
        resolved = _resolve_experiment(candidate)
        if (
            resolved.dataset_path == dataset_path
            and resolved.config["dataset"]["split"] == split_name
            and resolved.detector_path == detector_path
            and (explicit_checkpoint is None or resolved.config["detector"]["checkpoint"] == explicit_checkpoint)
            and resolved.reid_path == reid_path
        ):
            matches.append(candidate.resolve())

    detector_selector = str(detector_config["id"])
    if explicit_checkpoint is not None:
        detector_selector += f"/{explicit_checkpoint}"
    selector = (
        f'dataset "{dataset_config["id"]}", split "{split_name}", '
        f'detector "{detector_selector}", '
        f'ReID "{reid_id or "none"}"'
    )
    if not matches:
        raise ConfigurationError(
            f"No authored experiment matches {selector}. "
            "Create a matching experiment YAML or select one explicitly with --experiment."
        )
    if len(matches) > 1:
        catalog_root = EXPERIMENT_CONFIGS_DIR.resolve()
        choices = "\n  - ".join(str(path.relative_to(catalog_root)) for path in matches)
        raise ConfigurationError(
            f"Ambiguous direct evaluation selection; multiple authored experiments match {selector}:\n  - {choices}\n"
            "Select one explicitly with --experiment."
        )
    return matches[0]


__all__ = [
    "ConfigurationError",
    "EXPERIMENT_CONFIGS_DIR",
    "resolve_experiment_config",
    "resolve_experiment_path",
    "resolve_matching_experiment_path",
]
