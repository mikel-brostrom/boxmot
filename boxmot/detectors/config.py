"""Detector profile loading and runtime adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from boxmot.components.artifacts import resolve_artifact
from boxmot.components.resolution import (
    YAML_SUFFIXES,
    ArtifactResolver,
    artifact_provenance,
    component_options,
    explicit_artifact_path,
    fallback_artifact_path,
    load_component_mapping,
    required_artifact_values,
    resolve_component_artifact,
)
from boxmot.configs import CONFIG_ROOT
from boxmot.detectors.specs import DetectorSpec
from boxmot.utils.config import (
    ConfigurationError,
    iter_config_paths,
    load_yaml_mapping,
    resolve_config_path,
    validate_config_id,
)

DETECTOR_CONFIGS_DIR = CONFIG_ROOT / "detectors"


def _artifact_key(reference: str | Path) -> str:
    """Normalize an artifact filename for deterministic profile lookup."""

    return Path(str(reference)).stem.lower().replace("-", "").replace("_", "")


def _required_mapping(payload: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f'{context} must define a "{key}" mapping.')
    return dict(value)


def _required_text(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if value in (None, ""):
        raise ConfigurationError(f'{context} must define "{key}".')
    return str(value)


def _normalize_named_classes(raw_classes: Any, context: str) -> dict[str, int]:
    if not isinstance(raw_classes, Mapping) or not raw_classes:
        raise ConfigurationError(f"{context} must define at least one class.")

    classes: dict[str, int] = {}
    for key, value in raw_classes.items():
        if isinstance(value, bool):
            raise ConfigurationError(f"{context} class ids must be integers.")
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            name, class_id = str(value), int(key)
        else:
            name, class_id = str(key), value
            if isinstance(value, Mapping):
                class_id = value.get("id")
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise ConfigurationError(f'{context} class "{name}" must define an integer id.')
        if name in classes:
            raise ConfigurationError(f'{context} defines duplicate class name "{name}".')
        if int(class_id) in classes.values():
            raise ConfigurationError(f"{context} defines duplicate class id {class_id}.")
        classes[name] = int(class_id)
    return classes


def iter_detector_config_paths() -> list[Path]:
    """Return all built-in detector profile paths."""
    return iter_config_paths(DETECTOR_CONFIGS_DIR)


def resolve_detector_config_path(reference: str | Path) -> Path:
    """Resolve a detector profile by id, filename, or explicit path."""
    return resolve_config_path(DETECTOR_CONFIGS_DIR, reference, "detector")


def load_detector_config(reference: str | Path) -> dict[str, Any]:
    """Load and validate one detector profile with named checkpoints."""
    path = resolve_detector_config_path(reference)
    raw = load_yaml_mapping(path)
    context = f'Detector config "{path}"'
    detector_id = validate_config_id(_required_text(raw, "id", context), path=path, label="detector")
    box_type = _required_text(raw, "box_type", context).lower()
    if box_type not in {"aabb", "obb"}:
        raise ConfigurationError(f'{context} box_type must be "aabb" or "obb".')
    classes_by_name = _normalize_named_classes(raw.get("classes"), context)
    inference = _required_mapping(raw, "inference", context)

    image_size = inference.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
    ):
        raise ConfigurationError(f"{context} image_size must contain exactly two positive integers.")
    confidence = inference.get("confidence_threshold")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ConfigurationError(f"{context} confidence_threshold must be within [0, 1].")

    checkpoints = _required_mapping(raw, "checkpoints", context)
    if not checkpoints:
        raise ConfigurationError(f"{context} must define at least one checkpoint.")
    normalized_checkpoints: dict[str, dict[str, str]] = {}
    for checkpoint_name, checkpoint_value in checkpoints.items():
        if not isinstance(checkpoint_value, Mapping):
            raise ConfigurationError(f'{context} checkpoint "{checkpoint_name}" must be a mapping.')
        checkpoint_context = f'{context} checkpoint "{checkpoint_name}"'
        normalized_checkpoints[str(checkpoint_name)] = {
            "path": _required_text(checkpoint_value, "path", checkpoint_context),
            "uri": str(checkpoint_value.get("uri") or ""),
        }

    return {
        "id": detector_id,
        "box_type": box_type,
        "classes_by_name": classes_by_name,
        "classes": {class_id: name for name, class_id in classes_by_name.items()},
        "image_size": list(image_size),
        "confidence_threshold": float(confidence),
        "checkpoints": normalized_checkpoints,
        "config_path": path,
    }


def load_detector_artifact_profile(artifact: str | Path) -> dict[str, Any]:
    """Return runtime settings for the profile checkpoint matching ``artifact``.

    This is a configuration lookup, not a component registry.  Backend
    construction is owned exclusively by :func:`boxmot.detectors.create_detector`
    and the shared registry primitive in :mod:`boxmot.components`.
    """

    artifact_key = _artifact_key(artifact)
    if not artifact_key:
        return {}

    for config_path in iter_detector_config_paths():
        try:
            config = load_detector_config(config_path)
        except (ConfigurationError, FileNotFoundError, OSError):
            continue
        for checkpoint_name, checkpoint in config["checkpoints"].items():
            if _artifact_key(checkpoint["path"]) == artifact_key:
                return detector_config_to_runtime(config, checkpoint_name)
    return {}


def resolve_detector_download_url(uri: str | None) -> str:
    """Translate a detector checkpoint URI into a URL accepted by download helpers."""
    value = str(uri or "")
    if value.startswith("gdrive://"):
        file_id = value.removeprefix("gdrive://").strip("/")
        return f"https://drive.google.com/uc?id={file_id}"
    return value


def detector_config_to_runtime(config: Mapping[str, Any], checkpoint_name: str) -> dict[str, Any]:
    """Adapt a validated detector profile/checkpoint to the runtime mapping contract."""
    checkpoints = config["checkpoints"]
    if checkpoint_name not in checkpoints:
        available = ", ".join(sorted(checkpoints))
        raise ConfigurationError(
            f'Detector "{config["id"]}" has no checkpoint "{checkpoint_name}". Available checkpoints: {available}.'
        )
    checkpoint = checkpoints[checkpoint_name]
    download_url = resolve_detector_download_url(checkpoint["uri"])
    return {
        "id": config["id"],
        "checkpoint": checkpoint_name,
        "model": checkpoint["path"],
        "default_model": checkpoint["path"],
        "uri": checkpoint["uri"],
        "url": download_url,
        "model_url": download_url,
        "box_type": config["box_type"],
        "imgsz": list(config["image_size"]),
        "conf": config["confidence_threshold"],
        "classes": dict(config["classes"]),
        "config_path": config["config_path"],
    }


def load_detector_profile(reference: str | Path) -> dict[str, Any]:
    """Load a detector profile, requiring a checkpoint when the profile has several."""
    checkpoint_name: str | None = None
    try:
        config = load_detector_config(reference)
    except FileNotFoundError as original_error:
        reference_text = str(reference)
        if "/" not in reference_text:
            raise original_error
        detector_ref, checkpoint = reference_text.rsplit("/", 1)
        if not detector_ref or not checkpoint:
            raise original_error
        try:
            config = load_detector_config(detector_ref)
        except FileNotFoundError:
            raise original_error from None
        checkpoint_name = checkpoint

    checkpoints = config["checkpoints"]
    if checkpoint_name is None:
        if len(checkpoints) != 1:
            available = ", ".join(sorted(checkpoints))
            raise ConfigurationError(
                f'Detector "{config["id"]}" has multiple checkpoints. '
                f"Select one as detector/checkpoint. Available checkpoints: {available}."
            )
        checkpoint_name = next(iter(checkpoints))
    return detector_config_to_runtime(config, checkpoint_name)


def _detector_backend(identifier: str, artifact: str) -> str:
    normalized = f"{identifier} {Path(artifact).name}".lower()
    if "yolox" in normalized:
        return "yolox"
    if "rtdetr" in normalized or "rt-detr" in normalized:
        return "rtdetr"
    return "ultralytics"


def _detector_profile_payload(
    profile: Mapping[str, Any],
    *,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Adapt one validated detector profile to a component payload."""

    selected_artifact = str(profile["model"] if artifact_path is None else artifact_path)
    return {
        "backend": _detector_backend(str(profile["id"]), selected_artifact),
        "artifact": {
            "path": selected_artifact,
            "uri": profile.get("uri"),
        },
        "geometry_mode": profile["box_type"],
        "options": {
            "classes": tuple(sorted(int(key) for key in profile["classes"])),
            "confidence": float(profile["conf"]),
            "image_size": tuple(int(value) for value in profile["imgsz"]),
        },
    }


def _direct_detector_payload(
    artifact_path: Path,
    *,
    geometry: str,
    artifact_uri: str | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Build a detector payload for an explicit checkpoint or snapshot."""

    profile = load_detector_artifact_profile(artifact_path)
    if profile:
        config_path = Path(profile["config_path"])
        return _detector_profile_payload(profile, artifact_path=artifact_path), config_path
    return (
        {
            "backend": _detector_backend(artifact_path.stem, artifact_path.name),
            "artifact": {"path": str(artifact_path), "uri": artifact_uri},
            "geometry_mode": geometry,
            "options": {},
        },
        None,
    )


def _ultralytics_asset_uri(reference: str | Path) -> str | None:
    """Return the canonical URI for a bare official Ultralytics checkpoint.

    Ultralytics metadata is imported only while resolving a missing bare model
    selector, so importing this configuration module remains runtime-light.
    """

    selector = Path(reference)
    if selector.parent != Path("."):
        return None
    name = selector.name if selector.suffix else f"{selector.name}.pt"

    import inspect

    from ultralytics.utils.downloads import (
        GITHUB_ASSETS_NAMES,
        GITHUB_ASSETS_REPO,
        attempt_download_asset,
    )

    if name not in GITHUB_ASSETS_NAMES:
        return None
    release = inspect.signature(attempt_download_asset).parameters["release"].default
    if not isinstance(release, str) or not release:
        raise RuntimeError("Ultralytics does not declare a canonical checkpoint release.")
    return f"https://github.com/{GITHUB_ASSETS_REPO}/releases/download/{release}/{name}"


def resolve_detector_spec(
    reference: str | Path | Mapping[str, Any],
    *,
    geometry: str,
    allow_download: bool = True,
    artifact_resolver: ArtifactResolver = resolve_artifact,
) -> tuple[DetectorSpec, dict[str, Any]]:
    """Resolve a detector ID, YAML mapping, or artifact into a hashed spec."""

    config_path: Path | None = None
    if isinstance(reference, Mapping):
        payload = dict(reference)
    else:
        artifact_path = explicit_artifact_path(reference)
        if artifact_path is not None:
            payload, config_path = _direct_detector_payload(
                artifact_path,
                geometry=geometry,
            )
        else:
            authored = load_component_mapping(reference)
            if authored is not None and "backend" in authored[0]:
                payload, config_path = authored
            else:
                try:
                    profile = load_detector_profile(reference)
                except FileNotFoundError:
                    if Path(reference).suffix.lower() in YAML_SUFFIXES:
                        raise
                    artifact_path = fallback_artifact_path(reference)
                    payload, config_path = _direct_detector_payload(
                        artifact_path,
                        geometry=geometry,
                        artifact_uri=(
                            _ultralytics_asset_uri(reference)
                            if _detector_backend(str(reference), artifact_path.name) == "ultralytics"
                            else None
                        ),
                    )
                else:
                    config_path = Path(profile["config_path"])
                    payload = _detector_profile_payload(profile)
    configured_geometry = str(payload.get("geometry_mode") or geometry)
    if configured_geometry not in {"auto", geometry}:
        raise ConfigurationError(
            f"Detector geometry_mode {configured_geometry!r} does not match requested build geometry {geometry!r}."
        )
    backend = str(payload.get("backend") or "")
    path, uri, expected_hash = required_artifact_values(payload, component="Detector")
    artifact = resolve_component_artifact(
        path,
        uri=uri,
        expected_sha256=expected_hash,
        config_path=config_path,
        allow_download=allow_download,
        artifact_resolver=artifact_resolver,
    )
    if backend == "rtdetr" and not artifact.path.is_dir():
        raise ConfigurationError("RT-DETR requires a resolved local Hugging Face snapshot directory.")
    if backend in {"ultralytics", "yolox"} and not artifact.path.is_file():
        raise ConfigurationError(f"Detector backend {backend!r} requires a model file artifact.")
    spec = DetectorSpec(
        backend=backend,
        artifact=str(artifact.path),
        artifact_sha256=artifact.sha256,
        device=str(payload.get("device") or "cpu"),
        precision=str(payload.get("precision") or "fp32"),
        options=component_options(payload.get("options")),
        preprocessing=str(payload.get("preprocessing") or "default"),
        geometry_mode=geometry,
    )
    return spec, {"spec": asdict(spec), "artifact": artifact_provenance(artifact)}


__all__ = (
    "ConfigurationError",
    "DETECTOR_CONFIGS_DIR",
    "detector_config_to_runtime",
    "iter_detector_config_paths",
    "load_detector_artifact_profile",
    "load_detector_config",
    "load_detector_profile",
    "resolve_detector_spec",
    "resolve_detector_download_url",
    "resolve_detector_config_path",
)
