from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

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
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.utils.config import ConfigurationError, load_yaml_mapping, resolve_config_path, validate_config_id

REID_CONFIGS_DIR = CONFIG_ROOT / "reid"


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


def iter_reid_config_paths() -> list[Path]:
    """Return all built-in ReID runtime profile paths."""
    return sorted(REID_CONFIGS_DIR.glob("**/*.yaml"))


def resolve_reid_config_path(reference: str | Path) -> Path:
    """Resolve a ReID runtime profile by id, filename, or explicit path."""
    return resolve_config_path(REID_CONFIGS_DIR, reference, "ReID")


def load_reid_config(reference: str | Path) -> dict[str, Any]:
    """Load and validate one ReID runtime profile."""
    path = resolve_reid_config_path(reference)
    raw = load_yaml_mapping(path)
    context = f'ReID config "{path}"'
    reid_id = validate_config_id(_required_text(raw, "id", context), path=path, label="ReID")
    weights = _required_mapping(raw, "weights", context)
    runtime = _required_mapping(raw, "runtime", context)
    preprocessing = _required_mapping(raw, "preprocessing", context)
    model_path = _required_text(weights, "path", context)
    device = str(runtime.get("device") or "auto")
    precision = str(runtime.get("precision") or "fp32").lower()
    if precision not in {"fp16", "fp32", "bf16"}:
        raise ConfigurationError(f"{context} precision must be fp16, fp32, or bf16.")
    preprocess = _required_text(preprocessing, "mode", context)
    image_size = preprocessing.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
    ):
        raise ConfigurationError(f"{context} preprocessing.image_size must contain exactly two positive integers.")
    return {
        "id": reid_id,
        "model": model_path,
        "uri": str(weights.get("uri") or ""),
        "device": device,
        "precision": precision,
        "preprocess": preprocess,
        "image_size": list(image_size),
        "config_path": path,
    }


def reid_config_to_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a validated ReID profile to the runtime mapping contract."""
    device = "" if config["device"] == "auto" else config["device"]
    return {
        "id": config["id"],
        "model": config["model"],
        "default_model": config["model"],
        "uri": config["uri"],
        "url": config["uri"],
        "model_url": config["uri"],
        "device": device,
        "precision": config["precision"],
        "half": config["precision"] == "fp16",
        "preprocess": config["preprocess"],
        "imgsz": config["image_size"],
    }


def load_reid_profile(reference: str | Path) -> dict[str, Any]:
    """Load one ReID profile in runtime-friendly form."""
    return reid_config_to_runtime(load_reid_config(reference))


def find_reid_config_for_model(model: str | Path) -> Path | None:
    """Return the ReID profile whose model filename matches ``model``."""
    target_name = Path(str(model)).name.lower()
    for config_path in iter_reid_config_paths():
        try:
            config = load_reid_config(config_path)
        except (ConfigurationError, FileNotFoundError, OSError):
            continue
        if Path(str(config["model"])).name.lower() == target_name:
            return config_path
    return None


def load_runtime_reid_config(reference: str | Path | None) -> dict[str, Any]:
    """Load a ReID runtime profile by id/path or model filename."""
    if reference in (None, ""):
        return {}
    try:
        return load_reid_profile(str(reference))
    except FileNotFoundError:
        pass

    config_path = find_reid_config_for_model(str(reference))
    return load_reid_profile(config_path) if config_path is not None else {}


def _reid_backend(artifact: str) -> str:
    suffix = Path(artifact).suffix.lower()
    mapping = {
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".onnx": "onnx",
        ".xml": "openvino",
        ".tflite": "tflite",
        ".mlpackage": "coreml",
    }
    try:
        return mapping[suffix]
    except KeyError as exc:
        raise ConfigurationError(f"Cannot infer ReID backend from artifact {artifact!r}.") from exc


def _reid_profile_payload(
    profile: Mapping[str, Any],
    *,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Adapt one validated ReID profile to a component payload."""

    selected_artifact = str(profile["model"] if artifact_path is None else artifact_path)
    return {
        "backend": _reid_backend(selected_artifact),
        "artifact": {"path": selected_artifact, "uri": profile.get("uri")},
        "device": "cpu" if profile.get("device") == "auto" else profile.get("device"),
        "precision": profile["precision"],
        "preprocessing": profile["preprocess"],
        "crop_strategy": "aabb",
        "options": {"image_size": tuple(int(value) for value in profile["image_size"])},
    }


def _direct_reid_payload(artifact_path: Path) -> tuple[dict[str, Any], Path | None]:
    """Build a ReID payload, applying matching filename-profile defaults."""

    profile_path = find_reid_config_for_model(artifact_path)
    if profile_path is not None:
        profile = load_reid_config(profile_path)
        return _reid_profile_payload(profile, artifact_path=artifact_path), profile_path
    return (
        {
            "backend": _reid_backend(str(artifact_path)),
            "artifact": {"path": str(artifact_path)},
            "device": "cpu",
            "precision": "fp32",
            "preprocessing": "default",
            "crop_strategy": "aabb",
            "options": {},
        },
        None,
    )


def resolve_reid_spec(
    reference: str | Path | Mapping[str, Any],
    *,
    allow_download: bool = True,
    artifact_resolver: ArtifactResolver = resolve_artifact,
) -> tuple[ReIDEncoderSpec, dict[str, Any]]:
    """Resolve a ReID ID, YAML mapping, or artifact into a hashed spec."""

    config_path: Path | None = None
    if isinstance(reference, Mapping):
        payload = dict(reference)
    else:
        artifact_path = explicit_artifact_path(reference)
        if artifact_path is not None:
            payload, config_path = _direct_reid_payload(artifact_path)
        else:
            authored = load_component_mapping(reference)
            if authored is not None and "backend" in authored[0]:
                payload, config_path = authored
            else:
                try:
                    profile = load_reid_config(reference)
                except FileNotFoundError:
                    if Path(reference).suffix.lower() in YAML_SUFFIXES:
                        raise
                    payload, config_path = _direct_reid_payload(fallback_artifact_path(reference))
                else:
                    config_path = Path(profile["config_path"])
                    payload = _reid_profile_payload(profile)
    path, uri, expected_hash = required_artifact_values(payload, component="ReID encoder")
    artifact = resolve_component_artifact(
        path,
        uri=uri,
        expected_sha256=expected_hash,
        config_path=config_path,
        allow_download=allow_download,
        artifact_resolver=artifact_resolver,
    )
    spec = ReIDEncoderSpec(
        backend=str(payload.get("backend") or _reid_backend(path)),
        artifact=str(artifact.path),
        artifact_sha256=artifact.sha256,
        device=str(payload.get("device") or "cpu"),
        precision=str(payload.get("precision") or "fp32"),
        options=component_options(payload.get("options")),
        preprocessing=str(payload.get("preprocessing") or "default"),
        crop_strategy=str(payload.get("crop_strategy") or "aabb"),
    )
    return spec, {"spec": asdict(spec), "artifact": artifact_provenance(artifact)}


__all__ = (
    "ConfigurationError",
    "REID_CONFIGS_DIR",
    "find_reid_config_for_model",
    "iter_reid_config_paths",
    "load_reid_config",
    "load_reid_profile",
    "load_runtime_reid_config",
    "reid_config_to_runtime",
    "resolve_reid_spec",
    "resolve_reid_config_path",
)
