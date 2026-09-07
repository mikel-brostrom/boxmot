"""Shared mechanics for resolving authored component specifications."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml

from boxmot.components.artifacts import ResolvedArtifact, resolve_artifact
from boxmot.resources.paths import resolve_model_path
from boxmot.utils.config import ConfigurationError

YAML_SUFFIXES = frozenset({".yaml", ".yml"})


class ArtifactResolver(Protocol):
    """Callable contract for resolving one immutable component artifact."""

    def __call__(
        self,
        path: str | Path,
        *,
        source_uri: str | None = None,
        expected_sha256: str | None = None,
        allow_download: bool = False,
    ) -> ResolvedArtifact: ...


def freeze_json(value: Any, *, location: str = "value") -> Any:
    """Convert JSON/YAML containers to recursively immutable canonical values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must not contain non-finite numbers.")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, location=f"{location}[]") for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{location} mapping keys must be strings.")
        return tuple(
            (key, freeze_json(item, location=f"{location}.{key}"))
            for key, item in sorted(value.items())
        )
    raise TypeError(f"{location} contains unsupported value {type(value).__name__}.")


def component_options(value: object) -> tuple[tuple[str, Any], ...]:
    """Normalize an authored component options mapping."""

    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ConfigurationError("component options must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ConfigurationError("component option keys must be strings.")
    return tuple(
        (key, freeze_json(item, location=f"options.{key}"))
        for key, item in sorted(value.items())
    )


def load_component_mapping(reference: str | Path) -> tuple[dict[str, Any], Path] | None:
    """Load an explicit component YAML mapping, if ``reference`` names one."""

    path = Path(reference).expanduser()
    if path.suffix.lower() not in YAML_SUFFIXES or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ConfigurationError(f'Component config "{path}" must contain a mapping.')
    return dict(payload), path.resolve()


def explicit_artifact_path(reference: str | Path) -> Path | None:
    """Return an existing non-YAML artifact named explicitly by ``reference``."""

    path = Path(reference).expanduser()
    if path.suffix.lower() in YAML_SUFFIXES:
        return None
    resolved = resolve_model_path(path)
    if resolved.is_file() or resolved.is_dir():
        return resolved.resolve()
    return None


def fallback_artifact_path(reference: str | Path) -> Path:
    """Resolve an authored artifact selector after profile lookup has failed."""

    path = Path(reference).expanduser()
    if not path.suffix:
        path = path.with_suffix(".pt")
    return resolve_model_path(path)


def required_artifact_values(
    payload: Mapping[str, Any],
    *,
    component: str,
) -> tuple[str, str | None, str | None]:
    """Extract and require an artifact path plus optional URI and digest."""

    def optional_text(value: Any) -> str | None:
        return None if value in (None, "") else str(value)

    raw = payload.get("artifact")
    if isinstance(raw, Mapping):
        path = optional_text(raw.get("path"))
        uri = optional_text(raw.get("uri"))
        expected_sha256 = optional_text(raw.get("sha256"))
    else:
        path = optional_text(raw)
        uri = optional_text(payload.get("artifact_uri"))
        expected_sha256 = optional_text(payload.get("artifact_sha256"))
    if path is None:
        raise ConfigurationError(
            f"{component} requires an explicit local artifact path; "
            "model downloads are resolved before component construction."
        )
    return path, uri, expected_sha256


def resolve_component_artifact(
    path: str,
    *,
    uri: str | None,
    expected_sha256: str | None,
    config_path: Path | None,
    allow_download: bool,
    artifact_resolver: ArtifactResolver = resolve_artifact,
) -> ResolvedArtifact:
    """Resolve a component artifact relative to its authored configuration."""

    artifact_path = Path(path).expanduser()
    if not artifact_path.is_absolute():
        # Built-in profiles intentionally anchor their model paths at the
        # project invocation root. Custom YAML keeps relative artifacts beside
        # the authored config.
        if config_path is not None and "boxmot/configs" not in config_path.as_posix():
            artifact_path = config_path.parent / artifact_path
    return artifact_resolver(
        artifact_path,
        source_uri=uri,
        expected_sha256=expected_sha256,
        allow_download=allow_download,
    )


def artifact_provenance(artifact: ResolvedArtifact) -> dict[str, Any]:
    """Serialize a resolved artifact identity for build provenance."""

    return {
        "path": artifact.path.as_posix(),
        "uri": artifact.source_uri,
        "sha256": artifact.sha256,
    }


__all__ = (
    "ArtifactResolver",
    "YAML_SUFFIXES",
    "artifact_provenance",
    "component_options",
    "explicit_artifact_path",
    "fallback_artifact_path",
    "freeze_json",
    "load_component_mapping",
    "required_artifact_values",
    "resolve_component_artifact",
)
