"""Metadata sidecars shared by exported ReID artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

ARTIFACT_METADATA_SCHEMA_VERSION = 1
MODEL_KWARGS_SCHEMA_VERSION = 1
EXPORT_FINGERPRINT_SCHEMA_VERSION = 1
COREML_MANIFEST_NAME = "manifest.json"


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reid_code_sha256() -> str:
    """Fingerprint executable ReID Python sources used to build export graphs."""
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(file_sha256(source).encode("ascii"))
    return digest.hexdigest()


def export_content_fingerprint(
    source: str | Path,
    contract: Mapping[str, Any],
) -> str:
    """Fingerprint source weights, executable ReID code, and export settings."""
    source_path = Path(source)
    payload = {
        "schema_version": EXPORT_FINGERPRINT_SCHEMA_VERSION,
        "source_sha256": file_sha256(source_path) if source_path.is_file() else None,
        "reid_code_sha256": reid_code_sha256(),
        "contract": dict(contract),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_content_matches(path: str | Path, expected_sha256: str | None) -> bool:
    """Return whether an artifact exists and matches its recorded digest."""
    artifact = Path(path)
    if not expected_sha256 or not artifact.is_file():
        return False
    try:
        return file_sha256(artifact) == expected_sha256
    except OSError:
        return False


def artifact_metadata_path(path: str | Path) -> Path:
    """Return the canonical metadata path for a file or directory artifact."""
    artifact = Path(path)
    if artifact.is_dir() or artifact.name.lower().endswith("_coreml_model"):
        return artifact / COREML_MANIFEST_NAME
    return artifact.with_suffix(f"{artifact.suffix}.metadata.json")


def read_artifact_metadata(path: str | Path) -> dict[str, Any]:
    """Read artifact metadata, returning an empty mapping when unavailable."""
    metadata_path = artifact_metadata_path(path)
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_artifact_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Atomically write one artifact metadata sidecar."""
    metadata_path = artifact_metadata_path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ARTIFACT_METADATA_SCHEMA_VERSION,
        **dict(metadata),
    }
    temporary = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    return metadata_path


def source_artifact_metadata(source: str | Path) -> dict[str, Any]:
    """Build portable architecture metadata from a source checkpoint."""
    from boxmot.reid.core.registry import ReIDModelRegistry

    source_path = Path(source)
    model_kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(source_path)
    return {
        "source": str(source_path),
        "source_sha256": file_sha256(source_path) if source_path.is_file() else None,
        "model_name": ReIDModelRegistry.get_model_name(source_path),
        "num_classes": ReIDModelRegistry.get_nr_classes(source_path),
        "preprocess": ReIDModelRegistry.get_checkpoint_preprocess(source_path),
        "model_kwargs_schema_version": MODEL_KWARGS_SCHEMA_VERSION,
        "model_kwargs": model_kwargs,
        "img_size": list(model_kwargs["img_size"]) if model_kwargs.get("img_size") else None,
    }
