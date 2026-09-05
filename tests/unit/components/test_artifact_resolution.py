from __future__ import annotations

import hashlib

import pytest

from boxmot.components.artifacts import (
    require_resolved_artifact,
    resolve_artifact,
    sha256_artifact,
)
from boxmot.components.resolution import freeze_json


def test_resolve_artifact_returns_absolute_content_identity(tmp_path) -> None:
    path = tmp_path / "model.bin"
    path.write_bytes(b"stable artifact")

    artifact = resolve_artifact(path)

    expected = hashlib.sha256(b"stable artifact").hexdigest()
    assert artifact.path == path.resolve()
    assert artifact.sha256 == expected
    assert sha256_artifact(path) == expected


def test_resolve_artifact_rejects_missing_and_hash_mismatch(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="resolved local artifact"):
        resolve_artifact(tmp_path / "missing.pt")

    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_artifact(path, expected_sha256="0" * 64)


def test_resolve_artifact_hashes_snapshot_directories_and_revalidates_before_use(tmp_path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"weights")

    artifact = resolve_artifact(snapshot)

    assert artifact.path == snapshot.resolve()
    assert len(artifact.sha256) == 64
    assert require_resolved_artifact(
        str(artifact.path), artifact.sha256, component="test component"
    ) == snapshot.resolve()

    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        require_resolved_artifact(
            str(artifact.path), artifact.sha256, component="test component"
        )


def test_freeze_json_recursively_canonicalizes_mappings() -> None:
    assert freeze_json({"z": [1, {"b": True}], "a": None}) == (
        ("a", None),
        ("z", (1, (("b", True),))),
    )


def test_freeze_json_rejects_noncanonical_keys_and_numbers() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        freeze_json({1: "one"})
    with pytest.raises(ValueError, match="non-finite"):
        freeze_json({"threshold": float("nan")})
