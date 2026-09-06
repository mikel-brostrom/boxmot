from __future__ import annotations

import hashlib
from pathlib import Path

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


def test_resolve_artifact_downloads_huggingface_model_snapshot(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "rtdetr_v2_r18vd"
    calls: list[dict[str, str]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir()
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "model.safetensors").write_bytes(b"weights")
        metadata = local_dir / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True)
        (metadata / "config.json.metadata").write_text("unstable timestamp", encoding="utf-8")
        (local_dir / ".cache" / "required.bin").write_bytes(b"repository artifact")
        return str(local_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    artifact = resolve_artifact(
        destination,
        source_uri="hf://PekingU/rtdetr_v2_r18vd",
        allow_download=True,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == "PekingU/rtdetr_v2_r18vd"
    assert call["repo_type"] == "model"
    assert Path(call["local_dir"]).name.startswith(".rtdetr_v2_r18vd.")
    assert artifact.path == destination.resolve()
    assert artifact.source_uri == "hf://PekingU/rtdetr_v2_r18vd"
    assert artifact.sha256 == sha256_artifact(destination)
    assert not (destination / ".cache" / "huggingface").exists()
    assert (destination / ".cache" / "required.bin").read_bytes() == b"repository artifact"


def test_resolve_artifact_does_not_publish_an_empty_huggingface_snapshot(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "empty-snapshot"

    def fake_snapshot_download(**kwargs):
        metadata = Path(kwargs["local_dir"]) / ".cache" / "huggingface"
        metadata.mkdir(parents=True)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    with pytest.raises(ValueError, match="contains no artifact files"):
        resolve_artifact(
            destination,
            source_uri="hf://owner/empty-model",
            allow_download=True,
        )

    assert not destination.exists()


def test_resolve_artifact_rejects_malformed_huggingface_model_uri(tmp_path) -> None:
    with pytest.raises(ValueError, match="hf://owner/repository"):
        resolve_artifact(
            tmp_path / "snapshot",
            source_uri="hf://missing-repository-owner",
            allow_download=True,
        )


def test_resolve_artifact_hashes_snapshot_directories_and_revalidates_before_use(tmp_path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"weights")

    artifact = resolve_artifact(snapshot)

    assert artifact.path == snapshot.resolve()
    assert len(artifact.sha256) == 64
    assert (
        require_resolved_artifact(str(artifact.path), artifact.sha256, component="test component") == snapshot.resolve()
    )

    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        require_resolved_artifact(str(artifact.path), artifact.sha256, component="test component")


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
