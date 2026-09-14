"""EdgeTAM model-directory resolution and explicit checkpoint handling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from boxmot.resources import download, paths
from boxmot.segmentors.propagation import weights


@pytest.fixture
def model_directory(tmp_path, monkeypatch):
    """Isolate model discovery from the repository's real weights."""
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "models"
    monkeypatch.setattr(paths, "_default_weights_directory", lambda: directory)
    return directory


@pytest.mark.parametrize("authored", ["edgetam.pt", "models/edgetam.pt", "custom/edgetam.pt"])
def test_official_checkpoint_downloads_once_to_resolved_destination(model_directory, monkeypatch, authored):
    payload = b"complete official checkpoint"
    calls = []
    monkeypatch.setattr(weights, "_EDGETAM_SHA256", hashlib.sha256(payload).hexdigest())

    def download_file(url: str, destination: Path) -> Path:
        calls.append((url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return destination

    monkeypatch.setattr(download, "download_file", download_file)
    resolved = weights.resolve_edgetam_checkpoint(authored)
    expected = model_directory / authored if authored == "edgetam.pt" else Path(authored).resolve()
    assert resolved == expected
    assert weights.resolve_edgetam_checkpoint(authored) == resolved
    assert len(calls) == 1
    url, staged = calls[0]
    assert url == "https://huggingface.co/facebook/EdgeTAM/resolve/main/edgetam.pt"
    assert staged.name == "edgetam.pt" and staged.parent.parent == expected.parent
    assert list(expected.parent.iterdir()) == [resolved]


@pytest.mark.parametrize("authored", ["custom.pt", "custom/fine-tuned.pt", "models/edgetam.pt"])
def test_existing_checkpoints_are_preserved(model_directory, monkeypatch, authored):
    checkpoint = Path(authored)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"custom checkpoint")
    monkeypatch.setattr(weights, "resolve_artifact", lambda *args, **kwargs: pytest.fail("Already local"))
    assert weights.resolve_edgetam_checkpoint(authored) == checkpoint.resolve()


def test_bare_checkpoint_discovers_existing_model_directory(model_directory, monkeypatch):
    model_directory.mkdir()
    checkpoint = model_directory / "EdgeTAM.pt"
    checkpoint.touch()
    monkeypatch.setattr(weights, "resolve_artifact", lambda *args, **kwargs: pytest.fail("Already cached"))
    resolved = weights.resolve_edgetam_checkpoint("edgetam.pt")
    assert resolved.parent == model_directory and resolved.samefile(checkpoint)


def test_absolute_custom_checkpoint_is_honored(model_directory, tmp_path):
    checkpoint = tmp_path / "outside-model-directory.pt"
    checkpoint.touch()
    assert weights.resolve_edgetam_checkpoint(checkpoint) == checkpoint


def test_missing_unknown_checkpoint_does_not_download(model_directory, monkeypatch):
    monkeypatch.setattr(weights, "resolve_artifact", lambda *args, **kwargs: pytest.fail("Unknown weights"))
    with pytest.raises(FileNotFoundError, match="Use 'edgetam.pt' to download"):
        weights.resolve_edgetam_checkpoint("fine-tuned.pt")


def test_checkpoint_directory_is_rejected(model_directory):
    (model_directory / "edgetam.pt").mkdir(parents=True)
    with pytest.raises(ValueError, match="checkpoint must be a file"):
        weights.resolve_edgetam_checkpoint("edgetam.pt")


def test_checksum_failure_leaves_no_checkpoint_and_can_retry(model_directory, monkeypatch):
    valid_payload = b"complete official checkpoint"
    attempts = iter([b"truncated checkpoint", valid_payload])
    monkeypatch.setattr(weights, "_EDGETAM_SHA256", hashlib.sha256(valid_payload).hexdigest())

    def download_file(url: str, destination: Path) -> Path:
        destination.write_bytes(next(attempts))
        return destination

    monkeypatch.setattr(download, "download_file", download_file)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        weights.resolve_edgetam_checkpoint("edgetam.pt")
    assert list(model_directory.iterdir()) == []
    checkpoint = weights.resolve_edgetam_checkpoint("edgetam.pt")
    assert checkpoint.read_bytes() == valid_payload
    assert list(model_directory.iterdir()) == [checkpoint]


def test_checkpoint_supplied_during_download_is_preserved(model_directory, monkeypatch):
    payload = b"complete official checkpoint"
    checkpoint = model_directory / "edgetam.pt"
    monkeypatch.setattr(weights, "_EDGETAM_SHA256", hashlib.sha256(payload).hexdigest())

    def download_file(url: str, destination: Path) -> Path:
        destination.write_bytes(payload)
        checkpoint.write_bytes(b"user checkpoint supplied during download")
        return destination

    monkeypatch.setattr(download, "download_file", download_file)
    assert weights.resolve_edgetam_checkpoint("edgetam.pt") == checkpoint
    assert checkpoint.read_bytes() == b"user checkpoint supplied during download"
    assert list(model_directory.iterdir()) == [checkpoint]
