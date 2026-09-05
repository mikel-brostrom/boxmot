from __future__ import annotations

import pytest
import yaml

import boxmot.reid.config as reid_config
from boxmot.reid.config import resolve_reid_spec


def _artifact(tmp_path, name="model.pt"):
    path = tmp_path / name
    path.write_bytes(b"resolved model bytes")
    return path


def test_reid_yaml_resolves_relative_artifact_and_hash(tmp_path) -> None:
    artifact = _artifact(tmp_path, "appearance.pt")
    config = tmp_path / "reid.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "backend": "pytorch",
                "artifact": {"path": artifact.name},
                "precision": "fp16",
                "preprocessing": "resize",
                "crop_strategy": "mask",
                "options": {"image_size": [384, 128]},
            }
        ),
        encoding="utf-8",
    )

    spec, _ = resolve_reid_spec(config, allow_download=False)

    assert spec.artifact == str(artifact.resolve())
    assert spec.backend == "pytorch"
    assert spec.precision == "fp16"
    assert spec.preprocessing == "resize"
    assert spec.crop_strategy == "mask"
    assert spec.option_values() == {"image_size": (384, 128)}


def test_reid_binary_artifact_is_not_parsed_as_yaml(tmp_path) -> None:
    artifact = tmp_path / "custom_appearance.pt"
    artifact.write_bytes(b"\x80binary checkpoint")

    spec, _ = resolve_reid_spec(artifact, allow_download=False)

    assert spec.backend == "pytorch"
    assert spec.artifact == str(artifact.resolve())
    assert spec.precision == "fp32"
    assert spec.options == ()


def test_reid_artifact_uses_matching_filename_profile(tmp_path, monkeypatch) -> None:
    artifact = _artifact(tmp_path, "lmbn_n_fixture.pt")
    profile_config = tmp_path / "fixture-reid.yaml"
    profile_config.write_text(
        yaml.safe_dump(
            {
                "id": "fixture-reid",
                "weights": {"path": "models/ignored.pt", "uri": "https://example.test/reid.pt"},
                "runtime": {"device": "auto", "precision": "fp16"},
                "preprocessing": {"mode": "resize", "image_size": [384, 128]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(reid_config, "find_reid_config_for_model", lambda _artifact: profile_config)

    spec, provenance = resolve_reid_spec(artifact, allow_download=False)

    assert spec.artifact == str(artifact.resolve())
    assert spec.device == "cpu"
    assert spec.precision == "fp16"
    assert spec.preprocessing == "resize"
    assert spec.option_values() == {"image_size": (384, 128)}
    assert provenance["artifact"]["uri"] == "https://example.test/reid.pt"


def test_reid_profile_id_remains_a_profile_selector(tmp_path, monkeypatch) -> None:
    artifact = _artifact(tmp_path, "profile-reid.pt")
    requested: list[object] = []

    def load_profile(reference):
        requested.append(reference)
        return {
            "id": "fixture-reid",
            "model": str(artifact),
            "uri": None,
            "device": "auto",
            "precision": "fp16",
            "preprocess": "resize",
            "image_size": [256, 128],
            "config_path": tmp_path / "profile.yaml",
        }

    monkeypatch.setattr(reid_config, "load_reid_config", load_profile)

    spec, _ = resolve_reid_spec("fixture-reid", allow_download=False)

    assert requested == ["fixture-reid"]
    assert spec.artifact == str(artifact.resolve())
    assert spec.preprocessing == "resize"
    assert spec.option_values() == {"image_size": (256, 128)}


def test_reid_hash_mismatch_is_rejected_before_factory_use(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_reid_spec(
            {
                "backend": "pytorch",
                "artifact": {"path": str(artifact), "sha256": "0" * 64},
            },
            allow_download=False,
        )


def test_reid_resolution_requires_artifact_before_build_planning() -> None:
    with pytest.raises(ValueError, match="ReID encoder"):
        resolve_reid_spec({"backend": "fixture"}, allow_download=False)
