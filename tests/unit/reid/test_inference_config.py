"""Typed ReID model selection preserves profiles and portable configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from inspect import getdoc, signature
from pathlib import Path

import pytest
import yaml

from boxmot.reid import ReIDConfig
from boxmot.reid.config import resolve_reid_spec


def test_config_constructor_documents_every_editor_visible_argument() -> None:
    """The handwritten overloads must retain useful constructor tooltips."""
    documentation = getdoc(ReIDConfig.__init__)
    assert documentation is not None and "\nArgs:\n" in documentation
    for name in signature(ReIDConfig).parameters:
        assert f"    {name}:" in documentation


def test_config_is_immutable_and_does_not_resolve_model(tmp_path: Path) -> None:
    config = ReIDConfig(tmp_path / "unavailable.pt", allow_download=False)
    assert config.model == str(tmp_path / "unavailable.pt")
    assert config.device is None
    assert config.preprocessing is None
    assert config.batch_size is None
    with pytest.raises(FrozenInstanceError):
        config.device = "cuda:0"
    assert replace(config, batch_size=3).batch_size == 3
    assert config.batch_size is None


def test_config_default_matches_existing_lazy_reid_model() -> None:
    assert ReIDConfig().model == "osnet-x0-25-msmt17"
    assert ReIDConfig().allow_download is True


def test_config_round_trips_yaml_without_mutation(tmp_path: Path) -> None:
    config = ReIDConfig(
        tmp_path / "appearance.pt",
        device="cuda:2",
        precision="fp16",
        preprocessing="resize_pad",
        batch_size=3,
        image_size=(384, 128),
        embedding_dim=512,
        allow_download=False,
    )
    payload = yaml.safe_load(yaml.safe_dump(config.to_dict()))
    restored = ReIDConfig.from_mapping(payload)
    assert restored == config
    payload["image_size"][0] = 1
    assert restored.image_size == (384, 128)
    assert "device" not in ReIDConfig().to_dict()
    assert ReIDConfig.from_mapping({}) == ReIDConfig()


@pytest.mark.parametrize(
    "kwargs,exception,message",
    (
        ({"model": ""}, ValueError, "model"),
        ({"model": " weights.pt"}, ValueError, "model"),
        ({"model": object()}, TypeError, "model"),
        ({"device": ""}, ValueError, "device"),
        ({"device": 1}, ValueError, "device"),
        ({"precision": "int8"}, ValueError, "precision"),
        ({"precision": "bf16"}, ValueError, "precision"),
        ({"precision": []}, ValueError, "precision"),
        ({"preprocessing": "resize pad"}, ValueError, "preprocessing"),
        ({"batch_size": False}, ValueError, "batch_size"),
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"batch_size": 1.5}, ValueError, "batch_size"),
        ({"embedding_dim": 0}, ValueError, "embedding_dim"),
        ({"image_size": (0, 128)}, ValueError, "image_size"),
        ({"image_size": (256, True)}, ValueError, "image_size"),
        ({"image_size": [256, 128]}, ValueError, "image_size"),
        ({"image_size": (256,)}, ValueError, "image_size"),
        ({"allow_download": "false"}, TypeError, "allow_download"),
    ),
)
def test_config_rejects_invalid_settings(kwargs, exception, message) -> None:
    with pytest.raises(exception, match=message):
        ReIDConfig(**kwargs)


def test_config_rejects_unknown_and_non_mapping_yaml_fields() -> None:
    with pytest.raises(TypeError, match="Unexpected reid field 'half'"):
        ReIDConfig.from_mapping({"half": True})
    with pytest.raises(TypeError, match="configuration mapping"):
        ReIDConfig.from_mapping([])


def test_friendly_yaml_anchors_model_path_and_keeps_provenance(tmp_path: Path) -> None:
    directory = tmp_path / "config"
    directory.mkdir()
    artifact = directory / "appearance.pt"
    artifact.write_bytes(b"fixture weights")
    path = directory / "appearance.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "appearance.pt",
                "device": "cuda:2",
                "precision": "fp16",
                "batch_size": 5,
                "image_size": [384, 128],
                "allow_download": False,
            }
        )
    )
    spec, provenance = resolve_reid_spec(path)
    assert spec.artifact == str(artifact)
    assert spec.device == "cuda:2"
    assert spec.precision == "fp16"
    assert spec.option_values() == {"batch_size": 5, "image_size": (384, 128)}
    assert provenance["spec"]["device"] == "cuda:2"
    assert provenance["spec"]["options"] == spec.options


def test_friendly_yaml_can_wrap_resolved_component_yaml(tmp_path: Path) -> None:
    artifact = tmp_path / "appearance.onnx"
    artifact.write_bytes(b"fixture weights")
    component = tmp_path / "native.yaml"
    component.write_text(
        yaml.safe_dump({"backend": "native", "artifact": "appearance.onnx", "options": {"batch_size": 12}})
    )
    friendly = tmp_path / "settings.yaml"
    friendly.write_text(yaml.safe_dump({"model": "native.yaml", "embedding_dim": 8, "allow_download": False}))
    spec, _ = resolve_reid_spec(friendly)
    assert spec.backend == "native"
    assert spec.artifact == str(artifact)
    assert spec.option_values() == {"batch_size": 12, "embedding_dim": 8}


def test_friendly_yaml_rejects_reference_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("model: second.yaml\n")
    second.write_text("model: first.yaml\n")
    with pytest.raises(ValueError, match="configuration cycle"):
        resolve_reid_spec(first)
