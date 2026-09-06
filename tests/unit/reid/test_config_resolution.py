from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import boxmot.reid.config as reid_config
from boxmot.components.artifacts import ResolvedArtifact
from boxmot.reid.config import resolve_reid_spec
from boxmot.reid.core.catalog import TRAINED_URLS


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


@pytest.mark.parametrize(
    ("artifact", "expected_backend"),
    [
        ("model.pt", "pytorch"),
        ("model.torchscript", "torchscript"),
        ("model.onnx", "onnx"),
        ("model_openvino_model", "openvino"),
        ("model.engine", "tensorrt"),
        ("model_coreml_model", "coreml"),
        ("model.tflite", "tflite"),
    ],
)
def test_reid_backend_inference_uses_canonical_formats(artifact, expected_backend) -> None:
    assert reid_config._reid_backend(artifact) == expected_backend


@pytest.mark.parametrize(
    ("artifact_name", "is_directory", "expected_backend"),
    (
        ("model.torchscript", False, "torchscript"),
        ("model.onnx", False, "onnx"),
        ("model_openvino_model", True, "openvino"),
        ("model.engine", False, "tensorrt"),
        ("model_coreml_model", True, "coreml"),
        ("model.tflite", False, "tflite"),
    ),
)
def test_direct_deployed_reid_artifact_resolves_through_canonical_format(
    artifact_name,
    is_directory,
    expected_backend,
    tmp_path,
) -> None:
    artifact = tmp_path / artifact_name
    if is_directory:
        artifact.mkdir()
        (artifact / "artifact.bin").write_bytes(b"deployed model")
    else:
        artifact.write_bytes(b"deployed model")

    spec, _ = resolve_reid_spec(artifact, allow_download=False)

    assert spec.backend == expected_backend
    assert spec.artifact == str(artifact.resolve())


@pytest.mark.parametrize(
    "reference",
    [
        "osnet_x0_25_msmt17",
        "mobilenetv2_x1_4_market1501.pt",
    ],
)
def test_missing_catalog_artifact_uses_download_uri(reference, tmp_path, monkeypatch) -> None:
    filename = reference if Path(reference).suffix else f"{reference}.pt"
    artifact = tmp_path / filename
    calls: list[tuple[Path, str | None, bool]] = []

    monkeypatch.setattr(reid_config, "explicit_artifact_path", lambda _reference: None)
    monkeypatch.setattr(reid_config, "fallback_artifact_path", lambda _reference: artifact)
    monkeypatch.setattr(reid_config, "find_reid_config_for_model", lambda _artifact: None)

    def resolve_missing(
        path,
        *,
        source_uri,
        expected_sha256,
        allow_download,
    ) -> ResolvedArtifact:
        resolved = Path(path)
        resolved.write_bytes(b"downloaded model bytes")
        calls.append((resolved, source_uri, allow_download))
        assert expected_sha256 is None
        return ResolvedArtifact(
            path=resolved,
            sha256="0" * 64,
            source_uri=source_uri,
        )

    spec, provenance = resolve_reid_spec(reference, artifact_resolver=resolve_missing)

    assert spec.backend == "pytorch"
    assert calls == [(artifact, TRAINED_URLS[filename], True)]
    assert provenance["artifact"]["uri"] == TRAINED_URLS[filename]


def test_existing_catalog_named_artifact_does_not_claim_download_uri(tmp_path, monkeypatch) -> None:
    artifact = _artifact(tmp_path, "osnet_x0_25_msmt17.pt")
    monkeypatch.setattr(reid_config, "find_reid_config_for_model", lambda _artifact: None)

    payload, _ = reid_config._direct_reid_payload(artifact)

    assert payload["artifact"] == {"path": str(artifact)}


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
