from __future__ import annotations

import hashlib

import pytest
import yaml

import boxmot.components.resolution as component_resolution
import boxmot.detectors.config as detector_config
from boxmot.components.artifacts import ResolvedArtifact
from boxmot.detectors.config import resolve_detector_spec


def _artifact(tmp_path, name="model.pt"):
    path = tmp_path / name
    path.write_bytes(b"resolved model bytes")
    return path


def test_detector_yaml_resolves_relative_artifact_and_hash(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    config = tmp_path / "detector.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "backend": "ultralytics",
                "artifact": {"path": artifact.name},
                "geometry_mode": "aabb",
                "options": {"confidence": 0.25, "image_size": [640, 640]},
            }
        ),
        encoding="utf-8",
    )

    spec, provenance = resolve_detector_spec(config, geometry="aabb", allow_download=False)

    assert spec.artifact == str(artifact.resolve())
    assert spec.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert spec.options == (("confidence", 0.25), ("image_size", (640, 640)))
    assert provenance["artifact"]["sha256"] == spec.artifact_sha256


def test_detector_binary_artifact_is_not_parsed_as_yaml(tmp_path) -> None:
    artifact = tmp_path / "custom_yolo26n.pt"
    artifact.write_bytes(b"\x80binary checkpoint")

    spec, _ = resolve_detector_spec(artifact, geometry="aabb", allow_download=False)

    assert spec.backend == "ultralytics"
    assert spec.artifact == str(artifact.resolve())
    assert spec.options == ()


@pytest.mark.parametrize("selector", ("yolo26n", "yolov8n"))
def test_missing_bare_ultralytics_selector_resolves_before_spec_hashing(
    selector,
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / f"{selector}.pt"
    calls = []

    monkeypatch.setattr(component_resolution, "resolve_model_path", lambda _path: artifact)
    monkeypatch.setattr(detector_config, "load_detector_artifact_profile", lambda _artifact: {})

    def missing_profile(_reference):
        raise FileNotFoundError(selector)

    monkeypatch.setattr(detector_config, "load_detector_profile", missing_profile)
    monkeypatch.setattr(
        component_resolution.yaml,
        "safe_load",
        lambda _stream: pytest.fail("a bare detector selector must not be parsed as YAML"),
    )

    def resolve_missing(path, *, source_uri, expected_sha256, allow_download):
        calls.append((path, source_uri, expected_sha256, allow_download))
        assert not artifact.exists()
        artifact.write_bytes(b"downloaded official checkpoint")
        return ResolvedArtifact(
            path=artifact,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            source_uri=source_uri,
        )

    spec, provenance = resolve_detector_spec(
        selector,
        geometry="aabb",
        artifact_resolver=resolve_missing,
    )

    assert len(calls) == 1
    resolved_path, source_uri, expected_sha256, allow_download = calls[0]
    assert resolved_path == artifact
    assert source_uri == (
        "https://github.com/ultralytics/assets/releases/download/"
        f"v8.4.0/{selector}.pt"
    )
    assert expected_sha256 is None
    assert allow_download is True
    assert spec.backend == "ultralytics"
    assert spec.artifact == str(artifact.resolve())
    assert spec.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert provenance["artifact"]["uri"] == source_uri


def test_detector_artifact_uses_matching_profile_defaults(tmp_path, monkeypatch) -> None:
    artifact = _artifact(tmp_path, "yolox_x_fixture.pt")
    profile_config = tmp_path / "detector-profile.yaml"
    monkeypatch.setattr(
        detector_config,
        "load_detector_artifact_profile",
        lambda _artifact: {
            "id": "fixture-detector",
            "model": "models/ignored.pt",
            "uri": "https://example.test/detector.pt",
            "box_type": "aabb",
            "classes": {0: "person", 2: "car"},
            "conf": 0.15,
            "imgsz": [736, 1280],
            "config_path": profile_config,
        },
    )

    spec, provenance = resolve_detector_spec(
        artifact,
        geometry="aabb",
        allow_download=False,
    )

    assert spec.backend == "yolox"
    assert spec.artifact == str(artifact.resolve())
    assert spec.option_values() == {
        "classes": (0, 2),
        "confidence": 0.15,
        "image_size": (736, 1280),
    }
    assert provenance["artifact"]["uri"] == "https://example.test/detector.pt"


def test_detector_profile_id_remains_a_profile_selector(tmp_path, monkeypatch) -> None:
    artifact = _artifact(tmp_path, "profile-detector.pt")
    requested: list[object] = []

    def load_profile(reference):
        requested.append(reference)
        return {
            "id": "fixture-detector",
            "model": str(artifact),
            "uri": None,
            "box_type": "aabb",
            "classes": {0: "person"},
            "conf": 0.2,
            "imgsz": [640, 640],
            "config_path": tmp_path / "profile.yaml",
        }

    monkeypatch.setattr(detector_config, "load_detector_profile", load_profile)

    spec, _ = resolve_detector_spec(
        "fixture-detector",
        geometry="aabb",
        allow_download=False,
    )

    assert requested == ["fixture-detector"]
    assert spec.artifact == str(artifact.resolve())
    assert spec.option_values()["confidence"] == 0.2


def test_detector_resolution_requires_artifact_before_build_planning() -> None:
    with pytest.raises(ValueError, match="Detector"):
        resolve_detector_spec({"backend": "fixture"}, geometry="aabb", allow_download=False)
