"""Public detector shorthand preserves authored settings and artifact identity."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import boxmot.detectors.config as detector_config
import boxmot.detectors.factory as detector_factory
from boxmot.detectors import DetectorCapabilities, DetectorSpec, create_detector


class _Detector:
    """Capture construction while avoiding model runtime imports and downloads."""

    capabilities = DetectorCapabilities()

    def __init__(self, spec: DetectorSpec) -> None:
        self.spec = spec

    def predict(self, frames: Any) -> list:
        return []


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.pt"
    path.write_bytes(b"detector fixture")
    return path


@pytest.fixture(autouse=True)
def registry(monkeypatch: pytest.MonkeyPatch) -> list[DetectorSpec]:
    received = []

    def construct(spec: DetectorSpec) -> _Detector:
        received.append(spec)
        return _Detector(spec)

    monkeypatch.setattr(detector_factory, "_DETECTOR_FACTORIES", SimpleNamespace(resolve=lambda _backend: construct))
    return received


def _spec(artifact: Path, **kwargs: Any) -> DetectorSpec:
    return DetectorSpec(
        "ultralytics",
        artifact=str(artifact.resolve()),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        **kwargs,
    )


def _profile(artifact: Path, monkeypatch: pytest.MonkeyPatch, *, geometry: str = "obb") -> Path:
    """Resolve names through the actual catalog loader with a local checkpoint."""
    path = artifact.parent / "yolo26n.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "yolo26n",
                "box_type": geometry,
                "classes": {0: "person"},
                "inference": {"image_size": [640, 640], "confidence_threshold": 0.2},
                "checkpoints": {"default": {"path": str(artifact)}},
            }
        )
    )
    monkeypatch.setattr(detector_config, "DETECTOR_CONFIGS_DIR", artifact.parent)
    return path


@pytest.mark.parametrize("reference_kind", ("name", "profile_yaml", "component_yaml", "mapping", "artifact", "spec"))
def test_detector_accepts_public_reference_forms(
    artifact: Path, monkeypatch: pytest.MonkeyPatch, reference_kind: str
) -> None:
    payload = {"backend": "ultralytics", "artifact": {"path": artifact.name}, "geometry_mode": "obb"}
    if reference_kind in {"name", "profile_yaml"}:
        path = _profile(artifact, monkeypatch)
        reference = "yolo26n" if reference_kind == "name" else path
    elif reference_kind == "component_yaml":
        reference = artifact.parent / "component.yaml"
        reference.write_text(yaml.safe_dump(payload))
    elif reference_kind == "mapping":
        reference = {**payload, "artifact": str(artifact)}
    elif reference_kind == "spec":
        reference = _spec(artifact, geometry_mode="obb")
    else:
        reference = artifact

    detector = create_detector(reference, device="cpu", allow_download=False)

    assert detector.spec.artifact == str(artifact.resolve())
    assert detector.spec.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert detector.spec.device == "cpu"
    assert detector.spec.geometry_mode == ("auto" if reference_kind == "artifact" else "obb")


@pytest.mark.parametrize("as_spec", (False, True))
def test_runtime_overrides_merge_without_mutating_authored_settings(artifact: Path, as_spec: bool) -> None:
    authored = {
        "backend": "ultralytics",
        "artifact": str(artifact),
        "device": "cuda:0",
        "precision": "fp16",
        "preprocessing": "letterbox",
        "geometry_mode": "auto",
        "options": {"confidence": 0.2, "image_size": [640, 640]},
    }
    reference = (
        _spec(
            artifact,
            device="cuda:0",
            precision="fp16",
            preprocessing="letterbox",
            options=(("confidence", 0.2), ("image_size", (640, 640))),
        )
        if as_spec
        else authored
    )
    original = copy.deepcopy(reference)
    options = {"confidence": 0.4, "classes": [2, 7], "nested": {"sizes": [1, 2]}}

    detector = create_detector(
        reference,
        device="cpu",
        precision="fp32",
        preprocessing="default",
        geometry="obb",
        options=options,
        allow_download=False,
    )

    assert reference == original
    assert detector.spec.device == "cpu"
    assert detector.spec.precision == "fp32"
    assert detector.spec.preprocessing == "default"
    assert detector.spec.geometry_mode == "obb"
    assert detector.spec.option_values() == {
        "classes": (2, 7),
        "confidence": 0.4,
        "image_size": (640, 640),
        "nested": (("sizes", (1, 2)),),
    }
    options["classes"].append(8)
    options["nested"]["sizes"].append(3)
    assert detector.spec.option_values()["classes"] == (2, 7)
    assert detector.spec.option_values()["nested"] == (("sizes", (1, 2)),)


def test_advanced_spec_bypasses_reference_resolution(artifact: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_config, "resolve_detector_spec", lambda *a, **kw: pytest.fail("spec is resolved"))
    spec = _spec(artifact, device="cuda:0", precision="fp16", geometry_mode="obb")
    assert create_detector(spec=spec, allow_download=False).spec is spec
    assert create_detector(spec, device="cpu").spec == replace(spec, device="cpu")


@pytest.mark.parametrize("geometry", (None, "auto"))
def test_known_artifact_retains_profile_geometry(
    artifact: Path, monkeypatch: pytest.MonkeyPatch, geometry: str | None
) -> None:
    _profile(artifact, monkeypatch)
    detector = create_detector(artifact, geometry=geometry, allow_download=False)
    assert detector.spec.geometry_mode == "obb"
    assert detector.spec.option_values()["image_size"] == (640, 640)


@pytest.mark.parametrize("as_spec", (False, True))
def test_geometry_conflict_fails_before_artifact_resolution_or_construction(
    artifact: Path, monkeypatch: pytest.MonkeyPatch, registry: list[DetectorSpec], as_spec: bool
) -> None:
    reference = (
        _spec(artifact, geometry_mode="obb")
        if as_spec
        else {"backend": "ultralytics", "artifact": str(artifact), "geometry_mode": "obb"}
    )
    monkeypatch.setattr(
        detector_config, "resolve_component_artifact", lambda *a, **kw: pytest.fail("geometry is incompatible")
    )
    with pytest.raises(ValueError, match="does not match requested"):
        create_detector(reference, geometry="aabb")
    assert not registry


@pytest.mark.parametrize("reference", (None, 42, [], object()))
def test_invalid_reference_types_are_rejected(reference: Any, registry: list[DetectorSpec]) -> None:
    with pytest.raises(TypeError, match="spec must be"):
        create_detector(reference)
    assert not registry


@pytest.mark.parametrize("reference", ("", " ", " yolo26n"))
def test_empty_or_noncanonical_names_are_rejected(reference: str) -> None:
    with pytest.raises(ValueError, match="reference"):
        create_detector(reference)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    (
        ({"geometry": "rotated"}, ValueError, "geometry"),
        ({"precision": "int8"}, ValueError, "precision"),
        ({"device": ""}, ValueError, "device"),
        ({"preprocessing": "Invalid"}, ValueError, "preprocessing"),
        ({"allow_download": "false"}, TypeError, "allow_download"),
        ({"options": []}, ValueError, "mapping"),
        ({"options": {1: "invalid"}}, ValueError, "keys must be strings"),
        ({"options": {"confidence": float("nan")}}, ValueError, "non-finite"),
    ),
)
def test_invalid_overrides_never_reach_backend(
    artifact: Path, registry: list[DetectorSpec], options: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        create_detector(_spec(artifact), **options)
    assert not registry


def test_download_flag_controls_missing_artifact_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from boxmot.resources import download

    calls = []
    artifact = tmp_path / "missing.pt"

    def download_file(uri: str, destination: Path) -> None:
        calls.append((uri, destination))
        destination.write_bytes(b"downloaded fixture")

    monkeypatch.setattr(download, "download_file", download_file)
    payload = {
        "backend": "ultralytics",
        "artifact": {"path": str(artifact), "uri": "https://example.test/model.pt"},
    }
    with pytest.raises(FileNotFoundError, match="Model artifact does not exist"):
        create_detector(payload, allow_download=False)
    assert not calls
    detector = create_detector(payload)
    assert calls == [("https://example.test/model.pt", artifact)]
    assert detector.spec.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
