from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import boxmot.components.registry as registry_module
import boxmot.detectors.factory as detector_factory_module
import boxmot.reid.factory as reid_factory_module
import boxmot.segmentors.factory as segmentor_factory_module
from boxmot.components.registry import LazyComponentRegistry
from boxmot.detectors.factory import create_detector
from boxmot.detectors.protocols import Detector, DetectorCapabilities
from boxmot.detectors.specs import DetectorSpec
from boxmot.reid.core.formats import REID_FORMATS
from boxmot.reid.factory import create_reid_encoder
from boxmot.reid.protocols import AppearanceEncoder, EncoderRequirements
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.segmentors.factory import create_segmentor
from boxmot.segmentors.protocols import Segmentor
from boxmot.segmentors.specs import SegmentorSpec
from boxmot.structures import Boxes, Detections, Frame, MaskBatch
from tests._paths import REPO_ROOT


class _Detector:
    capabilities = DetectorCapabilities()

    def predict(self, frames):
        return []


class _Segmentor:
    def segment(self, frames, detections):
        return []


class _Encoder:
    embedding_dim = 8
    requirements = EncoderRequirements()

    def encode(self, frames, detections):
        return []


@pytest.mark.parametrize(
    "spec",
    (
        DetectorSpec("ultralytics"),
        SegmentorSpec("sam"),
        ReIDEncoderSpec("onnx"),
    ),
)
def test_component_specs_are_frozen_and_slotted(spec) -> None:
    assert not hasattr(spec, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.backend = "changed"


@pytest.mark.parametrize(
    ("contract", "field_name"),
    (
        (DetectorCapabilities(), "provides_masks"),
        (EncoderRequirements(), "masks"),
    ),
)
def test_component_capability_contracts_are_frozen_and_slotted(contract, field_name) -> None:
    assert not hasattr(contract, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(contract, field_name, True)


def test_component_capability_contracts_validate_exact_bools_and_geometry() -> None:
    with pytest.raises(TypeError, match="provides_masks"):
        DetectorCapabilities(provides_masks=1)
    with pytest.raises(ValueError, match="at least one geometry"):
        DetectorCapabilities(supports_aabb=False, supports_obb=False)
    with pytest.raises(TypeError, match="masks"):
        EncoderRequirements(masks=1)


@pytest.mark.parametrize("spec_type", (DetectorSpec, SegmentorSpec, ReIDEncoderSpec))
def test_component_specs_require_canonical_backend_precision_and_artifact_hash(spec_type) -> None:
    with pytest.raises(ValueError, match="canonical lowercase"):
        spec_type("ONNX")
    with pytest.raises(ValueError, match="precision"):
        spec_type("onnx", precision="float32")
    with pytest.raises(ValueError, match="SHA-256"):
        spec_type("onnx", artifact="model.onnx", artifact_sha256="not-a-hash")
    with pytest.raises(ValueError, match="requires an artifact"):
        spec_type("onnx", artifact_sha256="a" * 64)


@pytest.mark.parametrize("spec_type", (DetectorSpec, SegmentorSpec, ReIDEncoderSpec))
def test_component_specs_require_sorted_unique_immutable_options(spec_type) -> None:
    with pytest.raises(ValueError, match="sorted"):
        spec_type("onnx", options=(("zeta", 1), ("alpha", 2)))
    with pytest.raises(ValueError, match="unique"):
        spec_type("onnx", options=(("alpha", 1), ("alpha", 2)))
    with pytest.raises(TypeError, match="immutable JSON"):
        spec_type("onnx", options=(("alpha", [1, 2]),))

    spec = spec_type("onnx", options=(("alpha", (1, "two", None)), ("enabled", True)))
    first = spec.option_values()
    second = spec.option_values()
    first["alpha"] = "changed"
    assert second == {"alpha": (1, "two", None), "enabled": True}


def test_reid_spec_validates_preprocessing_and_has_no_crop_strategy() -> None:
    with pytest.raises(ValueError, match="preprocessing"):
        ReIDEncoderSpec("onnx", preprocessing="Letter Box")
    assert "crop_strategy" not in {field.name for field in dataclasses.fields(ReIDEncoderSpec)}
    with pytest.raises(TypeError, match="crop_strategy"):
        ReIDEncoderSpec("onnx", crop_strategy="perspective")


@pytest.mark.parametrize("spec_type", (DetectorSpec, SegmentorSpec))
def test_detector_and_segmentor_specs_fingerprint_preprocessing_and_geometry(spec_type) -> None:
    configured = spec_type("onnx", preprocessing="letterbox", geometry_mode="obb")
    assert configured.preprocessing == "letterbox"
    assert configured.geometry_mode == "obb"
    assert configured != spec_type("onnx")
    with pytest.raises(ValueError, match="preprocessing"):
        spec_type("onnx", preprocessing="Letter Box")
    with pytest.raises(ValueError, match="geometry_mode"):
        spec_type("onnx", geometry_mode="rotated")


def test_lazy_component_registry_copies_sorts_and_freezes_entries() -> None:
    entries = {
        "zeta": "example.backends:zeta_factory",
        "alpha": "example.backends:alpha_factory",
    }
    registry = LazyComponentRegistry("example", entries)

    entries["alpha"] = "changed.module:factory"

    assert list(registry.entries) == ["alpha", "zeta"]
    assert registry.entries["alpha"] == "example.backends:alpha_factory"
    with pytest.raises(TypeError):
        registry.entries["alpha"] = "changed.module:factory"


@pytest.mark.parametrize(
    ("component", "entries", "error_type", "message"),
    (
        ("", {}, ValueError, "component"),
        ("example", [], TypeError, "entries must be a mapping"),
        ("example", {"Not-Canonical": "example:factory"}, ValueError, "backend names"),
        ("example", {"valid": 42}, TypeError, "module:callable"),
        ("example", {"valid": "example.factory"}, ValueError, "module:callable"),
        ("example", {"valid": "not a module:factory"}, ValueError, "module:callable"),
        ("example", {"valid": "example:not-an-attribute"}, ValueError, "module:callable"),
    ),
)
def test_lazy_component_registry_validates_definitions(
    component,
    entries,
    error_type,
    message,
) -> None:
    with pytest.raises(error_type, match=message):
        LazyComponentRegistry(component, entries)


def test_lazy_component_registry_imports_only_when_resolved(monkeypatch) -> None:
    factory = lambda spec: spec
    imported = []
    registry = LazyComponentRegistry("example", {"backend": "optional.runtime:create"})
    monkeypatch.setattr(
        registry_module,
        "import_module",
        lambda module_name: imported.append(module_name) or SimpleNamespace(create=factory),
    )

    assert imported == []
    assert registry.resolve("backend") is factory
    assert imported == ["optional.runtime"]


def test_lazy_component_registry_reports_deterministic_unknown_backend_error() -> None:
    registry = LazyComponentRegistry(
        "detector",
        {
            "zeta": "example:zeta",
            "alpha": "example:alpha",
        },
    )

    with pytest.raises(ValueError) as error:
        registry.resolve("missing")

    assert str(error.value) == ("Unknown detector backend 'missing'. Available backends: alpha, zeta.")


def test_lazy_component_registry_rejects_non_callable_target(monkeypatch) -> None:
    registry = LazyComponentRegistry("example", {"backend": "optional.runtime:constant"})
    monkeypatch.setattr(
        registry_module,
        "import_module",
        lambda _module_name: SimpleNamespace(constant=object()),
    )

    with pytest.raises(TypeError, match="target .* is not callable"):
        registry.resolve("backend")


@pytest.mark.parametrize(
    ("package_name", "factory_module", "expected_exports", "implementation_prefixes"),
    (
        (
            "boxmot.detectors",
            "boxmot.detectors.factory",
            ["Detector", "DetectorCapabilities", "DetectorSpec", "create_detector"],
            ("boxmot.detectors.adapters", "boxmot.detectors.backends."),
        ),
        (
            "boxmot.segmentors",
            "boxmot.segmentors.factory",
            ["Segmentor", "SegmentorSpec", "create_segmentor"],
            ("boxmot.segmentors.backends.",),
        ),
        (
            "boxmot.reid",
            "boxmot.reid.factory",
            ["AppearanceEncoder", "EncoderRequirements", "ReIDEncoderSpec", "create_reid_encoder"],
            ("boxmot.reid.adapters", "boxmot.reid.backends."),
        ),
    ),
)
def test_public_component_exports_preserve_lazy_implementation_boundary(
    package_name,
    factory_module,
    expected_exports,
    implementation_prefixes,
) -> None:
    probe = (
        "import importlib, json, sys; "
        f"package = importlib.import_module({package_name!r}); "
        f"importlib.import_module({factory_module!r}); "
        f"prefixes = {implementation_prefixes!r}; "
        "print(json.dumps({'exports': sorted(package.__all__), "
        "'implementations': sorted(name for name in sys.modules "
        "if any(name == prefix or name.startswith(prefix) for prefix in prefixes))}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "exports": sorted(expected_exports),
        "implementations": [],
    }


@pytest.mark.parametrize(
    ("factory_module", "registry_name", "factory", "spec_type", "backend", "component"),
    (
        (
            detector_factory_module,
            "_DETECTOR_FACTORIES",
            create_detector,
            DetectorSpec,
            "ultralytics",
            _Detector(),
        ),
        (
            segmentor_factory_module,
            "_SEGMENTOR_FACTORIES",
            create_segmentor,
            SegmentorSpec,
            "sam",
            _Segmentor(),
        ),
        (
            reid_factory_module,
            "_REID_ENCODER_FACTORIES",
            create_reid_encoder,
            ReIDEncoderSpec,
            "onnx",
            _Encoder(),
        ),
    ),
)
def test_factories_accept_only_specs_and_delegate_to_lazy_registry(
    monkeypatch,
    factory_module,
    registry_name,
    factory,
    spec_type,
    backend,
    component,
    tmp_path,
) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"resolved model")
    spec = spec_type(
        backend,
        artifact=str(artifact.resolve()),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )
    calls = []
    monkeypatch.setattr(
        factory_module,
        registry_name,
        SimpleNamespace(resolve=lambda backend: lambda received: calls.append((backend, received)) or component),
    )

    assert factory(spec) is component
    assert calls == [(spec.backend, spec)]
    with pytest.raises(TypeError, match="spec must be"):
        factory(spec.backend)


@pytest.mark.parametrize(
    ("factory", "spec"),
    (
        (create_detector, DetectorSpec("ultralytics")),
        (create_segmentor, SegmentorSpec("sam")),
        (create_reid_encoder, ReIDEncoderSpec("onnx")),
    ),
)
def test_factories_reject_specs_without_resolved_artifact_identity(factory, spec) -> None:
    with pytest.raises(ValueError, match="resolved local artifact path and its SHA-256"):
        factory(spec)


def test_reid_factory_registers_every_runtime_artifact_format() -> None:
    """Keep exported formats constructible through the component factory."""

    assert set(reid_factory_module._REID_ENCODER_FACTORIES.entries) == {
        *(format_.id for format_ in REID_FORMATS),
        "native",
    }


def test_runtime_protocols_are_structural() -> None:
    assert isinstance(_Detector(), Detector)
    assert isinstance(_Segmentor(), Segmentor)
    assert isinstance(_Encoder(), AppearanceEncoder)


def test_protocol_shapes_are_usable_with_canonical_values() -> None:
    frame = Frame(torch.zeros((3, 8, 12), dtype=torch.uint8), "frame-1")
    detections = Detections(
        geometry=Boxes(torch.tensor([[1.0, 1.0, 4.0, 6.0]], dtype=torch.float32)),
        scores=torch.tensor([0.9], dtype=torch.float32),
        class_ids=torch.tensor([0], dtype=torch.int64),
        sample_id=frame.sample_id,
    )
    masks = MaskBatch(torch.zeros((1, 8, 12), dtype=torch.bool))
    assert len(detections.with_masks(masks)) == 1
