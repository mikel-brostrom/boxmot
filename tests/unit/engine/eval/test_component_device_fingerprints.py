from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from boxmot.detectors import DetectorSpec
from boxmot.engine.config.experiments import resolve_experiment_config
from boxmot.engine.eval import evaluator
from boxmot.engine.materialization import fingerprint
from boxmot.engine.materialization.builds import BuildCompatibilityError, validate_build_compatibility


def test_experiment_fingerprint_uses_build_execution_device(monkeypatch) -> None:
    authored = DetectorSpec(backend="fixture", device="cpu", geometry_mode="aabb")
    resolved_provenance = {"spec": asdict(authored), "artifact": None}
    built = replace(authored, device="mps")
    built_provenance = {"spec": asdict(built), "artifact": None}
    manifest = SimpleNamespace(
        metadata={
            "components": {
                "detector": built_provenance,
                "segmentor": None,
                "reid": None,
            },
            "component_fingerprints": {
                "detector": fingerprint(built_provenance),
                "segmentor": None,
                "reid": None,
            },
        }
    )
    monkeypatch.setattr(
        evaluator,
        "resolve_detector_spec",
        lambda _reference, *, geometry, allow_download: (authored, resolved_provenance),
    )
    experiment = {
        "dataset": {"box_type": "aabb"},
        "detector": {"ref": "fixture", "checkpoint": "default"},
        "segmentor": None,
        "reid": None,
    }

    expected = evaluator._experiment_component_fingerprints(experiment, manifest)

    assert expected == {"detector": fingerprint(built_provenance)}


def test_mmot_eval_reid_reference_has_no_crop_policy() -> None:
    resolved = resolve_experiment_config("mmot-obb/test-yolo11l-lmbn.yaml", mode="eval")

    reference = evaluator._reid_reference(resolved)

    assert reference is not None
    assert "crop_strategy" not in reference


def test_reid_fingerprint_uses_build_execution_device(monkeypatch) -> None:
    detector_spec = DetectorSpec(backend="fixture", device="cpu", geometry_mode="obb")
    detector_provenance = {"spec": asdict(detector_spec), "artifact": None}
    built_detector_provenance = {
        "spec": {**detector_provenance["spec"], "device": "mps"},
        "artifact": None,
    }
    reid_spec = {
        "backend": "pytorch",
        "device": "cpu",
        "precision": "fp16",
    }
    reid_provenance = {"spec": reid_spec, "artifact": {"sha256": "1" * 64}}
    built_reid_provenance = {
        **reid_provenance,
        "spec": {**reid_spec, "device": "mps"},
    }
    manifest = SimpleNamespace(
        metadata={
            "components": {
                "detector": built_detector_provenance,
                "segmentor": None,
                "reid": built_reid_provenance,
            },
            "component_fingerprints": {
                "detector": fingerprint(built_detector_provenance),
                "segmentor": None,
                "reid": fingerprint(built_reid_provenance),
            },
        },
    )
    monkeypatch.setattr(
        evaluator,
        "resolve_detector_spec",
        lambda *_args, **_kwargs: (detector_spec, detector_provenance),
    )
    monkeypatch.setattr(
        evaluator,
        "resolve_reid_spec",
        lambda _reference, **_kwargs: (None, reid_provenance),
    )
    resolved = resolve_experiment_config("mmot-obb/test-yolo11l-lmbn.yaml", mode="eval")

    expected = evaluator._experiment_component_fingerprints(resolved, manifest)

    assert expected["reid"] == fingerprint(built_reid_provenance)


def _runtime_validation_case(monkeypatch, *, recorded_precision: str = "fp16"):
    """Resolve authored components independently from immutable recorded provenance."""
    components = {
        name: {
            "spec": {
                "backend": "pytorch",
                "artifact": f"/{name}.pt",
                "artifact_sha256": str(index) * 64,
                "device": "cpu",
                "precision": "fp32",
                "preprocessing": "default",
                "options": (("image_size", (384, 128)),),
            },
            "artifact": {"sha256": str(index) * 64},
        }
        for index, name in enumerate(("detector", "segmentor", "reid"), 1)
    }
    recorded = deepcopy(components)
    for component in recorded.values():
        component["spec"]["device"] = "mps"
    recorded["reid"]["spec"]["precision"] = recorded_precision
    manifest = SimpleNamespace(
        build_id="a" * 64,
        box_type="aabb",
        artifacts_by_name={},
        metadata={
            "components": recorded,
            "component_fingerprints": {name: fingerprint(value) for name, value in recorded.items()},
        },
    )
    for name in components:
        monkeypatch.setattr(
            evaluator,
            f"resolve_{name}_spec",
            lambda _reference, *, _name=name, **_kwargs: (None, components[_name]),
        )
    experiment = {
        "dataset": {"box_type": "aabb"},
        "detector": {"ref": "fixture", "checkpoint": "default"},
        "segmentor": {"ref": "fixture"},
        "reid": {"model": "/reid.pt", "precision": "fp32"},
    }
    return experiment, manifest, components


@pytest.mark.parametrize("recorded_precision,requested_precision", [("fp16", "fp32"), ("fp32", "fp16")])
def test_experiment_validation_reuses_reid_precision_without_rewriting_provenance(
    monkeypatch, recorded_precision: str, requested_precision: str
) -> None:
    experiment, manifest, components = _runtime_validation_case(monkeypatch, recorded_precision=recorded_precision)
    experiment["reid"]["precision"] = requested_precision
    components["reid"]["spec"]["precision"] = requested_precision
    recorded_before = deepcopy(vars(manifest))
    authored_before = deepcopy(components)

    expected = evaluator._experiment_component_fingerprints(experiment, manifest)
    validate_build_compatibility(manifest, component_fingerprints=expected)

    assert expected == manifest.metadata["component_fingerprints"]
    assert vars(manifest) == recorded_before
    assert components == authored_before


@pytest.mark.parametrize("change", ["weights", "backend", "preprocessing", "image_size"])
def test_reid_precision_reuse_still_rejects_changed_embedding_content(monkeypatch, change: str) -> None:
    experiment, manifest, components = _runtime_validation_case(monkeypatch)
    spec = components["reid"]["spec"]
    if change == "weights":
        spec["artifact_sha256"] = "9" * 64
        components["reid"]["artifact"]["sha256"] = "9" * 64
    elif change == "backend":
        spec["backend"] = "onnx"
    elif change == "preprocessing":
        spec["preprocessing"] = "resize"
    else:
        spec["options"] = (("image_size", (256, 128)),)

    expected = evaluator._experiment_component_fingerprints(experiment, manifest)

    with pytest.raises(BuildCompatibilityError, match="component 'reid' mismatch"):
        validate_build_compatibility(manifest, component_fingerprints=expected)


@pytest.mark.parametrize("component_name", ["detector", "segmentor"])
def test_reid_precision_reuse_keeps_other_component_precision_strict(monkeypatch, component_name: str) -> None:
    experiment, manifest, components = _runtime_validation_case(monkeypatch)
    components[component_name]["spec"]["precision"] = "fp16"

    expected = evaluator._experiment_component_fingerprints(experiment, manifest)

    with pytest.raises(BuildCompatibilityError, match=f"component '{component_name}' mismatch"):
        validate_build_compatibility(manifest, component_fingerprints=expected)


@pytest.mark.parametrize("precision", [None, "", "float16", "int8", 16, ["fp16"]])
def test_reid_precision_reuse_rejects_invalid_recorded_precision(monkeypatch, precision) -> None:
    experiment, manifest, _ = _runtime_validation_case(monkeypatch)
    recorded_spec = manifest.metadata["components"]["reid"]["spec"]
    if precision is None:
        recorded_spec.pop("precision")
    else:
        recorded_spec["precision"] = precision

    with pytest.raises(ValueError, match=r"components\.reid\.spec\.precision"):
        evaluator._experiment_component_fingerprints(experiment, manifest)
