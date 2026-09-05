from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from boxmot.detectors import DetectorSpec
from boxmot.engine.eval import evaluator
from boxmot.engine.experiment_config import resolve_experiment_config
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
        "detections": {
            "source": "model",
            "model": {"ref": "fixture", "checkpoint": "default"},
        },
        "segmentor": None,
        "reid": None,
    }

    expected = evaluator._experiment_component_fingerprints(experiment, manifest)

    assert expected == {"detector": fingerprint(built_provenance)}


def test_mmot_eval_requires_perspective_obb_reid_provenance() -> None:
    resolved = resolve_experiment_config("mmot-obb-test-yolo11l-lmbn", mode="eval")

    reference = evaluator._reid_reference(resolved)

    assert reference is not None
    assert reference["crop_strategy"] == "perspective"


def test_mmot_perspective_recipe_rejects_aabb_crop_build_fingerprint(monkeypatch) -> None:
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
        "crop_strategy": "perspective",
    }
    perspective_provenance = {"spec": reid_spec, "artifact": {"sha256": "1" * 64}}
    built_perspective_provenance = {
        **perspective_provenance,
        "spec": {**reid_spec, "device": "mps"},
    }
    built_aabb_provenance = {
        **perspective_provenance,
        "spec": {**reid_spec, "device": "mps", "crop_strategy": "aabb"},
    }
    manifest = SimpleNamespace(
        build_id="c5d620",
        box_type="obb",
        artifacts_by_name={},
        metadata={
            "components": {
                "detector": built_detector_provenance,
                "segmentor": None,
                "reid": built_aabb_provenance,
            },
            "component_fingerprints": {
                "detector": fingerprint(built_detector_provenance),
                "segmentor": None,
                "reid": fingerprint(built_aabb_provenance),
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
        lambda reference, **_kwargs: (
            None,
            perspective_provenance,
        )
        if reference["crop_strategy"] == "perspective"
        else pytest.fail("MMOT evaluation must resolve the perspective crop recipe"),
    )
    resolved = resolve_experiment_config("mmot-obb-test-yolo11l-lmbn", mode="eval")

    expected = evaluator._experiment_component_fingerprints(resolved, manifest)

    assert expected["reid"] == fingerprint(built_perspective_provenance)
    assert expected["reid"] != manifest.metadata["component_fingerprints"]["reid"]
    with pytest.raises(BuildCompatibilityError, match="component 'reid' mismatch"):
        validate_build_compatibility(manifest, component_fingerprints=expected)
