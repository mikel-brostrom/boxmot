from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

from boxmot.detectors import DetectorSpec
from boxmot.engine.eval import evaluator
from boxmot.engine.experiment_config import resolve_experiment_config
from boxmot.engine.materialization import fingerprint


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
