"""Standalone EdgeTAM uses singleton prompts and releases frame embeddings."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import boxmot.segmentors.backends.edgetam as backend
from boxmot.components.artifacts import ResolvedArtifact, sha256_artifact
from boxmot.segmentors.config import resolve_segmentor_spec
from boxmot.segmentors.factory import create_segmentor
from boxmot.segmentors.specs import SegmentorSpec
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes


def _frame(sample_id: str = "sample") -> Frame:
    """Return distinguishable RGB channels so color conversion is observable."""
    image = torch.zeros((3, 6, 8), dtype=torch.uint8)
    image[0], image[1], image[2] = 10, 20, 30
    return Frame(image, sample_id)


def _detections(frame: Frame, *, empty: bool = False, obb: bool = False) -> Detections:
    """Make ordered boxes with differing sizes and positions."""
    if obb:
        geometry = OrientedBoxes(
            torch.tensor([[4.0, 3.0, 4.0, 2.0, math.pi / 2], [2.0, 2.0, 2.0, 2.0, 0.0]])
        )
    else:
        geometry = Boxes(torch.tensor([[1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 6.0, 4.0]]))
    if empty:
        geometry = Boxes(torch.empty((0, 4)))
    return Detections(
        geometry=geometry,
        scores=torch.full((len(geometry),), 0.9),
        class_ids=torch.zeros(len(geometry), dtype=torch.int64),
        sample_id=frame.sample_id,
    )


@pytest.fixture
def image_predictors(monkeypatch):
    """Replace only the optional image wrapper, preserving adapter behavior."""
    instances = []

    class ImagePredictor:
        def __init__(self, model, *, mask_threshold):
            self.model = model
            self.mask_threshold = mask_threshold
            self.images = []
            self.boxes = []
            self.resets = 0
            self.features = None
            self.fail = False
            instances.append(self)

        def set_image(self, image):
            assert torch.is_inference_mode_enabled()
            assert self.features is None
            self.images.append(image.copy())
            self.features = object()

        def predict(self, *, box, multimask_output, return_logits):
            assert torch.is_inference_mode_enabled()
            assert not multimask_output
            assert not return_logits
            assert box.shape == (4,)
            self.boxes.append(box.copy())
            if self.fail:
                raise RuntimeError("prediction failure")
            logits = np.full((1, 6, 8), -10, dtype=np.float32)
            logits[0, int(box[1]), int(box[0])] = 3
            return (logits > self.mask_threshold).astype(np.float32), np.ones(1), np.zeros((1, 2, 2))

        def reset_predictor(self):
            self.features = None
            self.resets += 1

    monkeypatch.setattr(
        backend, "import_module", lambda name: SimpleNamespace(SAM2ImagePredictor=ImagePredictor)
    )
    return instances


def test_singleton_boxes_preserve_alignment_rgb_and_release_each_frame(image_predictors) -> None:
    """Each frame is encoded once; masks retain detection and frame order."""
    segmentor = backend.EdgeTAMSegmentor(SegmentorSpec("edgetam", artifact="edgetam.pt"), model=object())
    first, second, empty = _frame("first"), _frame("second"), _frame("empty")

    masks = segmentor.segment(
        [first, second, empty], [_detections(first), _detections(second), _detections(empty, empty=True)]
    )

    predictor = image_predictors[0]
    assert len(predictor.images) == 2
    assert len(predictor.boxes) == 4
    assert predictor.resets == 2
    assert predictor.features is None
    assert predictor.mask_threshold == 0.0
    np.testing.assert_array_equal(predictor.images[0][0, 0], [10, 20, 30])
    assert all(isinstance(result, MaskBatch) for result in masks)
    assert masks[0].values.shape == masks[1].values.shape == (2, 6, 8)
    assert masks[2].values.shape == (0, 6, 8)
    assert all(result.values.dtype == torch.bool for result in masks)
    assert all(result.values.device.type == "cpu" for result in masks)
    assert masks[0].values[0, 1, 1]
    assert masks[0].values[1, 2, 2]
    assert masks[0].values.sum() == 2


def test_obb_prompts_are_enclosing_aabbs_in_original_order(image_predictors) -> None:
    """Rotated boxes use the common enclosing geometry calculation."""
    frame = _frame()
    segmentor = backend.EdgeTAMSegmentor(
        SegmentorSpec("edgetam", artifact="edgetam.pt", geometry_mode="obb"), model=object()
    )

    masks = segmentor.segment([frame], [_detections(frame, obb=True)])

    np.testing.assert_allclose(image_predictors[0].boxes, [[3, 1, 5, 5], [1, 1, 3, 3]], atol=1e-6)
    assert masks[0].values.shape == (2, 6, 8)


def test_separate_images_can_change_resolution_and_keep_full_frame_outputs(image_predictors) -> None:
    """Standalone image inference has no temporal fixed-resolution restriction."""
    first = _frame("first")
    second = Frame(torch.zeros((3, 12, 16), dtype=torch.uint8), "second")
    empty = Frame(torch.zeros((3, 5, 7), dtype=torch.uint8), "empty")
    segmentor = backend.EdgeTAMSegmentor(SegmentorSpec("edgetam", artifact="edgetam.pt"), model=object())

    results = segmentor.segment(
        [first, second, empty], [_detections(first), _detections(second), _detections(empty, empty=True)]
    )

    assert [tuple(result.values.shape) for result in results] == [(2, 6, 8), (2, 12, 16), (0, 5, 7)]
    assert image_predictors[0].resets == 2
    assert image_predictors[0].features is None


def test_predictor_cleanup_runs_on_inference_failure(image_predictors) -> None:
    """Failed predictions do not pin the previous image features."""
    frame = _frame()
    segmentor = backend.EdgeTAMSegmentor(SegmentorSpec("edgetam", artifact="edgetam.pt"), model=object())
    predictor = image_predictors[0]
    predictor.fail = True

    with pytest.raises(RuntimeError, match="prediction failure"):
        segmentor.segment([frame], [_detections(frame)])

    assert predictor.features is None
    assert predictor.resets == 1


def test_binary_predictions_do_not_apply_logit_threshold_twice(image_predictors) -> None:
    """Positive custom logit thresholds retain upstream binary results."""
    frame = _frame()
    segmentor = backend.EdgeTAMSegmentor(
        SegmentorSpec("edgetam", artifact="edgetam.pt", options=(("mask_threshold", 2.5),)), model=object()
    )

    results = segmentor.segment([frame], [_detections(frame)])

    assert image_predictors[0].mask_threshold == 2.5
    assert results[0].values.sum() == 2


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_reduced_precision_on_cpu_fails_before_checkpoint_loading(monkeypatch, precision) -> None:
    """Portable CPU inference requires FP32."""
    build = Mock()
    monkeypatch.setattr(backend, "build_edgetam_predictor", build)

    with pytest.raises(ValueError, match="[Cc][Pp][Uu]|CUDA|cuda|fp32|FP32"):
        backend.EdgeTAMSegmentor(SegmentorSpec("edgetam", artifact="edgetam.pt", precision=precision))

    build.assert_not_called()


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"artifact": None}, "requires an artifact"),
        ({"preprocessing": "custom"}, "preprocessing"),
        ({"options": (("unknown", 1),)}, "Unsupported"),
        ({"options": (("mask_threshold", "nan"),)}, "finite logit"),
    ],
)
def test_rejects_unsupported_settings_before_loading(monkeypatch, overrides, message) -> None:
    """Invalid backend settings produce actionable failures without allocation."""
    build = Mock()
    monkeypatch.setattr(backend, "build_edgetam_predictor", build)

    with pytest.raises(ValueError, match=message):
        backend.EdgeTAMSegmentor(replace(SegmentorSpec("edgetam", artifact="edgetam.pt"), **overrides))

    build.assert_not_called()


def test_missing_optional_dependency_has_install_instruction(monkeypatch) -> None:
    """Core installation can explain the optional dependency when requested."""
    def missing(_name):
        raise ModuleNotFoundError("No module named 'sam2'", name="sam2")

    monkeypatch.setattr(backend, "import_module", missing)
    with pytest.raises(ImportError, match="uv sync --extra cpu --group mask-guidance"):
        backend.EdgeTAMSegmentor(SegmentorSpec("edgetam", artifact="edgetam.pt"), model=object())


def test_input_alignment_and_geometry_fail_before_inference(image_predictors) -> None:
    """Invalid canonical input cannot leave any upstream inference state."""
    frame = _frame()
    segmentor = backend.EdgeTAMSegmentor(
        SegmentorSpec("edgetam", artifact="edgetam.pt", geometry_mode="aabb"), model=object()
    )

    with pytest.raises(ValueError, match="aligned"):
        segmentor.segment([frame], [])
    with pytest.raises(ValueError, match="sample_id"):
        segmentor.segment([frame], [_detections(_frame("other"))])
    with pytest.raises(ValueError, match="OBB"):
        segmentor.segment([frame], [_detections(frame, obb=True)])
    assert image_predictors[0].images == []


def test_factory_injects_same_model_and_still_verifies_checkpoint(monkeypatch, image_predictors, tmp_path) -> None:
    """Sharing avoids another model allocation without bypassing content identity."""
    artifact = tmp_path / "edgetam.pt"
    artifact.write_bytes(b"test checkpoint")
    spec = SegmentorSpec("edgetam", artifact=str(artifact), artifact_sha256=sha256_artifact(artifact))
    build = Mock()
    monkeypatch.setattr(backend, "build_edgetam_predictor", build)
    shared_model = torch.nn.Linear(1, 1)

    first = create_segmentor(spec, model=shared_model)
    second = create_segmentor(spec, model=shared_model)

    assert first.model is second.model is shared_model
    assert image_predictors[0] is not image_predictors[1]
    assert image_predictors[0].model is image_predictors[1].model is shared_model
    build.assert_not_called()
    artifact.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        create_segmentor(spec, model=shared_model)


def test_factory_does_not_share_models_into_other_backends() -> None:
    """The new injection keyword has an explicit backend contract."""
    with pytest.raises(ValueError, match="only for backend='edgetam'"):
        create_segmentor(SegmentorSpec("sam"), model=object())


def test_config_resolution_uses_native_logit_threshold_and_artifact_identity(tmp_path) -> None:
    """Defaults become part of the immutable backend specification."""
    artifact = tmp_path / "edgetam.pt"
    artifact.write_bytes(b"test checkpoint")

    spec, provenance = resolve_segmentor_spec(
        {"backend": "edgetam", "artifact": str(artifact)}, geometry="obb", allow_download=False
    )

    assert spec.backend == "edgetam"
    assert spec.option_values() == {"mask_threshold": 0.0}
    assert spec.geometry_mode == "obb"
    assert spec.precision == "fp32"
    assert provenance["artifact"]["sha256"] == spec.artifact_sha256


def test_config_resolution_rejects_directory_checkpoint(tmp_path) -> None:
    """An EdgeTAM artifact must contain checkpoint bytes, not a directory."""
    (tmp_path / "weights.pt").write_bytes(b"test checkpoint")
    with pytest.raises(ValueError, match="checkpoint file artifact"):
        resolve_segmentor_spec(
            {"backend": "edgetam", "artifact": str(tmp_path)}, geometry="aabb", allow_download=False
        )


def test_example_resolves_checkpoint_from_project_models_directory(tmp_path) -> None:
    """Built-in YAML is executable without making the cloned EdgeTAM a dependency."""
    requests = []
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.write_bytes(b"test checkpoint")

    def resolve(path, **kwargs):
        requests.append((path, kwargs))
        return ResolvedArtifact(checkpoint, sha256_artifact(checkpoint), kwargs["source_uri"])

    example = Path(__file__).resolve().parents[3] / "boxmot/configs/segmentors/edgetam.yaml"
    spec, _ = resolve_segmentor_spec(example, geometry="aabb", artifact_resolver=resolve)

    assert requests[0][0] == Path("models/edgetam.pt")
    assert requests[0][1]["expected_sha256"] == "ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df"
    assert spec.backend == "edgetam"
    assert spec.precision == "fp32"
    assert spec.option_values() == {"mask_threshold": 0.0}
