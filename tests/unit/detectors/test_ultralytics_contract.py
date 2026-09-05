from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot.components.timing import timing_event_sink
from boxmot.detectors import DetectorCapabilities, DetectorSpec
from boxmot.detectors.backends.ultralytics import UltralyticsDetector
from boxmot.structures import Detections, Frame


def _frame(sample_id: str = "sample") -> Frame:
    image = torch.zeros((3, 6, 8), dtype=torch.uint8)
    image[0] = 10
    image[1] = 20
    image[2] = 30
    return Frame(image, sample_id)


def _detector(predictor) -> UltralyticsDetector:
    detector = UltralyticsDetector.__new__(UltralyticsDetector)
    detector._predictor = predictor
    detector._is_obb = False
    detector._prediction_options = {
        "conf": 0.4,
        "iou": 0.7,
        "classes": None,
        "agnostic_nms": False,
    }
    detector.capabilities = DetectorCapabilities(provides_masks=True, supports_obb=True)
    return detector


def test_ultralytics_predict_consumes_rgb_frames_and_returns_canonical_detections() -> None:
    calls = {}

    class Predictor:
        args = SimpleNamespace(conf=0.25, iou=0.7, classes=None, agnostic_nms=False)

        def preprocess(self, images):
            calls["images"] = images
            return torch.zeros((1, 3, 8, 8), dtype=torch.float32)

        def inference(self, preprocessed):
            calls["preprocessed"] = preprocessed
            return object()

        def postprocess(self, raw, preprocessed, images):
            assert raw is not None and preprocessed is calls["preprocessed"]
            assert images is calls["images"]

            class ResultBoxes:
                xyxy = torch.tensor([[1, 2, 4, 5]], dtype=torch.float32)
                conf = torch.tensor([0.75], dtype=torch.float32)
                cls = torch.tensor([3], dtype=torch.float32)

                def __len__(self):
                    return 1

            class ResultMasks:
                data = torch.ones((1, 6, 8), dtype=torch.float32)
                orig_shape = (6, 8)

                def __len__(self):
                    return 1

            boxes = ResultBoxes()
            masks = ResultMasks()
            return [SimpleNamespace(obb=None, boxes=boxes, masks=masks, orig_img=images[0])]

    detector = _detector(Predictor())
    events = []
    with timing_event_sink(events.append):
        result = detector.predict([_frame()])[0]

    assert isinstance(result, Detections)
    assert [(event.component, event.phase) for event in events] == [
        ("detector", "preprocess"),
        ("detector", "process"),
        ("detector", "postprocess"),
    ]
    assert calls["images"][0][0, 0].tolist() == [30, 20, 10]
    assert result.sample_id == "sample"
    assert result.geometry.values.tolist() == [[1.0, 2.0, 4.0, 5.0]]
    assert result.scores.tolist() == [0.75]
    assert result.class_ids.tolist() == [3]
    assert result.masks is not None
    assert result.masks.values.dtype == torch.bool
    assert result.masks.image_size == (6, 8)


def test_ultralytics_predict_empty_sequence_does_not_touch_model() -> None:
    detector = _detector(None)
    events = []

    with timing_event_sink(events.append):
        assert detector.predict(()) == []
    assert events == []


def test_ultralytics_predict_preserves_obb_geometry_and_radians() -> None:
    class ResultObb:
        xywhr = torch.tensor([[4.0, 3.0, 2.0, 1.0, 3.5]], dtype=torch.float32)
        conf = torch.tensor([0.8], dtype=torch.float32)
        cls = torch.tensor([2], dtype=torch.float32)

        def __len__(self):
            return 1

    class Predictor:
        args = SimpleNamespace(conf=0.25, iou=0.7, classes=None, agnostic_nms=False)

        def preprocess(self, _images):
            return object()

        def inference(self, _preprocessed):
            return object()

        def postprocess(self, _raw, _preprocessed, images):
            return [SimpleNamespace(obb=ResultObb(), boxes=None, masks=None, orig_img=images[0])]

    detector = _detector(Predictor())
    detector._is_obb = True
    detector.capabilities = DetectorCapabilities(supports_aabb=False, supports_obb=True)

    result = detector.predict([_frame()])[0]

    assert result.is_obb is True
    torch.testing.assert_close(
        result.to_obb_rows(),
        torch.tensor([[4.0, 3.0, 2.0, 1.0, 3.5, 0.8, 2.0]], dtype=torch.float32),
    )


def test_ultralytics_rejects_raw_array_input() -> None:
    detector = _detector(None)

    with pytest.raises(TypeError, match=r"Sequence\[Frame\]"):
        detector.predict(np.zeros((6, 8, 3), dtype=np.uint8))


def test_ultralytics_predictor_clears_previous_class_filter() -> None:
    detector = UltralyticsDetector.__new__(UltralyticsDetector)
    detector._predictor = SimpleNamespace(args=SimpleNamespace(conf=0.25, iou=0.7, classes=None, agnostic_nms=False))

    detector._ensure_predictor(classes=[0])
    assert detector._predictor.args.classes == [0]

    detector._ensure_predictor(classes=None)
    assert detector._predictor.args.classes is None


def test_ultralytics_uses_backend_default_image_size_when_spec_omits_it() -> None:
    calls = {}
    predictor = SimpleNamespace(args=SimpleNamespace(conf=0.25, iou=0.7, classes=None, agnostic_nms=False))

    class Yolo:
        def predict(self, **kwargs):
            calls.update(kwargs)
            self.predictor = predictor

    detector = UltralyticsDetector.__new__(UltralyticsDetector)
    detector._predictor = None
    detector._yolo = Yolo()
    detector.device = torch.device("cpu")
    detector.imgsz = None

    detector._ensure_predictor()

    assert "imgsz" not in calls


def test_ultralytics_empty_fallback_preserves_obb_mode() -> None:
    detector = UltralyticsDetector.__new__(UltralyticsDetector)
    detector._is_obb = True

    rows, masks = detector._extract_rows(SimpleNamespace(obb=None, boxes=[]))

    assert rows.shape == (0, 7)
    assert rows.dtype == np.float32
    assert masks is None


def test_ultralytics_never_replaces_or_downloads_a_resolved_corrupt_artifact(monkeypatch, tmp_path) -> None:
    import boxmot.detectors.backends.ultralytics as ultralytics_module

    model_path = tmp_path / "yolo11_custom.pt"
    model_path.write_bytes(b"original-corrupt-weights")
    monkeypatch.setattr(
        ultralytics_module,
        "YOLO",
        lambda _path: (_ for _ in ()).throw(RuntimeError("failed finding central directory")),
    )

    with pytest.raises(RuntimeError, match="failed finding central directory"):
        UltralyticsDetector(DetectorSpec("ultralytics", artifact=str(model_path)))

    assert model_path.read_bytes() == b"original-corrupt-weights"
