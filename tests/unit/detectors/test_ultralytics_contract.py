from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
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


@pytest.mark.parametrize(
    ("filename", "wrapper", "task"),
    (
        ("yolo26n.pt", "YOLO", "detect"),
        ("yolov8n-pose.pt", "YOLO", "pose"),
        ("yolo11n-obb.pt", "YOLO", "obb"),
        ("yolov8s-worldv2.pt", "YOLO", "detect"),
        ("yoloe-26n-seg-pf.pt", "YOLO", "segment"),
        ("rtdetr-l.pt", "YOLO", "detect"),
        ("yolo_nas_s.pt", "NAS", "detect"),
        ("FastSAM-x.pt", "FastSAM", "segment"),
        ("custom_checkpoint.pt", "YOLO", "segment"),
    ),
)
def test_ultralytics_selects_family_wrapper_and_actual_output_capabilities(
    tmp_path: Path, monkeypatch, filename: str, wrapper: str, task: str
) -> None:
    """Specialized upstream wrappers retain canonical geometry and mask contracts."""
    import ultralytics

    import boxmot.detectors.backends.ultralytics as backend

    artifact = tmp_path / filename
    artifact.write_bytes(b"fixture checkpoint")
    calls = []
    names = {0: "person"}

    def construct(selected: str):
        def model(path: str):
            calls.append((selected, path))
            return SimpleNamespace(task=task, names=names)

        return model

    monkeypatch.setattr(backend, "YOLO", construct("YOLO"))
    monkeypatch.setattr(ultralytics, "NAS", construct("NAS"))
    monkeypatch.setattr(ultralytics, "FastSAM", construct("FastSAM"))
    detector = UltralyticsDetector(DetectorSpec("ultralytics", artifact=str(artifact)))

    assert calls == [(wrapper, str(artifact))]
    assert detector.names == names
    assert detector.capabilities == DetectorCapabilities(
        provides_masks=task == "segment", supports_aabb=task != "obb", supports_obb=task == "obb"
    )


@pytest.mark.parametrize("task", ("classify", "semantic", "", "unknown"))
def test_ultralytics_rejects_tasks_without_object_boxes(tmp_path: Path, monkeypatch, task: str) -> None:
    import boxmot.detectors.backends.ultralytics as backend

    artifact = tmp_path / "custom.pt"
    artifact.write_bytes(b"fixture checkpoint")
    monkeypatch.setattr(backend, "YOLO", lambda _: SimpleNamespace(task=task))

    with pytest.raises(ValueError, match="does not provide supported tracking detections"):
        UltralyticsDetector(DetectorSpec("ultralytics", artifact=str(artifact)))


@pytest.mark.parametrize(("task", "geometry"), (("obb", "aabb"), ("detect", "obb"), ("segment", "obb")))
def test_ultralytics_rejects_loaded_geometry_conflicting_with_spec(
    tmp_path: Path, monkeypatch, task: str, geometry: str
) -> None:
    import boxmot.detectors.backends.ultralytics as backend

    artifact = tmp_path / "custom.pt"
    artifact.write_bytes(b"fixture checkpoint")
    monkeypatch.setattr(backend, "YOLO", lambda _: SimpleNamespace(task=task))

    with pytest.raises(ValueError, match="conflicts with geometry_mode"):
        UltralyticsDetector(DetectorSpec("ultralytics", artifact=str(artifact), geometry_mode=geometry))


def test_ultralytics_nas_preserves_optional_dependency_failure(tmp_path: Path, monkeypatch) -> None:
    """Selecting NAS must preserve upstream dependency requirements without installers."""
    import ultralytics

    artifact = tmp_path / "yolo_nas_s.pt"
    artifact.write_bytes(b"fixture checkpoint")
    failure = ModuleNotFoundError("No module named 'super_gradients'", name="super_gradients")

    def unavailable(path: str):
        raise failure

    monkeypatch.setattr(ultralytics, "NAS", unavailable)
    with pytest.raises(ModuleNotFoundError) as raised:
        UltralyticsDetector(DetectorSpec("ultralytics", artifact=str(artifact)))
    assert raised.value is failure


def test_ultralytics_grayscale_setup_and_prediction_keep_original_rgb_frames(tmp_path: Path, monkeypatch) -> None:
    """The first-call warm-up and later inference both respect one-channel weights."""
    import boxmot.detectors.backends.ultralytics as backend

    observed = {}

    class Predictor:
        args = SimpleNamespace(conf=0.25, iou=0.7, classes=None, agnostic_nms=False)
        model = SimpleNamespace(channels=1)

        def preprocess(self, images):
            observed["model_images"] = images
            assert len(images) == 2
            assert all(image.shape == (6, 8, 1) for image in images)
            return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()

        def inference(self, preprocessed):
            assert preprocessed.shape == (2, 1, 6, 8)
            assert preprocessed[0, 0, 0, 0].item() == 18
            return object()

        def postprocess(self, raw, preprocessed, images):
            observed["original_images"] = images
            assert self.batch[1] is images
            assert images[0].shape == (6, 8, 3)
            assert images[0][0, 0].tolist() == [30, 20, 10]
            return [SimpleNamespace(obb=None, boxes=[], masks=None) for _ in images]

    class Yolo:
        task = "detect"
        names = {0: "person"}
        model = SimpleNamespace(yaml={"channels": 1})

        def predict(self, **kwargs):
            observed["warmup"] = kwargs["source"]
            assert kwargs["source"].shape == (32, 32, 1)
            self.predictor = Predictor()

    artifact = tmp_path / "custom-gray.pt"
    artifact.write_bytes(b"fixture grayscale weights")
    monkeypatch.setattr(backend, "YOLO", lambda _: Yolo())
    detector = UltralyticsDetector(DetectorSpec("ultralytics", artifact=str(artifact)))
    frames = [_frame("first"), _frame("second")]
    original_pixels = [frame.image.clone() for frame in frames]
    assert detector._predictor is None
    results = detector.predict(frames)

    assert observed["warmup"].dtype == np.uint8
    assert [result.sample_id for result in results] == ["first", "second"]
    assert all(len(result) == 0 and not result.is_obb for result in results)
    for grayscale, original, frame, pixels in zip(
        observed["model_images"], observed["original_images"], frames, original_pixels, strict=True
    ):
        np.testing.assert_array_equal(grayscale[..., 0], cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        torch.testing.assert_close(frame.image, pixels)


def test_ultralytics_rejects_unsupported_model_channel_count_before_preprocessing() -> None:
    predictor = SimpleNamespace(
        args=SimpleNamespace(conf=0.25, iou=0.7, classes=None, agnostic_nms=False),
        model=SimpleNamespace(channels=5),
    )
    with pytest.raises(ValueError, match="requires 5 image channels"):
        _detector(predictor).predict([_frame()])
