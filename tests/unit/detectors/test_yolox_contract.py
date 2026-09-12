from __future__ import annotations

import os

import pytest
import torch

import boxmot.detectors.backends.yolox as yolox_module
from boxmot.components.timing import timing_event_sink
from boxmot.detectors import DetectorCapabilities, DetectorSpec
from boxmot.structures import Detections, Frame


def _frame() -> Frame:
    return Frame(torch.zeros((3, 32, 48), dtype=torch.uint8), "sample")


@pytest.mark.parametrize("device", ("cpu", "1", "cuda:1"))
def test_yolox_uses_profile_class_count_and_names(monkeypatch, tmp_path, device) -> None:
    model_path = tmp_path / "yolox_x_visdrone.pt"
    model_path.touch()
    selected_devices = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            selected_devices.append(device)
            return self

        def load_state_dict(self, state):
            assert state == {"weights": "stub"}

    class FakeExp:
        num_classes = 80

        def get_model(self):
            return FakeModel()

    experiment = FakeExp()
    names = {index: f"class-{index}" for index in range(10)}
    monkeypatch.setattr(yolox_module, "get_exp", lambda *_args: experiment)
    monkeypatch.setattr(yolox_module, "load_detector_artifact_profile", lambda _path: {"classes": names})
    monkeypatch.setattr(yolox_module.torch, "load", lambda *_args, **_kwargs: {"model": {"weights": "stub"}})
    monkeypatch.setattr(yolox_module, "fuse_model", lambda model: model)

    detector = yolox_module.YoloXDetector(
        DetectorSpec(
            "yolox",
            artifact=str(model_path),
            device=device,
            options=(("image_size", (64, 96)),),
        )
    )

    assert detector.num_classes == 10
    assert detector.names == names
    assert detector.imgsz == [64, 96]
    assert experiment.num_classes == 10
    assert detector.capabilities == DetectorCapabilities()
    expected_device = torch.device("cpu" if device == "cpu" else "cuda:1")
    assert detector.device == expected_device
    assert selected_devices == [expected_device]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,7"


def test_yolox_predict_filters_classes_on_device_and_returns_canonical_rows(monkeypatch) -> None:
    raw = torch.tensor(
        [
            [1, 2, 11, 12, 0.8, 0.5, 0],
            [3, 4, 13, 14, 0.9, 0.8, 1],
        ],
        dtype=torch.float32,
    )

    class Model:
        def __call__(self, inputs):
            assert inputs.shape == (1, 3, 8, 8)
            return [raw]

    detector = yolox_module.YoloXDetector.__new__(yolox_module.YoloXDetector)
    detector.device = torch.device("cpu")
    detector.model = Model()
    detector.num_classes = 2
    detector.capabilities = DetectorCapabilities()
    detector._prediction_options = {
        "conf": 0.25,
        "iou": 0.7,
        "classes": (1,),
        "agnostic_nms": False,
    }
    monkeypatch.setattr(
        detector,
        "_preprocess_images",
        lambda _images: (torch.zeros((1, 3, 8, 8), dtype=torch.float32), [1.0]),
    )
    monkeypatch.setattr(yolox_module, "yolox_postprocess", lambda *_args, **_kwargs: [raw.clone()])

    def reject_numpy_mask(_values):
        raise AssertionError("class filtering must keep its mask on the detection tensor device")

    monkeypatch.setattr(yolox_module.torch, "from_numpy", reject_numpy_mask)
    events = []
    with timing_event_sink(events.append):
        result = detector.predict([_frame()])[0]

    assert isinstance(result, Detections)
    assert [(event.component, event.phase) for event in events] == [
        ("detector", "preprocess"),
        ("detector", "process"),
        ("detector", "postprocess"),
    ]
    assert result.sample_id == "sample"
    torch.testing.assert_close(
        result.to_aabb_rows(),
        torch.tensor([[3, 4, 13, 14, 0.72, 1]], dtype=torch.float32),
    )
    assert result.geometry.values.is_contiguous()
    assert result.scores.is_contiguous()


def test_yolox_predict_empty_sequence_skips_model() -> None:
    detector = yolox_module.YoloXDetector.__new__(yolox_module.YoloXDetector)
    detector.model = None

    assert detector.predict(()) == []


def test_yolox_decode_clones_inference_tensor_before_mutating_postprocess(monkeypatch) -> None:
    """Upstream postprocess mutates boxes and cannot receive an inference tensor."""

    detector = yolox_module.YoloXDetector.__new__(yolox_module.YoloXDetector)
    detector.num_classes = 2
    detector._prediction_options = {
        "conf": 0.25,
        "iou": 0.7,
        "classes": None,
        "agnostic_nms": False,
    }
    with torch.inference_mode():
        predictions = torch.tensor([[[4, 5, 8, 10, 0.8, 0.5, 1]]], dtype=torch.float32)
    expected_predictions = predictions.clone()
    observed: dict[str, bool] = {}

    def mutating_postprocess(values: torch.Tensor, *_args, **_kwargs):
        observed["is_inference"] = torch.is_inference(values)
        values[..., :4].add_(100)
        return [torch.tensor([[1, 2, 11, 12, 0.8, 0.5, 1]], dtype=torch.float32)]

    monkeypatch.setattr(yolox_module, "yolox_postprocess", mutating_postprocess)

    result = detector._decode([_frame()], predictions, [1.0])[0]

    assert observed == {"is_inference": False}
    torch.testing.assert_close(predictions, expected_predictions)
    torch.testing.assert_close(
        result.to_aabb_rows(),
        torch.tensor([[1, 2, 11, 12, 0.4, 1]], dtype=torch.float32),
    )
