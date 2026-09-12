from __future__ import annotations

import importlib
import os
import sys
import types

import pytest
import torch

from boxmot.components.timing import timing_event_sink
from boxmot.detectors import DetectorCapabilities, DetectorSpec
from boxmot.structures import Detections, Frame


def _import_rtdetr_with_stubbed_transformers(monkeypatch):
    """Import the backend without making transformers a unit-test dependency."""

    transformers = types.ModuleType("transformers")
    transformers.RTDetrImageProcessor = object
    transformers.RTDetrV2ForObjectDetection = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    package = importlib.import_module("boxmot.detectors.backends")
    monkeypatch.delattr(package, "rtdetr", raising=False)
    monkeypatch.delitem(sys.modules, "boxmot.detectors.backends.rtdetr", raising=False)
    return importlib.import_module("boxmot.detectors.backends.rtdetr")


def _frame(sample_id: str, height: int, width: int) -> Frame:
    return Frame(torch.zeros((3, height, width), dtype=torch.uint8), sample_id)


def test_rtdetr_preserves_two_frame_batch_with_mixed_empty_results(monkeypatch) -> None:
    rtdetr_module = _import_rtdetr_with_stubbed_transformers(monkeypatch)

    class Inputs(dict):
        def to(self, device):
            assert device == torch.device("cpu")
            return self

    class FakeModel:
        def __call__(self, **preprocessed):
            assert "pixel_values" in preprocessed
            return object()

    class FakeProcessor:
        def __call__(self, *, images, return_tensors):
            assert [image.size for image in images] == [(48, 32), (60, 40)]
            assert return_tensors == "pt"
            return Inputs(pixel_values=torch.zeros((2, 3, 8, 8)))

        def post_process_object_detection(self, outputs, target_sizes, threshold):
            assert outputs is not None
            assert target_sizes.tolist() == [[32, 48], [40, 60]]
            assert threshold == 0.0
            return [
                {
                    "boxes": torch.tensor([[1, 2, 11, 12], [3, 4, 13, 14]], dtype=torch.float32),
                    "scores": torch.tensor([0.9, 0.2], dtype=torch.float32),
                    "labels": torch.tensor([2, 3], dtype=torch.int64),
                },
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                },
            ]

    detector = rtdetr_module.RTDetrDetector.__new__(rtdetr_module.RTDetrDetector)
    detector.device = torch.device("cpu")
    detector.model = FakeModel()
    detector.image_processor = FakeProcessor()
    detector.names = {2: "car", 3: "bus"}
    detector._confidence = 0.5
    detector._classes = None
    detector.capabilities = DetectorCapabilities()

    events = []
    with timing_event_sink(events.append):
        results = detector.predict([_frame("first", 32, 48), _frame("second", 40, 60)])

    assert len(results) == 2
    assert [(event.component, event.phase) for event in events] == [
        ("detector", "preprocess"),
        ("detector", "process"),
        ("detector", "postprocess"),
    ]
    assert all(isinstance(result, Detections) for result in results)
    torch.testing.assert_close(
        results[0].to_aabb_rows(),
        torch.tensor([[1, 2, 11, 12, 0.9, 2]], dtype=torch.float32),
    )
    assert results[1].sample_id == "second"
    assert results[1].geometry.values.shape == (0, 4)
    assert results[1].scores.dtype == torch.float32


@pytest.mark.parametrize("device", ("cpu", "1", "cuda:1"))
def test_rtdetr_loads_only_a_resolved_local_snapshot(monkeypatch, tmp_path, device) -> None:
    rtdetr_module = _import_rtdetr_with_stubbed_transformers(monkeypatch)
    snapshot = tmp_path / "rtdetr-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    class FakeProcessorClass:
        @classmethod
        def from_pretrained(cls, reference, **kwargs):
            calls.append(("processor", reference, kwargs))
            return object()

    class FakeModel:
        config = types.SimpleNamespace(id2label={0: "person"})

        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            return self

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, reference, **kwargs):
            calls.append(("model", reference, kwargs))
            return FakeModel()

    monkeypatch.setattr(
        rtdetr_module,
        "_transformers_classes",
        lambda: (FakeProcessorClass, FakeModelClass),
    )
    detector = rtdetr_module.RTDetrDetector(DetectorSpec("rtdetr", artifact=str(snapshot), device=device))

    assert detector.model_id == str(snapshot.resolve())
    assert calls[:2] == [
        ("processor", str(snapshot.resolve()), {"local_files_only": True}),
        ("model", str(snapshot.resolve()), {"local_files_only": True}),
    ]
    expected_device = torch.device("cpu" if device == "cpu" else "cuda:1")
    assert detector.device == expected_device
    assert calls[2] == ("to", expected_device)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,7"

    checkpoint = tmp_path / "rtdetr.pt"
    checkpoint.write_bytes(b"not a snapshot")
    with pytest.raises(ValueError, match="local Hugging Face snapshot directory"):
        rtdetr_module.RTDetrDetector(DetectorSpec("rtdetr", artifact=str(checkpoint)))


def test_rtdetr_predict_empty_sequence_skips_processor_and_model(monkeypatch) -> None:
    rtdetr_module = _import_rtdetr_with_stubbed_transformers(monkeypatch)
    detector = rtdetr_module.RTDetrDetector.__new__(rtdetr_module.RTDetrDetector)

    assert detector.predict(()) == []
