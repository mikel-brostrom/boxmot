from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import torch


def _import_rtdetr_with_stubbed_transformers(monkeypatch):
    """Import the adapter without making transformers a unit-test dependency."""
    transformers = types.ModuleType("transformers")
    transformers.RTDetrImageProcessor = object
    transformers.RTDetrV2ForObjectDetection = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    package = importlib.import_module("boxmot.detectors")
    monkeypatch.delattr(package, "rtdetr", raising=False)
    monkeypatch.delitem(sys.modules, "boxmot.detectors.rtdetr", raising=False)
    return importlib.import_module("boxmot.detectors.rtdetr")


def test_rtdetr_preserves_two_image_batch_with_mixed_empty_results(monkeypatch):
    rtdetr_module = _import_rtdetr_with_stubbed_transformers(monkeypatch)

    class FakeModel:
        def __call__(self, **preprocessed):
            assert "pixel_values" in preprocessed
            return object()

    class FakeProcessor:
        def post_process_object_detection(self, outputs, target_sizes, threshold):
            assert outputs is not None
            assert tuple(target_sizes.shape) == (2, 2)
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

    images = [
        np.zeros((32, 48, 3), dtype=np.uint8),
        np.zeros((40, 60, 3), dtype=np.uint8),
    ]
    detector = rtdetr_module.RTDetrDetector.__new__(rtdetr_module.RTDetrDetector)
    detector.device = torch.device("cpu")
    detector.model = FakeModel()
    detector.image_processor = FakeProcessor()
    detector.names = {2: "car", 3: "bus"}
    detector._images = images
    detector._target_sizes = torch.tensor([[32, 48], [40, 60]], dtype=torch.int64)

    raw = detector.process({"pixel_values": torch.zeros((2, 3, 8, 8))})
    results = detector.postprocess(raw, conf=0.5, classes=None, iou=0.7, agnostic_nms=False)

    assert len(results) == 2
    assert results[0].orig_img is images[0]
    assert results[1].orig_img is images[1]
    np.testing.assert_array_equal(
        results[0].dets,
        np.array([[1, 2, 11, 12, 0.9, 2]], dtype=np.float32),
    )
    assert results[1].dets.shape == (0, 6)
    assert results[1].dets.dtype == np.float32
