from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import boxmot.detectors.ultralytics as ultralytics_module
from boxmot.detectors.ultralytics import UltralyticsDetector


def test_ultralytics_predictor_clears_previous_class_filter():
    detector = UltralyticsDetector.__new__(UltralyticsDetector)
    detector._predictor = SimpleNamespace(args=SimpleNamespace(conf=0.25, iou=0.7, classes=None, agnostic_nms=False))

    detector._ensure_predictor(classes=[0])
    assert detector._predictor.args.classes == [0]

    detector._ensure_predictor(classes=None)
    assert detector._predictor.args.classes is None


def test_ultralytics_empty_fallback_preserves_obb_mode():
    detector = UltralyticsDetector.__new__(UltralyticsDetector)
    detector.is_obb = True

    detections, masks = detector._extract_dets(SimpleNamespace(obb=None, boxes=[]))

    assert detections.shape == (0, 7)
    assert detections.dtype == np.float32
    assert masks is None


def test_ultralytics_restores_corrupt_custom_weights_when_recovery_fails(monkeypatch, tmp_path):
    model_path = tmp_path / "yolo11_custom.pt"
    model_path.write_bytes(b"original-corrupt-weights")
    detector = UltralyticsDetector.__new__(UltralyticsDetector)

    monkeypatch.setattr(
        ultralytics_module,
        "YOLO",
        lambda _path: (_ for _ in ()).throw(RuntimeError("failed finding central directory")),
    )
    monkeypatch.setattr(
        detector,
        "_download_weights",
        lambda _path, _url: (_ for _ in ()).throw(OSError("download failed")),
    )

    with pytest.raises(OSError, match="download failed"):
        detector._load_yolo(model_path, "https://example.test/custom.pt")

    assert model_path.read_bytes() == b"original-corrupt-weights"
    assert not list(tmp_path.glob("*.backup"))
