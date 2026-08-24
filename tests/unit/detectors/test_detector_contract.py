from __future__ import annotations

from pathlib import Path

import numpy as np

import boxmot.detectors.detector as detector_module
from boxmot.detectors.base import Detections


class _StagedBackend:
    models = []

    def __init__(self, model, device, imgsz):
        self.models.append(model)
        self.device = device
        self.imgsz = imgsz
        self._images = []

    def preprocess(self, images):
        self._images = images
        return images

    def process(self, preprocessed):
        return preprocessed

    def postprocess(self, _raw, **_kwargs):
        return [
            Detections(
                dets=np.empty((0, 6), dtype=np.float32),
                orig_img=image,
                path="",
                names={0: "person"},
            )
            for image in self._images
        ]


def test_detector_normalizes_string_model_path_before_backend_construction(monkeypatch):
    _StagedBackend.models = []
    monkeypatch.setattr(detector_module.Detector, "_get_backend_class", classmethod(lambda cls, path: _StagedBackend))

    detector_module.Detector("models/yolox_s.pt", device="cpu", imgsz=[64, 64], conf=0.25)

    assert _StagedBackend.models == [Path("models/yolox_s.pt")]


def test_detector_returns_one_metadata_aligned_result_per_input_for_every_batch_size(monkeypatch):
    frames = [
        ("frame-1.jpg", np.zeros((16, 20, 3), dtype=np.uint8)),
        ("frame-2.jpg", np.ones((16, 20, 3), dtype=np.uint8)),
        ("frame-3.jpg", np.full((16, 20, 3), 2, dtype=np.uint8)),
    ]
    monkeypatch.setattr(detector_module.Detector, "_get_backend_class", classmethod(lambda cls, path: _StagedBackend))
    monkeypatch.setattr(detector_module, "iter_source", lambda _source, vid_stride=1: iter(frames))

    outputs_by_batch = []
    for batch_size in (1, 2):
        detector = detector_module.Detector(
            "models/yolo11n.pt",
            device="cpu",
            imgsz=[64, 64],
            conf=0.25,
            batch=batch_size,
        )
        outputs_by_batch.append(detector("stream://unit-test", as_detections=True))

    for outputs in outputs_by_batch:
        assert len(outputs) == len(frames)
        assert all(isinstance(result, Detections) for result in outputs)
        assert [result.path for result in outputs] == [path for path, _frame in frames]
        assert all(result.orig_img is frame for result, (_path, frame) in zip(outputs, frames))
        assert all(result.dets.shape == (0, 6) for result in outputs)

    raw_outputs = detector_module.Detector(
        "models/yolo11n.pt",
        device="cpu",
        imgsz=[64, 64],
        conf=0.25,
        batch=2,
    )("stream://unit-test")
    assert len(raw_outputs) == len(frames)
    assert all(isinstance(result, np.ndarray) and result.shape == (0, 6) for result in raw_outputs)


def test_per_call_options_keep_the_staged_pipeline_and_do_not_preprocess_twice(monkeypatch):
    calls = []

    class RecordingBackend(_StagedBackend):
        def preprocess(self, images):
            calls.append("preprocess")
            return super().preprocess(images)

        def process(self, preprocessed):
            calls.append("process")
            return preprocessed

        def postprocess(self, raw, **kwargs):
            calls.append(("postprocess", kwargs["conf"], kwargs["classes"]))
            return super().postprocess(raw, **kwargs)

    monkeypatch.setattr(
        detector_module.Detector,
        "_get_backend_class",
        classmethod(lambda cls, path: RecordingBackend),
    )
    detector = detector_module.Detector("models/yolo11n.pt", imgsz=[64, 64], conf=0.25)

    result = detector(
        np.zeros((16, 20, 3), dtype=np.uint8),
        conf=0.8,
        classes=[2],
        as_detections=True,
    )

    assert isinstance(result, Detections)
    assert calls == ["preprocess", "process", ("postprocess", 0.8, [2])]
