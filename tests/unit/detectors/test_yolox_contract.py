from __future__ import annotations

import numpy as np
import torch

import boxmot.detectors.yolox as yolox_module


def test_yolox_uses_profile_class_count_and_names(monkeypatch, tmp_path):
    model_path = tmp_path / "yolox_x_visdrone.pt"
    model_path.touch()

    class FakeModel:
        def eval(self):
            return self

        def to(self, _device):
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
    monkeypatch.setattr(yolox_module, "load_detector_cfg", lambda _path: {"classes": names})
    monkeypatch.setattr(yolox_module.torch, "load", lambda *_args, **_kwargs: {"model": {"weights": "stub"}})
    monkeypatch.setattr(yolox_module, "fuse_model", lambda model: model)

    detector = yolox_module.YoloXDetector(model_path, device="cpu", imgsz=[64, 96])

    assert detector.num_classes == 10
    assert detector.names == names
    assert experiment.num_classes == 10


def test_yolox_filters_classes_without_cpu_numpy_index_roundtrip(monkeypatch):
    raw = torch.tensor(
        [
            [1, 2, 11, 12, 0.8, 0.5, 0],
            [3, 4, 13, 14, 0.9, 0.8, 1],
        ],
        dtype=torch.float32,
    )
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    detector = yolox_module.YoloXDetector.__new__(yolox_module.YoloXDetector)
    detector._im0s = [image]
    detector._preproc_data = [1.0]

    monkeypatch.setattr(yolox_module, "postprocess", lambda *_args, **_kwargs: [raw.clone()])

    def reject_numpy_mask(_values):
        raise AssertionError("class filtering must keep its mask on the detection tensor device")

    monkeypatch.setattr(yolox_module.torch, "from_numpy", reject_numpy_mask)

    results = detector.postprocess(
        [raw],
        conf=0.25,
        iou=0.7,
        classes=[1],
        agnostic_nms=False,
    )

    assert len(results) == 1
    assert results[0].orig_img is image
    np.testing.assert_allclose(
        results[0].dets,
        np.array([[3, 4, 13, 14, 0.72, 1]], dtype=np.float32),
        atol=1e-6,
    )
