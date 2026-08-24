from __future__ import annotations

import subprocess
import sys

import pytest

from boxmot.detectors.registry import (
    get_detector_url,
    is_rtdetr_model,
    is_seg_model,
    is_ultralytics_model,
    is_yolox_model,
    load_detector_cfg,
)


@pytest.mark.parametrize(
    ("matcher", "model"),
    [
        (is_ultralytics_model, "models/YOLO11N.PT"),
        (is_ultralytics_model, "custom/YoloV8_people.onnx"),
        (is_yolox_model, "models/YOLOX_S.PT"),
        (is_yolox_model, "weights/YoloX_X_MOT17.pt"),
        (is_rtdetr_model, "PekingU/RTDETR_V2_R50VD"),
        (is_rtdetr_model, "models/RtDetr_V2_R101VD.pt"),
    ],
)
def test_detector_family_matchers_are_case_insensitive(matcher, model):
    assert matcher(model) is True


def test_detector_family_matching_ignores_parent_directories_and_segmentation_substrings():
    assert is_yolox_model("runs/yolox_s_experiment/custom.pt") is False
    assert is_seg_model("noseg.pt") is False
    assert is_seg_model("models/YOLO11N-SEG.PT") is True


@pytest.mark.parametrize(
    ("model", "checkpoint", "file_id"),
    [
        ("yolox_n.pt", "n", "1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX"),
        ("yolox_s.pt", "s", "1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj"),
        ("yolox_m.pt", "m", "11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun"),
        ("yolox_l.pt", "l", "1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz"),
        ("yolox_x.pt", "x", "1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5"),
    ],
)
def test_generic_yolox_checkpoint_metadata_comes_from_detector_config(model, checkpoint, file_id):
    config = load_detector_cfg(model)

    assert config["id"] == "yolox"
    assert config["checkpoint"] == checkpoint
    assert config["model"] == f"models/{model}"
    assert get_detector_url(model) == f"https://drive.google.com/uc?id={file_id}"


def test_public_detector_exports_do_not_eagerly_import_backend_modules():
    code = """
import sys
import boxmot.detectors as detectors

backend_modules = {
    "boxmot.detectors.rtdetr",
    "boxmot.detectors.ultralytics",
    "boxmot.detectors.yolox",
}
assert backend_modules.isdisjoint(sys.modules)
assert {"Detector", "Detections", "get_detector_class"} <= set(detectors.__all__)
_ = detectors.Detector
_ = detectors.Detections
_ = detectors.get_detector_class
assert backend_modules.isdisjoint(sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
