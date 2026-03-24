import pytest

from boxmot.detectors import get_detector_class


def test_get_yolo_inferer_routes_yolox_model():
    inferer_cls = get_detector_class("yolox_s.pt")
    assert inferer_cls.__name__ == "YoloXDetector"


def test_get_yolo_inferer_returns_callable_strategy_for_yolox():
    inferer_cls = get_detector_class("yolox_n.pt")
    assert callable(inferer_cls)


@pytest.mark.parametrize("name", ["yolox_s.pt", "yolox_x_MOT17_ablation.pt"])
def test_get_yolo_inferer_accepts_common_yolox_names(name):
    inferer_cls = get_detector_class(name)
    assert inferer_cls.__name__ == "YoloXDetector"
