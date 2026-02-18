import pytest

from boxmot.detectors import get_yolo_inferer


def test_get_yolo_inferer_routes_yolox_model():
    inferer_cls = get_yolo_inferer("yolox_s.pt")
    assert inferer_cls.__name__ == "YoloXStrategy"


def test_get_yolo_inferer_returns_callable_strategy_for_yolox():
    inferer_cls = get_yolo_inferer("yolox_n.pt")
    assert callable(inferer_cls)


@pytest.mark.parametrize("name", ["yolox_s.pt", "yolox_x_MOT17_ablation.pt"])
def test_get_yolo_inferer_accepts_common_yolox_names(name):
    inferer_cls = get_yolo_inferer(name)
    assert inferer_cls.__name__ == "YoloXStrategy"
