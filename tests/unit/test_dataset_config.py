import boxmot.utils.dataset_config as dataset_config
from pathlib import Path
from types import SimpleNamespace

from boxmot.utils.dataset_config import (
    apply_dataset_benchmark_config,
    load_dataset_cfg,
    resolve_required_yolo_model,
    should_use_dataset_detector,
)


def test_mot17_dataset_uses_new_schema():
    cfg = load_dataset_cfg("MOT17-ablation")
    assert cfg["id"] == "mot17-ablation"
    assert cfg["storage"] == {
        "root": "boxmot/engine/trackeval/data/MOT17-ablation",
        "split": "train",
    }
    assert cfg["evaluation"] == {
        "box_type": "aabb",
        "layout": "mot",
        "tracker_eval": "mot_challenge",
        "classes": {
            "eval": {1: "pedestrian"},
            "distractor": {
                2: "person_on_vehicle",
                7: "static_person",
                8: "distractor",
                12: "reflection",
            },
            "mapping": {},
        },
    }


def test_mot17_dataset_exposes_default_detector():
    cfg = load_dataset_cfg("MOT17-ablation")
    assert resolve_required_yolo_model(cfg) == Path("models/yolox_x_MOT17_ablation.pt")


def test_dataset_detector_is_used_for_default_model_selection():
    cfg = load_dataset_cfg("MOT17-ablation")
    args = SimpleNamespace(yolo_model=[Path("models/yolov8n.pt")], yolo_model_explicit=False)
    assert should_use_dataset_detector(args, cfg) is True


def test_dataset_detector_is_used_when_same_model_is_explicit():
    cfg = load_dataset_cfg("MMOT-OBB")
    args = SimpleNamespace(yolo_model=[Path("models/yolo11l-3ch.pt")], yolo_model_explicit=True)
    assert should_use_dataset_detector(args, cfg) is True


def test_dataset_detector_is_not_used_for_other_explicit_models():
    cfg = load_dataset_cfg("visdrone-ablation")
    args = SimpleNamespace(yolo_model=[Path("models/yolov8x.pt")], yolo_model_explicit=True)
    assert should_use_dataset_detector(args, cfg) is False


def test_apply_dataset_benchmark_config_preserves_runtime_benchmark_name(monkeypatch):
    monkeypatch.setattr(dataset_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(source="dancetrack-ablation")
    cfg = apply_dataset_benchmark_config(args)
    assert cfg["id"] == "dancetrack-ablation"
    assert args.dataset_id == "dancetrack-ablation"
    assert args.benchmark == "dancetrack-ablation"
    assert args.source == Path("boxmot/engine/trackeval/data/test1/val")


def test_apply_dataset_benchmark_config_preserves_case_matched_storage_name(monkeypatch):
    monkeypatch.setattr(dataset_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(source="MOT17-ablation")
    cfg = apply_dataset_benchmark_config(args)
    assert cfg["id"] == "mot17-ablation"
    assert args.dataset_id == "mot17-ablation"
    assert args.benchmark == "MOT17-ablation"
    assert args.source == Path("boxmot/engine/trackeval/data/MOT17-ablation/train")
