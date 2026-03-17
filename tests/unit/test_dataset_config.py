from pathlib import Path
from types import SimpleNamespace

import boxmot.utils.benchmark_config as benchmark_config

from boxmot.utils.benchmark_config import (
    apply_benchmark_config,
    apply_reid_runtime_defaults,
    apply_dataset_benchmark_config,
    ensure_benchmark_detector_model,
    get_benchmark_detector_url,
    get_benchmark_reid_cfg,
    load_benchmark_only_cfg,
    load_detector_component_cfg,
    load_benchmark_cfg,
    load_dataset_cfg,
    load_reid_component_cfg,
    load_runtime_reid_component_cfg,
    resolve_benchmark_cfg_path,
    resolve_dataset_cfg_path,
    resolve_required_reid_device,
    resolve_required_reid_half,
    resolve_required_reid_model,
    resolve_required_yolo_model,
    should_use_benchmark_detector,
    should_use_benchmark_reid,
)


def test_mot17_benchmark_uses_split_schema():
    cfg = load_benchmark_only_cfg("MOT17-ablation")
    assert cfg["id"] == "mot17-ablation"
    assert cfg["dataset_config"] == "mot17-ablation"
    assert cfg["path"] == "boxmot/engine/trackeval/data/MOT17-ablation"
    assert cfg["split"] == "train"
    assert cfg["train"] == "train"
    assert cfg["detector_config"] == "yolox_x_mot17_ablation"
    assert cfg["reid_config"] == "lmbn_n_duke"
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


def test_mot17_mini_uses_its_own_benchmark_id():
    cfg = load_benchmark_only_cfg("mot17-mini")
    assert cfg["id"] == "mot17-mini"
    assert cfg["dataset_config"] == "mot17-mini"
    assert cfg["detector_config"] == "yolox_x_mot17_ablation"
    assert cfg["reid_config"] == "lmbn_n_duke"
    assert cfg["storage"] == {
        "root": "assets/MOT17-mini",
        "split": "train",
    }


def test_dataset_path_stays_dataset_yaml():
    cfg_path = resolve_dataset_cfg_path("boxmot/configs/datasets/mot17-ablation.yaml")
    assert cfg_path.name == "mot17-ablation.yaml"
    assert cfg_path.parent.name == "datasets"


def test_legacy_dataset_path_resolves_to_benchmark_yaml():
    cfg_path = resolve_benchmark_cfg_path("boxmot/configs/datasets/mot17-ablation.yaml")
    assert cfg_path.name == "mot17-ablation.yaml"
    assert cfg_path.parent.name == "benchmarks"


def test_dataset_config_loads_without_model_bindings():
    cfg = load_dataset_cfg("MOT17-ablation")
    assert cfg["id"] == "mot17-ablation"
    assert cfg["path"] == "boxmot/engine/trackeval/data/MOT17-ablation"
    assert cfg["detector_config"] is None
    assert cfg["reid_config"] is None
    assert cfg["download"] == {
        "dataset": "https://github.com/mikel-brostrom/boxmot/releases/download/v13.0.9/MOT17-ablation.zip",
        "runs": "",
    }


def test_obb_dataset_derives_trackeval_from_box_type():
    cfg = load_dataset_cfg("MMOT-OBB")
    assert cfg["layout"] == "mot"
    assert cfg["box_type"] == "obb"
    assert cfg["trackeval"] == "mmot_rgb"
    assert cfg["evaluation"]["tracker_eval"] == "mmot_rgb"


def test_detector_and_reid_component_configs_load_separately():
    detector_cfg = load_detector_component_cfg("yolox_x_mot17_ablation")
    reid_cfg = load_reid_component_cfg("lmbn_n_duke")

    assert detector_cfg["id"] == "yolox_x_mot17_ablation"
    assert detector_cfg["model"] == "models/yolox_x_MOT17_ablation.pt"
    assert reid_cfg["id"] == "lmbn_n_duke"
    assert reid_cfg["model"] == "models/lmbn_n_duke.pt"
    assert reid_cfg["url"] == "https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_duke.pth"
    assert reid_cfg["device"] == ""
    assert reid_cfg["half"] is True


def test_runtime_reid_component_cfg_matches_model_stem():
    reid_cfg = load_runtime_reid_component_cfg("models/lmbn_n_duke.pt")

    assert reid_cfg["id"] == "lmbn_n_duke"
    assert reid_cfg["half"] is True


def test_mot17_dataset_exposes_default_detector():
    cfg = load_benchmark_cfg("MOT17-ablation")
    assert resolve_required_yolo_model(cfg) == Path("models/yolox_x_MOT17_ablation.pt")


def test_mot17_dataset_exposes_default_reid():
    cfg = load_benchmark_cfg("MOT17-ablation")
    assert get_benchmark_reid_cfg(cfg) == {
        "id": "lmbn_n_duke",
        "default_model": "models/lmbn_n_duke.pt",
        "model": "models/lmbn_n_duke.pt",
        "model_url": "https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_duke.pth",
        "url": "https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_duke.pth",
        "device": "",
        "half": True,
    }
    assert resolve_required_reid_model(cfg) == Path("models/lmbn_n_duke.pt")
    assert resolve_required_reid_device(cfg) is None
    assert resolve_required_reid_half(cfg) is True


def test_dataset_detector_is_used_for_default_model_selection():
    cfg = load_benchmark_cfg("MOT17-ablation")
    args = SimpleNamespace(yolo_model=[Path("models/yolov8n.pt")], yolo_model_explicit=False)
    assert should_use_benchmark_detector(args, cfg) is True


def test_dataset_reid_is_used_for_default_model_selection():
    cfg = load_benchmark_cfg("MOT17-ablation")
    args = SimpleNamespace(reid_model=[Path("models/osnet_x0_25_msmt17.pt")], reid_model_explicit=False)
    assert should_use_benchmark_reid(args, cfg) is True


def test_dataset_detector_is_used_when_same_model_is_explicit():
    cfg = load_benchmark_cfg("MMOT-OBB")
    args = SimpleNamespace(yolo_model=[Path("models/yolo11l-3ch.pt")], yolo_model_explicit=True)
    assert should_use_benchmark_detector(args, cfg) is True


def test_dataset_reid_is_not_used_for_other_explicit_models():
    cfg = load_benchmark_cfg("MOT17-ablation")
    args = SimpleNamespace(reid_model=[Path("models/mobilenetv2_x1_4_dukemtmcreid.pt")], reid_model_explicit=True)
    assert should_use_benchmark_reid(args, cfg) is False


def test_reid_runtime_defaults_follow_model_config_when_cli_not_explicit():
    cfg = load_benchmark_cfg("MOT17-ablation")
    args = SimpleNamespace(device="", half=False, device_explicit=False, half_explicit=False)

    apply_reid_runtime_defaults(args, cfg, use_config=True)

    assert args.reid_device == ""
    assert args.reid_half is True


def test_reid_runtime_defaults_respect_explicit_cli_flags():
    cfg = {
        "reid": {
            "model": "models/lmbn_n_duke.pt",
            "device": "cpu",
            "half": True,
        }
    }
    args = SimpleNamespace(device="cuda:0", half=False, device_explicit=True, half_explicit=True)

    apply_reid_runtime_defaults(args, cfg, use_config=True)

    assert args.reid_device == "cuda:0"
    assert args.reid_half is False


def test_mmot_obb_detector_exposes_download_url():
    cfg = load_benchmark_cfg("MMOT-OBB")
    assert get_benchmark_detector_url(cfg) == "https://drive.google.com/uc?id=15gmA4-Yclvh5EZvTJYhcyV1CVdNRGIkR"


def test_mot17_detector_exposes_download_url():
    cfg = load_benchmark_cfg("MOT17-ablation")
    assert get_benchmark_detector_url(cfg) == "https://drive.google.com/uc?id=1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob"


def test_dataset_detector_is_not_used_for_other_explicit_models():
    cfg = load_benchmark_cfg("visdrone-ablation")
    args = SimpleNamespace(yolo_model=[Path("models/yolov8x.pt")], yolo_model_explicit=True)
    assert should_use_benchmark_detector(args, cfg) is False


def test_apply_benchmark_config_preserves_runtime_benchmark_name(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(data="dancetrack-ablation", source=None)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "dancetrack-ablation"
    assert cfg["detector_config_id"] == "yolox_x_dancetrack_ablation"
    assert cfg["reid_config_id"] == "lmbn_n_duke"
    assert args.benchmark_id == "dancetrack-ablation"
    assert args.dataset_id == "dancetrack-ablation"
    assert args.benchmark == "dancetrack-ablation"
    assert args.source == Path("boxmot/engine/trackeval/data/test1/val")


def test_apply_benchmark_config_preserves_case_matched_storage_name(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(data="MOT17-ablation", source=None)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "mot17-ablation"
    assert cfg["detector_config_id"] == "yolox_x_mot17_ablation"
    assert cfg["reid_config_id"] == "lmbn_n_duke"
    assert args.benchmark_id == "mot17-ablation"
    assert args.dataset_id == "mot17-ablation"
    assert args.benchmark == "MOT17-ablation"
    assert args.source == Path("boxmot/engine/trackeval/data/MOT17-ablation/train")


def test_apply_benchmark_config_ignores_source_without_data(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(source="MOT17-ablation")
    assert apply_benchmark_config(args) is None


def test_apply_dataset_benchmark_config_accepts_legacy_source_fallback(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(source="MOT17-ablation")
    cfg = apply_dataset_benchmark_config(args)
    assert cfg["id"] == "mot17-ablation"
    assert args.benchmark == "MOT17-ablation"
    assert args.source == Path("boxmot/engine/trackeval/data/MOT17-ablation/train")


def test_ensure_benchmark_detector_model_downloads_missing_weight(monkeypatch, tmp_path):
    cfg = load_benchmark_cfg("MMOT-OBB")
    target = tmp_path / "yolo11l-3ch.pt"
    calls = {}

    monkeypatch.setattr(benchmark_config, "resolve_model_path", lambda *_args, **_kwargs: target)
    monkeypatch.setattr(
        benchmark_config,
        "download_file",
        lambda url, dest, overwrite=False, **_kwargs: calls.update({"url": url, "dest": dest, "overwrite": overwrite}) or dest,
    )

    resolved = ensure_benchmark_detector_model(cfg)
    assert resolved == target
    assert calls == {
        "url": "https://drive.google.com/uc?id=15gmA4-Yclvh5EZvTJYhcyV1CVdNRGIkR",
        "dest": target,
        "overwrite": False,
    }
