from pathlib import Path
from types import SimpleNamespace

import boxmot.utils.benchmark_config as benchmark_config

from boxmot.utils.benchmark_config import (
    apply_benchmark_config,
    apply_reid_runtime_defaults,
    ensure_dataset_source_available,
    ensure_benchmark_detector_model,
    find_dataset_cfg_for_source,
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
    cfg = load_benchmark_only_cfg("mot17-mini")
    assert cfg["id"] == "mot17-mini"
    assert cfg["dataset_config"] == "mot17-mini"
    assert cfg["path"] == "assets/MOT17-mini"
    assert cfg["split"] == "train"
    assert cfg["train"] == "train"
    assert cfg["detector_config"] == "yolox_x_mot17_ablation"
    assert cfg["reid_config"] == "lmbn_n_duke"
    assert cfg["storage"] == {
        "root": "assets/MOT17-mini",
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
    cfg_path = resolve_dataset_cfg_path("boxmot/configs/datasets/mot17-mini.yaml")
    assert cfg_path.name == "mot17-mini.yaml"
    assert cfg_path.parent.name == "datasets"


def test_benchmark_path_stays_dataset_yaml():
    cfg_path = resolve_benchmark_cfg_path("boxmot/configs/datasets/mot17-mini.yaml")
    assert cfg_path.name == "mot17-mini.yaml"
    assert cfg_path.parent.name == "datasets"


def test_dataset_config_loads_with_model_bindings():
    cfg = load_dataset_cfg("mot17-mini")
    assert cfg["id"] == "mot17-mini"
    assert cfg["path"] == "assets/MOT17-mini"
    assert cfg["detector_config"] == "yolox_x_mot17_ablation"
    assert cfg["reid_config"] == "lmbn_n_duke"


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
    cfg = load_benchmark_cfg("mot17-mini")
    assert resolve_required_yolo_model(cfg) == Path("models/yolox_x_MOT17_ablation.pt")


def test_mot17_dataset_exposes_default_reid():
    cfg = load_benchmark_cfg("mot17-mini")
    assert get_benchmark_reid_cfg(cfg) == {
        "id": "lmbn_n_duke",
        "default_model": "models/lmbn_n_duke.pt",
        "model": "models/lmbn_n_duke.pt",
        "model_url": "https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_duke.pth",
        "url": "https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_duke.pth",
        "device": "",
        "half": True,
        "preprocess": "resize",
    }
    assert resolve_required_reid_model(cfg) == Path("models/lmbn_n_duke.pt")
    assert resolve_required_reid_device(cfg) is None
    assert resolve_required_reid_half(cfg) is True


def test_dataset_detector_is_used_for_default_model_selection():
    cfg = load_benchmark_cfg("mot17-mini")
    args = SimpleNamespace(detector=[Path("models/yolov8n.pt")], detector_explicit=False)
    assert should_use_benchmark_detector(args, cfg) is True


def test_dataset_reid_is_used_for_default_model_selection():
    cfg = load_benchmark_cfg("mot17-mini")
    args = SimpleNamespace(reid=[Path("models/osnet_x0_25_msmt17.pt")], reid_explicit=False)
    assert should_use_benchmark_reid(args, cfg) is True


def test_dataset_detector_is_used_when_same_model_is_explicit():
    cfg = load_benchmark_cfg("MMOT-OBB")
    args = SimpleNamespace(detector=[Path("models/yolo11l-3ch.pt")], detector_explicit=True)
    assert should_use_benchmark_detector(args, cfg) is True


def test_dataset_reid_is_not_used_for_other_explicit_models():
    cfg = load_benchmark_cfg("mot17-mini")
    args = SimpleNamespace(reid=[Path("models/mobilenetv2_x1_4_dukemtmcreid.pt")], reid_explicit=True)
    assert should_use_benchmark_reid(args, cfg) is False


def test_reid_runtime_defaults_follow_model_config_when_cli_not_explicit():
    cfg = load_benchmark_cfg("mot17-mini")
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
    cfg = load_benchmark_cfg("mot17-mini")
    assert get_benchmark_detector_url(cfg) == "https://drive.google.com/uc?id=1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob"


def test_dataset_detector_is_not_used_for_other_explicit_models():
    cfg = load_benchmark_cfg("visdrone-ablation")
    args = SimpleNamespace(detector=[Path("models/yolov8x.pt")], detector_explicit=True)
    assert should_use_benchmark_detector(args, cfg) is False


def test_apply_benchmark_config_preserves_runtime_benchmark_name(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(data="dancetrack-ablation", source=None, split=None, split_explicit=False)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "dancetrack-ablation"
    assert cfg["detector_config_id"] == "yolox_x_dancetrack_ablation"
    assert cfg["reid_config_id"] == "lmbn_n_duke"
    assert args.benchmark_id == "dancetrack-ablation"
    assert args.dataset_id == "dancetrack-ablation"
    assert args.benchmark == "dancetrack-ablation"
    assert args.source == Path("boxmot/engine/trackeval/data/test1/val")


def test_apply_benchmark_config_normalizes_benchmark_name_to_lowercase(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(data="MOT17-mini", source=None, split=None, split_explicit=False)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "mot17-mini"
    assert cfg["detector_config_id"] == "yolox_x_mot17_ablation"
    assert cfg["reid_config_id"] == "lmbn_n_duke"
    assert args.benchmark_id == "mot17-mini"
    assert args.dataset_id == "mot17-mini"
    assert args.benchmark == "mot17-mini"
    assert args.source == Path("assets/MOT17-mini/train")


def test_apply_benchmark_config_ignores_source_without_data(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(source="MOT17-ablation")
    assert apply_benchmark_config(args) is None


def test_find_dataset_cfg_for_nested_source_path():
    cfg = find_dataset_cfg_for_source("boxmot/engine/trackeval/data/MMOT-OBB/train/data44-3/img1")

    assert cfg is not None
    assert cfg["id"] == "mmot-obb"
    assert cfg["path"] == "boxmot/engine/trackeval/data/MMOT-OBB"


def test_ensure_dataset_source_available_downloads_missing_dataset(monkeypatch):
    calls = {}
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: calls.update(kwargs))
    source = "boxmot/engine/trackeval/data/MMOT-OBB/train/data44-3/img1"
    real_exists = Path.exists

    def fake_exists(self):
        if self == Path(source):
            return False
        return real_exists(self)

    monkeypatch.setattr(benchmark_config.Path, "exists", fake_exists)

    args = SimpleNamespace(
        source=source,
        eval_box_type=None,
    )

    cfg = ensure_dataset_source_available(args)

    assert cfg is not None
    assert cfg["id"] == "mmot-obb"
    assert args.source == "boxmot/engine/trackeval/data/MMOT-OBB/train/data44-3/img1"
    assert args.dataset_id == "mmot-obb"
    assert args.eval_box_type == "obb"
    assert calls == {
        "runs_url": "",
        "dataset_url": "https://github.com/mikel-brostrom/boxmot/releases/download/v16.0.11/MMOT-OBB.zip",
        "dataset_dest": Path("boxmot/engine/trackeval/data/MMOT-OBB.zip"),
        "overwrite": False,
        "runs_check_path": None,
        "status_fn": None,
    }


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


def test_sportsmot_benchmark_uses_split_schema():
    cfg = load_benchmark_only_cfg("sportsmot")
    assert cfg["id"] == "sportsmot"
    assert cfg["dataset_config"] == "sportsmot"
    assert cfg["path"] == "boxmot/engine/trackeval/data/SportsMOT"
    assert cfg["split"] == "val"
    assert cfg["train"] == "train"
    assert cfg["test"] == "test"
    assert cfg["detector_config"] == "yolox_x_sportsmot"
    assert cfg["reid_config"] == "lmbn_n_duke"
    assert cfg["storage"] == {
        "root": "boxmot/engine/trackeval/data/SportsMOT",
        "split": "val",
    }
    assert cfg["evaluation"] == {
        "box_type": "aabb",
        "layout": "mot",
        "tracker_eval": "mot_challenge",
        "classes": {
            "eval": {1: "player"},
            "distractor": {},
            "mapping": {},
        },
    }


def test_sportsmot_dataset_loads_with_model_bindings():
    cfg = load_dataset_cfg("sportsmot")
    assert cfg["id"] == "sportsmot"
    assert cfg["path"] == "boxmot/engine/trackeval/data/SportsMOT"
    assert cfg["box_type"] == "aabb"
    assert cfg["layout"] == "mot"
    assert cfg["trackeval"] == "mot_challenge"
    assert cfg["detector_config"] == "yolox_x_sportsmot"
    assert cfg["reid_config"] == "lmbn_n_duke"


def test_sportsmot_full_benchmark_loads_detector_and_reid():
    cfg = load_benchmark_cfg("sportsmot")
    assert resolve_required_yolo_model(cfg) == Path("models/yolox_x_sportsmot.pt")
    assert resolve_required_reid_model(cfg) == Path("models/lmbn_n_duke.pt")


def test_apply_benchmark_config_resolves_sportsmot(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(data="sportsmot", source=None, split=None, split_explicit=False)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "sportsmot"
    assert args.benchmark_id == "sportsmot"
    assert args.dataset_id == "sportsmot"
    assert args.source == Path("boxmot/engine/trackeval/data/SportsMOT/val")


def test_find_dataset_cfg_for_sportsmot_source():
    cfg = find_dataset_cfg_for_source("boxmot/engine/trackeval/data/SportsMOT/test/SNMOT-001/img1")
    assert cfg is not None
    assert cfg["id"] == "sportsmot"
    assert cfg["path"] == "boxmot/engine/trackeval/data/SportsMOT"


def test_mot17_ablation_split_resolves_to_train_with_seq_pattern(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(data="mot17", source=None, split="ablation", split_explicit=True)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "mot17"
    assert args.split == "ablation"
    # When the base dir exists, a filtered split dir is built with symlinks
    assert args.source == Path("boxmot/engine/trackeval/data/MOT17/ablation")
    assert args.seq_pattern == "*-FRCNN"
    # Verify the symlink dir only contains FRCNN sequences
    seq_names = [p.name for p in args.source.iterdir() if p.is_dir()]
    assert all(name.endswith("-FRCNN") for name in seq_names)
    assert len(seq_names) == 7


def test_mot17_ablation_split_respects_cli_detection_source(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(
        data="mot17", source=None, split="ablation", split_explicit=True,
        detection_source="public",
    )
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "mot17"
    assert args.split == "ablation"
    assert args.source == Path("boxmot/engine/trackeval/data/MOT17/ablation")
    assert args.seq_pattern == "*-FRCNN"
    # CLI --detection-source takes precedence
    assert args.detection_source == "public"
