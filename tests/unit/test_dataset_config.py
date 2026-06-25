from pathlib import Path
from types import SimpleNamespace

import pytest

import boxmot.configs.benchmark as benchmark_config
from boxmot.configs.benchmark import (
    apply_benchmark_config,
    apply_reid_runtime_defaults,
    ensure_benchmark_detector_model,
    ensure_dataset_source_available,
    find_dataset_cfg_for_source,
    get_benchmark_detector_url,
    get_benchmark_reid_cfg,
    load_benchmark_cfg,
    load_benchmark_only_cfg,
    load_dataset_cfg,
    load_detector_component_cfg,
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


def test_benchmark_yaml_supports_inline_dataset_detector_reid_blocks():
    cfg = load_benchmark_cfg("mot17-mini")

    assert cfg["dataset_config"] == "mot17-mini"
    assert cfg["detector"]["id"] == "yolox_x_mot17_ablation"
    assert cfg["reid"]["id"] == "lmbn_n_duke"


def test_dataset_path_stays_dataset_yaml():
    cfg_path = resolve_dataset_cfg_path("mot17-mini")
    assert cfg_path.name == "mot17-mini.yaml"
    assert cfg_path.parent.name == "benchmarks"


def test_benchmark_path_stays_dataset_yaml():
    cfg_path = resolve_benchmark_cfg_path("mot17-mini")
    assert cfg_path.name == "mot17-mini.yaml"
    assert cfg_path.parent.name == "benchmarks"


def test_dataset_config_loads_with_model_bindings():
    cfg = load_dataset_cfg("mot17-mini")
    assert cfg["id"] == "mot17-mini"
    assert cfg["path"] == "assets/MOT17-mini"
    assert cfg["detector_config"] == "yolox_x_mot17_ablation"
    assert cfg["reid_config"] == "lmbn_n_duke"


def test_obb_dataset_derives_trackeval_from_box_type():
    cfg = load_dataset_cfg("mmot")
    assert cfg["layout"] == "mot"
    assert cfg["box_type"] == "obb"
    assert cfg["trackeval"] == "mot_challenge_obb"
    assert cfg["evaluation"]["tracker_eval"] == "mot_challenge_obb"


def test_mmot_mini_uses_mmot_mini_root():
    cfg = load_benchmark_only_cfg("mmot-mini")
    assert cfg["id"] == "mmot-mini"
    assert cfg["dataset_config"] == "mmot-mini"
    assert cfg["path"] == "assets/mmot-mini"
    assert cfg["split"] == "train"
    assert cfg["train"] == "train/npy"
    assert cfg["trackeval"] == "mot_challenge_obb"


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
    cfg = load_benchmark_cfg("mmot")
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
    cfg = load_benchmark_cfg("mmot")
    assert get_benchmark_detector_url(cfg) == "https://drive.google.com/uc?id=15gmA4-Yclvh5EZvTJYhcyV1CVdNRGIkR"


def test_mot17_detector_exposes_download_url():
    cfg = load_benchmark_cfg("mot17-mini")
    assert get_benchmark_detector_url(cfg) == "https://huggingface.co/Lekim89/yolox/resolve/main/yolox_x_MOT17_ablation.pt"


def test_dataset_detector_is_not_used_for_other_explicit_models():
    cfg = load_benchmark_cfg("visdrone")
    args = SimpleNamespace(detector=[Path("models/yolov8x.pt")], detector_explicit=True)
    assert should_use_benchmark_detector(args, cfg) is False


def test_apply_benchmark_config_preserves_runtime_benchmark_name(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(data="dancetrack", source=None, split=None, split_explicit=False)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "dancetrack"
    assert cfg["detector_config_id"] == "yolox_x_dancetrack"
    assert cfg["reid_config_id"] == "lmbn_n_duke"
    assert args.benchmark_id == "dancetrack"
    assert args.dataset_id == "dancetrack"
    assert args.benchmark == "dancetrack"
    assert args.source == Path("data/benchmarks/test1/val")


def test_apply_benchmark_config_normalizes_benchmark_name_to_lowercase(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(data="mot17-mini", source=None, split=None, split_explicit=False)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "mot17-mini"
    assert cfg["detector_config_id"] == "yolox_x_mot17_ablation"
    assert cfg["reid_config_id"] == "lmbn_n_duke"
    assert args.benchmark_id == "mot17-mini"
    assert args.dataset_id == "mot17-mini"
    assert args.benchmark == "mot17-mini"
    assert args.source == Path("assets/MOT17-mini/train")


def test_apply_benchmark_config_resolves_split_specific_runs_url(monkeypatch):
    calls = {}

    def _capture_download(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(benchmark_config, "download_eval_data", _capture_download)
    monkeypatch.setattr("boxmot.data.mot17_parquet.setup_mot17_from_parquet", lambda **kwargs: None)
    args = SimpleNamespace(data="mot17", source=None, split="ablation", split_explicit=True)

    apply_benchmark_config(args)

    assert calls["runs_url"] == "hf://Lekim89/runs/runs/dets_n_embs/mot17/ablation"
    assert calls["dataset_url"] == ""
    assert calls["runs_check_path"] == Path("runs/dets_n_embs/mot17/ablation")


def test_apply_benchmark_config_resolves_mot17_test_dataset_url(monkeypatch):
    parquet_calls = {}

    def _capture_parquet(**kwargs):
        parquet_calls.update(kwargs)

    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    monkeypatch.setattr("boxmot.data.mot17_parquet.setup_mot17_from_parquet", _capture_parquet)
    args = SimpleNamespace(data="mot17", source=None, split="test", split_explicit=True)

    apply_benchmark_config(args)

    assert parquet_calls["split"] == "test"
    assert parquet_calls["detector"] == "FRCNN"


def test_apply_benchmark_config_applies_ablation_component_overrides(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    monkeypatch.setattr("boxmot.data.mot17_parquet.setup_mot17_from_parquet", lambda **kwargs: None)
    args = SimpleNamespace(data="mot17", source=None, split="ablation", split_explicit=True)

    cfg = apply_benchmark_config(args)

    assert cfg["detector"]["id"] == "yolox_x_mot17_ablation"
    assert cfg["reid"]["id"] == "lmbn_n_duke"
    assert resolve_required_yolo_model(cfg) == Path("models/yolox_x_MOT17_ablation.pt")
    assert resolve_required_reid_model(cfg) == Path("models/lmbn_n_duke.pt")


def test_apply_benchmark_config_ignores_source_without_data(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(source="MOT17-ablation")
    assert apply_benchmark_config(args) is None


def test_find_dataset_cfg_for_nested_source_path():
    cfg = find_dataset_cfg_for_source("data/benchmarks/MMOT-OBB/train/data44-3/img1")

    assert cfg is not None
    assert cfg["id"] == "mmot"
    assert cfg["path"] == "data/benchmarks/MMOT-OBB"


def test_ensure_dataset_source_available_downloads_missing_dataset(monkeypatch):
    calls = {}
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: calls.update(kwargs))
    source = "data/benchmarks/MMOT-OBB/train/data44-3/img1"
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
    assert cfg["id"] == "mmot"
    assert args.source == "data/benchmarks/MMOT-OBB/train/data44-3/img1"
    assert args.dataset_id == "mmot"
    assert args.eval_box_type == "obb"
    assert calls == {
        "runs_url": "",
        "dataset_url": "https://github.com/mikel-brostrom/boxmot/releases/download/v16.0.11/MMOT-OBB.zip",
        "dataset_dest": Path("data/benchmarks/MMOT-OBB.zip"),
        "overwrite": False,
        "runs_check_path": None,
        "status_fn": None,
    }


def test_apply_benchmark_config_resolves_mmot_test_runs_url(monkeypatch):
    calls = {}

    def _capture_download(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(benchmark_config, "download_eval_data", _capture_download)
    args = SimpleNamespace(data="mmot", source=None, split="test", split_explicit=True)

    apply_benchmark_config(args)

    assert calls["runs_url"] == "hf://Lekim89/runs/runs/dets_n_embs/mmot/test"
    assert calls["runs_check_path"] == Path("runs/dets_n_embs/mmot/test")


def test_ensure_benchmark_detector_model_downloads_missing_weight(monkeypatch, tmp_path):
    cfg = load_benchmark_cfg("mmot")
    target = tmp_path / "yolo11l-3ch.pt"
    calls = {}

    monkeypatch.setattr(benchmark_config, "resolve_model_path", lambda *_args, **_kwargs: target)

    def fake_download_file(url, dest, overwrite=False, **_kwargs):
        calls.update({"url": url, "dest": dest, "overwrite": overwrite})
        return dest

    monkeypatch.setattr(
        benchmark_config,
        "download_file",
        fake_download_file,
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
    assert cfg["path"] == "data/benchmarks/SportsMOT"
    assert cfg["split"] == "val"
    assert cfg["train"] == "train"
    assert cfg["test"] == "test"
    assert cfg["detector_config"] == "yolox_x_sportsmot"
    assert cfg["reid_config"] == "lmbn_n_duke"
    assert cfg["storage"] == {
        "root": "data/benchmarks/SportsMOT",
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
    assert cfg["path"] == "data/benchmarks/SportsMOT"
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
    calls = {}
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: calls.update(kwargs))
    args = SimpleNamespace(data="sportsmot", source=None, split=None, split_explicit=False)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "sportsmot"
    assert args.benchmark_id == "sportsmot"
    assert args.dataset_id == "sportsmot"
    assert args.source == Path("data/benchmarks/SportsMOT/val")
    assert calls["runs_url"] == "hf://Lekim89/runs/runs/dets_n_embs/sportsmot/val"
    assert calls["runs_check_path"] == Path("runs/dets_n_embs/sportsmot/val")


def test_apply_benchmark_config_resolves_sportsmot_test_runs_url(monkeypatch):
    calls = {}

    def _capture_download(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(benchmark_config, "download_eval_data", _capture_download)
    args = SimpleNamespace(data="sportsmot", source=None, split="test", split_explicit=True)

    apply_benchmark_config(args)

    assert calls["runs_url"] == "hf://Lekim89/runs/runs/dets_n_embs/sportsmot/test"
    assert calls["runs_check_path"] == Path("runs/dets_n_embs/sportsmot/test")


def test_apply_benchmark_config_skips_dataset_download_when_split_is_populated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    split_dir = tmp_path / "data" / "benchmarks" / "SportsMOT" / "val"
    (split_dir / "SNMOT-001" / "img1").mkdir(parents=True)

    calls = {}

    def _capture_download(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(benchmark_config, "download_eval_data", _capture_download)
    args = SimpleNamespace(data="sportsmot", source=None, split="val", split_explicit=True)

    apply_benchmark_config(args)

    assert calls["dataset_url"] == ""
    assert calls["dataset_dest"] == Path("data/benchmarks/SportsMOT")
    assert args.source == Path("data/benchmarks/SportsMOT/val")


def test_find_dataset_cfg_for_sportsmot_source():
    cfg = find_dataset_cfg_for_source("data/benchmarks/SportsMOT/test/SNMOT-001/img1")
    assert cfg is not None
    assert cfg["id"] == "sportsmot"
    assert cfg["path"] == "data/benchmarks/SportsMOT"


@pytest.mark.skipif(
    not Path("data/benchmarks/MOT17/train").is_dir(),
    reason="MOT17 train data not available",
)
def test_mot17_ablation_split_resolves_to_ablation_dir(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(data="mot17", source=None, split="ablation", split_explicit=True)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "mot17"
    assert args.split == "ablation"
    assert args.source == Path("data/benchmarks/MOT17/ablation")
    # Verify the dir only contains FRCNN sequences
    seq_names = [p.name for p in args.source.iterdir() if p.is_dir()]
    assert all(name.endswith("-FRCNN") for name in seq_names)
    assert len(seq_names) == 7


@pytest.mark.skipif(
    not Path("data/benchmarks/MOT17/train").is_dir(),
    reason="MOT17 train data not available",
)
def test_mot17_ablation_split_respects_cli_detection_source(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(
        data="mot17", source=None, split="ablation", split_explicit=True,
        detection_source="public",
    )
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "mot17"
    assert args.split == "ablation"
    assert args.source == Path("data/benchmarks/MOT17/ablation")
    # CLI --detection-source takes precedence
    assert args.detection_source == "public"
