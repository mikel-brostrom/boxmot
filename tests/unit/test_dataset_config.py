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
from boxmot.data.benchmark import build_gt_class_remap, load_benchmark_cfg_from_args, resolve_obb_eval_class_pairs
from boxmot.engine.eval.motmetrics import build_dataset_eval_settings


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
        "metric_eval": "mot_challenge",
        "classes": {
            "eval": {1: "pedestrian"},
            "distractor": {
                2: "person_on_vehicle",
                7: "static_person",
                8: "distractor",
                12: "reflection",
            },
            "mapping": {"pedestrian": "person"},
            "bridge": [
                {
                    "name": "pedestrian",
                    "dataset_id": 1,
                    "detector_id": 0,
                    "detector_name": "person",
                }
            ],
            "ignore_dataset_ids": [2, 7, 8, 12],
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


def test_load_benchmark_cfg_from_args_falls_back_to_data_path(tmp_path):
    cfg_path = tmp_path / "local-mmot.yaml"
    cfg_path.write_text(
        """
id: local-mmot
dataset:
  id: local-mmot
  root: /tmp/local-mmot
  layout: mot
  box_type: obb
  splits:
    test: test/npy
  default_split: test
  classes:
    1: car
evaluation:
  classes:
    - {name: car, dataset_id: 1, detector_id: 0, detector_name: car}
download:
  runs: ""
detector:
  id: yolo11l_3ch
  model: yolo11l_3ch.pt
  box_type: obb
  imgsz: [1280, 1280]
  conf: 0.2
reid:
  id: lmbn_n_duke
  model: models/lmbn_n_duke.pt
  half: true
  preprocess: resize
""",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        benchmark_id=None,
        dataset_id=None,
        benchmark="local-mmot",
        data=str(cfg_path),
    )

    cfg = load_benchmark_cfg_from_args(args)

    assert cfg["id"] == "local-mmot"
    assert cfg["detector"]["id"] == "yolo11l_3ch"
    assert cfg["detector"]["default_model"] == "yolo11l_3ch.pt"
    assert cfg["reid"]["id"] == "lmbn_n_duke"
    assert cfg["benchmark"]["box_type"] == "obb"


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


def test_obb_dataset_derives_metric_backend_from_box_type():
    cfg = load_dataset_cfg("mmot")
    assert cfg["layout"] == "mot"
    assert cfg["box_type"] == "obb"
    assert cfg["metric_backend"] == "mot_challenge_obb"
    assert cfg["evaluation"]["metric_eval"] == "mot_challenge_obb"


def test_all_benchmarks_define_explicit_class_bridge():
    for cfg_path in sorted(Path("boxmot/configs/benchmarks").glob("*.yaml")):
        cfg = load_benchmark_cfg(cfg_path)
        bridge = cfg["benchmark"]["class_bridge"]

        assert bridge, f"{cfg_path.name} must define evaluation.classes"
        assert cfg["evaluation"]["classes"]["bridge"] == bridge
        assert cfg["benchmark"]["eval_classes"] == {
            entry["dataset_id"]: entry["name"] for entry in bridge
        }
        assert cfg["benchmark"]["class_mapping"] == {
            entry["name"]: entry.get("detector_name", entry["name"]) for entry in bridge
        }


def test_mot20_class_bridge_remaps_gt_to_detector_person():
    cfg = load_benchmark_cfg("mot20")

    remap, class_ids, class_names = build_gt_class_remap(
        cfg["benchmark"],
        cfg["detector"],
        benchmark_name="mot20",
        model_stem="yolox_x_MOT20_ablation",
    )

    assert remap == {1: 1}
    assert class_ids == [1]
    assert class_names == ["person"]
    assert cfg["benchmark"]["ignore_dataset_ids"] == [2, 6, 7, 8, 12]


def test_mmot_obb_class_bridge_uses_detector_ids():
    cfg = load_benchmark_cfg("mmot")
    args = SimpleNamespace(
        classes=None,
        remapped_class_ids=None,
        remapped_class_names=None,
        translated_benchmark_class_names=None,
    )

    pairs = resolve_obb_eval_class_pairs(args, cfg["benchmark"])

    assert pairs == [
        ("car", 0),
        ("bike", 1),
        ("pedestrian", 2),
        ("van", 3),
        ("truck", 4),
        ("bus", 5),
        ("tricycle", 6),
        ("awning-bike", 7),
    ]


def test_visdrone_metric_backend_uses_explicit_ignore_dataset_ids():
    args = SimpleNamespace(
        benchmark="visdrone",
        benchmark_id=None,
        dataset_id=None,
        remapped_class_ids=[1],
        remapped_class_names=["pedestrian"],
        classes=None,
    )

    settings = build_dataset_eval_settings(args, Path("gt"), {"uav0000013_00000_v": 100})

    assert settings["classes_to_eval"] == ["pedestrian"]
    assert settings["class_ids"] == [1]
    assert settings["distractor_ids"] == [0, 11]
    assert settings["gt_loc_format"] == "{gt_folder}/{seq}.txt"


def test_mmot_mini_uses_mmot_mini_root():
    cfg = load_benchmark_only_cfg("mmot-mini")
    assert cfg["id"] == "mmot-mini"
    assert cfg["dataset_config"] == "mmot-mini"
    assert cfg["path"] == "assets/mmot-mini"
    assert cfg["split"] == "train"
    assert cfg["train"] == "train/npy"
    assert cfg["metric_backend"] == "mot_challenge_obb"


def test_detector_and_reid_component_configs_load_separately():
    detector_cfg = load_detector_component_cfg("yolox_x_mot17_ablation")
    reid_cfg = load_reid_component_cfg("lmbn_n_duke")

    assert detector_cfg["id"] == "yolox_x_mot17_ablation"
    assert detector_cfg["model"] == "models/yolox_x_MOT17_ablation.pt"
    assert reid_cfg["id"] == "lmbn_n_duke"
    assert reid_cfg["model"] == "models/lmbn_n_duke.pt"
    assert reid_cfg["url"] == "https://github.com/mikel-brostrom/boxmot/releases/download/v21.0.0/lmbn_n_duke.pt"
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
        "model_url": "https://github.com/mikel-brostrom/boxmot/releases/download/v21.0.0/lmbn_n_duke.pt",
        "url": "https://github.com/mikel-brostrom/boxmot/releases/download/v21.0.0/lmbn_n_duke.pt",
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
    assert args.source == Path("boxmot/datasets/mot/test1/val")


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
    cfg = find_dataset_cfg_for_source("boxmot/datasets/mot/MMOT-OBB/train/data44-3/img1")

    assert cfg is not None
    assert cfg["id"] == "mmot"
    assert cfg["path"] == "boxmot/datasets/mot/MMOT-OBB"


def test_ensure_dataset_source_available_downloads_missing_dataset(monkeypatch):
    calls = {}
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: calls.update(kwargs))
    source = "boxmot/datasets/mot/MMOT-OBB/train/data44-3/img1"
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
    assert args.source == "boxmot/datasets/mot/MMOT-OBB/train/data44-3/img1"
    assert args.dataset_id == "mmot"
    assert args.eval_box_type == "obb"
    assert calls == {
        "runs_url": "",
        "dataset_url": "https://github.com/mikel-brostrom/boxmot/releases/download/v16.0.11/MMOT-OBB.zip",
        "dataset_dest": Path("boxmot/datasets/mot/MMOT-OBB.zip"),
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
    assert cfg["path"] == "boxmot/datasets/mot/SportsMOT"
    assert cfg["split"] == "val"
    assert cfg["train"] == "train"
    assert cfg["test"] == "test"
    assert cfg["detector_config"] == "yolox_x_sportsmot"
    assert cfg["reid_config"] == "lmbn_n_duke"
    assert cfg["storage"] == {
        "root": "boxmot/datasets/mot/SportsMOT",
        "split": "val",
    }
    assert cfg["evaluation"] == {
        "box_type": "aabb",
        "layout": "mot",
        "metric_eval": "mot_challenge",
        "classes": {
            "eval": {1: "player"},
            "distractor": {},
            "mapping": {"player": "person"},
            "bridge": [
                {
                    "name": "player",
                    "dataset_id": 1,
                    "detector_id": 0,
                    "detector_name": "person",
                }
            ],
            "ignore_dataset_ids": [],
        },
    }


def test_sportsmot_dataset_loads_with_model_bindings():
    cfg = load_dataset_cfg("sportsmot")
    assert cfg["id"] == "sportsmot"
    assert cfg["path"] == "boxmot/datasets/mot/SportsMOT"
    assert cfg["box_type"] == "aabb"
    assert cfg["layout"] == "mot"
    assert cfg["metric_backend"] == "mot_challenge"
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
    assert args.source == Path("boxmot/datasets/mot/SportsMOT/val")
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
    split_dir = tmp_path / "boxmot" / "datasets" / "mot" / "SportsMOT" / "val"
    (split_dir / "SNMOT-001" / "img1").mkdir(parents=True)

    calls = {}

    def _capture_download(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(benchmark_config, "download_eval_data", _capture_download)
    args = SimpleNamespace(data="sportsmot", source=None, split="val", split_explicit=True)

    apply_benchmark_config(args)

    assert calls["dataset_url"] == ""
    assert calls["dataset_dest"] == Path("boxmot/datasets/mot/SportsMOT")
    assert args.source == Path("boxmot/datasets/mot/SportsMOT/val")


def test_find_dataset_cfg_for_sportsmot_source():
    cfg = find_dataset_cfg_for_source("boxmot/datasets/mot/SportsMOT/test/SNMOT-001/img1")
    assert cfg is not None
    assert cfg["id"] == "sportsmot"
    assert cfg["path"] == "boxmot/datasets/mot/SportsMOT"


@pytest.mark.skipif(
    not Path("boxmot/datasets/mot/MOT17/train").is_dir(),
    reason="MOT17 train data not available",
)
def test_mot17_ablation_split_resolves_to_ablation_dir(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    args = SimpleNamespace(data="mot17", source=None, split="ablation", split_explicit=True)
    cfg = apply_benchmark_config(args)
    assert cfg["id"] == "mot17"
    assert args.split == "ablation"
    assert args.source == Path("boxmot/datasets/mot/MOT17/ablation")
    # Verify the dir only contains FRCNN sequences
    seq_names = [p.name for p in args.source.iterdir() if p.is_dir()]
    assert all(name.endswith("-FRCNN") for name in seq_names)
    assert len(seq_names) == 7


@pytest.mark.skipif(
    not Path("boxmot/datasets/mot/MOT17/train").is_dir(),
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
    assert args.source == Path("boxmot/datasets/mot/MOT17/ablation")
    # CLI --detection-source takes precedence
    assert args.detection_source == "public"
