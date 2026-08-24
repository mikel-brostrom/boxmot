from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import boxmot.engine.workflows.benchmark as benchmark_config
from boxmot.data.benchmark import build_gt_class_remap, resolve_obb_eval_class_pairs
from boxmot.data.config import (
    load_artifact_config,
    load_dataset_config,
    resolve_artifact_config_path,
)
from boxmot.detectors.config import load_detector_config
from boxmot.engine.experiment import (
    ConfigurationError,
    resolve_experiment_config,
    write_experiment_snapshots,
)
from boxmot.engine.workflows.benchmark import (
    apply_evaluation_config,
    apply_reid_runtime_defaults,
    ensure_benchmark_detector_model,
    eval_init,
    find_dataset_cfg_for_source,
    get_benchmark_detector_url,
    load_dataset_cfg,
    load_detector_component_cfg,
    load_evaluation_config_from_args,
    load_experiment_cfg,
    load_reid_component_cfg,
    load_runtime_reid_component_cfg,
    resolve_dataset_cfg_path,
    resolve_experiment_cfg_path,
    resolve_required_reid_model,
    resolve_required_yolo_model,
    should_use_benchmark_detector,
    should_use_benchmark_reid,
)


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _apply_args(experiment: str, *, mode: str = "eval") -> SimpleNamespace:
    return SimpleNamespace(
        experiment=experiment,
        source=None,
        split="",
        split_explicit=False,
        workflow_mode=mode,
        detection_source=None,
    )


def test_registry_paths_are_separate_from_experiments():
    dataset_path = resolve_dataset_cfg_path("mot17")
    artifact_path = resolve_artifact_config_path("mot17")
    experiment_path = resolve_experiment_cfg_path("mot17-ablation-yolox-lmbn")

    assert dataset_path == Path("boxmot/configs/datasets/mot17.yaml").resolve()
    assert artifact_path == Path("boxmot/configs/artifacts/mot17.yaml").resolve()
    assert experiment_path == Path("boxmot/configs/experiments/mot17/ablation-yolox-lmbn.yaml").resolve()


def test_dataset_registry_contains_dataset_facts_and_download():
    raw = yaml.safe_load(Path("boxmot/configs/datasets/mot17.yaml").read_text())
    dataset = load_dataset_config("mot17")

    assert "detector" not in raw
    assert "reid" not in raw
    assert set(raw["resources"]) == {"dataset"}
    assert raw["classes"] == {
        "target": {"pedestrian": 1},
        "ignore": {
            "person_on_vehicle": 2,
            "static_person": 7,
            "distractor": 8,
            "reflection": 12,
        },
    }
    assert dataset["splits"]["test"]["has_ground_truth"] is False
    assert dataset["classes"]["pedestrian"] == {"id": 1, "evaluation": "target"}
    assert dataset["classes"]["reflection"] == {"id": 12, "evaluation": "ignore"}
    assert dataset["resources"]["dataset"] == raw["resources"]["dataset"]
    assert dataset["artifacts"]["precomputed"] == load_artifact_config("mot17")["artifacts"]["precomputed"]


def test_dataset_configs_only_embed_their_own_download_resources():
    for path in Path("boxmot/configs/datasets").glob("*.yaml"):
        resources = yaml.safe_load(path.read_text()).get("resources") or {}
        assert set(resources) <= {"dataset"}, path


def test_component_uris_are_owned_by_their_component_configs():
    dataset = yaml.safe_load(Path("boxmot/configs/datasets/mot17.yaml").read_text())
    detector = yaml.safe_load(Path("boxmot/configs/detectors/yolox-x-mot17.yaml").read_text())
    reid = yaml.safe_load(Path("boxmot/configs/reid/lmbn-n-duke.yaml").read_text())
    artifacts = load_artifact_config("mot17")["artifacts"]

    assert dataset["resources"]["dataset"]["uris"]["train"].startswith("hf://")
    assert detector["checkpoints"]["ablation"]["uri"].startswith("https://")
    assert reid["weights"]["uri"].startswith("https://")
    assert "dataset" not in artifacts


def test_detector_registry_uses_explicit_named_checkpoints():
    detector = load_detector_config("yolox-x-mot17")

    assert detector["id"] == "yolox-x-mot17"
    assert detector["image_size"] == [800, 1440]
    assert detector["confidence_threshold"] == 0.01
    assert detector["checkpoints"]["ablation"]["path"] == "models/yolox_x_MOT17_ablation.pt"
    assert detector["checkpoints"]["test"]["path"] == "models/yolox_x_MOT17_test.pt"


def test_model_experiment_resolves_semantic_class_map_to_numeric_bridge():
    cfg = resolve_experiment_config("mot17-ablation-yolox-lmbn", mode="evaluation")

    assert cfg["dataset"]["id"] == "mot17"
    assert cfg["dataset"]["split"] == "ablation"
    assert cfg["detections"] == {
        "source": "model",
        "model": {"ref": "yolox-x-mot17", "checkpoint": "ablation"},
    }
    assert cfg["evaluation"]["classes"] == [
        {
            "name": "pedestrian",
            "dataset_id": 1,
            "detector_name": "person",
            "detector_id": 0,
        }
    ]
    assert cfg["evaluation"]["ignore_dataset_ids"] == [2, 7, 8, 12]


def test_auto_class_map_resolves_identical_multiclass_names():
    cfg = resolve_experiment_config("mmot-obb-test-yolo11l-lmbn", mode="evaluation")

    assert [(item["name"], item["dataset_id"], item["detector_id"]) for item in cfg["evaluation"]["classes"]] == [
        ("car", 1, 0),
        ("bike", 2, 1),
        ("pedestrian", 3, 2),
        ("van", 4, 3),
        ("truck", 5, 4),
        ("bus", 6, 5),
        ("tricycle", 7, 6),
        ("awning-bike", 8, 7),
    ]


def test_public_detection_experiment_resolves_dataset_resource():
    cfg = resolve_experiment_config("mot17-ablation-frcnn-lmbn", mode="evaluation")

    assert cfg["detections"] == {"source": "public", "name": "frcnn"}
    assert cfg["detector"]["id"] == "mot17-public-frcnn"
    assert cfg["detector"]["model"] is None


def test_precomputed_experiment_validates_contents_and_producer():
    cfg = resolve_experiment_config("mot17-ablation-precomputed", mode="evaluation")

    assert cfg["detections"]["source"] == "precomputed"
    assert cfg["detections"]["contains"] == ["detections", "embeddings"]
    assert cfg["detector"]["checkpoint"] == "ablation"
    assert cfg["reid"]["id"] == "lmbn-n-duke"


def test_runtime_adapter_preserves_existing_evaluator_contract():
    cfg = load_experiment_cfg("mot17-ablation-yolox-lmbn")

    assert cfg["id"] == "mot17-ablation-yolox-lmbn"
    assert cfg["dataset_config"] == "mot17"
    assert cfg["path"] == "boxmot/datasets/mot/MOT17"
    assert cfg["split"] == "ablation"
    assert cfg["detector"]["id"] == "yolox-x-mot17"
    assert cfg["reid"]["id"] == "lmbn-n-duke"
    assert cfg["benchmark"]["class_bridge"] == cfg["evaluation"]["classes"]["bridge"]


def test_dataset_loader_has_no_model_bindings():
    cfg = load_dataset_cfg("sportsmot")

    assert cfg["id"] == "sportsmot"
    assert cfg["path"] == "boxmot/datasets/mot/SportsMOT"
    assert "detector_config" not in cfg
    assert "reid_config" not in cfg


def test_mmot_mini_owns_the_mini_archive_resource():
    archive = "https://github.com/mikel-brostrom/boxmot/releases/download/v16.0.11/MMOT-OBB.zip"

    assert load_dataset_cfg("mmot-mini")["download"]["dataset"] == archive
    assert load_dataset_cfg("mmot")["download"]["dataset"] == ""


def test_component_loaders_read_standalone_registries():
    detector = load_detector_component_cfg("yolox-x-sportsmot")
    reid = load_reid_component_cfg("lmbn-n-duke")

    assert detector["model"] == "models/yolox_x_sportsmot.pt"
    assert detector["imgsz"] == [800, 1440]
    assert reid["model"] == "models/lmbn_n_duke.pt"
    assert reid["half"] is True
    assert reid["device"] == ""
    assert reid["imgsz"] == [384, 128]


def test_detector_with_multiple_checkpoints_requires_explicit_checkpoint():
    with pytest.raises(ConfigurationError, match="multiple checkpoints"):
        load_detector_component_cfg("yolox-x-mot17")

    cfg = load_detector_component_cfg("yolox-x-mot17/ablation")
    assert cfg["checkpoint"] == "ablation"


def test_runtime_reid_component_matches_model_filename():
    cfg = load_runtime_reid_component_cfg("models/lmbn_n_duke.pt")

    assert cfg["id"] == "lmbn-n-duke"
    assert cfg["precision"] == "fp16"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda cfg: cfg["dataset"].update(split="missing"), "has no split"),
        (lambda cfg: cfg["detections"]["model"].update(checkpoint="missing"), "has no checkpoint"),
        (lambda cfg: cfg["evaluation"].update(class_map={"missing": "person"}), "does not exist"),
        (lambda cfg: cfg["evaluation"].update(class_map={"pedestrian": "missing"}), "does not exist"),
    ],
)
def test_experiment_reference_validation_fails_early(tmp_path, mutator, message):
    source = yaml.safe_load(Path("boxmot/configs/experiments/mot17/ablation-yolox-lmbn.yaml").read_text())
    mutator(source)
    path = _write_yaml(tmp_path / "invalid.yaml", source)

    with pytest.raises(ConfigurationError, match=message):
        resolve_experiment_config(path, mode="evaluation")


def test_evaluation_rejects_split_without_ground_truth():
    with pytest.raises(ConfigurationError, match="has no ground truth"):
        resolve_experiment_config("mot17-test-yolox-lmbn", mode="evaluation")

    cfg = resolve_experiment_config("mot17-test-yolox-lmbn", mode="inference")
    assert cfg["dataset"]["has_ground_truth"] is False


def test_public_source_must_exist_for_dataset(tmp_path):
    source = yaml.safe_load(Path("boxmot/configs/experiments/mot17/ablation-frcnn-lmbn.yaml").read_text())
    source["detections"]["name"] = "missing"
    path = _write_yaml(tmp_path / "invalid-public.yaml", source)

    with pytest.raises(ConfigurationError, match="not available"):
        resolve_experiment_config(path)


def test_precomputed_artifact_requires_detections_and_embeddings(tmp_path):
    dataset = yaml.safe_load(Path("boxmot/configs/datasets/mot17.yaml").read_text())
    dataset["id"] = "invalid-artifact-dataset"
    dataset_dir = tmp_path / "configs" / "datasets"
    artifact_dir = tmp_path / "configs" / "artifacts"
    dataset_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    dataset_path = _write_yaml(dataset_dir / "invalid-artifact-dataset.yaml", dataset)
    artifact = yaml.safe_load(Path("boxmot/configs/artifacts/mot17.yaml").read_text())
    artifact["id"] = "invalid-artifact-dataset"
    artifact["artifacts"]["precomputed"]["ablation"]["contains"] = ["detections"]
    _write_yaml(artifact_dir / "invalid-artifact-dataset.yaml", artifact)
    experiment = yaml.safe_load(Path("boxmot/configs/experiments/mot17/ablation-precomputed.yaml").read_text())
    experiment["dataset"]["ref"] = str(dataset_path)
    experiment_path = _write_yaml(tmp_path / "experiment.yaml", experiment)

    with pytest.raises(ConfigurationError, match="missing required content: embeddings"):
        resolve_experiment_config(experiment_path)


@pytest.mark.parametrize(
    ("inference", "message"),
    [
        ({"image_size": [640], "confidence_threshold": 0.1}, "exactly two"),
        ({"image_size": [640, 640], "confidence_threshold": 1.1}, "within"),
    ],
)
def test_detector_inference_validation(tmp_path, inference, message):
    detector = yaml.safe_load(Path("boxmot/configs/detectors/yolox-x-sportsmot.yaml").read_text())
    detector["inference"] = inference
    path = _write_yaml(tmp_path / "detector.yaml", detector)

    with pytest.raises(ConfigurationError, match=message):
        load_detector_config(path)


def test_detector_and_dataset_box_types_must_match(tmp_path):
    source = yaml.safe_load(Path("boxmot/configs/experiments/mot17/ablation-yolox-lmbn.yaml").read_text())
    source["detections"]["model"]["ref"] = "yolo11l-mmot-obb"
    source["detections"]["model"]["checkpoint"] = "default"
    path = _write_yaml(tmp_path / "mismatch.yaml", source)

    with pytest.raises(ConfigurationError, match="uses obb boxes"):
        resolve_experiment_config(path)


def test_class_bridge_remains_usable_by_evaluator():
    cfg = load_experiment_cfg("mot20-ablation-yolox-lmbn")

    remap, class_ids, class_names = build_gt_class_remap(
        cfg["benchmark"],
        cfg["detector"],
        benchmark_name="mot20",
        model_stem="yolox_x_MOT20_ablation",
    )

    assert remap == {1: 1}
    assert class_ids == [1]
    assert class_names == ["person"]


def test_obb_class_pairs_use_resolved_detector_ids():
    cfg = load_experiment_cfg("mmot-obb-test-yolo11l-lmbn")
    args = SimpleNamespace(
        classes=None,
        remapped_class_ids=None,
        remapped_class_names=None,
        translated_benchmark_class_names=None,
    )

    assert resolve_obb_eval_class_pairs(args, cfg["benchmark"]) == [
        ("car", 0),
        ("bike", 1),
        ("pedestrian", 2),
        ("van", 3),
        ("truck", 4),
        ("bus", 5),
        ("tricycle", 6),
        ("awning-bike", 7),
    ]


def test_apply_experiment_sets_runtime_source_and_model_defaults(monkeypatch):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    monkeypatch.setattr("boxmot.data.mot17_parquet.setup_mot17_from_parquet", lambda **kwargs: None)
    args = _apply_args("mot17-ablation-yolox-lmbn")

    cfg = apply_evaluation_config(args)

    assert cfg["id"] == "mot17-ablation-yolox-lmbn"
    assert args.experiment_id == "mot17-ablation-yolox-lmbn"
    assert args.dataset_id == "mot17"
    assert args.benchmark == "mot17"
    assert args.source == Path("boxmot/datasets/mot/MOT17/ablation")
    assert args.detection_source == "private"
    assert args.experiment_source_path == resolve_experiment_cfg_path("mot17-ablation-yolox-lmbn")


def test_eval_init_resolves_model_free_dataset_split(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    monkeypatch.setattr("boxmot.data.mot17_parquet.setup_mot17_from_parquet", lambda **kwargs: None)
    args = SimpleNamespace(
        experiment=None,
        dataset="mot17",
        source=None,
        split="ablation",
        split_explicit=True,
        workflow_mode="eval",
        detection_source=None,
        project=tmp_path / "runs",
    )

    eval_init(args)

    assert args.source == Path("boxmot/datasets/mot/MOT17/ablation").resolve()
    assert args.project == (tmp_path / "runs").resolve()
    assert args.dataset_id == "mot17"
    assert args.experiment_id is None
    assert args.benchmark == "mot17"
    assert args.split == "ablation"

    runtime_cfg = load_evaluation_config_from_args(args)
    assert "detector" not in runtime_cfg
    assert "reid" not in runtime_cfg
    assert runtime_cfg["benchmark"]["eval_classes"] == {1: "pedestrian"}
    assert runtime_cfg["benchmark"]["ignore_dataset_ids"] == [2, 7, 8, 12]


def test_dataset_reference_passed_as_experiment_fails_before_path_normalization(tmp_path):
    args = SimpleNamespace(
        experiment="mot17",
        dataset=None,
        source=None,
        split="ablation",
        split_explicit=True,
        workflow_mode="eval",
        detection_source=None,
        project=tmp_path / "runs",
    )

    with pytest.raises(ConfigurationError, match=r"Use --dataset mot17 instead of --experiment mot17"):
        eval_init(args)


def test_apply_public_experiment_sets_named_detection_source(monkeypatch):
    parquet_calls = {}
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: None)
    monkeypatch.setattr(
        "boxmot.data.mot17_parquet.setup_mot17_from_parquet",
        lambda **kwargs: parquet_calls.update(kwargs),
    )
    args = _apply_args("mot17-ablation-frcnn-lmbn")

    apply_evaluation_config(args)

    assert args.detection_source == "frcnn"
    assert parquet_calls["detector"] == "FRCNN"


def test_apply_precomputed_experiment_downloads_only_selected_artifact(monkeypatch):
    calls = {}
    monkeypatch.setattr(benchmark_config, "download_eval_data", lambda **kwargs: calls.update(kwargs))
    monkeypatch.setattr("boxmot.data.mot17_parquet.setup_mot17_from_parquet", lambda **kwargs: None)
    args = _apply_args("mot17-ablation-precomputed")

    apply_evaluation_config(args)

    assert calls["runs_url"] == "hf://Lekim89/runs/runs/dets_n_embs/mot17/ablation"
    assert calls["dataset_url"] == ""


def test_find_dataset_config_uses_registry_roots():
    cfg = find_dataset_cfg_for_source("boxmot/datasets/mot/MMOT-OBB/train/data44-3/img1")

    assert cfg is not None
    assert cfg["id"] == "mmot"
    assert cfg["path"] == "boxmot/datasets/mot/MMOT-OBB"


def test_model_and_reid_defaults_are_resolved_from_experiment():
    cfg = load_experiment_cfg("sportsmot-val-yolox-lmbn")

    assert resolve_required_yolo_model(cfg) == Path("models/yolox_x_sportsmot.pt")
    assert resolve_required_reid_model(cfg) == Path("models/lmbn_n_duke.pt")
    assert should_use_benchmark_detector(
        SimpleNamespace(detector=[Path("models/yolov8n.pt")], detector_explicit=False), cfg
    )
    assert should_use_benchmark_reid(
        SimpleNamespace(reid=[Path("models/osnet_x0_25_msmt17.pt")], reid_explicit=False), cfg
    )


def test_reid_runtime_defaults_alias_auto_and_fp16():
    cfg = load_experiment_cfg("sportsmot-val-yolox-lmbn")
    args = SimpleNamespace(device="cpu", half=False, device_explicit=False, half_explicit=False)

    apply_reid_runtime_defaults(args, cfg)

    assert args.reid_device == "cpu"
    assert args.reid_half is True
    assert args.reid_preprocess == "resize"


def test_gdrive_uri_is_normalized_for_download():
    cfg = load_experiment_cfg("mmot-obb-test-yolo11l-lmbn")

    assert get_benchmark_detector_url(cfg) == ("https://drive.google.com/uc?id=15gmA4-Yclvh5EZvTJYhcyV1CVdNRGIkR")


def test_ensure_detector_model_uses_registry_uri(monkeypatch, tmp_path):
    cfg = load_experiment_cfg("mmot-obb-test-yolo11l-lmbn")
    target = tmp_path / "yolo11l_3ch.pt"
    calls = {}
    monkeypatch.setattr(benchmark_config, "resolve_model_path", lambda *_args, **_kwargs: target)
    monkeypatch.setattr(
        benchmark_config,
        "download_file",
        lambda url, dest, overwrite=False, **kwargs: calls.update({"url": url, "dest": dest, "overwrite": overwrite})
        or dest,
    )

    assert ensure_benchmark_detector_model(cfg) == target
    assert calls["url"].startswith("https://drive.google.com/uc?id=")


def test_run_snapshots_include_source_and_resolved_runtime(tmp_path):
    resolved = resolve_experiment_config("mot17-ablation-yolox-lmbn")
    args = SimpleNamespace(
        experiment_source_path=resolved["source_path"],
        resolved_experiment_config=resolved,
        tracker="bytetrack",
        tracker_backend="python",
        detection_source="private",
        detector=[Path("models/yolox_x_MOT17_ablation.pt")],
        reid=[Path("models/lmbn_n_duke.pt")],
        device="cpu",
        imgsz=[800, 1440],
        conf=0.01,
    )

    source_path, resolved_path = write_experiment_snapshots(
        args,
        tmp_path,
        tracker_config={"track_thresh": 0.55},
    )
    source = yaml.safe_load(source_path.read_text())
    recorded = yaml.safe_load(resolved_path.read_text())

    assert source["dataset"] == {"ref": "mot17", "split": "ablation"}
    assert recorded["dataset"]["has_ground_truth"] is True
    assert recorded["detector"]["model"] == "models/yolox_x_MOT17_ablation.pt"
    assert recorded["runtime"]["tracker"] == "bytetrack"
    assert recorded["runtime"]["tracker_config"]["track_thresh"] == 0.55
    assert recorded["runtime"]["tracker_config"]["frame_rate"] == 30
