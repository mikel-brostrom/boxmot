from pathlib import Path

import pytest

from boxmot.datasets.config import ConfigurationError, load_dataset_config
from boxmot.engine.config import (
    BOXMOT_DEFAULTS,
    DEFAULT_DETECTOR,
    DEFAULT_REID,
    build_mode_namespace,
    get_mode_default,
    get_mode_defaults,
)
from boxmot.reid.exporters.config import (
    build_export_namespace,
    load_export_defaults,
    resolve_export_weights,
)
from boxmot.reid.training.presets import build_training_namespace, load_training_recipe
from boxmot.utils import WEIGHTS


def _write_dataset_config(
    path: Path,
    *,
    split_path: str,
    annotations: str | None = None,
    storage_root: str = "data",
) -> Path:
    annotation_line = "" if annotations is None else f"    annotations: {annotations}\n"
    path.write_text(
        """id: fixture
format:
  layout: mot
  box_type: obb
storage:
  root: """
        + storage_root
        + """
default_split: test
splits:
  test:
    path: """
        + split_path
        + "\n"
        + annotation_line
        + """    has_ground_truth: true
classes:
  target:
    car: 1
""",
        encoding="utf-8",
    )
    return path


def test_dataset_config_preserves_valid_split_annotation_path(tmp_path):
    config = load_dataset_config(
        _write_dataset_config(
            tmp_path / "fixture.yaml",
            split_path="test/npy",
            annotations="test/mot",
        )
    )

    assert config["splits"]["test"]["annotations"] == "test/mot"


@pytest.mark.parametrize("dataset_id", ("mmot", "mmot-mini"))
def test_mmot_dataset_configs_use_native_zero_based_class_ids(dataset_id: str) -> None:
    config = load_dataset_config(dataset_id)

    assert {name: metadata["id"] for name, metadata in config["classes"].items()} == {
        "car": 0,
        "bike": 1,
        "pedestrian": 2,
        "van": 3,
        "truck": 4,
        "bus": 5,
        "tricycle": 6,
        "awning-bike": 7,
    }


def test_mmot_dataset_config_matches_standard_directory_layout():
    config = load_dataset_config("mmot")

    assert config["root"] == "data"
    assert config["splits"]["train"]["path"] == "train/npy"
    assert config["splits"]["train"]["annotations"] == "train/mot"
    assert config["splits"]["test"]["path"] == "test/npy"
    assert config["splits"]["test"]["annotations"] == "test/mot"


def test_mot17_dataset_config_uses_canonical_hugging_face_splits() -> None:
    config = load_dataset_config("mot17")

    assert config["root"] == "MOT17"
    assert {name: split["path"] for name, split in config["splits"].items()} == {
        "train": "train",
        "val": "val",
        "test": "test",
        "ablation": "ablation",
    }
    assert config["resources"]["dataset"] == {
        "type": "per_split",
        "uris": {
            "train": "hf://Lekim89/MOT17/train",
            "val": "hf://Lekim89/MOT17/val",
            "test": "hf://Lekim89/MOT17/test",
            "ablation": "hf://Lekim89/MOT17/ablation",
        },
    }


def test_kitti_mots_dataset_config_uses_native_classes_paths_and_official_splits() -> None:
    """Keep the packaged profile aligned with the original KITTI MOTS downloads."""

    config = load_dataset_config("kitti-mots")

    assert config["layout"] == "kitti-mots"
    assert config["root"] == "KITTI-MOTS"
    assert config["classes"]["car"] == {"id": 1, "evaluation": "target"}
    assert config["classes"]["pedestrian"] == {"id": 2, "evaluation": "target"}
    for name in ("train", "val", "fulltrain"):
        assert config["splits"][name]["path"] == "data_tracking_image_2/training/image_02"
        assert config["splits"][name]["annotations"] == "instances"
        assert config["splits"][name]["has_ground_truth"] is True
    assert config["splits"]["test"]["path"] == "data_tracking_image_2/testing/image_02"
    assert config["splits"]["test"]["has_ground_truth"] is False
    assert "annotations" not in config["splits"]["test"]
    assert "sequences" not in config["splits"]["fulltrain"]
    assert config["splits"]["train"]["sequences"] == [
        "0000",
        "0001",
        "0003",
        "0004",
        "0005",
        "0009",
        "0011",
        "0012",
        "0015",
        "0017",
        "0019",
        "0020",
    ]
    assert config["splits"]["val"]["sequences"] == [
        "0002",
        "0006",
        "0007",
        "0008",
        "0010",
        "0013",
        "0014",
        "0016",
        "0018",
    ]


@pytest.mark.parametrize(("split_path", "annotations"), (("../frames", None), ("test/npy", "../mot")))
def test_dataset_config_rejects_split_paths_outside_storage_root(tmp_path, split_path, annotations):
    path = _write_dataset_config(
        tmp_path / "fixture.yaml",
        split_path=split_path,
        annotations=annotations,
    )

    with pytest.raises(ConfigurationError, match="must remain beneath storage.root"):
        load_dataset_config(path)


@pytest.mark.parametrize("storage_root", ("../outside", "/outside", r"C:\outside", r"nested\outside"))
def test_dataset_config_rejects_unsafe_storage_root(tmp_path, storage_root):
    path = _write_dataset_config(
        tmp_path / "fixture.yaml",
        storage_root=storage_root,
        split_path="test/npy",
    )

    with pytest.raises(ConfigurationError, match="must remain beneath storage.root"):
        load_dataset_config(path)


def test_resolve_export_weights_preserves_explicit_export_paths():
    model_path = "models/osnet_x0_25_msmt17_saved_model/osnet_x0_25_msmt17_float32.tflite"

    resolved = resolve_export_weights(model_path)

    assert resolved == Path(model_path)


def test_resolve_export_weights_keeps_bare_reid_names_in_weights_dir():
    resolved = resolve_export_weights("osnet_x0_25_msmt17")

    assert resolved == WEIGHTS / "osnet_x0_25_msmt17.pt"


def test_build_mode_namespace_uses_shared_runtime_defaults():
    args = build_mode_namespace("eval", {"experiment": "mot17-mini"}, explicit_keys=set())

    assert not hasattr(args, "detector")
    assert not hasattr(args, "reid")
    assert args.tracker == get_mode_default("eval", "tracker")
    assert args.tracker_backend == "python"
    assert not hasattr(args, "detector_explicit")
    assert not hasattr(args, "reid_explicit")
    assert args.project == Path(get_mode_default("eval", "project"))
    assert args.show_timing is False


def test_materialize_namespace_allows_only_canonical_inputs() -> None:
    canonical = {
        "build_root": "builds",
        "data_root": "datasets",
        "device": "mps",
        "experiment": "experiment-one",
        "fps": 2.5,
        "plan_overrides": ("detect.batch_size=8",),
        "plan_path": "executor.yaml",
        "publish_embeddings": True,
        "publish_image_refs": True,
        "publish_masks": False,
        "resume": True,
    }
    rejected = {
        "conf": 0.1,
        "dataset": "other-dataset",
        "detector": "other-detector",
        "geometry": "obb",
        "reid": "other-reid",
        "segmentor": "other-segmentor",
        "source": "other-source",
        "split": "other-split",
        "tracker": "botsort",
    }

    args = build_mode_namespace(
        "materialize",
        {**canonical, **rejected, "materialize_explicit_keys": ("source",)},
        explicit_keys={*canonical, *rejected},
    )

    assert set(vars(args)) == {*canonical, "materialize_explicit_keys"}
    assert args.data_root == Path("datasets")
    assert args.build_root == Path("builds")
    assert args.plan_path == Path("executor.yaml")
    assert args.fps == canonical["fps"]
    assert args.materialize_explicit_keys == tuple(sorted(canonical))


@pytest.mark.parametrize("mode", ("train", "export", "eval-reid"))
def test_engine_config_rejects_domain_owned_modes(mode: str):
    with pytest.raises(ValueError, match="Unknown runtime mode"):
        build_mode_namespace(mode, {})


@pytest.mark.parametrize("cpu_count, expected_workers", [(None, 1), (1, 1), (4, 4), (32, 8)])
def test_get_mode_defaults_returns_normalized_merged_defaults(monkeypatch, cpu_count, expected_workers):
    monkeypatch.setattr("boxmot.engine.config.os.cpu_count", lambda: cpu_count)
    defaults = get_mode_defaults("eval")

    assert defaults["detector"] == DEFAULT_DETECTOR
    assert defaults["reid"] == DEFAULT_REID
    assert defaults["tracker"] == get_mode_default("eval", "tracker")
    assert defaults["project"] == Path(get_mode_default("eval", "project"))
    assert defaults["show_timing"] is False
    assert isinstance(defaults["sequence_workers"], int)
    assert defaults["sequence_workers"] == expected_workers
    assert "n_threads" not in defaults


def test_boxmot_defaults_bundle_exposes_typed_mode_defaults():
    assert BOXMOT_DEFAULTS.shared.detector == DEFAULT_DETECTOR
    assert BOXMOT_DEFAULTS.shared.reid == DEFAULT_REID
    assert BOXMOT_DEFAULTS.track.tracker == get_mode_default("track", "tracker")
    assert BOXMOT_DEFAULTS.track.tracker_backend == "python"
    assert BOXMOT_DEFAULTS.eval.project == Path(get_mode_default("eval", "project"))
    assert load_export_defaults().include == ("onnx",)


def test_default_11m_training_uses_promoted_speed_recipe():
    args = build_training_namespace({"data_dir": "."}, explicit_keys={"data_dir"})

    assert args.model == "csl_tinyvit_11m_v20"
    assert args.num_workers == 4
    assert args.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert args.spatial_conv_mode == "depthwise_separable"
    assert args.head_parts == (1, 2, 4)
    assert args.scale_balanced_branches is True
    assert args.p_ids == 12
    assert args.k_instances == 8
    assert args.pk_steps_per_epoch == 0
    assert args.camera_aware_sampler is False
    assert args.attention_window_layout == "rect"
    assert args.interpolate_pretrained_attention_bias is True
    assert args.attention_mask is True
    assert args.flip_tta is False
    assert args.background_mosaic is False
    assert args.background_mosaic_probability == 0.3
    assert args.background_mosaic_start_epoch == 10
    assert args.background_mosaic_ramp_end_epoch == 30
    assert args.same_id_part_mosaic is False
    assert args.same_id_part_mosaic_probability == 0.35
    assert args.same_id_part_mosaic_min_unaltered == 0.5
    assert args.pav_mosaic is False
    assert args.pav_mosaic_probability == 0.25
    assert args.pav_consistency_weight == 0.0


def test_11m_training_recipe_omits_experimental_mosaic_policy():
    recipe = load_training_recipe("csl_tinyvit_11m")

    experimental_keys = {
        "background_mosaic",
        "background_mosaic_mask_dir",
        "background_mosaic_probability",
        "same_id_part_mosaic",
        "same_id_part_mosaic_probability",
        "pav_mosaic",
        "pav_metadata_dir",
        "pav_mosaic_probability",
        "pav_consistency_weight",
    }
    assert recipe.keys().isdisjoint(experimental_keys)


def test_11m_training_recipe_exposes_fixed_camera_aware_sampler_controls():
    recipe = load_training_recipe("csl_tinyvit_11m")

    assert recipe["model"] == "csl_tinyvit_11m_v20"
    assert recipe["num_workers"] == 4
    assert recipe["pk_steps_per_epoch"] == 0
    assert recipe["camera_aware_sampler"] is False


def test_explicit_11m_v20_model_uses_canonical_training_recipe():
    args = build_training_namespace(
        {"data_dir": ".", "model": "csl_tinyvit_11m_v20"},
        explicit_keys={"data_dir", "model"},
    )

    assert args.model == "csl_tinyvit_11m_v20"
    assert args.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert args.spatial_conv_mode == "depthwise_separable"
    assert args.head_parts == (1, 2, 4)
    assert args.num_workers == 4


def test_explicit_non_11m_training_model_keeps_generic_defaults():
    args = build_training_namespace(
        {"data_dir": ".", "model": "csl_tinyvit_7m"},
        explicit_keys={"data_dir", "model"},
    )

    assert args.model == "csl_tinyvit_7m"
    assert args.feature_fusion == "last2"
    assert args.head_parts == (1, 2)
    assert args.scale_balanced_branches is False
    assert args.attention_window_layout == "legacy"


def test_runtime_and_export_namespaces_normalize_models():
    track_args = build_mode_namespace("track", {"source": "0"}, explicit_keys=set())
    export_args = build_export_namespace(
        {"weights": "osnet_x0_25_msmt17", "include": ["onnx"]},
    )

    assert track_args.detector == DEFAULT_DETECTOR
    assert track_args.reid == DEFAULT_REID
    assert track_args.tracker_backend == "python"
    assert export_args.weights == WEIGHTS / "osnet_x0_25_msmt17.pt"
    assert export_args.include == ("onnx",)


def test_build_mode_namespace_preserves_authored_component_selectors():
    args = build_mode_namespace(
        "track",
        {
            "detector": "yolox-x-dancetrack",
            "reid": "configs/custom-reid.yaml",
            "source": "0",
        },
        explicit_keys={"detector", "reid", "source"},
    )

    assert args.detector == "yolox-x-dancetrack"
    assert args.reid == "configs/custom-reid.yaml"


def test_build_mode_namespace_uses_explicit_tracker_backend():
    args = build_mode_namespace(
        "eval",
        {"experiment": "mot17-mini", "tracker": "botsort", "tracker_backend": "cpp"},
        explicit_keys={"tracker", "tracker_backend"},
    )

    assert args.tracker == "botsort"
    assert args.tracker_backend == "cpp"
