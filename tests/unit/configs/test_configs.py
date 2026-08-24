from pathlib import Path

from boxmot.engine.config import (
    BOXMOT_DEFAULTS,
    DEFAULT_DETECTOR,
    DEFAULT_REID,
    build_mode_namespace,
    ensure_model_extension,
    get_mode_default,
    get_mode_defaults,
    load_training_recipe,
)
from boxmot.utils import WEIGHTS


def test_ensure_model_extension_preserves_explicit_export_paths():
    model_path = "models/osnet_x0_25_msmt17_saved_model/osnet_x0_25_msmt17_float32.tflite"

    resolved = ensure_model_extension(model_path)

    assert resolved == Path(model_path)


def test_ensure_model_extension_keeps_bare_reid_names_in_weights_dir():
    resolved = ensure_model_extension("osnet_x0_25_msmt17")

    assert resolved == WEIGHTS / "osnet_x0_25_msmt17.pt"


def test_build_mode_namespace_uses_shared_runtime_defaults():
    args = build_mode_namespace("eval", {"experiment": "mot17-mini"}, explicit_keys=set())

    assert args.detector == [DEFAULT_DETECTOR]
    assert args.reid == [DEFAULT_REID]
    assert args.tracker == get_mode_default("eval", "tracker")
    assert args.tracker_backend == "python"
    assert args.detector_explicit is False
    assert args.reid_explicit is False
    assert args.project == Path(get_mode_default("eval", "project"))
    assert args.show_timing is False


def test_get_mode_defaults_returns_normalized_merged_defaults():
    defaults = get_mode_defaults("eval")

    assert defaults["detector"] == DEFAULT_DETECTOR
    assert defaults["reid"] == DEFAULT_REID
    assert defaults["tracker"] == get_mode_default("eval", "tracker")
    assert defaults["project"] == Path(get_mode_default("eval", "project"))
    assert defaults["show_timing"] is False
    assert isinstance(defaults["n_threads"], int)
    assert defaults["n_threads"] >= 1


def test_boxmot_defaults_bundle_exposes_typed_mode_defaults():
    assert BOXMOT_DEFAULTS.shared.detector == DEFAULT_DETECTOR
    assert BOXMOT_DEFAULTS.shared.reid == DEFAULT_REID
    assert BOXMOT_DEFAULTS.track.tracker == get_mode_default("track", "tracker")
    assert BOXMOT_DEFAULTS.track.tracker_backend == "python"
    assert BOXMOT_DEFAULTS.eval.project == Path(get_mode_default("eval", "project"))
    assert BOXMOT_DEFAULTS.export.include == tuple(get_mode_default("export", "include"))


def test_default_11m_training_uses_promoted_speed_recipe():
    args = build_mode_namespace("train", {"data_dir": "."}, explicit_keys={"data_dir"})

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
    args = build_mode_namespace(
        "train",
        {"data_dir": ".", "model": "csl_tinyvit_11m_v20"},
        explicit_keys={"data_dir", "model"},
    )

    assert args.model == "csl_tinyvit_11m_v20"
    assert args.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert args.spatial_conv_mode == "depthwise_separable"
    assert args.head_parts == (1, 2, 4)
    assert args.num_workers == 4


def test_explicit_non_11m_training_model_keeps_generic_defaults():
    args = build_mode_namespace(
        "train",
        {"data_dir": ".", "model": "csl_tinyvit_7m"},
        explicit_keys={"data_dir", "model"},
    )

    assert args.model == "csl_tinyvit_7m"
    assert args.feature_fusion == "last2"
    assert args.head_parts == (1, 2)
    assert args.scale_balanced_branches is False
    assert args.attention_window_layout == "legacy"


def test_build_mode_namespace_normalizes_track_and_export_models():
    track_args = build_mode_namespace("track", {"source": "0"}, explicit_keys=set())
    export_args = build_mode_namespace(
        "export",
        {"weights": "osnet_x0_25_msmt17", "include": ["onnx"]},
        explicit_keys={"weights", "include"},
    )

    assert track_args.detector == DEFAULT_DETECTOR
    assert track_args.reid == DEFAULT_REID
    assert track_args.tracker_backend == "python"
    assert export_args.weights == WEIGHTS / "osnet_x0_25_msmt17.pt"
    assert export_args.include == ("onnx",)


def test_build_mode_namespace_uses_explicit_tracker_backend():
    args = build_mode_namespace(
        "eval",
        {"experiment": "mot17-mini", "tracker": "botsort", "tracker_backend": "cpp"},
        explicit_keys={"tracker", "tracker_backend"},
    )

    assert args.tracker == "botsort"
    assert args.tracker_backend == "cpp"
