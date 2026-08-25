import sys
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from boxmot.engine.cli import boxmot


def test_eval_requires_dataset_or_experiment():
    result = CliRunner().invoke(boxmot, ["eval"])
    assert result.exit_code != 0
    assert "requires either --dataset <dataset-id-or-yaml> or --experiment <experiment-id-or-yaml>" in result.output


def test_eval_rejects_source_option():
    result = CliRunner().invoke(boxmot, ["eval", "--source", "boxmot/datasets/mot/MOT17-mini/train"])
    assert result.exit_code != 0
    assert "No such option" in result.output and "--source" in result.output


def test_eval_passes_experiment_config_via_experiment(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "--experiment", "mot17-mini"])
    assert result.exit_code == 0, result.output
    assert captured["args"].experiment == "mot17-mini"
    assert captured["args"].dataset is None
    assert captured["args"].source is None
    assert captured["args"].tracker == "bytetrack"


def test_eval_accepts_dataset_with_explicit_split(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["eval", "--dataset", "mot17", "--split", "ablation", "--tracker", "boosttrack"],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].dataset == "mot17"
    assert captured["args"].experiment is None
    assert captured["args"].source is None
    assert captured["args"].split == "ablation"
    assert captured["args"].split_explicit is True
    assert captured["args"].tracker == "boosttrack"


def test_eval_rejects_dataset_and_experiment_together():
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--dataset",
            "mot17",
            "--experiment",
            "mot17-ablation-yolox-lmbn",
        ],
    )

    assert result.exit_code != 0
    assert "accepts either --dataset" in result.output
    assert "not both" in result.output


def test_eval_accepts_tracker_option(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "--experiment", "mot17-mini", "--tracker", "boosttrack"])
    assert result.exit_code == 0, result.output
    assert captured["args"].tracker == "boosttrack"


def test_eval_accepts_trackeval_comparison(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["eval", "--experiment", "mot17-mini", "--compare-trackeval"],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].compare_trackeval is True


def test_eval_accepts_tracker_backend_option(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["eval", "--experiment", "mot17-mini", "--tracker", "botsort", "--tracker-backend", "cpp"],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].tracker == "botsort"
    assert captured["args"].tracker_backend == "cpp"


def test_track_accepts_tracker_backend_option(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracking.workflow", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["track", "--source", "0", "--tracker", "botsort", "--tracker-backend", "cpp"],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].tracker == "botsort"
    assert captured["args"].tracker_backend == "cpp"
    assert captured["args"].show is True


def test_track_rejects_non_positive_saved_video_fps():
    result = CliRunner().invoke(boxmot, ["track", "--source", "0", "--fps", "0"])

    assert result.exit_code != 0
    assert "Invalid value for '--fps'" in result.output
    assert "x>=1" in result.output


def test_track_accepts_visualization_flags(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracking.workflow", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["track", "--source", "0", "--show-trajectories", "--show-kf-preds"],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].show_trajectories is True
    assert captured["args"].show_kf_preds is True


def test_track_accepts_dataset_with_selected_models(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracking.workflow", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "track",
            "--detector", "yolo26n",
            "--reid", "lmbn_n_duke",
            "--tracker", "occluboost",
            "--dataset", "mot17",
            "--split", "ablation",
            "--save",
            "--show",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.dataset == "mot17"
    assert args.source is None
    assert args.split == "ablation"
    assert args.split_explicit is True
    assert args.detector.name == "yolo26n.pt"
    assert args.reid.name == "lmbn_n_duke.pt"
    assert args.tracker == "occluboost"
    assert args.save is True
    assert args.show is True
    assert args.workflow_mode == "track"


def test_track_rejects_dataset_with_explicit_source():
    result = CliRunner().invoke(
        boxmot,
        ["track", "--dataset", "mot17", "--source", "video.mp4"],
    )

    assert result.exit_code != 0
    assert "accepts either --dataset" in result.output
    assert "--source <input>, not both" in result.output


def test_track_default_webcam_source_is_preserved_without_dataset(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracking.workflow", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["track"])

    assert result.exit_code == 0, result.output
    assert captured["args"].dataset is None
    assert captured["args"].source == "0"
    assert captured["args"].show is True
    assert captured["args"].workflow_mode == "track"


def test_track_live_source_keeps_show_false_when_save_is_explicit(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracking.workflow", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["track", "--source", "0", "--tracker", "botsort", "--save"])
    assert result.exit_code == 0, result.output
    assert captured["args"].save is True
    assert captured["args"].show is False


def test_eval_passes_show_timing_flag(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "--experiment", "mot17-mini", "--show-timing"])
    assert result.exit_code == 0, result.output
    assert captured["args"].show_timing is True


def test_train_preserves_explicit_hparam_keys(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--model",
            "csl_tinyvit_7m",
            "--lr",
            "3.5e-4",
            "--center-loss-weight",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].lr == 3.5e-4
    assert captured["args"].center_loss_weight == 0.0
    assert set(captured["args"].train_explicit_keys) >= {"data_dir", "model", "lr", "center_loss_weight"}


def test_train_accepts_boxmot_training_cfg(monkeypatch, tmp_path):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    train_cfg = tmp_path / "custom_config.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "run:",
                "  model_name: csl_tinyvit_7m",
                "data:",
                "  dataset: duke",
                f"  data_dir: {tmp_path}",
                "  img_size: [384, 128]",
                "model:",
                "  head:",
                "    parts: [1, 2, 4]",
                "optimization:",
                "  epochs: 99",
                "  lr: 0.001",
                "system:",
                "  device: cpu",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(boxmot, ["train", "--cfg", str(train_cfg), "--epochs", "3"])

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_7m"
    assert args.dataset == "duke"
    assert args.data_dir == str(tmp_path)
    assert args.imgsz == (384, 128)
    assert args.head_parts == (1, 2, 4)
    assert args.lr == 0.001
    assert args.epochs == 3
    assert args.device == "cpu"
    assert {"cfg", "epochs"} <= set(args.train_explicit_keys)


def test_train_accepts_anatomical_branch_distillation_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.reid.trainer",
        SimpleNamespace(main=fake_main),
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--anatomical-auxiliary",
            "--anatomical-metadata-dir",
            ".",
            "--anatomical-target-type",
            "learned_pose_concat_ema",
            "--anatomical-multiscale",
            "--anatomical-branch-distill-weight",
            "0.05",
            "--anatomical-branch-global-coefficient",
            "0.20",
            "--anatomical-branch-coarse-coefficient",
            "0.30",
            "--anatomical-branch-fine-coefficient",
            "0.50",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.anatomical_target_type == "learned_pose_concat_ema"
    assert args.anatomical_branch_distill_weight == 0.05
    assert args.anatomical_branch_global_coefficient == 0.20
    assert args.anatomical_branch_coarse_coefficient == 0.30
    assert args.anatomical_branch_fine_coefficient == 0.50


def test_train_accepts_pose_semantic_teacher_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.reid.trainer",
        SimpleNamespace(main=fake_main),
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--anatomical-auxiliary",
            "--anatomical-metadata-dir",
            "pose",
            "--anatomical-person-mask-dir",
            "masks",
            "--anatomical-target-type",
            "learned_pose_semantic_fused_ema",
            "--anatomical-multiscale",
            "--anatomical-foreground-weight",
            "0.03",
            "--anatomical-semantic-part-weight",
            "0.05",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.anatomical_target_type == (
        "learned_pose_semantic_fused_ema"
    )
    assert args.anatomical_person_mask_dir == "masks"
    assert args.anatomical_foreground_weight == 0.03
    assert args.anatomical_semantic_part_weight == 0.05


def test_train_accepts_anatomical_deployment_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.reid.trainer",
        SimpleNamespace(main=fake_main),
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--anatomical-auxiliary",
            "--anatomical-target-type",
            "learned_pose_concat_ema",
            "--anatomical-multiscale",
            "--anatomical-deployment",
            "--anatomical-deployment-dim",
            "64",
            "--anatomical-deployment-alpha",
            "0.25",
            "--anatomical-deployment-id-weight",
            "0.25",
            "--anatomical-deployment-metric-weight",
            "0.10",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.anatomical_deployment is True
    assert args.anatomical_deployment_dim == 64
    assert args.anatomical_deployment_alpha == 0.25
    assert args.anatomical_deployment_id_weight == 0.25
    assert args.anatomical_deployment_metric_weight == 0.10


def test_train_accepts_reid_data_yaml_list(monkeypatch, tmp_path):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    data_root = tmp_path / "datasets"
    market_root = data_root / "Market-1501-v15.09.15"
    duke_root = data_root / "DukeMTMC-reID"
    market_root.mkdir(parents=True)
    duke_root.mkdir(parents=True)
    market_yaml = tmp_path / "market1501.yaml"
    duke_yaml = tmp_path / "duke.yaml"
    market_yaml.write_text(
        f"dataset: market1501\npath: {market_root}\ntrain: bounding_box_train\nval: query\n",
        encoding="utf-8",
    )
    duke_yaml.write_text(
        f"dataset: duke\npath: {duke_root}\ntrain: bounding_box_train\nval: query\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        boxmot,
        ["train", "--data", str(market_yaml), "--data", str(duke_yaml), "--epochs", "1"],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.dataset == "market1501,duke"
    assert args.data_dir == str(data_root)
    assert args.data == (str(market_yaml), str(duke_yaml))
    assert len(args.data_specs) == 2
    assert args.data_specs[0]["name"] == "market1501"
    assert args.data_specs[0]["root"] == str(market_root)
    assert args.data_specs[1]["name"] == "duke"
    assert args.data_specs[1]["root"] == str(duke_root)
    assert {"data", "dataset", "data_dir", "data_specs"} <= set(args.train_explicit_keys)


def test_train_data_yaml_runs_download_script(monkeypatch, tmp_path):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    market_root = tmp_path / "downloaded" / "Market-1501-v15.09.15"
    market_yaml = tmp_path / "market1501.yaml"
    market_yaml.write_text(
        "\n".join(
            [
                "dataset: market1501",
                f"path: {market_root}",
                "download: |",
                "  Path(yaml['path']).mkdir(parents=True, exist_ok=True)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(boxmot, ["train", "--data", str(market_yaml), "--epochs", "1"])

    assert result.exit_code == 0, result.output
    assert market_root.exists()
    assert captured["args"].dataset == "market1501"
    assert captured["args"].data_specs[0]["root"] == str(market_root)


def test_train_accepts_reid_data_names_with_data_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["train", "--data", "market1501", "--data", "duke", "--data-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.dataset == "market1501,duke"
    assert args.data_dir == str(tmp_path)
    assert args.data_specs == (
        {"name": "market1501", "root": str(tmp_path.resolve())},
        {"name": "duke", "root": str(tmp_path.resolve())},
    )


def test_train_accepts_global_seed_and_deterministic_flags(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["train", "--data-dir", ".", "--seed", "123", "--no-deterministic"],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.seed == 123
    assert args.deterministic is False
    assert {"seed", "deterministic"} <= set(args.train_explicit_keys)


def test_train_accepts_composed_head_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--head-type",
            "gpc_lite",
            "--head-parts",
            "1,3",
            "--stripe-visibility",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.head_type == "gpc_lite"
    assert args.head_parts == (1, 3)
    assert args.stripe_visibility is True
    assert {"head_type", "head_parts", "stripe_visibility"} <= set(args.train_explicit_keys)


def test_train_accepts_head_and_branch_toggles(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--inference-feature",
            "dse_mix",
            "--metric-feature",
            "global",
            "--feature-fusion",
            "normpres_last3",
            "--pyramid-resize-mode",
            "pool_nearest",
            "--spatial-conv-mode",
            "depthwise_separable",
            "--post-fusion-mixer",
            "dwconv",
            "--post-fusion-mixer-reduction",
            "4",
            "--post-fusion-mixer-kernel",
            "5,3",
            "--post-fusion-mixer-gamma-init",
            "1e-4",
            "--aux-ce-weight",
            "0.05",
            "--aux-ce-drop-epoch",
            "120",
            "--drop-path-rate",
            "0.1",
            "--vit-lr-profile",
            "reid_lrd",
            "--backbone-freeze-epochs",
            "20",
            "--attention-window-layout",
            "rect",
            "--attention-bias",
            "signed_factorized",
            "--attention-mask",
            "--attention-shift",
            "--stage3-global",
            "--reid-adapter-stages",
            "2,3",
            "--reid-adapter-reduction",
            "8",
            "--head-pool",
            "dse",
            "--head-parts",
            "1,2,4",
            "--part-pooling",
            "tokens",
            "--num-part-tokens",
            "4",
            "--decouple-patterns",
            "--pattern-adapter-dim",
            "128",
            "--drop-global-aux",
            "--drop-global-aux-ratio",
            "0.25",
            "--branch-aware-metric",
            "--branch-metric-part-weight",
            "0.25",
            "--head-warmup-epochs",
            "5",
            "--head-warmup-lr-mult",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.inference_feature == "dse_mix"
    assert args.metric_feature == "global"
    assert args.feature_fusion == "normpres_last3"
    assert args.pyramid_resize_mode == "pool_nearest"
    assert args.spatial_conv_mode == "depthwise_separable"
    assert args.post_fusion_mixer == "dwconv"
    assert args.post_fusion_mixer_reduction == 4
    assert args.post_fusion_mixer_kernel == (5, 3)
    assert args.post_fusion_mixer_gamma_init == 1e-4
    assert args.aux_ce_weight == 0.05
    assert args.aux_ce_drop_epoch == 120
    assert args.drop_path_rate == 0.1
    assert args.vit_lr_profile == "reid_lrd"
    assert args.backbone_freeze_epochs == 20
    assert args.attention_window_layout == "rect"
    assert args.attention_bias == "signed_factorized"
    assert args.attention_mask is True
    assert args.attention_shift is True
    assert args.stage3_global is True
    assert args.reid_adapter_stages == (2, 3)
    assert args.reid_adapter_reduction == 8
    assert args.head_pool == "dse"
    assert args.head_parts == (1, 2, 4)
    assert args.part_pooling == "tokens"
    assert args.num_part_tokens == 4
    assert args.decouple_patterns is True
    assert args.pattern_adapter_dim == 128
    assert args.drop_global_aux is True
    assert args.drop_global_aux_ratio == 0.25
    assert args.branch_aware_metric is True
    assert args.branch_metric_part_weight == 0.25
    assert args.head_warmup_epochs == 5
    assert args.head_warmup_lr_mult == 3.0
    assert {
        "inference_feature",
        "metric_feature",
        "feature_fusion",
        "pyramid_resize_mode",
        "spatial_conv_mode",
        "post_fusion_mixer",
        "post_fusion_mixer_reduction",
        "post_fusion_mixer_kernel",
        "post_fusion_mixer_gamma_init",
        "aux_ce_weight",
        "aux_ce_drop_epoch",
        "drop_path_rate",
        "vit_lr_profile",
        "backbone_freeze_epochs",
        "attention_window_layout",
        "attention_bias",
        "attention_mask",
        "attention_shift",
        "stage3_global",
        "reid_adapter_stages",
        "reid_adapter_reduction",
        "head_pool",
        "head_parts",
        "part_pooling",
        "num_part_tokens",
        "decouple_patterns",
        "pattern_adapter_dim",
        "drop_global_aux",
        "drop_global_aux_ratio",
        "branch_aware_metric",
        "branch_metric_part_weight",
        "head_warmup_epochs",
        "head_warmup_lr_mult",
    } <= set(args.train_explicit_keys)


def test_train_accepts_pafpn_feature_fusion(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--feature-fusion",
            "last3_pafpn_stage2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].feature_fusion == "last3_pafpn_stage2"
    assert "feature_fusion" in set(captured["args"].train_explicit_keys)


def test_train_accepts_stage1_ablation_feature_fusions(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))
    modes = (
        "last3_stage1_concat",
        "last3_fpn_stage1_split",
        "last3_panet_stage1_split",
        "last3_panet_stage1_shared",
        "last3_bifpn_stage1_split",
        "global_final_parts_stage1_concat",
        "global_final_parts_fpn_layer0",
        "last3_panet_stage1_scale_aware",
        "last3_bifpn_stage1_branch_aware",
        "global_final_parts_stage2_semantic_residual",
        "global_final_parts_stage2_hierarchical_control",
        "global_final_parts_stage0_semantic_fine_reference",
        "global_final_parts_stage0_semantic_fine",
        "global_final_parts_stage0_fine_lite",
        "global_final_parts_stage0_panet_lite",
        "global_final_parts_stage0_bifpn_lite",
        "global_final_parts_stage0_native_pyramid",
        "global_final_parts_stage0_pool_first",
        "global_final_parts_hierarchical_fpn",
    )

    for mode in modes:
        result = CliRunner().invoke(boxmot, ["train", "--data-dir", ".", "--feature-fusion", mode])
        assert result.exit_code == 0, result.output
        assert captured["args"].feature_fusion == mode


def test_train_accepts_gradual_unfreeze_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--backbone-freeze-epochs",
            "0",
            "--gradual-unfreeze",
            "--gradual-unfreeze-head-epochs",
            "5",
            "--gradual-unfreeze-stage-epochs",
            "10",
            "--gradual-unfreeze-backbone-lr-mult",
            "0.1",
            "--gradual-unfreeze-backbone-lr-epochs",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.backbone_freeze_epochs == 0
    assert args.gradual_unfreeze is True
    assert args.gradual_unfreeze_head_epochs == 5
    assert args.gradual_unfreeze_stage_epochs == 10
    assert args.gradual_unfreeze_backbone_lr_mult == 0.1
    assert args.gradual_unfreeze_backbone_lr_epochs == 5
    assert {
        "gradual_unfreeze",
        "backbone_freeze_epochs",
        "gradual_unfreeze_head_epochs",
        "gradual_unfreeze_stage_epochs",
        "gradual_unfreeze_backbone_lr_mult",
        "gradual_unfreeze_backbone_lr_epochs",
    } <= set(args.train_explicit_keys)


def test_train_accepts_loss_ablation_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--loss",
            "circle",
            "--classifier-loss",
            "arcface",
            "--triplet-hard-margin",
            "--arcface-scale",
            "30",
            "--arcface-margin",
            "0.5",
            "--cosface-scale",
            "30",
            "--cosface-margin",
            "0.35",
            "--id-loss-weight",
            "1.25",
            "--metric-loss-weight",
            "1.0",
            "--early-id-loss-weight",
            "1.25",
            "--early-id-loss-epochs",
            "40",
            "--center-loss-ramp-start-epoch",
            "10",
            "--center-loss-ramp-end-epoch",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.loss == "circle"
    assert args.classifier_loss == "arcface"
    assert args.triplet_soft_margin is False
    assert args.arcface_scale == 30.0
    assert args.arcface_margin == 0.5
    assert args.cosface_scale == 30.0
    assert args.cosface_margin == 0.35
    assert args.id_loss_weight == 1.25
    assert args.metric_loss_weight == 1.0
    assert args.early_id_loss_weight == 1.25
    assert args.early_id_loss_epochs == 40
    assert args.center_loss_ramp_start_epoch == 10
    assert args.center_loss_ramp_end_epoch == 20
    assert {
        "loss",
        "classifier_loss",
        "triplet_soft_margin",
        "arcface_scale",
        "arcface_margin",
        "cosface_scale",
        "cosface_margin",
        "id_loss_weight",
        "metric_loss_weight",
        "early_id_loss_weight",
        "early_id_loss_epochs",
        "center_loss_ramp_start_epoch",
        "center_loss_ramp_end_epoch",
    } <= set(args.train_explicit_keys)


def test_train_accepts_weighted_regularized_triplet(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["train", "--data-dir", ".", "--loss", "wrt"],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].loss == "wrt"
    assert "loss" in captured["args"].train_explicit_keys


def test_train_recipe_values_apply_but_cli_flags_win(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--recipe",
            "csl_tinyvit_23m",
            "--data-dir",
            ".",
            "--epochs",
            "200",
            "--lr",
            "3.5e-4",
            "--center-loss-weight",
            "0",
            "--metric-feature",
            "raw_mean",
            "--feature-fusion",
            "weighted_last2",
            "--feat-dim",
            "384",
            "--neck-dim",
            "384",
            "--no-color-jitter",
            "--random-erasing",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_23m"
    assert args.weight_decay == 0.1
    assert args.imgsz == (384, 128)
    assert args.warmup_epochs == 20
    assert args.head_pool == "gelu_gem"
    assert args.head_parts == (1, 2)
    assert args.inference_feature == "norm_concat_bn"
    assert args.post_fusion_mixer == "none"
    assert args.post_fusion_mixer_reduction == 4
    assert args.post_fusion_mixer_kernel == (5, 3)
    assert args.post_fusion_mixer_gamma_init == 0.0
    assert args.feat_dim == 384
    assert args.neck_dim == 384
    assert args.drop_path_rate == 0.2
    assert args.attention_window_layout == "legacy"
    assert args.attention_bias == "absolute"
    assert args.attention_mask is False
    assert args.attention_shift is False
    assert args.stage3_global is False
    assert args.backbone_freeze_epochs == 10
    assert args.gradual_unfreeze is False
    assert args.gradual_unfreeze_head_epochs == 0
    assert args.gradual_unfreeze_stage_epochs == 0
    assert args.gradual_unfreeze_backbone_lr_mult == 1.0
    assert args.gradual_unfreeze_backbone_lr_epochs == 0
    assert args.early_id_loss_weight == 0.0
    assert args.early_id_loss_epochs == 0
    assert args.center_loss_ramp_start_epoch == 0
    assert args.center_loss_ramp_end_epoch == 0
    assert args.branch_aware_metric is False
    assert args.drop_global_aux is False
    assert args.drop_global_aux_ratio == 0.25
    assert args.branch_metric_part_weight == 0.5
    assert args.head_warmup_epochs == 0
    assert args.head_warmup_lr_mult == 2.0
    assert args.label_smooth == 0.05
    assert args.batch_size == 64
    assert args.p_ids == 16
    assert args.k_instances == 4
    assert args.random_grayscale == 0.1
    assert args.seed == 0
    assert args.deterministic is True
    assert args.epochs == 200
    assert args.lr == 3.5e-4
    assert args.center_loss_weight == 0.0
    assert args.metric_feature == "raw_mean"
    assert args.feature_fusion == "weighted_last2"
    assert args.color_jitter is False
    assert args.random_erasing == 0.0

    explicit = set(args.train_explicit_keys)
    assert {
        "recipe",
        "data_dir",
        "epochs",
        "lr",
        "center_loss_weight",
        "metric_feature",
        "feature_fusion",
        "feat_dim",
        "neck_dim",
        "color_jitter",
        "random_erasing",
    } <= explicit
    assert "weight_decay" not in explicit
    assert "imgsz" not in explicit
    assert "warmup_epochs" not in explicit
    assert "head_pool" not in explicit
    assert "head_parts" not in explicit
    assert "inference_feature" not in explicit
    assert "feature_fusion" in explicit
    assert "feat_dim" in explicit
    assert "neck_dim" in explicit
    assert "branch_aware_metric" not in explicit
    assert "branch_metric_part_weight" not in explicit
    assert "head_warmup_epochs" not in explicit
    assert "head_warmup_lr_mult" not in explicit
    assert "label_smooth" not in explicit
    assert "random_grayscale" not in explicit


def test_train_accepts_scale_balanced_branch_training_and_retrieval(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--model",
            "csl_tinyvit_11m",
            "--head-parts",
            "1,2,4",
            "--metric-feature",
            "raw_concat",
            "--inference-feature",
            "norm_concat_bn",
            "--scale-balanced-branches",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.scale_balanced_branches is True
    assert args.head_parts == (1, 2, 4)
    assert args.metric_feature == "raw_concat"
    assert args.inference_feature == "norm_concat_bn"
    assert "scale_balanced_branches" in args.train_explicit_keys


def test_train_recipe_can_supply_data_dir(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["train", "--recipe", "csl_tinyvit_23m"])

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.data_dir == "./Market-1501-v15.09.15"
    assert args.drop_path_rate == 0.2
    assert args.attention_window_layout == "legacy"
    assert args.attention_bias == "absolute"
    assert args.attention_mask is False
    assert args.attention_shift is False
    assert args.stage3_global is False
    assert args.head_pool == "gelu_gem"
    assert args.metric_feature == "raw_concat"
    assert args.inference_feature == "norm_concat_bn"
    assert args.post_fusion_mixer == "none"
    assert args.post_fusion_mixer_reduction == 4
    assert args.post_fusion_mixer_kernel == (5, 3)
    assert args.post_fusion_mixer_gamma_init == 0.0
    assert args.backbone_freeze_epochs == 10
    assert args.gradual_unfreeze is False
    assert args.gradual_unfreeze_stage_epochs == 0
    assert args.early_id_loss_weight == 0.0
    assert args.center_loss_ramp_start_epoch == 0


def test_train_can_disable_recipe_flip_tta(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["train", "--recipe", "csl_tinyvit_11m", "--no-flip-tta"])

    assert result.exit_code == 0, result.output
    assert captured["args"].flip_tta is False
    assert "flip_tta" in captured["args"].train_explicit_keys


def test_train_mobilenetv4_recipes_use_mobile_safe_baselines(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    cases = {
        "mobilenetv4": {
            "model_name": "mobilenetv4_conv_small",
            "feature_dim": 384,
            "feature_fusion": "final",
            "head_pool": "avg",
            "head_parts": (1,),
            "metric_feature": "auto",
            "inference_feature": "concat_bn",
            "epochs": 120,
            "lr": 3.5e-4,
        },
        "mobilenetv4_conv_small": {
            "model_name": "mobilenetv4_conv_small",
            "feature_dim": 384,
            "feature_fusion": "final",
            "head_pool": "avg",
            "head_parts": (1,),
            "metric_feature": "auto",
            "inference_feature": "concat_bn",
            "epochs": 120,
            "lr": 3.5e-4,
        },
        "mobilenetv4_conv_medium": {
            "model_name": "mobilenetv4_conv_medium",
            "feature_dim": 512,
            "feature_fusion": "last2",
            "head_pool": "gelu_gem",
            "head_parts": (1, 3),
            "metric_feature": "raw_concat",
            "inference_feature": "norm_concat_bn",
            "epochs": 120,
            "lr": 5e-4,
        },
        "mobilenetv4_conv_large": {
            "model_name": "mobilenetv4_conv_large",
            "feature_dim": 768,
            "feature_fusion": "final",
            "head_pool": "avg",
            "head_parts": (1,),
            "metric_feature": "auto",
            "inference_feature": "concat_bn",
            "epochs": 120,
            "lr": 3.5e-4,
        },
    }

    for recipe, expected in cases.items():
        result = CliRunner().invoke(boxmot, ["train", "--recipe", recipe, "--data-dir", "."])

        assert result.exit_code == 0, result.output
        args = captured["args"]
        assert args.model == expected["model_name"]
        assert args.pretrained is True
        assert args.imgsz == (384, 128)
        assert args.batch_size == 64
        assert args.feature_fusion == expected["feature_fusion"]
        assert args.post_fusion_mixer == "none"
        assert args.post_fusion_mixer_reduction == 4
        assert args.post_fusion_mixer_kernel == (5, 3)
        assert args.post_fusion_mixer_gamma_init == 0.0
        assert args.feat_dim == expected["feature_dim"]
        assert args.neck_dim == expected["feature_dim"]
        assert args.head_pool == expected["head_pool"]
        assert args.head_parts == expected["head_parts"]
        assert args.head_type == "standard"
        assert args.part_pooling == "stripes"
        assert args.metric_feature == expected["metric_feature"]
        assert args.inference_feature == expected["inference_feature"]
        assert args.drop_path_rate == 0.0
        assert args.drop_global_aux is False
        assert args.drop_global_aux_ratio == 0.25
        assert args.branch_aware_metric is False
        assert args.epochs == expected["epochs"]
        assert args.lr == expected["lr"]
        assert args.weight_decay == 1e-4
        assert args.warmup_epochs == 10
        assert args.eta_min == 1e-7
        assert args.center_loss_weight == 5e-4
        assert args.label_smooth == 0.1
        assert args.triplet_soft_margin is True
        assert args.backbone_freeze_epochs == 10
        assert args.head_warmup_epochs == 0
        assert args.head_warmup_lr_mult == 2.0
        assert args.gradual_unfreeze is False
        assert args.gradual_unfreeze_head_epochs == 0
        assert args.gradual_unfreeze_stage_epochs == 0
        assert args.gradual_unfreeze_backbone_lr_mult == 1.0
        assert args.gradual_unfreeze_backbone_lr_epochs == 0
        assert args.ema_decay == 0.999
        assert args.color_jitter is False
        assert args.gaussian_blur is False
        assert args.random_grayscale == 0.0
        assert args.random_erasing == 0.35
        assert args.random_patch is False
        assert args.flip_tta is False


def test_train_mobilenetv4_a11k_recipe_transfers_scale_balanced_procedure(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["train", "--recipe", "mobilenetv4_conv_small_a11k", "--data-dir", "."],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "mobilenetv4_conv_small"
    assert args.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert args.pyramid_resize_mode == "pool_bilinear"
    assert args.head_pool == "gelu_gem"
    assert args.head_parts == (1, 2, 4)
    assert args.feat_dim == 384
    assert args.neck_dim == 384
    assert args.scale_balanced_branches is True
    assert args.metric_feature == "raw_concat"
    assert args.inference_feature == "norm_concat_bn"
    assert args.p_ids == 12
    assert args.k_instances == 8
    assert args.epochs == 120
    assert args.lr == 5e-4
    assert args.weight_decay == 1e-4
    assert args.backbone_freeze_epochs == 10
    assert args.warmup_epochs == 20
    assert args.ema_decay == 0.999
    assert args.flip_tta is False


def test_train_accepts_csl_tinyvit_speed_architecture_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))
    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--attention-window-layout",
            "rect",
            "--stage3-downsample",
            "--stage2-mlp-ratio",
            "3.5",
            "--stage3-mlp-ratio",
            "3",
            "--stage2-depth",
            "7",
            "--stage3-depth",
            "1",
            "--native-branch-widths",
            "--fine-map-dim",
            "0",
            "--spatial-conv-mode",
            "bottleneck_depthwise",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.attention_window_layout == "rect"
    assert args.stage3_downsample is True
    assert args.stage2_mlp_ratio == 3.5
    assert args.stage3_mlp_ratio == 3.0
    assert args.stage2_depth == 7
    assert args.stage3_depth == 1
    assert args.native_branch_widths is True
    assert args.fine_map_dim == 0
    assert args.spatial_conv_mode == "bottleneck_depthwise"
    assert {
        "attention_window_layout",
        "stage3_downsample",
        "stage2_mlp_ratio",
        "stage3_mlp_ratio",
        "stage2_depth",
        "stage3_depth",
        "native_branch_widths",
        "fine_map_dim",
        "spatial_conv_mode",
    } <= set(args.train_explicit_keys)


def test_train_accepts_reid_x_register_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.reid.trainer",
        SimpleNamespace(main=fake_main),
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--no-width-first-hierarchy",
            "--stage2-depth",
            "6",
            "--stage3-depth",
            "2",
            "--identity-registers",
            "--identity-register-count",
            "4",
            "--identity-register-dim",
            "128",
            "--identity-register-num-heads",
            "4",
            "--identity-register-dropout",
            "0.10",
            "--identity-register-gate-init",
            "0",
            "--identity-register-diversity-weight",
            "0.01",
            "--identity-register-diversity-margin",
            "0.10",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.width_first_hierarchy is False
    assert args.stage2_depth == 6
    assert args.stage3_depth == 2
    assert args.identity_registers is True
    assert args.identity_register_count == 4
    assert args.identity_register_dim == 128
    assert args.identity_register_num_heads == 4
    assert args.identity_register_dropout == 0.10
    assert args.identity_register_gate_init == 0.0
    assert args.identity_register_diversity_weight == 0.01
    assert args.identity_register_diversity_margin == 0.10
    assert {
        "width_first_hierarchy",
        "stage2_depth",
        "stage3_depth",
        "identity_registers",
        "identity_register_count",
        "identity_register_dim",
        "identity_register_num_heads",
        "identity_register_dropout",
        "identity_register_gate_init",
        "identity_register_diversity_weight",
        "identity_register_diversity_margin",
    } <= set(args.train_explicit_keys)


def test_train_accepts_pretrained_attention_bias_interpolation(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--attention-window-layout",
            "rect",
            "--interpolate-pretrained-attention-bias",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].interpolate_pretrained_attention_bias is True


def test_train_accepts_width_merge_and_compact_distillation_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))
    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--stage2-width-merge-after",
            "2",
            "--compact-deployment-head",
            "--compact-metric-loss-weight",
            "0.75",
            "--compact-cosine-distill-weight",
            "0.5",
            "--compact-pairwise-distill-weight",
            "0.25",
            "--csmm-loss-weight",
            "0.1",
            "--csmm-margin",
            "0.1",
            "--csmm-temperature",
            "0.05",
            "--csmm-topk-negatives",
            "8",
            "--csmm-start-epoch",
            "20",
            "--csmm-ramp-end-epoch",
            "40",
            "--treeboost-loss-weight",
            "0.15",
            "--treeboost-coarse-coefficient",
            "1.0",
            "--treeboost-fine-coefficient",
            "1.0",
            "--treeboost-node-coefficient",
            "0.25",
            "--treeboost-regression-coefficient",
            "0.1",
            "--treeboost-difficulty-floor",
            "0.25",
            "--treeboost-regression-tolerance",
            "0.02",
            "--treeboost-temperature",
            "0.05",
            "--treeboost-start-epoch",
            "30",
            "--treeboost-ramp-end-epoch",
            "60",
            "--hierarchical-branch-attention",
            "--branch-attention-token-dim",
            "96",
            "--branch-attention-num-heads",
            "4",
            "--branch-attention-num-layers",
            "1",
            "--branch-attention-mlp-ratio",
            "2.0",
            "--branch-attention-dropout",
            "0.0",
            "--branch-set-attention",
            "--branch-set-attention-token-dim",
            "128",
            "--branch-set-attention-num-heads",
            "4",
            "--branch-set-attention-num-layers",
            "1",
            "--branch-set-attention-mlp-ratio",
            "2.0",
            "--branch-set-attention-dropout",
            "0.0",
            "--multiscale-query-decoder",
            "--query-decoder-dim",
            "128",
            "--query-decoder-num-heads",
            "4",
            "--query-decoder-num-layers",
            "1",
            "--query-decoder-mlp-ratio",
            "2.0",
            "--query-decoder-dropout",
            "0.0",
            "--hierarchical-late-interaction",
            "--late-interaction-dim",
            "128",
            "--late-interaction-num-heads",
            "4",
            "--late-interaction-num-layers",
            "1",
            "--late-interaction-sinkhorn-iters",
            "5",
            "--late-interaction-null-tokens",
            "1",
            "--late-interaction-negative-identities",
            "16",
            "--late-interaction-rerank-topk",
            "100",
            "--late-interaction-base-score-init",
            "0.9",
            "--late-interaction-loss-weight",
            "0.2",
            "--late-interaction-distill-weight",
            "0.05",
            "--late-interaction-temperature",
            "0.07",
            "--late-interaction-start-epoch",
            "20",
            "--late-interaction-ramp-end-epoch",
            "50",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.stage2_width_merge_after == 2
    assert args.compact_deployment_head is True
    assert args.compact_metric_loss_weight == 0.75
    assert args.compact_cosine_distill_weight == 0.5
    assert args.compact_pairwise_distill_weight == 0.25
    assert args.csmm_loss_weight == 0.1
    assert args.csmm_margin == 0.1
    assert args.csmm_temperature == 0.05
    assert args.csmm_topk_negatives == 8
    assert args.csmm_start_epoch == 20
    assert args.csmm_ramp_end_epoch == 40
    assert args.treeboost_loss_weight == 0.15
    assert args.treeboost_coarse_coefficient == 1.0
    assert args.treeboost_fine_coefficient == 1.0
    assert args.treeboost_node_coefficient == 0.25
    assert args.treeboost_regression_coefficient == 0.1
    assert args.treeboost_difficulty_floor == 0.25
    assert args.treeboost_regression_tolerance == 0.02
    assert args.treeboost_temperature == 0.05
    assert args.treeboost_start_epoch == 30
    assert args.treeboost_ramp_end_epoch == 60
    assert args.hierarchical_branch_attention is True
    assert args.branch_attention_token_dim == 96
    assert args.branch_attention_num_heads == 4
    assert args.branch_attention_num_layers == 1
    assert args.branch_attention_mlp_ratio == 2.0
    assert args.branch_attention_dropout == 0.0
    assert args.branch_set_attention is True
    assert args.branch_set_attention_token_dim == 128
    assert args.branch_set_attention_num_heads == 4
    assert args.branch_set_attention_num_layers == 1
    assert args.branch_set_attention_mlp_ratio == 2.0
    assert args.branch_set_attention_dropout == 0.0
    assert args.multiscale_query_decoder is True
    assert args.query_decoder_dim == 128
    assert args.query_decoder_num_heads == 4
    assert args.query_decoder_num_layers == 1
    assert args.query_decoder_mlp_ratio == 2.0
    assert args.query_decoder_dropout == 0.0
    assert args.hierarchical_late_interaction is True
    assert args.late_interaction_dim == 128
    assert args.late_interaction_num_heads == 4
    assert args.late_interaction_num_layers == 1
    assert args.late_interaction_sinkhorn_iters == 5
    assert args.late_interaction_null_tokens == 1
    assert args.late_interaction_negative_identities == 16
    assert args.late_interaction_rerank_topk == 100
    assert args.late_interaction_base_score_init == 0.9
    assert args.late_interaction_loss_weight == 0.2
    assert args.late_interaction_distill_weight == 0.05
    assert args.late_interaction_temperature == 0.07
    assert args.late_interaction_start_epoch == 20
    assert args.late_interaction_ramp_end_epoch == 50


def test_train_accepts_multiscale_channel_head_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.reid.trainer",
        SimpleNamespace(main=fake_main),
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--data-dir",
            ".",
            "--head-type",
            "multiscale_channel2",
            "--multiscale-channel-alpha",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].head_type == "multiscale_channel2"
    assert captured["args"].multiscale_channel_alpha == 0.5


def test_train_short_run_caps_inherited_backbone_freeze(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--model",
            "mobilenetv2_x1_0",
            "--data-dir",
            ".",
            "--epochs",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].epochs == 1
    assert captured["args"].backbone_freeze_epochs == 1
    assert "epochs" in set(captured["args"].train_explicit_keys)
    assert "backbone_freeze_epochs" not in set(captured["args"].train_explicit_keys)


def test_train_short_run_preserves_explicit_backbone_freeze(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--model",
            "mobilenetv2_x1_0",
            "--data-dir",
            ".",
            "--epochs",
            "1",
            "--backbone-freeze-epochs",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].epochs == 1
    assert captured["args"].backbone_freeze_epochs == 10
    assert {"epochs", "backbone_freeze_epochs"} <= set(captured["args"].train_explicit_keys)


def test_train_csl_tinyvit_7m_recipe_keeps_small_model(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["train", "--recipe", "csl_tinyvit_7m", "--data-dir", "."])

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_7m"
    assert args.weight_decay == 0.1
    assert args.warmup_epochs == 20
    assert args.feat_dim == 512
    assert args.neck_dim == 512
    assert args.head_pool == "gelu_gem"
    assert args.head_parts == (1, 2)
    assert args.metric_feature == "raw_concat"
    assert args.inference_feature == "norm_concat_bn"
    assert args.feature_fusion == "last2"
    assert args.drop_path_rate == 0.1
    assert args.branch_aware_metric is False
    assert args.head_warmup_epochs == 0
    assert args.backbone_freeze_epochs == 10
    assert args.gradual_unfreeze is False
    assert args.gradual_unfreeze_head_epochs == 0
    assert args.gradual_unfreeze_stage_epochs == 0
    assert args.gradual_unfreeze_backbone_lr_mult == 1.0
    assert args.gradual_unfreeze_backbone_lr_epochs == 0


def test_train_default_model_is_csl_tinyvit_11m(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["train", "--data-dir", "."])

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_11m_v20"
    assert args.num_workers == 4
    assert args.imgsz == (384, 128)
    assert args.weight_decay == 0.1
    assert args.warmup_epochs == 20
    assert args.feat_dim == 512
    assert args.neck_dim == 512
    assert args.head_pool == "gelu_gem"
    assert args.head_parts == (1, 2, 4)
    assert args.metric_feature == "raw_concat"
    assert args.inference_feature == "norm_concat_bn"
    assert args.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert args.spatial_conv_mode == "depthwise_separable"
    assert args.scale_balanced_branches is True
    assert args.p_ids == 12
    assert args.k_instances == 8
    assert args.post_fusion_mixer == "none"
    assert args.post_fusion_mixer_reduction == 4
    assert args.post_fusion_mixer_kernel == (5, 3)
    assert args.post_fusion_mixer_gamma_init == 0.0
    assert args.attention_window_layout == "rect"
    assert args.attention_bias == "absolute"
    assert args.interpolate_pretrained_attention_bias is True
    assert args.attention_mask is True
    assert args.attention_shift is False
    assert args.stage3_global is False
    assert args.backbone_freeze_epochs == 10
    assert args.gradual_unfreeze is False
    assert args.gradual_unfreeze_head_epochs == 5
    assert args.gradual_unfreeze_stage_epochs == 20
    assert args.gradual_unfreeze_backbone_lr_mult == 0.1
    assert args.gradual_unfreeze_backbone_lr_epochs == 5
    assert args.early_id_loss_weight == 0.0
    assert args.early_id_loss_epochs == 0
    assert args.center_loss_ramp_start_epoch == 0
    assert args.center_loss_ramp_end_epoch == 0
    assert args.drop_global_aux is False
    assert args.drop_global_aux_ratio == 0.25
    assert args.flip_tta is False
    assert args.epochs == 200
    assert args.seed == 0
    assert args.deterministic is True


def test_train_csl_tinyvit_11m_recipe_is_normal_model(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["train", "--recipe", "csl_tinyvit_11m", "--data-dir", "."])

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_11m_v20"
    assert args.num_workers == 4
    assert args.head_pool == "gelu_gem"
    assert args.head_parts == (1, 2, 4)
    assert args.metric_feature == "raw_concat"
    assert args.inference_feature == "norm_concat_bn"
    assert args.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert args.pyramid_resize_mode == "bilinear"
    assert args.spatial_conv_mode == "depthwise_separable"
    assert args.scale_balanced_branches is True
    assert args.p_ids == 12
    assert args.k_instances == 8
    assert args.post_fusion_mixer == "none"
    assert args.drop_global_aux is False
    assert args.attention_window_layout == "rect"
    assert args.attention_bias == "absolute"
    assert args.interpolate_pretrained_attention_bias is True
    assert args.attention_mask is True
    assert args.attention_shift is False
    assert args.stage3_global is False
    assert args.branch_aware_metric is False
    assert args.head_warmup_epochs == 0
    assert args.backbone_freeze_epochs == 10
    assert args.gradual_unfreeze is False
    assert args.gradual_unfreeze_head_epochs == 5
    assert args.gradual_unfreeze_stage_epochs == 20
    assert args.gradual_unfreeze_backbone_lr_mult == 0.1
    assert args.gradual_unfreeze_backbone_lr_epochs == 5
    assert args.flip_tta is False
    assert args.early_id_loss_weight == 0.0
    assert args.early_id_loss_epochs == 0
    assert args.center_loss_ramp_start_epoch == 0
    assert args.center_loss_ramp_end_epoch == 0


def test_train_explicit_11m_v20_model_uses_canonical_recipe(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["train", "--model", "csl_tinyvit_11m_v20", "--data-dir", "."],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_11m_v20"
    assert args.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert args.spatial_conv_mode == "depthwise_separable"
    assert args.num_workers == 4


def test_train_accepts_csl_tinyvit_23m_lmbn_model(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["train", "--data-dir", ".", "--model", "csl_tinyvit_23m_lmbn"],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].model == "csl_tinyvit_23m_lmbn"


def test_eval_reid_accepts_scientific_feature_override_options(monkeypatch, tmp_path):
    captured = {}
    weights = tmp_path / "best.pt"
    data_dir = tmp_path / "market1501"
    weights.write_bytes(b"checkpoint")
    data_dir.mkdir()

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "eval-reid",
            "--weights",
            str(weights),
            "--dataset",
            "market1501",
            "--data-dir",
            str(data_dir),
            "--preprocess",
            "resize",
            "--imgsz",
            "384,128",
            "--inference-feature",
            "dse_mix",
            "--flip-tta",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.preprocess == "resize"
    assert args.imgsz == (384, 128)
    assert args.inference_feature == "dse_mix"
    assert args.flip_tta is True


def test_compare_reid_accepts_multiple_models_and_targets(monkeypatch, tmp_path):
    captured = {}
    weights_a = tmp_path / "a" / "best.pt"
    weights_b = tmp_path / "b" / "best.pt"
    market_data = tmp_path / "market1501"
    duke_data = tmp_path / "duke"
    weights_a.parent.mkdir()
    weights_b.parent.mkdir()
    weights_a.write_bytes(b"checkpoint-a")
    weights_b.write_bytes(b"checkpoint-b")
    market_data.mkdir()
    duke_data.mkdir()

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.comparison", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "compare-reid",
            "--weights",
            str(weights_a),
            "--weights",
            str(weights_b),
            "--target",
            f"market1501={market_data}",
            "--target",
            f"duke={duke_data}",
            "--label",
            "market-model",
            "--label",
            "duke-model",
            "--model",
            "csl_tinyvit_23m",
            "--inference-feature",
            "evidence_sinkhorn",
            "--include-same-dataset",
            "--continue-on-error",
            "--latency-warmup",
            "2",
            "--latency-iters",
            "7",
            "--output",
            str(tmp_path / "comparison"),
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.weights == (str(weights_a), str(weights_b))
    assert args.target == (f"market1501={market_data}", f"duke={duke_data}")
    assert args.label == ("market-model", "duke-model")
    assert args.model == ("csl_tinyvit_23m",)
    assert args.inference_feature == "evidence_sinkhorn"
    assert args.include_same_dataset is True
    assert args.continue_on_error is True
    assert args.latency_warmup == 2
    assert args.latency_iters == 7


def test_eval_rejects_positional_tracker_shim():
    result = CliRunner().invoke(boxmot, ["eval", "boosttrack", "--experiment", "mot17-mini"])
    assert result.exit_code != 0
    assert "Got unexpected extra argument (boosttrack)" in result.output


def test_generate_requires_data_or_source():
    result = CliRunner().invoke(boxmot, ["generate"])
    assert result.exit_code != 0
    assert (
        "requires --experiment <experiment-id-or-yaml> for config-driven runs or --source <dataset-path>"
        in result.output
    )


def test_generate_rejects_data_and_source_together():
    result = CliRunner().invoke(
        boxmot,
        ["generate", "--experiment", "mot17-mini", "--source", "boxmot/datasets/mot/MOT17-mini/train"],
    )
    assert result.exit_code != 0
    assert "accepts either --experiment <experiment-id-or-yaml> or --source <dataset-path>, not both" in result.output


def test_generate_passes_experiment_config_via_experiment(monkeypatch):
    captured = {}

    def fake_generate(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.cache",
        SimpleNamespace(main=fake_generate),
    )

    result = CliRunner().invoke(boxmot, ["generate", "--experiment", "mot17-mini"])
    assert result.exit_code == 0, result.output
    assert captured["args"].experiment == "mot17-mini"
    assert captured["args"].source is None


def test_tune_rejects_positional_tracker_shim():
    result = CliRunner().invoke(boxmot, ["tune", "boosttrack", "--experiment", "mot17-mini"])
    assert result.exit_code != 0
    assert "Got unexpected extra argument (boosttrack)" in result.output


def test_tune_requires_benchmark():
    result = CliRunner().invoke(boxmot, ["tune"])
    assert result.exit_code != 0
    assert "requires --experiment <experiment-id-or-yaml>" in result.output


def test_tune_rejects_source_option():
    result = CliRunner().invoke(boxmot, ["tune", "--source", "boxmot/datasets/mot/MOT17-mini/train"])
    assert result.exit_code != 0
    assert "No such option" in result.output and "--source" in result.output


def test_tune_accepts_space_separated_metric_lists(monkeypatch):
    captured = {}

    def fake_tune(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tuning.tuner", SimpleNamespace(main=fake_tune))

    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--experiment",
            "mot17-mini",
            "--tracker",
            "botsort",
            "--n-trials",
            "100",
            "--maximize",
            "HOTA",
            "MOTA",
            "IDF1",
            "--minimize",
            "IDSW_rate",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].maximize == ("HOTA,MOTA,IDF1",)
    assert captured["args"].minimize == ("IDSW_rate",)


def test_research_requires_benchmark():
    result = CliRunner().invoke(boxmot, ["research"])
    assert result.exit_code != 0
    assert "requires --experiment <experiment-id-or-yaml>" in result.output


def test_research_passes_experiment_config_via_flags(monkeypatch):
    captured = {}

    def fake_run(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.research",
        SimpleNamespace(main=fake_run),
    )

    result = CliRunner().invoke(
        boxmot,
        [
            "research",
            "--experiment",
            "mot17-mini",
            "--tracker",
            "boosttrack",
            "--proposal-model",
            "openai/gpt-5.4",
            "--proposal-api-key",
            "sk-test",
            "--proposal-api-key-env",
            "OPENAI_API_KEY",
            "--max-metric-calls",
            "5",
            "--eval-timeout",
            "12",
            "--keep-workspace",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].tracker == "boosttrack"
    assert captured["args"].experiment == "mot17-mini"
    assert captured["args"].proposal_model == "openai/gpt-5.4"
    assert captured["args"].proposal_api_key == "sk-test"
    assert captured["args"].proposal_api_key_env == "OPENAI_API_KEY"
    assert captured["args"].max_metric_calls == 5
    assert captured["args"].eval_timeout == 12.0
    assert captured["args"].keep_workspace is True


def test_research_help_shows_proposal_model_examples():
    result = CliRunner().invoke(boxmot, ["research", "--help"])
    assert result.exit_code == 0
    assert "openai/gpt-5.4" in result.output
    assert "anthropic/claude-sonnet-4-20250514" in result.output
    assert "--proposal-api-key" in result.output
    assert "--proposal-api-key-env" in result.output


def test_research_rejects_positional_tracker_shim():
    result = CliRunner().invoke(boxmot, ["research", "boosttrack", "--experiment", "mot17-mini"])
    assert result.exit_code != 0
    assert "Got unexpected extra argument (boosttrack)" in result.output


def test_eval_rejects_legacy_data_alias():
    result = CliRunner().invoke(boxmot, ["eval", "--data", "mot17-mini"])
    assert result.exit_code != 0
    assert "No such option" in result.output and "--data" in result.output


def test_generate_rejects_benchmark_names_passed_through_source():
    result = CliRunner().invoke(
        boxmot,
        ["generate", "--source", "mot17-mini-train-yolox-lmbn"],
    )
    assert result.exit_code != 0
    assert "uses --experiment <experiment-id-or-yaml> for experiment configs" in result.output


def test_generate_accepts_component_flags_with_source(monkeypatch):
    captured = {}

    def fake_generate(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.cache",
        SimpleNamespace(main=fake_generate),
    )

    result = CliRunner().invoke(
        boxmot,
        [
            "generate",
            "--source", ".",
            "--detector", "yolo11s-obb.pt",
            "--reid", "lmbn_n_duke.pt",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].source == "."
    assert captured["args"].experiment is None


def test_track_keeps_source_literal(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracking.workflow", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["track", "--source", "mot17-mini"])
    assert result.exit_code == 0, result.output
    assert captured["args"].source == "mot17-mini"
    assert captured["args"].benchmark == ""


def test_track_rejects_legacy_detector_alias():
    result = CliRunner().invoke(boxmot, ["track", "--yolo-model", "yolov8n.pt"])
    assert result.exit_code != 0
    assert "No such option" in result.output and "--yolo-model" in result.output


def test_track_rejects_legacy_reid_alias():
    result = CliRunner().invoke(boxmot, ["track", "--reid-model", "osnet_x0_25_msmt17.pt"])
    assert result.exit_code != 0
    assert "No such option" in result.output and "--reid-model" in result.output


def test_eval_rejects_legacy_tracking_method_alias():
    result = CliRunner().invoke(boxmot, ["eval", "--experiment", "mot17-mini", "--tracking-method", "boosttrack"])
    assert result.exit_code != 0
    assert "No such option" in result.output and "--tracking-method" in result.output


def test_track_help_lists_current_component_options():
    result = CliRunner().invoke(boxmot, ["track", "--help"])
    assert result.exit_code == 0, result.output
    assert "--detector PATH" in result.output
    assert "--reid PATH" in result.output
    assert "--tracker TEXT" in result.output
    assert "--dataset TEXT" in result.output
    assert "--yolo-model" not in result.output
    assert "--reid-model" not in result.output
    assert "--tracking-method" not in result.output
    assert "[default: bytetrack]" in result.output
    assert "--save" in result.output
    assert "--save-txt" in result.output
    assert "--save-crop" in result.output
    normalized_help = " ".join(result.output.split())
    assert "botsort, bytetrack, occluboost, ocsort, and sfsort" in normalized_help


def test_train_help_describes_identity_only_retrieval_labels():
    result = CliRunner().invoke(boxmot, ["train", "--help"])

    assert result.exit_code == 0, result.output
    normalized_help = " ".join(result.output.split())
    assert "distinct-camera instances within each identity; cameras affect sampling only" in normalized_help
    assert "Identity-defined dataset-memory SmoothAP" in normalized_help
    assert "Different-identity hard negatives" in normalized_help


def test_root_help_lists_research_mode():
    result = CliRunner().invoke(boxmot, ["--help"])
    assert result.exit_code == 0, result.output
    assert "research" in result.output
    assert (
        "boxmot research --experiment mot17-ablation-yolox-lmbn --tracker bytetrack "
        "--proposal-model openai/gpt-5.4"
        in result.output
    )


def test_export_builds_shared_namespace(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.export", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        [
            "export",
            "--weights",
            "osnet_x0_25_msmt17.pt",
            "--include",
            "onnx",
            "--coreml-batch-buckets",
            "1,4,16",
            "--coreml-minimum-deployment-target",
            "macOS15",
            "--coreml-compute-units",
            "CPUAndGPU",
            "--coreml-timeout",
            "120",
            "--coreml-max-memory-gb",
            "8",
            "--tflite-quantize",
            "static",
            "--tflite-calibration-data",
            "calibration",
            "--tflite-calibration-samples",
            "64",
            "--tflite-calibration-seed",
            "7",
            "--tflite-calibration-update",
            "moving_average",
            "--tflite-static-activation-bits",
            "8",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].weights.name == "osnet_x0_25_msmt17.pt"
    assert captured["args"].include == ("onnx",)
    assert captured["args"].device == "cpu"
    assert captured["args"].coreml_batch_buckets == (1, 4, 16)
    assert captured["args"].coreml_minimum_deployment_target == "macOS15"
    assert captured["args"].coreml_compute_units == "CPUAndGPU"
    assert captured["args"].coreml_timeout == 120.0
    assert captured["args"].coreml_max_memory_gb == 8.0
    assert captured["args"].tflite_quantize == "static"
    assert captured["args"].tflite_calibration_data == Path("calibration")
    assert captured["args"].tflite_calibration_samples == 64
    assert captured["args"].tflite_calibration_seed == 7
    assert captured["args"].tflite_calibration_update == "moving_average"
    assert captured["args"].tflite_static_activation_bits == 8
