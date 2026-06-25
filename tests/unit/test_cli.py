import sys
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from boxmot.engine.cli import boxmot


def test_eval_requires_benchmark():
    result = CliRunner().invoke(boxmot, ["eval"])
    assert result.exit_code != 0
    assert "requires --benchmark <benchmark.yaml>" in result.output


def test_eval_rejects_source_option():
    result = CliRunner().invoke(boxmot, ["eval", "--source", "data/benchmarks/MOT17-mini/train"])
    assert result.exit_code != 0
    assert "No such option" in result.output and "--source" in result.output


def test_eval_passes_benchmark_config_via_benchmark(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "--benchmark", "mot17-mini"])
    assert result.exit_code == 0, result.output
    assert captured["args"].data == "mot17-mini"
    assert captured["args"].source is None
    assert captured["args"].tracker == "bytetrack"


def test_eval_accepts_tracker_option(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "--benchmark", "mot17-mini", "--tracker", "boosttrack"])
    assert result.exit_code == 0, result.output
    assert captured["args"].tracker == "boosttrack"


def test_eval_accepts_tracker_backend_option(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["eval", "--benchmark", "mot17-mini", "--tracker", "botsort", "--tracker-backend", "cpp"],
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

    result = CliRunner().invoke(boxmot, ["eval", "--benchmark", "mot17-mini", "--show-timing"])
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
    assert args.branch_aware_metric is True
    assert args.branch_metric_part_weight == 0.25
    assert args.head_warmup_epochs == 5
    assert args.head_warmup_lr_mult == 3.0
    assert {
        "inference_feature",
        "metric_feature",
        "feature_fusion",
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
        "branch_aware_metric",
        "branch_metric_part_weight",
        "head_warmup_epochs",
        "head_warmup_lr_mult",
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
    assert {
        "loss",
        "classifier_loss",
        "triplet_soft_margin",
        "arcface_scale",
        "arcface_margin",
        "cosface_scale",
        "cosface_margin",
    } <= set(args.train_explicit_keys)


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
    assert args.feat_dim == 384
    assert args.neck_dim == 384
    assert args.drop_path_rate == 0.2
    assert args.attention_window_layout == "rect"
    assert args.attention_bias == "signed_factorized"
    assert args.attention_mask is True
    assert args.attention_shift is True
    assert args.stage3_global is False
    assert args.branch_aware_metric is False
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
    assert args.attention_window_layout == "rect"
    assert args.attention_bias == "signed_factorized"
    assert args.attention_mask is True
    assert args.attention_shift is True
    assert args.stage3_global is False
    assert args.head_pool == "gelu_gem"
    assert args.metric_feature == "raw_concat"
    assert args.inference_feature == "norm_concat_bn"


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
    assert args.head_pool == "gem"
    assert args.head_parts == (1, 2)
    assert args.inference_feature == "concat_bn"
    assert args.feature_fusion == "last2"
    assert args.branch_aware_metric is False
    assert args.head_warmup_epochs == 0


def test_train_default_model_is_csl_tinyvit_11m(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.reid.trainer", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["train", "--data-dir", "."])

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_11m"
    assert args.imgsz == (384, 128)
    assert args.weight_decay == 0.1
    assert args.warmup_epochs == 20
    assert args.feat_dim == 512
    assert args.neck_dim == 512
    assert args.head_pool == "gem"
    assert args.head_parts == (1, 2)
    assert args.inference_feature == "concat_bn"
    assert args.feature_fusion == "last3"
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
    assert args.model == "csl_tinyvit_11m"
    assert args.head_pool == "gem"
    assert args.head_parts == (1, 2)
    assert args.inference_feature == "concat_bn"
    assert args.feature_fusion == "last3"
    assert args.branch_aware_metric is False
    assert args.head_warmup_epochs == 0


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


def test_eval_rejects_positional_tracker_shim():
    result = CliRunner().invoke(boxmot, ["eval", "boosttrack", "--benchmark", "mot17-mini"])
    assert result.exit_code != 0
    assert "Got unexpected extra argument (boosttrack)" in result.output


def test_generate_requires_data_or_source():
    result = CliRunner().invoke(boxmot, ["generate"])
    assert result.exit_code != 0
    assert "requires --benchmark <benchmark.yaml> for config-driven runs or --source <dataset-path>" in result.output


def test_generate_rejects_data_and_source_together():
    result = CliRunner().invoke(
        boxmot,
        ["generate", "--benchmark", "mot17-mini", "--source", "data/benchmarks/MOT17-mini/train"],
    )
    assert result.exit_code != 0
    assert "accepts either --benchmark <benchmark.yaml> or --source <dataset-path>, not both" in result.output


def test_generate_passes_benchmark_config_via_benchmark(monkeypatch):
    captured = {}

    def fake_generate(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.cache",
        SimpleNamespace(main=fake_generate),
    )

    result = CliRunner().invoke(boxmot, ["generate", "--benchmark", "mot17-mini"])
    assert result.exit_code == 0, result.output
    assert captured["args"].data == "mot17-mini"
    assert captured["args"].source is None


def test_tune_rejects_positional_tracker_shim():
    result = CliRunner().invoke(boxmot, ["tune", "boosttrack", "--benchmark", "mot17-mini"])
    assert result.exit_code != 0
    assert "Got unexpected extra argument (boosttrack)" in result.output


def test_tune_requires_benchmark():
    result = CliRunner().invoke(boxmot, ["tune"])
    assert result.exit_code != 0
    assert "requires --benchmark <benchmark.yaml>" in result.output


def test_tune_rejects_source_option():
    result = CliRunner().invoke(boxmot, ["tune", "--source", "data/benchmarks/MOT17-mini/train"])
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
            "--benchmark",
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
    assert "requires --benchmark <benchmark.yaml>" in result.output


def test_research_passes_benchmark_config_via_flags(monkeypatch):
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
            "--benchmark",
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
    assert captured["args"].data == "mot17-mini"
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
    result = CliRunner().invoke(boxmot, ["research", "boosttrack", "--benchmark", "mot17-mini"])
    assert result.exit_code != 0
    assert "Got unexpected extra argument (boosttrack)" in result.output


def test_eval_rejects_legacy_data_alias():
    result = CliRunner().invoke(boxmot, ["eval", "--data", "mot17-mini"])
    assert result.exit_code != 0
    assert "No such option" in result.output and "--data" in result.output


def test_generate_rejects_benchmark_names_passed_through_source():
    result = CliRunner().invoke(boxmot, ["generate", "--source", "mot17-mini"])
    assert result.exit_code != 0
    assert "uses --benchmark <benchmark.yaml> for benchmark configs" in result.output


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
    assert captured["args"].data is None


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
    result = CliRunner().invoke(boxmot, ["eval", "--benchmark", "mot17-mini", "--tracking-method", "boosttrack"])
    assert result.exit_code != 0
    assert "No such option" in result.output and "--tracking-method" in result.output


def test_track_help_lists_current_component_options():
    result = CliRunner().invoke(boxmot, ["track", "--help"])
    assert result.exit_code == 0, result.output
    assert "--detector PATH" in result.output
    assert "--reid PATH" in result.output
    assert "--tracker TEXT" in result.output
    assert "--yolo-model" not in result.output
    assert "--reid-model" not in result.output
    assert "--tracking-method" not in result.output
    assert "[default: bytetrack]" in result.output
    assert "--save" in result.output
    assert "--save-txt" in result.output
    assert "--save-crop" in result.output


def test_root_help_lists_research_mode():
    result = CliRunner().invoke(boxmot, ["--help"])
    assert result.exit_code == 0, result.output
    assert "research" in result.output
    assert (
        "boxmot research --benchmark mot17 --split ablation --tracker bytetrack --proposal-model openai/gpt-5.4"
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
    assert captured["args"].tflite_quantize == "static"
    assert captured["args"].tflite_calibration_data == Path("calibration")
    assert captured["args"].tflite_calibration_samples == 64
    assert captured["args"].tflite_calibration_seed == 7
    assert captured["args"].tflite_calibration_update == "moving_average"
    assert captured["args"].tflite_static_activation_bits == 8
