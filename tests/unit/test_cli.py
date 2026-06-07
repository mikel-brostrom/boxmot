import sys
from types import SimpleNamespace

from click.testing import CliRunner

from boxmot.engine.cli import boxmot


def test_eval_requires_benchmark():
    result = CliRunner().invoke(boxmot, ["eval"])
    assert result.exit_code != 0
    assert "requires --benchmark <benchmark.yaml>" in result.output


def test_eval_rejects_source_option():
    result = CliRunner().invoke(boxmot, ["eval", "--source", "boxmot/engine/eval/trackeval/data/MOT17-mini/train"])
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


def test_track_accepts_inline_tracker_backend(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracking.tracker", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["track", "--source", "0", "--tracker", "botsort:cpp"])
    assert result.exit_code == 0, result.output
    assert captured["args"].tracker == "botsort"
    assert captured["args"].tracker_backend == "cpp"
    assert captured["args"].show is True


def test_track_live_source_keeps_show_false_when_save_is_explicit(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracking.tracker", SimpleNamespace(main=fake_main))

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
            "csl_tinyvit_5m",
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
            "csl_tinyvit_5m",
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
            "--no-color-jitter",
            "--random-erasing",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_11m"
    assert args.weight_decay == 0.1
    assert args.imgsz == (384, 128)
    assert args.warmup_epochs == 20
    assert args.label_smooth == 0.05
    assert args.batch_size == 64
    assert args.p_ids == 16
    assert args.k_instances == 4
    assert args.random_grayscale == 0.1
    assert args.epochs == 200
    assert args.lr == 3.5e-4
    assert args.center_loss_weight == 0.0
    assert args.metric_feature == "raw_mean"
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
        "color_jitter",
        "random_erasing",
    } <= explicit
    assert "weight_decay" not in explicit
    assert "imgsz" not in explicit
    assert "warmup_epochs" not in explicit
    assert "label_smooth" not in explicit
    assert "random_grayscale" not in explicit


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
        ["generate", "--benchmark", "mot17-mini", "--source", "boxmot/engine/eval/trackeval/data/MOT17-mini/train"],
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
    result = CliRunner().invoke(boxmot, ["tune", "--source", "boxmot/engine/eval/trackeval/data/MOT17-mini/train"])
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

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracking.tracker", SimpleNamespace(main=fake_main))

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
        ["export", "--weights", "osnet_x0_25_msmt17.pt", "--include", "onnx"],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].weights.name == "osnet_x0_25_msmt17.pt"
    assert captured["args"].include == ("onnx",)
    assert captured["args"].device == "cpu"
