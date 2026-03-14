import sys
from types import SimpleNamespace

from click.testing import CliRunner

from boxmot.engine.cli import boxmot


def test_eval_requires_data_or_source():
    result = CliRunner().invoke(boxmot, ["eval"])
    assert result.exit_code != 0
    assert "requires --data <benchmark.yaml> for benchmark runs or --source <dataset-path>" in result.output


def test_eval_rejects_data_and_source_together():
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--data", "mot17-ablation", "--source", "boxmot/engine/trackeval/data/MOT17-ablation/train"],
    )
    assert result.exit_code != 0
    assert "accepts either --data <benchmark.yaml> or --source <dataset-path>, not both" in result.output


def test_eval_passes_benchmark_config_via_data(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "--data", "mot17-ablation"])
    assert result.exit_code == 0, result.output
    assert captured["args"].data == "mot17-ablation"
    assert captured["args"].source is None


def test_eval_resolves_benchmark_source_name_automatically(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "--source", "MOT17-ablation"])
    assert result.exit_code == 0, result.output
    assert "use '--data MOT17-ablation' instead of '--source MOT17-ablation'" in result.output
    assert captured["args"].data == "MOT17-ablation"
    assert captured["args"].source is None


def test_generate_requires_data_or_source():
    result = CliRunner().invoke(boxmot, ["generate"])
    assert result.exit_code != 0
    assert "requires --data <benchmark.yaml> for benchmark runs or --source <dataset-path>" in result.output


def test_generate_rejects_data_and_source_together():
    result = CliRunner().invoke(
        boxmot,
        ["generate", "--data", "mot17-ablation", "--source", "boxmot/engine/trackeval/data/MOT17-ablation/train"],
    )
    assert result.exit_code != 0
    assert "accepts either --data <benchmark.yaml> or --source <dataset-path>, not both" in result.output


def test_generate_passes_benchmark_config_via_data(monkeypatch):
    captured = {}

    def fake_generate(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(run_generate_dets_embs=fake_generate))

    result = CliRunner().invoke(boxmot, ["generate", "--data", "mot17-ablation"])
    assert result.exit_code == 0, result.output
    assert captured["args"].data == "mot17-ablation"
    assert captured["args"].source is None


def test_track_keeps_source_literal(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracker", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["track", "--source", "MOT17-ablation"])
    assert result.exit_code == 0, result.output
    assert captured["args"].source == "MOT17-ablation"
    assert captured["args"].benchmark == ""


def test_track_help_lists_legacy_save_options():
    result = CliRunner().invoke(boxmot, ["track", "--help"])
    assert result.exit_code == 0, result.output
    assert "--save" in result.output
    assert "--save-txt" in result.output
    assert "--save-crop" in result.output
