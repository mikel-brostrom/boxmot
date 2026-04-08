import sys
from types import SimpleNamespace

from click.testing import CliRunner

from boxmot.engine.cli import boxmot


def _workflow_stub(run_impl):
    class WorkflowStub:
        def __init__(self, args):
            self.args = args

        def run(self):
            return run_impl(self.args)

    return WorkflowStub


def test_eval_requires_benchmark():
    result = CliRunner().invoke(boxmot, ["eval"])
    assert result.exit_code != 0
    assert "requires --benchmark <benchmark.yaml>" in result.output


def test_eval_rejects_source_option():
    result = CliRunner().invoke(boxmot, ["eval", "--source", "boxmot/engine/trackeval/data/MOT17-ablation/train"])
    assert result.exit_code != 0
    assert "No such option: --source" in result.output


def test_eval_passes_benchmark_config_via_data(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "--benchmark", "mot17-ablation"])
    assert result.exit_code == 0, result.output
    assert captured["args"].data == "mot17-ablation"
    assert captured["args"].source is None
    assert captured["args"].tracking_method == "bytetrack"


def test_eval_accepts_tracker_option(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "--benchmark", "mot17-ablation", "--tracker", "boosttrack"])
    assert result.exit_code == 0, result.output
    assert captured["args"].tracking_method == "boosttrack"


def test_eval_accepts_tracker_only_for_benchmark_runs(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "boosttrack", "--benchmark", "mot17-ablation"])
    assert result.exit_code == 0, result.output
    assert captured["args"].tracking_method == "boosttrack"
    assert captured["args"].data == "mot17-ablation"
    assert captured["args"].yolo_model_explicit is False
    assert captured["args"].reid_model_explicit is False


def test_generate_requires_data_or_source():
    result = CliRunner().invoke(boxmot, ["generate"])
    assert result.exit_code != 0
    assert "requires --benchmark <benchmark.yaml> for config-driven runs or --source <dataset-path>" in result.output


def test_generate_rejects_data_and_source_together():
    result = CliRunner().invoke(
        boxmot,
        ["generate", "--benchmark", "mot17-ablation", "--source", "boxmot/engine/trackeval/data/MOT17-ablation/train"],
    )
    assert result.exit_code != 0
    assert "accepts either --benchmark <benchmark.yaml> or --source <dataset-path>, not both" in result.output


def test_generate_passes_benchmark_config_via_data(monkeypatch):
    captured = {}

    def fake_generate(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.cache",
        SimpleNamespace(DetectionsEmbeddingsGenerator=_workflow_stub(fake_generate)),
    )

    result = CliRunner().invoke(boxmot, ["generate", "--benchmark", "mot17-ablation"])
    assert result.exit_code == 0, result.output
    assert captured["args"].data == "mot17-ablation"
    assert captured["args"].source is None


def test_tune_accepts_tracker_only_for_benchmark_runs(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.tuner", SimpleNamespace(TrackerTuner=_workflow_stub(fake_main)))

    result = CliRunner().invoke(boxmot, ["tune", "boosttrack", "--benchmark", "mot17-ablation"])
    assert result.exit_code == 0, result.output
    assert captured["args"].tracking_method == "boosttrack"
    assert captured["args"].data == "mot17-ablation"
    assert captured["args"].yolo_model_explicit is False
    assert captured["args"].reid_model_explicit is False


def test_tune_requires_benchmark():
    result = CliRunner().invoke(boxmot, ["tune"])
    assert result.exit_code != 0
    assert "requires --benchmark <benchmark.yaml>" in result.output


def test_tune_rejects_source_option():
    result = CliRunner().invoke(boxmot, ["tune", "--source", "boxmot/engine/trackeval/data/MOT17-ablation/train"])
    assert result.exit_code != 0
    assert "No such option: --source" in result.output


def test_eval_accepts_legacy_data_alias(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(boxmot, ["eval", "--data", "mot17-ablation"])
    assert result.exit_code == 0, result.output
    assert captured["args"].data == "mot17-ablation"


def test_generate_accepts_component_flags_with_source(monkeypatch):
    captured = {}

    def fake_generate(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.cache",
        SimpleNamespace(DetectionsEmbeddingsGenerator=_workflow_stub(fake_generate)),
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

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracker", SimpleNamespace(TrackingSession=_workflow_stub(fake_main)))

    result = CliRunner().invoke(boxmot, ["track", "--source", "MOT17-ablation"])
    assert result.exit_code == 0, result.output
    assert captured["args"].source == "MOT17-ablation"
    assert captured["args"].benchmark == ""


def test_track_rejects_legacy_detector_alias():
    result = CliRunner().invoke(boxmot, ["track", "--yolo-model", "yolov8n.pt"])
    assert result.exit_code != 0
    assert "No such option: --yolo-model" in result.output


def test_track_rejects_legacy_reid_alias():
    result = CliRunner().invoke(boxmot, ["track", "--reid-model", "osnet_x0_25_msmt17.pt"])
    assert result.exit_code != 0
    assert "No such option: --reid-model" in result.output


def test_eval_rejects_legacy_tracking_method_alias():
    result = CliRunner().invoke(boxmot, ["eval", "--benchmark", "mot17-ablation", "--tracking-method", "boosttrack"])
    assert result.exit_code != 0
    assert "No such option: --tracking-method" in result.output


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


def test_export_builds_shared_namespace(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(sys.modules, "boxmot.engine.export", SimpleNamespace(main=fake_main))

    result = CliRunner().invoke(
        boxmot,
        ["export", "--weights", "osnet_x0_25_msmt17.pt", "--include", "onnx"],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].weights.name == "osnet_x0_25_msmt17.pt"
    assert captured["args"].include == ("onnx",)
    assert captured["args"].device == "cpu"
