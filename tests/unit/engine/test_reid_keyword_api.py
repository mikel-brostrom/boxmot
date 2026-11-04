"""Exercise keyword ReID workflows without training or exporting real models."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from boxmot.engine.commands.reid import compare, evaluate, export, train


@pytest.fixture
def training_dispatch(monkeypatch, tmp_path):
    """Capture the typed configuration passed to the training domain."""
    from boxmot.reid.training.trainer import ReIDTrainer

    captured = {}
    result = SimpleNamespace(weights_path=tmp_path / "best.pt", best_mAP=0.75, best_rank1=0.85)

    def from_config(config):
        captured["config"] = config
        return SimpleNamespace(run=lambda: result)

    monkeypatch.setattr(ReIDTrainer, "from_config", from_config)
    return captured, result


def test_train_keywords_resolve_recipes_configs_and_dataset_yaml(training_dispatch, tmp_path) -> None:
    captured, expected = training_dispatch
    data_root = tmp_path / "market"
    data_root.mkdir()
    (data_root / "readme.txt").write_text("local dataset fixture")
    data_yaml = tmp_path / "market.yaml"
    data_yaml.write_text("dataset: market1501\npath: market\n")
    config_yaml = tmp_path / "training.yaml"
    config_yaml.write_text("epochs: 90\nlr: 0.002\ndevice: mps\n")

    result = train.main(
        recipe="csl_tinyvit_11m",
        cfg=config_yaml,
        data=(str(data_yaml),),
        epochs=5,
        device="cpu",
    )

    assert result is expected
    config = captured["config"]
    assert config.model.model_name == "csl_tinyvit_11m_v20"
    assert config.data.img_size == (384, 128)
    assert config.optimization.epochs == 5
    assert config.optimization.lr == 0.002
    assert config.run.device == "cpu"
    assert config.data.dataset_name == "market1501"
    assert Path(config.data.data_specs[0]["root"]) == data_root


def test_train_keywords_resolve_builtin_dataset_and_model_defaults(training_dispatch, monkeypatch, tmp_path) -> None:
    from boxmot.reid.datasets import resolution

    captured, expected = training_dispatch
    acquired = []
    monkeypatch.setattr(resolution, "default_reid_data_root", lambda: tmp_path)
    monkeypatch.setattr(
        resolution,
        "ensure_builtin_reid_dataset",
        lambda dataset, root: acquired.append((dataset, root)),
    )

    result = train.main(model="osnet_x0_25", dataset="market1501", imgsz=[256, 128])

    assert result is expected
    assert acquired == [("market1501", tmp_path)]
    config = captured["config"]
    assert config.model.model_name == "osnet_x0_25"
    assert config.data.data_dir == str(tmp_path)
    assert config.data.img_size == (256, 128)
    assert config.run.device == "cpu"


def test_evaluate_keywords_preserve_checkpoint_defaults_and_save_report(monkeypatch, tmp_path) -> None:
    from boxmot.reid.evaluation import runner

    expected = {"model": "osnet_x0_25", "dataset": "market1501", "mAP": 0.75}
    captured = {}

    def run_evaluation(config):
        captured["config"] = config
        return expected

    monkeypatch.setattr(runner, "evaluate_reid", run_evaluation)
    result = evaluate.main(weights=tmp_path / "best.pt", dataset="market1501", data_dir=tmp_path)

    assert result is expected
    assert json.loads((tmp_path / "eval_market1501.json").read_text()) == expected
    config = captured["config"]
    assert config.device == "cpu"
    assert config.batch_size == 64
    assert config.preprocess is None
    assert config.imgsz is None
    assert config.inference_feature is None
    assert config.flip_tta is None


def test_compare_keywords_parse_targets_and_use_default_report_directory(monkeypatch, tmp_path) -> None:
    from boxmot.reid.evaluation import comparison

    captured = {}
    expected = {"summary": {"evaluated": 0}, "results": []}

    def run_comparison(config):
        captured["config"] = config
        return expected

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(comparison, "compare_reid", run_comparison)
    result = compare.main(weights=("first.pt", "second.pt"), target=(f"market1501={tmp_path}",))

    assert result is expected
    output = tmp_path / "runs" / "reid_cross_domain"
    assert json.loads((output / "cross_domain_results.json").read_text()) == expected
    assert (output / "cross_domain_results.md").is_file()
    config = captured["config"]
    assert config.weights == ("first.pt", "second.pt")
    assert config.targets[0].dataset == "market1501"
    assert config.targets[0].data_dir == tmp_path
    assert config.include_same_dataset is False
    assert config.continue_on_error is False
    assert config.labels == ()
    assert config.device == "cpu"
    assert config.flip_tta is None


def test_export_keywords_normalize_paths_and_defaults_before_domain_dispatch(monkeypatch, tmp_path) -> None:
    from boxmot.engine.ui.reporters import export as reporter
    from boxmot.reid.exporters import workflow

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"checkpoint")
    captured = {}
    pipeline = MagicMock()
    pipeline.__enter__.return_value = pipeline
    monkeypatch.setattr(reporter, "ExportWorkflowReporter", lambda args: SimpleNamespace(pipeline=lambda: pipeline))

    def prepare_export(args):
        captured["args"] = args
        return torch.nn.Identity(), torch.zeros(1, 3, 8, 4)

    monkeypatch.setattr(workflow, "prepare_export", prepare_export)
    monkeypatch.setattr(workflow, "execute_export", lambda *args: {})
    result = export.main(weights=str(weights), include=["onnx"], tflite_calibration_data=str(tmp_path))

    assert result.weights == weights
    args = captured["args"]
    assert args.weights == weights
    assert args.include == ("onnx",)
    assert args.tflite_calibration_data == tmp_path
    assert args.batch_size == 1
    assert args.verbose is False
    assert args.half is False
    pipeline.finish.assert_called_once()


@pytest.mark.parametrize("workflow", [train.main, evaluate.main, compare.main, export.main])
def test_reid_workflows_reject_mixed_namespace_and_keywords(workflow) -> None:
    with pytest.raises(TypeError, match="either resolved arguments or keyword options"):
        workflow(SimpleNamespace(), device="cpu")
