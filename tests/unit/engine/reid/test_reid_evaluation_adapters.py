"""Thin-adapter tests for ReID evaluation and comparison persistence."""

from __future__ import annotations

import json
from types import SimpleNamespace

import click

from boxmot.engine.commands.reid import compare as compare_command
from boxmot.engine.commands.reid import evaluate as evaluate_command
from boxmot.reid.evaluation import comparison, runner


def test_evaluation_adapter_defaults_match_the_domain_configuration() -> None:
    defaults = runner.ReIDEvaluationConfig(weights="weights.pt", dataset="market1501", data_dir="data")
    expected = {
        "preprocess": defaults.preprocess,
        "imgsz": defaults.imgsz,
        "inference_feature": defaults.inference_feature,
        "flip_tta": defaults.flip_tta,
        "device": defaults.device,
        "batch_size": defaults.batch_size,
        "num_workers": defaults.num_workers,
        "latency_warmup": defaults.latency_warmup,
        "latency_iters": defaults.latency_iters,
    }

    for command in (evaluate_command.eval_reid, compare_command.compare_reid):
        options = {param.name: param for param in command.params if isinstance(param, click.Option)}
        assert {name: options[name].default for name in expected} == expected


def test_evaluate_adapter_persists_domain_result(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.touch()
    output = tmp_path / "reports"
    expected = {
        "model": "osnet_x0_25",
        "weights": str(weights),
        "dataset": "market1501",
        "preprocess": "resize",
        "img_size": [256, 128],
        "inference_feature": "global",
        "feature_dim": 512,
        "flip_tta": False,
        "mAP": 0.75,
        "rank1": 0.85,
        "rank5": 0.9,
        "rank10": 0.95,
    }
    captured = {}

    def fake_evaluate(config):
        captured["config"] = config
        return dict(expected)

    monkeypatch.setattr(runner, "evaluate_reid", fake_evaluate)
    result = evaluate_command.main(
        SimpleNamespace(
            weights=weights,
            dataset="market1501",
            data_dir=tmp_path,
            output=output,
        )
    )

    report = output / "eval_osnet_x0_25_market1501_global.json"
    assert result == expected
    assert json.loads(report.read_text()) == expected
    assert captured["config"].weights == weights
    assert not hasattr(captured["config"], "output")


def test_compare_adapter_persists_aggregate_table_and_pair_results(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.touch()
    data_dir = tmp_path / "market"
    data_dir.mkdir()
    output = tmp_path / "comparison"
    evaluation = {
        "model": "osnet_x0_25",
        "weights": str(weights),
        "dataset": "market1501",
        "preprocess": "resize",
        "img_size": [256, 128],
        "inference_feature": "global",
        "feature_dim": 512,
        "flip_tta": False,
        "mAP": 0.75,
        "rank1": 0.85,
        "rank5": 0.9,
        "rank10": 0.95,
        "latency_device": "cpu",
        "latency_ms_per_image": 2.5,
    }
    row = {
        "label": "candidate",
        "train_dataset": "duke",
        "eval_dataset": "market1501",
        "eval_dataset_key": "market1501",
        "data_dir": str(data_dir),
        "cross_domain": True,
        "status": "ok",
        **evaluation,
    }
    captured = {}

    def fake_compare(config):
        captured["config"] = config
        return {
            "summary": {
                "models": 1,
                "targets": 1,
                "rows": 1,
                "evaluated": 1,
                "skipped": 0,
                "failed": 0,
                "cross_domain_only": True,
            },
            "results": [dict(row)],
        }

    monkeypatch.setattr(comparison, "compare_reid", fake_compare)
    monkeypatch.setattr(compare_command, "_write_map_latency_plot", lambda *_args: None)
    result = compare_command.main(
        SimpleNamespace(
            weights=(weights,),
            target=(f"market1501={data_dir}",),
            label=("candidate",),
            model=(),
            output=output,
        )
    )

    pair_path = output / "candidate" / "eval_osnet_x0_25_market1501_global.json"
    assert json.loads(pair_path.read_text()) == evaluation
    assert json.loads((output / "cross_domain_results.json").read_text()) == result
    assert (output / "cross_domain_results.md").read_text().startswith("# ReID Model Comparison")
    assert not hasattr(captured["config"], "output")
