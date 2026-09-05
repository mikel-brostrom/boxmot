"""Delegation tests for standalone ReID offline-tool adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace

from boxmot.engine.commands.reid import human_pretrain, privileged_cache, teacher_extract
from boxmot.reid.training import human_pretraining_runner, teacher_extraction
from boxmot.reid.training import privileged_cache as privileged_cache_domain


def test_privileged_cache_index_adapter_delegates(monkeypatch, tmp_path, capsys) -> None:
    captured = {}

    def fake_export(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(summary=lambda: {"dataset_index": str(kwargs["output"]), "sample_count": 2})

    monkeypatch.setattr(privileged_cache_domain, "export_dataset_index", fake_export)
    output = tmp_path / "index.json"
    assert (
        privileged_cache.main(
            [
                "index",
                "--dataset",
                "market1501",
                "--data-dir",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert captured["dataset_name"] == "market1501"
    assert captured["output"] == output
    assert json.loads(capsys.readouterr().out)["sample_count"] == 2


def test_teacher_extract_adapter_builds_domain_config(monkeypatch, tmp_path, capsys) -> None:
    captured = {}

    def fake_extract(config):
        captured["config"] = config
        return SimpleNamespace(summary=lambda: {"model_name": "teacher", "global_dim": 5})

    monkeypatch.setattr(teacher_extraction, "run_teacher_extraction", fake_extract)
    assert (
        teacher_extract.main(
            [
                "--teacher",
                str(tmp_path / "teacher.pt"),
                "--dataset-index",
                str(tmp_path / "index.json"),
                "--image-root",
                str(tmp_path),
                "--part-mask-input",
                str(tmp_path / "masks.pt"),
                "--output",
                str(tmp_path / "signals.pt"),
                "--storage-dtype",
                "float16",
            ]
        )
        == 0
    )

    assert str(captured["config"].storage_dtype) == "torch.float16"
    assert json.loads(capsys.readouterr().out)["global_dim"] == 5


def test_human_pretrain_adapter_maps_imgsz_and_lr(monkeypatch, tmp_path) -> None:
    captured = {}
    sentinel = object()

    def fake_run(config):
        captured["config"] = config
        return sentinel

    monkeypatch.setattr(human_pretraining_runner, "run_human_pretraining", fake_run)
    result = human_pretrain.main(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(tmp_path / "encoder.pt"),
            "--imgsz",
            "192,64",
            "--lr",
            "0.001",
        ]
    )

    assert result is sentinel
    assert captured["config"].img_size == (192, 64)
    assert captured["config"].learning_rate == 0.001
