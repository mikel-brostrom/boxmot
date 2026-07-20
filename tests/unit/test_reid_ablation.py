"""Resume-contract and ablation-orchestration regression tests."""

from __future__ import annotations

import json

import pytest
import torch

from boxmot.engine.reid.ablation import AblationSpec, discover_run
from boxmot.reid.training.resume import (
    contract_differences,
    contract_fingerprint,
    run_fingerprint,
)
from boxmot.reid.training.trainer import ReIDTrainer


def _trainer(tmp_path, **kwargs) -> ReIDTrainer:
    values = {
        "model_name": "csl_tinyvit_11m",
        "dataset_name": "market1501",
        "data_dir": str(tmp_path),
        "epochs": 200,
        "p": 8,
        "k": 8,
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "head_parts": (1, 2, 4),
        "scale_balanced_branches": True,
        "metric_feature": "raw_concat",
        "inference_feature": "norm_concat_bn",
        "attention_window_layout": "rect",
        "interpolate_pretrained_attention_bias": True,
        "attention_mask": True,
        "flip_tta": False,
    }
    values.update(kwargs)
    trainer = ReIDTrainer(**values)
    recipe = trainer._resolve_training_recipe_for_model_name()
    recipe.apply_pre_build_defaults(trainer)
    recipe.apply_defaults(trainer)
    return trainer


def _spec(trainer: ReIDTrainer) -> AblationSpec:
    contract = trainer._resume_contract()
    return AblationSpec(
        trainer=trainer,
        contract=contract,
        fingerprint=contract_fingerprint(contract),
        run_fingerprint=run_fingerprint(contract, trainer.epochs),
    )


def _write_hparams(run_dir, spec: AblationSpec) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "hparams.json").write_text(
        json.dumps(
            {
                "resume": {
                    "contract": spec.contract,
                    "fingerprint": spec.fingerprint,
                    "run_fingerprint": spec.run_fingerprint,
                    "target_epochs": spec.trainer.epochs,
                }
            }
        ),
        encoding="utf-8",
    )


def test_resume_contract_ignores_runtime_placement_but_catches_sampler(tmp_path):
    original = _spec(_trainer(tmp_path, device="cpu"))
    moved = _spec(_trainer(tmp_path / "moved", device="mps"))

    assert contract_differences(original.contract, moved.contract) == []

    changed_sampler = _spec(_trainer(tmp_path, p=12, k=8))
    assert contract_differences(original.contract, changed_sampler.contract) == [
        "data.p: saved=8, requested=12"
    ]


def test_trainer_rejects_incompatible_and_complete_resume(tmp_path):
    trainer = _trainer(tmp_path)
    checkpoint = {
        "resume_contract": trainer._resume_contract(),
        "epochs": 200,
        "epoch": 50,
    }
    trainer._assert_resume_compatible(checkpoint, tmp_path / "last.pt")

    changed = _trainer(tmp_path, stage2_width_merge_after=2)
    with pytest.raises(ValueError, match="model.stage2_width_merge_after"):
        changed._assert_resume_compatible(checkpoint, tmp_path / "last.pt")

    checkpoint["epoch"] = 200
    with pytest.raises(ValueError, match="already complete"):
        trainer._assert_resume_compatible(checkpoint, tmp_path / "last.pt")


def test_resume_directory_refuses_best_only_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best.pt").write_bytes(b"weights")
    trainer = _trainer(tmp_path, resume=str(run_dir))

    with pytest.raises(ValueError, match="inference-only checkpoint"):
        trainer._resolve_resume_path()


def test_discover_run_selects_only_matching_resumable_state(tmp_path):
    spec = _spec(_trainer(tmp_path))
    project = tmp_path / "runs"
    run_dir = project / "variant"
    _write_hparams(run_dir, spec)
    torch.save(
        {
            "epoch": 40,
            "epochs": 200,
            "resumable": True,
            "resume_contract": spec.contract,
            "optimizer": {},
            "optimizer_center": {},
            "scheduler": {},
            "rng_state": {},
            "center_loss_state_dict": {},
        },
        run_dir / "last.pt",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"train": [{"epoch": 40}]}),
        encoding="utf-8",
    )

    action, path, reason = discover_run(project, "variant", spec)
    assert (action, path, reason) == ("resume", run_dir / "last.pt", "epoch 40")

    incompatible = _spec(_trainer(tmp_path, p=12, k=8))
    action, path, reason = discover_run(project, "variant", incompatible)
    assert action == "incompatible"
    assert path == run_dir
    assert "data.p" in reason


def test_discover_run_accepts_completed_metrics_without_retained_weights(tmp_path):
    spec = _spec(_trainer(tmp_path))
    project = tmp_path / "runs"
    run_dir = project / "variant"
    _write_hparams(run_dir, spec)
    (run_dir / "metrics.json").write_text(
        json.dumps({"train": [{"epoch": 200}]}),
        encoding="utf-8",
    )

    assert discover_run(project, "variant", spec)[:2] == ("complete", run_dir)
