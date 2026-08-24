"""Resume-contract and ablation-orchestration regression tests."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from boxmot.engine.reid.experimental.ablation import AblationSpec, discover_run
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


def test_resume_contract_and_checkpoint_bind_anatomical_metadata_bytes(
    tmp_path,
) -> None:
    metadata_root = tmp_path / "anatomy"
    mask_path = metadata_root / "person" / "0001.png"
    mask_path.parent.mkdir(parents=True)
    mask_path.write_bytes(b"mask-v1")
    manifest_path = metadata_root / "metadata.json"
    manifest_path.write_text(
        json.dumps(
            {
                "images": {
                    "bounding_box_train/0001.jpg": {
                        "keypoints": [[0.0, 0.0, 1.0]] * 17,
                        "person_mask": "person/0001.png",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    original_trainer = _trainer(
        tmp_path,
        anatomical_auxiliary=True,
        anatomical_metadata_dir=str(metadata_root),
    )
    original = original_trainer._resume_contract()
    digest = original["data"]["anatomical_metadata_sha256"]
    checkpoint_metadata = original_trainer._checkpoint_metadata(
        torch.nn.Linear(2, 2)
    )

    assert len(digest) == 64
    assert (
        checkpoint_metadata["anatomical_metadata_provenance"]["sha256"]
        == digest
    )

    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    changed_trainer = _trainer(
        tmp_path,
        anatomical_auxiliary=True,
        anatomical_metadata_dir=str(metadata_root),
    )
    changed = changed_trainer._resume_contract()
    assert changed["data"]["anatomical_metadata_sha256"] != digest
    with pytest.raises(ValueError, match="anatomical_metadata_sha256"):
        changed_trainer._assert_resume_compatible(
            {
                "resume_contract": original,
                "epochs": original_trainer.epochs,
                "epoch": 1,
            },
            tmp_path / "last.pt",
        )

    moved_root = tmp_path / "anatomy-moved"
    moved_mask = moved_root / "person" / "0001.png"
    moved_mask.parent.mkdir(parents=True)
    moved_mask.write_bytes(mask_path.read_bytes())
    (moved_root / "metadata.json").write_bytes(manifest_path.read_bytes())
    moved = _trainer(
        tmp_path,
        anatomical_auxiliary=True,
        anatomical_metadata_dir=str(moved_root),
    )._resume_contract()
    assert (
        moved["data"]["anatomical_metadata_sha256"]
        != changed["data"]["anatomical_metadata_sha256"]
    )


def test_resume_contract_ignores_missing_default_pk_policy(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    del historical["data"]["pk_steps_per_epoch"]
    del historical["data"]["camera_aware_sampler"]

    assert contract_differences(historical, current) == []

    fixed_camera_aware = _spec(
        _trainer(
            tmp_path,
            pk_steps_per_epoch=62,
            camera_aware_sampler=True,
        )
    ).contract
    assert contract_differences(historical, fixed_camera_aware) == [
        "data.camera_aware_sampler: saved='<missing>', requested=True",
        "data.pk_steps_per_epoch: saved='<missing>', requested=62",
    ]


def test_resume_contract_defaults_historical_timm_head_to_pooled(tmp_path):
    pooled = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(pooled)
    del historical["model"]["timm_head_mode"]

    assert contract_differences(historical, pooled) == []

    spatial = copy.deepcopy(pooled)
    spatial["model"]["timm_head_mode"] = "spatial"
    assert contract_differences(historical, spatial) == [
        "model.timm_head_mode: saved='pooled', requested='spatial'"
    ]


def test_resume_contract_normalizes_new_mobilenet_ablation_defaults(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    del historical["model"]["timm_model_name"]
    del historical["model"]["mobilenetv4_last_stride"]
    del historical["model"]["mobilenetv4_neck_mode"]
    del historical["optimization"]["backbone_lr_mult"]

    assert contract_differences(historical, current) == []

    requested = copy.deepcopy(current)
    requested["model"]["timm_model_name"] = (
        "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k"
    )
    requested["model"]["mobilenetv4_last_stride"] = 1
    requested["model"]["mobilenetv4_neck_mode"] = "spatial_ln"
    requested["optimization"]["backbone_lr_mult"] = 0.25
    assert contract_differences(historical, requested) == [
        "model.mobilenetv4_last_stride: saved=2, requested=1",
        "model.mobilenetv4_neck_mode: saved='cnn', requested='spatial_ln'",
        "model.timm_model_name: saved='', "
        "requested='mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k'",
        "optimization.backbone_lr_mult: saved=1.0, requested=0.25",
    ]


def test_resume_contract_ignores_missing_disabled_auxiliary_losses(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    for key in tuple(historical["loss"]):
        if key.startswith(("csmm_", "treeboost_")):
            del historical["loss"][key]

    assert contract_differences(historical, current) == []

    enabled_csmm = _spec(_trainer(tmp_path, csmm_loss_weight=0.1)).contract
    differences = contract_differences(historical, enabled_csmm)
    assert "loss.csmm_loss_weight: saved='<missing>', requested=0.1" in differences

    enabled_treeboost = _spec(_trainer(tmp_path, treeboost_loss_weight=0.15)).contract
    differences = contract_differences(historical, enabled_treeboost)
    assert "loss.treeboost_loss_weight: saved='<missing>', requested=0.15" in differences


def test_resume_contract_normalizes_missing_part_objective_defaults(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    for key in tuple(historical["loss"]):
        if key.startswith(("adasp_", "part_relation_")) or key in {
            "part_to_global_weight",
            "coarse_branch_ce_weight",
            "fine_branch_ce_weight",
        }:
            del historical["loss"][key]

    assert contract_differences(historical, current) == []

    reduced_fine_ce = copy.deepcopy(current)
    reduced_fine_ce["loss"]["fine_branch_ce_weight"] = 0.0
    assert contract_differences(historical, reduced_fine_ce) == [
        "loss.fine_branch_ce_weight: saved=1.0, requested=0.0"
    ]

    enabled_adasp = _spec(
        _trainer(tmp_path, adasp_loss_weight=0.1)
    ).contract
    assert "loss.adasp_loss_weight: saved='<missing>', requested=0.1" in contract_differences(
        historical,
        enabled_adasp,
    )

    enabled_part_relation = copy.deepcopy(current)
    enabled_part_relation["loss"]["part_relation_weight"] = 0.25
    assert (
        "loss.part_relation_weight: saved='<missing>', requested=0.25"
        in contract_differences(historical, enabled_part_relation)
    )


def test_resume_contract_ignores_missing_disabled_background_mosaic(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    for key in tuple(historical["augmentation"]):
        if key == "background_mosaic" or key.startswith("background_mosaic_"):
            del historical["augmentation"][key]

    assert contract_differences(historical, current) == []

    enabled = _spec(
        _trainer(
            tmp_path,
            background_mosaic=True,
            background_mosaic_mask_dir=str(tmp_path / "masks"),
        )
    ).contract
    differences = contract_differences(historical, enabled)
    assert "augmentation.background_mosaic: saved='<missing>', requested=True" in differences
    assert (
        "augmentation.background_mosaic_probability: saved='<missing>', requested=0.3"
        in differences
    )


def test_resume_contract_ignores_missing_disabled_same_id_part_mosaic(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    for key in tuple(historical["augmentation"]):
        if key == "same_id_part_mosaic" or key.startswith("same_id_part_mosaic_"):
            del historical["augmentation"][key]

    assert contract_differences(historical, current) == []

    enabled = _spec(_trainer(tmp_path, same_id_part_mosaic=True)).contract
    differences = contract_differences(historical, enabled)
    assert "augmentation.same_id_part_mosaic: saved='<missing>', requested=True" in differences
    assert (
        "augmentation.same_id_part_mosaic_probability: saved='<missing>', requested=0.35"
        in differences
    )


def test_resume_contract_ignores_missing_disabled_pav_mosaic(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    for key in tuple(historical["augmentation"]):
        if key == "pav_mosaic" or key.startswith("pav_mosaic_") or key == "pav_consistency_weight":
            del historical["augmentation"][key]

    assert contract_differences(historical, current) == []

    enabled = _spec(
        _trainer(
            tmp_path,
            pav_mosaic=True,
            pav_metadata_dir=str(tmp_path / "pose"),
        )
    ).contract
    differences = contract_differences(historical, enabled)
    assert "augmentation.pav_mosaic: saved='<missing>', requested=True" in differences
    assert (
        "augmentation.pav_mosaic_probability: saved='<missing>', requested=0.25"
        in differences
    )


def test_resume_contract_ignores_missing_disabled_context_occluder(tmp_path):
    current = _spec(
        _trainer(
            tmp_path,
            background_mosaic=True,
            background_mosaic_mask_dir=str(tmp_path / "masks"),
        )
    ).contract
    historical = copy.deepcopy(current)
    for key in tuple(historical["augmentation"]):
        if key.startswith("background_mosaic_occluder_"):
            del historical["augmentation"][key]

    assert contract_differences(historical, current) == []

    enabled = _spec(
        _trainer(
            tmp_path,
            background_mosaic=True,
            background_mosaic_mask_dir=str(tmp_path / "masks"),
            background_mosaic_occluder_probability=0.15,
        )
    ).contract
    differences = contract_differences(historical, enabled)
    assert (
        "augmentation.background_mosaic_occluder_probability: "
        "saved='<missing>', requested=0.15"
    ) in differences


def test_resume_contract_ignores_missing_disabled_hierarchical_attention(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    for key in tuple(historical["model"]):
        if key == "hierarchical_branch_attention" or key.startswith("branch_attention_"):
            del historical["model"][key]

    assert contract_differences(historical, current) == []

    enabled = _spec(_trainer(tmp_path, hierarchical_branch_attention=True)).contract
    differences = contract_differences(historical, enabled)
    assert "model.hierarchical_branch_attention: saved='<missing>', requested=True" in differences
    assert "model.branch_attention_token_dim: saved='<missing>', requested=96" in differences


def test_resume_contract_ignores_missing_disabled_late_interaction(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    for section in ("model", "loss", "evaluation"):
        for key in tuple(historical[section]):
            if key == "hierarchical_late_interaction" or key.startswith("late_interaction_"):
                del historical[section][key]

    assert contract_differences(historical, current) == []

    enabled = _spec(
        _trainer(
            tmp_path,
            spatial_conv_mode="depthwise_separable",
            hierarchical_late_interaction=True,
        )
    ).contract
    differences = contract_differences(historical, enabled)
    assert "model.hierarchical_late_interaction: saved='<missing>', requested=True" in differences
    assert "model.late_interaction_dim: saved='<missing>', requested=128" in differences
    assert "loss.late_interaction_loss_weight: saved='<missing>', requested=0.2" in differences


def test_resume_contract_ignores_missing_disabled_branch_set_attention(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    for key in tuple(historical["model"]):
        if key == "branch_set_attention" or key.startswith("branch_set_attention_"):
            del historical["model"][key]

    assert contract_differences(historical, current) == []

    enabled = _spec(
        _trainer(
            tmp_path,
            spatial_conv_mode="depthwise_separable",
            branch_set_attention=True,
        )
    ).contract
    differences = contract_differences(historical, enabled)
    assert "model.branch_set_attention: saved='<missing>', requested=True" in differences
    assert "model.branch_set_attention_token_dim: saved='<missing>', requested=128" in differences


def test_resume_contract_ignores_missing_disabled_multiscale_query_decoder(tmp_path):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    for key in tuple(historical["model"]):
        if key == "multiscale_query_decoder" or key.startswith("query_decoder_"):
            del historical["model"][key]

    assert contract_differences(historical, current) == []

    enabled = _spec(
        _trainer(
            tmp_path,
            spatial_conv_mode="depthwise_separable",
            multiscale_query_decoder=True,
        )
    ).contract
    differences = contract_differences(historical, enabled)
    assert "model.multiscale_query_decoder: saved='<missing>', requested=True" in differences
    assert "model.query_decoder_dim: saved='<missing>', requested=128" in differences


def test_resume_contract_tracks_multiscale_channel_power_only_when_enabled(
    tmp_path,
):
    current = _spec(_trainer(tmp_path)).contract
    historical = copy.deepcopy(current)
    del historical["model"]["multiscale_channel_alpha"]
    assert contract_differences(historical, current) == []

    enabled = _spec(
        _trainer(
            tmp_path,
            head_type="multiscale_channel2",
            multiscale_channel_alpha=0.5,
        )
    ).contract
    changed = copy.deepcopy(enabled)
    changed["model"]["multiscale_channel_alpha"] = 0.35
    assert contract_differences(enabled, changed) == [
        "model.multiscale_channel_alpha: saved=0.5, requested=0.35"
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
