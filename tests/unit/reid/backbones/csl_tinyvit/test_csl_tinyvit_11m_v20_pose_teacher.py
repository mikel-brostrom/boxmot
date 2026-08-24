"""Checks for the canonical 11M V20 A11v8 pose-teacher recipe."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from click.testing import CliRunner

from boxmot.engine.config import build_mode_namespace, load_training_recipe
from boxmot.engine.cli import boxmot
from boxmot.reid.backbones.families.csl_tinyvit.deployment import (
    optimize_csl_tinyvit_for_inference,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.training.config import ReIDTrainConfig, trainer_kwargs_from_args
from boxmot.reid.training.model_options import build_reid_model_kwargs
from boxmot.reid.training.trainer import ReIDTrainer

RECIPE_NAME = "csl_tinyvit_11m_v20_pose_teacher"


def test_11m_v20_pose_teacher_recipe_reproduces_a11v8_policy():
    recipe = load_training_recipe(RECIPE_NAME)

    expected = {
        "model": "csl_tinyvit_11m_v20",
        "num_workers": 4,
        "p_ids": 12,
        "k_instances": 8,
        "epochs": 200,
        "lr": 0.0007,
        "weight_decay": 0.1,
        "layer_decay": 0.95,
        "warmup_epochs": 20,
        "eta_min": 1e-7,
        "backbone_freeze_epochs": 10,
        "anatomical_auxiliary": True,
        "anatomical_metadata_dir": "Market-1501-pav-metadata-clean",
        "anatomical_target_type": "learned_pose_concat_ema",
        "anatomical_token_dim": 128,
        "anatomical_multiscale": True,
        "anatomical_accessory_query": False,
        "anatomical_deployment": False,
        "anatomical_distill_weight": 0.1,
        "anatomical_attention_weight": 0.1,
        "anatomical_visibility_weight": 0.05,
        "anatomical_contrastive_weight": 0.1,
        "anatomical_descriptor_distill_weight": 0.0,
        "anatomical_pose_teacher_weight": 0.03,
        "anatomical_local_scale_weight": 0.6,
        "anatomical_fine_scale_weight": 0.4,
        "anatomical_cross_scale_weight": 0.05,
        "anatomical_pose_only_reliability": 0.0,
        "anatomical_student_start_epoch": 0,
        "anatomical_student_ramp_end_epoch": 50,
        "anatomical_decay_start_epoch": 120,
        "anatomical_decay_end_epoch": 170,
        "anatomical_temperature": 0.07,
        "anatomical_teacher_momentum": 0.999,
        "flip_tta": False,
    }
    assert {key: recipe[key] for key in expected} == expected
    assert recipe["background_mosaic"] is False
    assert recipe["same_id_part_mosaic"] is False
    assert recipe["pav_mosaic"] is False
    assert recipe["pav_consistency_weight"] == 0.0


def test_11m_v20_pose_teacher_cli_selects_dedicated_policy(monkeypatch):
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
            "--recipe",
            RECIPE_NAME,
            "--data-dir",
            ".",
            "--anatomical-metadata-dir",
            ".",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_11m_v20"
    assert args.anatomical_auxiliary is True
    assert args.anatomical_target_type == "learned_pose_concat_ema"
    assert args.anatomical_multiscale is True
    assert args.anatomical_deployment is False
    assert args.anatomical_pose_teacher_weight == 0.03
    assert args.anatomical_metadata_dir == "."
    assert args.num_workers == 4


def test_11m_v20_pose_teacher_model_is_exact_and_training_only(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    args = build_mode_namespace(
        "train",
        {
            "recipe": RECIPE_NAME,
            "data_dir": str(tmp_path),
            "anatomical_metadata_dir": str(metadata_dir),
        },
        explicit_keys={"recipe", "data_dir", "anatomical_metadata_dir"},
    )
    trainer_kwargs = trainer_kwargs_from_args(args, {})
    trainer_kwargs["pretrained"] = False
    trainer = ReIDTrainer.from_config(ReIDTrainConfig.from_flat_kwargs(**trainer_kwargs))
    model = ReIDModelRegistry.build_model(
        trainer.model_name,
        weights=None,
        num_classes=751,
        loss=trainer.loss_type,
        pretrained=False,
        use_gpu=False,
        **build_reid_model_kwargs(trainer),
    )

    assert trainer.model_name == "csl_tinyvit_11m_v20"
    assert trainer.anatomical_auxiliary is True
    assert model.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert model.head.head_parts == (1, 2, 4)
    assert model.head.metric_dim == 1536
    assert model.head.inference_feature == "norm_concat_bn"
    assert model.head.anatomical_auxiliary_pool is not None
    assert model.head.anatomical_deployment_enabled is False
    assert sum(parameter.numel() for parameter in model.parameters()) == 13_887_539

    optimize_csl_tinyvit_for_inference(model.eval())

    assert model.head.anatomical_auxiliary_pool is None
    assert model.head.anatomical_auxiliary_enabled is False
    with torch.inference_mode():
        descriptor = model(torch.randn(1, 3, 384, 128))
    assert descriptor.shape == (1, 1536)
    torch.testing.assert_close(
        descriptor.norm(dim=1),
        torch.ones(1),
        rtol=1e-5,
        atol=1e-6,
    )
