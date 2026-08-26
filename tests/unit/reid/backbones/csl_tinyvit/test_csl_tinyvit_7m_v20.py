"""Checks for the width-adapted V20 recipe on CSL-TinyViT 7M."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from click.testing import CliRunner

from boxmot.engine.config import load_training_recipe
from boxmot.engine.cli import boxmot
from boxmot.reid.backbones.families.csl_tinyvit import csl_tinyvit_7m_v20
from boxmot.reid.training.config import ReIDTrainConfig, trainer_kwargs_from_args
from boxmot.reid.training.resume import contract_fingerprint
from boxmot.reid.training.trainer import ReIDTrainer

EXP_14_RESUME_FINGERPRINT = (
    "854459c023aad4ac4254820dd6978f9ed113376b9bc4eda753ed05354d296b95"
)


def test_7m_v20_recipe_preserves_v8_policy_with_scaled_widths():
    recipe = load_training_recipe("csl_tinyvit_7m_v20")

    assert recipe["model"] == "csl_tinyvit_7m_v20"
    assert recipe["feature_fusion"] == (
        "global_final_parts_stage0_semantic_fine"
    )
    assert recipe["feat_dim"] == 384
    assert recipe["neck_dim"] == 384
    assert recipe["anatomical_token_dim"] == 96
    assert recipe["head_parts"] == [1, 2, 4]
    assert recipe["part_pooling"] == "stripes"
    assert recipe["scale_balanced_branches"] is True
    assert recipe["anatomical_target_type"] == "learned_pose_concat_ema"
    assert recipe["anatomical_multiscale"] is True
    assert recipe["p_ids"] == 12
    assert recipe["k_instances"] == 8
    assert recipe["num_workers"] == 4
    assert recipe["flip_tta"] is False
    assert recipe["layer_decay"] == 0.95


def test_7m_v20_model_stays_in_7m_budget_and_balances_scale_power():
    model = csl_tinyvit_7m_v20(
        num_classes=751,
        loss="triplet",
        pretrained=False,
        anatomical_auxiliary=True,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
    )

    assert [layer.dim for layer in model.layers] == [64, 128, 160, 320]
    assert model.feature_fusion_module.stage_indices == (1, 2, 0)
    assert model.feature_fusion_module.local_channels == 384
    assert model.feature_fusion_module.fine_output_channels == 384
    assert model.head.bn_global.reduction.out_channels == 384
    assert model.head.bn_part0.reduction.out_channels == 192
    assert model.head.bn_part1.reduction.out_channels == 192
    assert all(
        getattr(model.head, f"bn_part{index}").reduction.out_channels == 96
        for index in range(2, 6)
    )
    assert model.head.anatomical_auxiliary_pool.token_dim == 96
    assert model.head.metric_dim == 1152
    assert model.head.classifier_dim == 1152
    assert model.head.center_dim == 1152
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_165_011


def test_7m_v20_direct_preset_is_rgb_only_and_unit_normalized():
    model = csl_tinyvit_7m_v20(
        num_classes=751,
        loss="triplet",
        pretrained=False,
    ).eval()

    assert model.head.anatomical_auxiliary_pool is None
    assert model.head.metric_feature == "raw_concat"
    assert model.head.metric_dim == 1152
    assert model.head.classifier_dim == 1152
    assert model.head.center_dim == 1152
    assert sum(parameter.numel() for parameter in model.parameters()) == 6_937_893
    with torch.inference_mode():
        descriptor = model(torch.randn(2, 3, 384, 128))
    assert descriptor.shape == (2, 1152)
    torch.testing.assert_close(
        descriptor.norm(dim=1),
        torch.ones(2),
        rtol=1e-5,
        atol=1e-6,
    )


def test_7m_v20_model_selection_resolves_promoted_training_contract(
    monkeypatch,
    tmp_path,
):
    captured = {}
    anatomical_metadata = tmp_path / "anatomical-metadata"
    anatomical_metadata.mkdir()
    (anatomical_metadata / "metadata.json").write_text(
        '{"images": {}}',
        encoding="utf-8",
    )

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
            "train-reid",
            "--model",
            "csl_tinyvit_7m_v20",
            "--anatomical-metadata-dir",
            str(anatomical_metadata),
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_7m_v20"
    assert args.data_dir == "boxmot/datasets/reid/Market-1501-v15.09.15"
    assert args.feat_dim == 384
    assert args.neck_dim == 384
    assert args.anatomical_token_dim == 96
    assert args.anatomical_auxiliary is True
    assert args.anatomical_multiscale is True
    assert args.p_ids == 12
    assert args.k_instances == 8
    assert args.flip_tta is False
    assert args.layer_decay == 0.95

    trainer_kwargs = trainer_kwargs_from_args(args, {})
    config = ReIDTrainConfig.from_flat_kwargs(**trainer_kwargs)
    trainer = ReIDTrainer.from_config(config)

    contract = trainer._resume_contract()
    assert len(contract["data"].pop("anatomical_metadata_sha256")) == 64
    contract["augmentation"].pop("anatomical_min_effective_coverage")
    assert contract_fingerprint(contract) == (
        EXP_14_RESUME_FINGERPRINT
    )
