"""Checks for the Stage-3 Hi-AFA-lite transfer to the 7M V20 recipe."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner

from boxmot.engine.config import load_training_recipe
from boxmot.engine.cli import boxmot
from boxmot.reid.backbones.families.csl_tinyvit import csl_tinyvit_7m
from boxmot.reid.training.ablation import resolve_csl_tinyvit_ablation
from boxmot.reid.training.config import ReIDTrainConfig, trainer_kwargs_from_args


def _v20_model_kwargs() -> dict[str, object]:
    """Return the exact architecture settings shared by V20 and Hi-AFA-lite."""
    return {
        "num_classes": 751,
        "loss": "triplet",
        "pretrained": False,
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "pyramid_resize_mode": "bilinear",
        "spatial_conv_mode": "depthwise_separable",
        "feat_dim": 384,
        "neck_dim": 384,
        "head_pool": "gelu_gem",
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "inference_feature": "norm_concat_bn",
        "scale_balanced_branches": True,
        "anatomical_auxiliary": True,
        "anatomical_token_dim": 96,
        "anatomical_multiscale": True,
        "anatomical_target_type": "learned_pose_concat_ema",
        "attention_window_layout": "rect",
        "attention_bias": "absolute",
        "interpolate_pretrained_attention_bias": True,
        "attention_mask": True,
        "drop_path_rate": 0.1,
    }


def test_hi_afa_lite_recipe_is_v20_plus_stage3_suppression() -> None:
    """Keep the comparison controlled to the requested adapter treatment."""
    baseline = load_training_recipe("csl_tinyvit_7m_v20")
    treatment = load_training_recipe("csl_tinyvit_7m_hi_afa_lite")
    expected = dict(baseline)
    expected["reid_adapter_stages"] = [3]
    expected["reid_adapter_suppression_tau"] = 0.7
    expected["n_params"] = 7_268_373

    assert treatment == expected
    assert treatment["reid_adapter_reduction"] == 4


def test_hi_afa_lite_model_has_exact_adapter_contract_and_parameter_count() -> None:
    """Build the V20 treatment and guard its architecture metadata."""
    model = csl_tinyvit_7m(
        **_v20_model_kwargs(),
        reid_adapter_stages=(3,),
        reid_adapter_reduction=4,
        reid_adapter_suppression_tau=0.7,
    )

    assert model.reid_adapter_stages == (3,)
    assert model.reid_adapter_reduction == 4
    assert model.reid_adapter_suppression_tau == pytest.approx(0.7)
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_268_373


def test_zero_gated_adapters_preserve_same_seed_v20_shared_initialization() -> None:
    """Keep the treatment bit-exact to V20 outside its private adapter state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        baseline = csl_tinyvit_7m(**_v20_model_kwargs())
        torch.manual_seed(2026)
        treatment = csl_tinyvit_7m(
            **_v20_model_kwargs(),
            reid_adapter_stages=(3,),
            reid_adapter_reduction=4,
            reid_adapter_suppression_tau=0.7,
        )

    baseline_state = baseline.state_dict()
    treatment_state = treatment.state_dict()
    extra_keys = treatment_state.keys() - baseline_state.keys()
    assert extra_keys
    assert all(".reid_adapters." in key for key in extra_keys)
    mismatched = [
        key
        for key, value in baseline_state.items()
        if not torch.equal(value, treatment_state[key])
    ]
    assert mismatched == []

    baseline.eval()
    treatment.eval()
    generator = torch.Generator().manual_seed(73)
    inputs = torch.randn(1, 3, 384, 128, generator=generator)
    with torch.no_grad():
        baseline_descriptor = baseline(inputs)
        treatment_descriptor = treatment(inputs)
    torch.testing.assert_close(
        treatment_descriptor,
        baseline_descriptor,
        rtol=0,
        atol=0,
    )


def test_hi_afa_lite_ablation_metadata_records_suppression_tau() -> None:
    plan = resolve_csl_tinyvit_ablation(
        {
            "model_name": "csl_tinyvit_7m",
            "head_type": "standard",
            "reid_adapter_stages": (3,),
            "reid_adapter_reduction": 4,
            "reid_adapter_suppression_tau": 0.7,
        }
    )

    treatment = next(
        addon
        for addon in plan.addons
        if addon.spec.name == "architecture.reid_adapters"
    )
    assert treatment.settings == {
        "reid_adapter_stages": (3,),
        "reid_adapter_reduction": 4,
        "reid_adapter_suppression_tau": pytest.approx(0.7),
    }


def test_hi_afa_lite_recipe_resolves_through_train_cli(monkeypatch) -> None:
    """Exercise the named-recipe path used by the launch script."""
    captured = {}

    def fake_main(args) -> None:
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
            "csl_tinyvit_7m_hi_afa_lite",
            "--data-dir",
            ".",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_7m_v20"
    assert args.num_workers == 4
    assert args.reid_adapter_stages == (3,)
    assert args.reid_adapter_reduction == 4
    assert args.reid_adapter_suppression_tau == pytest.approx(0.7)
    assert args.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert args.feat_dim == args.neck_dim == 384
    assert args.head_parts == (1, 2, 4)
    assert args.scale_balanced_branches is True
    assert args.anatomical_auxiliary is True
    assert args.anatomical_target_type == "learned_pose_concat_ema"
    assert args.p_ids == 12
    assert args.k_instances == 8
    assert args.flip_tta is False

    trainer_values = trainer_kwargs_from_args(args)
    config = ReIDTrainConfig.from_flat_kwargs(**trainer_values)
    assert config.model.reid_adapter_stages == (3,)
    assert config.model.reid_adapter_reduction == 4
    assert config.model.reid_adapter_suppression_tau == pytest.approx(0.7)
    assert config.to_trainer_kwargs()["reid_adapter_suppression_tau"] == pytest.approx(0.7)
