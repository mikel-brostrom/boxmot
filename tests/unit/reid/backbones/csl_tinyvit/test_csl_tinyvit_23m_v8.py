"""Checks for the width-adapted V8 recipe on CSL-TinyViT 23M."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from click.testing import CliRunner

from boxmot.engine.config import load_training_recipe
from boxmot.engine.cli import boxmot
from boxmot.reid.backbones.families.csl_tinyvit import csl_tinyvit_23m


def _model(*, anatomical_auxiliary: bool):
    return csl_tinyvit_23m(
        num_classes=751,
        loss="triplet",
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        pyramid_resize_mode="bilinear",
        spatial_conv_mode="depthwise_separable",
        feat_dim=640,
        neck_dim=640,
        head_pool="gelu_gem",
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=anatomical_auxiliary,
        anatomical_token_dim=160,
        anatomical_multiscale=anatomical_auxiliary,
        anatomical_target_type="learned_pose_concat_ema",
        attention_window_layout="rect",
        attention_bias="absolute",
        interpolate_pretrained_attention_bias=True,
        attention_mask=True,
        drop_path_rate=0.2,
    )


def test_23m_v8_recipe_preserves_policy_with_scaled_widths():
    recipe = load_training_recipe("csl_tinyvit_23m_v8")

    expected = {
        "model": "csl_tinyvit_23m",
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "spatial_conv_mode": "depthwise_separable",
        "feat_dim": 640,
        "neck_dim": 640,
        "head_parts": [1, 2, 4],
        "part_pooling": "stripes",
        "scale_balanced_branches": True,
        "metric_feature": "raw_concat",
        "inference_feature": "norm_concat_bn",
        "attention_window_layout": "rect",
        "attention_bias": "absolute",
        "interpolate_pretrained_attention_bias": True,
        "attention_mask": True,
        "attention_shift": False,
        "stage3_downsample": False,
        "stage2_depth": 6,
        "stage3_depth": 2,
        "drop_path_rate": 0.2,
        "lr": 7e-4,
        "weight_decay": 0.1,
        "vit_lr_profile": "layer_decay",
        "backbone_freeze_epochs": 10,
        "warmup_epochs": 20,
        "label_smooth": 0.05,
        "triplet_soft_margin": True,
        "center_loss_weight": 0.005,
        "random_crop_scale": 1.05,
        "random_erasing": 0.5,
        "random_patch": True,
        "color_jitter": True,
        "gaussian_blur": True,
        "random_grayscale": 0.1,
        "anatomical_auxiliary": True,
        "anatomical_token_dim": 160,
        "anatomical_target_type": "learned_pose_concat_ema",
        "anatomical_multiscale": True,
        "anatomical_teacher_momentum": 0.999,
        "anatomical_distill_weight": 0.10,
        "anatomical_attention_weight": 0.10,
        "anatomical_visibility_weight": 0.05,
        "anatomical_contrastive_weight": 0.10,
        "anatomical_pose_teacher_weight": 0.03,
        "anatomical_local_scale_weight": 0.60,
        "anatomical_fine_scale_weight": 0.40,
        "anatomical_cross_scale_weight": 0.05,
        "anatomical_student_start_epoch": 0,
        "anatomical_student_ramp_end_epoch": 50,
        "anatomical_decay_start_epoch": 120,
        "anatomical_decay_end_epoch": 170,
        "anatomical_temperature": 0.07,
        "p_ids": 12,
        "k_instances": 8,
        "epochs": 200,
        "flip_tta": False,
    }
    for key, value in expected.items():
        assert recipe[key] == value, (key, recipe[key], value)


def test_23m_v8_model_balances_1920d_scale_power():
    model = _model(anatomical_auxiliary=True)

    assert [layer.dim for layer in model.layers] == [96, 192, 384, 576]
    assert [layer.depth for layer in model.layers] == [2, 2, 6, 2]
    assert model.feature_fusion_module.stage_indices == (1, 2, 0)
    assert model.feature_fusion_module.local_channels == 640
    assert model.feature_fusion_module.fine_output_channels == 640
    assert model.head.bn_global.reduction.out_channels == 640
    assert model.head.bn_part0.reduction.out_channels == 320
    assert model.head.bn_part1.reduction.out_channels == 320
    assert all(
        getattr(model.head, f"bn_part{index}").reduction.out_channels == 160
        for index in range(2, 6)
    )
    assert model.head.anatomical_auxiliary_pool.token_dim == 160
    assert sum(parameter.numel() for parameter in model.parameters()) == 25_518_147


def test_23m_pose_teacher_is_training_only_and_preserves_rgb_initialization():
    torch.manual_seed(23)
    control = _model(anatomical_auxiliary=False)
    torch.manual_seed(23)
    treatment = _model(anatomical_auxiliary=True)

    treatment_state = treatment.state_dict()
    assert all(
        torch.equal(value, treatment_state[key])
        for key, value in control.state_dict().items()
    )
    control.eval()
    treatment.eval()
    image = torch.randn(1, 3, 384, 128)
    with torch.inference_mode():
        control_descriptor = control(image)
        treatment_descriptor = treatment(image)

    assert control_descriptor.shape == treatment_descriptor.shape == (1, 1920)
    assert torch.equal(control_descriptor, treatment_descriptor)
    assert torch.allclose(
        treatment_descriptor.norm(dim=1),
        torch.ones(1),
        atol=1e-5,
    )


def test_23m_v8_cli_recipe_resolves_training_contract(monkeypatch):
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
            "csl_tinyvit_23m_v8",
            "--data-dir",
            ".",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_23m"
    assert args.feat_dim == 640
    assert args.neck_dim == 640
    assert args.anatomical_token_dim == 160
    assert args.anatomical_auxiliary is True
    assert args.anatomical_multiscale is True
    assert args.p_ids == 12
    assert args.k_instances == 8
    assert args.flip_tta is False
