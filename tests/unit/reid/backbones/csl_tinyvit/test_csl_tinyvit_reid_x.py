"""Regression tests for the CSL-TinyViT-ReID-X ablation."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from boxmot.reid.backbones.families.csl_tinyvit.blocks import (
    IdentityRegisterCommunication,
    PatchMerging,
)
from boxmot.reid.backbones.families.csl_tinyvit.model import CSLTinyViT
from boxmot.reid.backbones.families.csl_tinyvit.variants import (
    csl_tinyvit_11m,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.training.config import ReIDTrainConfig
from boxmot.reid.training.resume import contract_differences
from boxmot.reid.training.trainer import ReIDTrainer


def _reid_x_model(
    *,
    registers: bool = False,
    width_first: bool = True,
) -> CSLTinyViT:
    """Build a channel-reduced model with production spatial geometry."""
    stage2_depth = 5 if width_first else 6
    stage3_depth = 3 if width_first else 2
    return CSLTinyViT(
        num_classes=10,
        loss="triplet",
        pretrained=False,
        img_size=(384, 128),
        embed_dims=[8, 16, 20, 40],
        depths=[1, 2, stage2_depth, stage3_depth],
        num_heads=[1, 2, 2, 4],
        attention_window_layout="rect",
        attention_mask=True,
        interpolate_pretrained_attention_bias=True,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        head_pool="gelu_gem",
        scale_balanced_branches=True,
        neck_dim=16,
        feat_dim=16,
        inference_feature="norm_concat_bn",
        stage2_depth=stage2_depth,
        stage3_depth=stage3_depth,
        width_first_hierarchy=width_first,
        identity_registers=registers,
        identity_register_count=4,
        identity_register_dim=128,
        identity_register_num_heads=4,
        identity_register_dropout=0.10,
        identity_register_gate_init=0.0,
    )


def test_patch_merging_can_reduce_height_without_reducing_width():
    merge = PatchMerging(
        input_resolution=(48, 8),
        dim=8,
        out_dim=16,
        activation=nn.GELU,
        stride=(2, 1),
    )

    tokens, size = merge(
        torch.randn(2, 48 * 8, 8),
        (48, 8),
    )

    assert size == (24, 8)
    assert tokens.shape == (2, 24 * 8, 16)


def test_width_first_hierarchy_preserves_the_requested_geometry():
    model = _reid_x_model().eval()
    transitions = []
    hooks = []

    for stage_index, layer in enumerate(model.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda _module, inputs, output, index=stage_index: (
                    transitions.append(
                        (index, inputs[1], output[1])
                    )
                )
            )
        )
    hooks.append(
        model.stage1_width_merge.register_forward_hook(
            lambda _module, inputs, output: transitions.append(
                ("width", inputs[1], output[1])
            )
        )
    )
    with torch.no_grad():
        descriptor = model(torch.randn(1, 3, 384, 128))
    for hook in hooks:
        hook.remove()

    assert transitions == [
        (0, (96, 32), (48, 16)),
        ("width", (48, 16), (48, 8)),
        (1, (48, 8), (24, 8)),
        (2, (24, 8), (24, 8)),
        (3, (24, 8), (24, 8)),
    ]
    assert [
        block.window_size for block in model.layers[1].blocks
    ] == [(12, 4), (16, 4)]
    assert descriptor.shape == (1, 3 * 16)


def test_zero_gated_registers_begin_from_the_exact_x1_descriptor():
    inputs = torch.randn(2, 3, 384, 128)
    torch.manual_seed(123)
    x1 = _reid_x_model(registers=False).eval()
    torch.manual_seed(123)
    x2 = _reid_x_model(registers=True).eval()

    with torch.no_grad():
        x1_descriptor = x1(inputs)
        x2_descriptor = x2(inputs)

    assert torch.equal(x1_descriptor, x2_descriptor)
    assert all(
        module.broadcast_gate.item() == 0.0
        for module in x2.identity_register_modules
    )


def test_zero_gated_registers_begin_from_the_exact_v8_descriptor():
    inputs = torch.randn(2, 3, 384, 128)
    torch.manual_seed(123)
    v8 = _reid_x_model(
        registers=False,
        width_first=False,
    ).eval()
    torch.manual_seed(123)
    treatment = _reid_x_model(
        registers=True,
        width_first=False,
    ).eval()

    with torch.no_grad():
        v8_descriptor = v8(inputs)
        treatment_descriptor = treatment(inputs)

    assert torch.equal(v8_descriptor, treatment_descriptor)
    assert treatment.stage2_depth == 6
    assert treatment.stage3_depth == 2
    assert treatment.stage1_width_merge is None


def test_register_communication_uses_the_lightweight_128d_bottleneck():
    modules = nn.ModuleList(
        [
            IdentityRegisterCommunication(
                448,
                register_dim=128,
                num_registers=4,
                num_heads=4,
                window_size=(12, 4),
                dropout=0.10,
                gate_init=0.0,
            )
            for _ in range(2)
        ]
    )
    register_seed_parameters = 4 * 128
    added_parameters = sum(
        parameter.numel() for parameter in modules.parameters()
    ) + register_seed_parameters

    assert added_parameters == 628_098
    assert all(module.spatial_dim == 448 for module in modules)
    assert all(module.register_dim == 128 for module in modules)
    assert all(
        not module.summary_norm.elementwise_affine
        for module in modules
    )
    assert all(
        module.summary_projection.weight.shape == (128, 448)
        for module in modules
    )
    assert all(
        module.broadcast_projection.weight.shape == (448, 128)
        for module in modules
    )


def test_registers_receive_retrieval_and_diversity_gradients():
    model = _reid_x_model(registers=True).train()
    scores, features = model(torch.randn(2, 3, 384, 128))
    register_tokens = features["_identity_register_tokens"]
    trainer = ReIDTrainer.__new__(ReIDTrainer)
    trainer.device = torch.device("cpu")
    trainer.identity_registers = True
    trainer.identity_register_count = 4
    trainer.identity_register_diversity_margin = -1.0

    diversity = trainer._identity_register_diversity_loss(
        features
    )
    classification = sum(score.square().mean() for score in scores)
    (classification + diversity).backward()

    assert len(register_tokens) == 2
    assert all(tokens.shape == (2, 4, 128) for tokens in register_tokens)
    assert diversity.item() > 0
    assert model.identity_register_seed.grad is not None
    assert model.identity_register_seed.grad.abs().sum().item() > 0
    for module in model.identity_register_modules:
        assert module.broadcast_gate.grad is not None
        assert module.broadcast_gate.grad.abs().item() > 0
        assert (
            module.register_attention.in_proj_weight.grad is not None
        )


def test_registers_are_reid_modules_during_backbone_warm_start(
    tmp_path,
):
    model = _reid_x_model(registers=True)
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
    )

    trainer._set_backbone_freeze_trainability(model, True)

    assert model.identity_register_seed.requires_grad
    assert all(
        parameter.requires_grad
        for module in model.identity_register_modules
        for parameter in module.parameters()
    )
    assert trainer._is_head_or_neck_param("identity_register_seed")
    assert trainer._is_head_or_neck_param(
        "identity_register_modules.0.broadcast_gate"
    )
    assert not model.patch_embed.seq[0].c.weight.requires_grad
    assert not model.layers[1].blocks[0].attn.qkv.weight.requires_grad


def test_register_seed_uses_head_lr_without_weight_decay(tmp_path):
    model = _reid_x_model(registers=True)
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        weight_decay=0.1,
    )
    groups = trainer._build_vit_param_groups(model)
    seed_group = next(
        group
        for group in groups
        if any(
            parameter is model.identity_register_seed
            for parameter in group["params"]
        )
    )

    assert seed_group["lr_scale"] == 1.0
    assert seed_group["weight_decay"] == 0.0
    assert seed_group["is_head"] is True
    assert seed_group["is_backbone"] is False


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is required for the anisotropic-backward regression",
)
def test_width_first_register_backward_is_mps_safe():
    model = _reid_x_model(registers=True).to("mps").train()
    scores, features = model(
        torch.randn(2, 3, 384, 128, device="mps")
    )
    loss = sum(score.square().mean() for score in scores)
    loss = loss + 0.01 * sum(
        tokens.float().square().mean()
        for tokens in features["_identity_register_tokens"]
    )

    loss.backward()
    torch.mps.synchronize()

    assert all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_reid_x_configuration_round_trips_and_resume_is_strict():
    config = ReIDTrainConfig.from_flat_kwargs(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir="Market-1501-v15.09.15",
        width_first_hierarchy=True,
        identity_registers=True,
        identity_register_count=4,
        identity_register_dim=128,
        identity_register_num_heads=4,
        identity_register_dropout=0.10,
        identity_register_gate_init=0.0,
        identity_register_diversity_weight=0.01,
        identity_register_diversity_margin=0.10,
    )
    assert config.model.width_first_hierarchy is True
    assert config.model.identity_registers is True
    assert config.model.identity_register_dim == 128
    assert config.model.identity_register_diversity_weight == 0.01

    current = {
        "model": {
            "width_first_hierarchy": False,
            "identity_registers": False,
            "identity_register_count": 4,
            "identity_register_dim": 128,
            "identity_register_num_heads": 4,
            "identity_register_dropout": 0.10,
            "identity_register_gate_init": 0.0,
            "identity_register_diversity_weight": 0.0,
            "identity_register_diversity_margin": 0.10,
        }
    }
    historical = {"model": {}}
    assert contract_differences(historical, current) == []

    enabled = copy.deepcopy(current)
    enabled["model"]["width_first_hierarchy"] = True
    enabled["model"]["identity_registers"] = True
    differences = contract_differences(historical, enabled)
    assert (
        "model.width_first_hierarchy: saved='<missing>', requested=True"
        in differences
    )
    assert (
        "model.identity_registers: saved='<missing>', requested=True"
        in differences
    )
    assert (
        "model.identity_register_dim: saved='<missing>', requested=128"
        in differences
    )


def test_register_bottleneck_checkpoint_reconstructs_strictly(tmp_path):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        pretrained=False,
        img_size=(384, 128),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        attention_window_layout="rect",
        attention_mask=True,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        head_pool="gelu_gem",
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        width_first_hierarchy=True,
        stage2_depth=5,
        stage3_depth=3,
        identity_registers=True,
        identity_register_count=4,
        identity_register_dim=128,
        identity_register_num_heads=4,
        identity_register_dropout=0.10,
        identity_register_gate_init=0.0,
        identity_register_diversity_weight=0.01,
        identity_register_diversity_margin=0.10,
        neck_dim=32,
        feat_dim=16,
    )
    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)
    weights = tmp_path / "reid_x_d128.pt"
    torch.save(
        {**metadata, "state_dict": model.state_dict()},
        weights,
    )

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)
    reconstructed = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        **kwargs,
    )
    reconstructed.load_state_dict(model.state_dict(), strict=True)

    assert metadata["identity_register_dim"] == 128
    assert (
        metadata["model"]["transformer"]["identity_registers"]["dim"]
        == 128
    )
    assert kwargs["identity_register_dim"] == 128
    assert reconstructed.identity_register_seed.shape == (1, 4, 128)
