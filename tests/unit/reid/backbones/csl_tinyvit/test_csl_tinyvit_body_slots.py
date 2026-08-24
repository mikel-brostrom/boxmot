"""Regression tests for the privileged recurrent BodySlot descriptor."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from boxmot.engine.config import BOXMOT_DEFAULTS
from boxmot.reid.backbones.anatomical_registry import (
    get_anatomical_target_spec,
)
from boxmot.reid.backbones.families.csl_tinyvit.blocks import (
    BodySlotReadWrite,
)
from boxmot.reid.backbones.families.csl_tinyvit.variants import (
    csl_tinyvit_11m,
)
from boxmot.reid.backbones.head_registry import (
    HeadImplementation,
    get_reid_head_spec,
)
from boxmot.reid.training.model_options import (
    REID_MODEL_OPTION_GROUPS,
    build_reid_model_kwargs,
)
from boxmot.reid.training.trainer import ReIDTrainer


def _body_slot_model(
    *,
    mode: str = "recurrent_read",
    num_classes: int = 8,
):
    return csl_tinyvit_11m(
        num_classes=num_classes,
        loss="triplet",
        pretrained=False,
        img_size=(96, 32),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        attention_window_layout="rect",
        attention_mask=True,
        interpolate_pretrained_attention_bias=True,
        head_pool="gelu_gem",
        head_parts=(1, 2, 4),
        head_type="body_slot",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=True,
        anatomical_multiscale=True,
        anatomical_target_type="body_slot_privileged_ema",
        body_slot_mode=mode,
    )


def _targets(batch: int = 4, height: int = 96, width: int = 32):
    targets = {
        "masks": torch.zeros(batch, 6, height, width),
        "foreground_mask": torch.zeros(batch, 1, height, width),
        "accessory_mask": torch.zeros(batch, 1, height, width),
    }
    targets["masks"][:, 0, :20] = 1
    targets["masks"][:, 1, 20:52] = 1
    targets["masks"][:, 2, 24:60, :8] = 1
    targets["masks"][:, 3, 24:60, -8:] = 1
    targets["masks"][:, 4, 52:, :16] = 1
    targets["masks"][:, 5, 52:, 16:] = 1
    targets["foreground_mask"] = (
        targets["masks"].amax(dim=1, keepdim=True) > 0
    ).float()
    targets["accessory_mask"][:, :, 28:52, 20:30] = 1
    return targets


def _loss_trainer() -> ReIDTrainer:
    trainer = object.__new__(ReIDTrainer)
    settings = {
        "anatomical_auxiliary": True,
        "anatomical_distill_weight": 0.10,
        "anatomical_attention_weight": 0.10,
        "anatomical_visibility_weight": 0.05,
        "anatomical_query_diversity_weight": 0.02,
        "anatomical_foreground_weight": 0.10,
        "anatomical_part_triplet_weight": 0.10,
        "anatomical_query_diversity_margin": 0.10,
        "anatomical_branch_global_coefficient": 0.20,
        "anatomical_branch_coarse_coefficient": 0.30,
        "anatomical_branch_fine_coefficient": 0.50,
        "anatomical_cross_scale_weight": 0.05,
        "anatomical_student_start_epoch": 0,
        "anatomical_student_ramp_end_epoch": 0,
        "anatomical_decay_start_epoch": 0,
        "anatomical_decay_end_epoch": 0,
        "margin": 0.30,
    }
    for name, value in settings.items():
        setattr(trainer, name, value)
    return trainer


def test_body_slot_registry_and_model_options_are_canonical():
    train_defaults = BOXMOT_DEFAULTS.train
    spec = get_reid_head_spec("body_slot", family="csl_tinyvit")
    assert spec.implementation == HeadImplementation.BODY_SLOT
    assert get_anatomical_target_spec(
        "body_slot_privileged_ema"
    ).uses_body_slots
    assert train_defaults.body_slot_mode == "recurrent_read"

    option_names = {
        option.source or option.kwarg
        for group in REID_MODEL_OPTION_GROUPS
        for option in group.options
    }
    options = SimpleNamespace(
        **{
            name: getattr(train_defaults, name, False)
            for name in option_names
        }
    )
    kwargs = build_reid_model_kwargs(options)
    assert kwargs["body_slot_mode"] == "recurrent_read"
    assert kwargs["body_slot_alpha"] == 0.45
    assert kwargs["body_slot_visibility_floor"] == 0.05


def test_zero_gated_body_slot_writeback_preserves_spatial_path_exactly():
    block = BodySlotReadWrite(
        32,
        slot_dim=16,
        num_slots=8,
        num_heads=4,
        writeback=True,
        gate_init=0.0,
    )
    spatial = torch.randn(2, 12, 32, requires_grad=True)
    slots = torch.randn(2, 8, 16, requires_grad=True)
    roles = torch.randn(1, 8, 16)
    masks = torch.ones(2, 8, 12, 4)

    output = block(
        spatial,
        (3, 4),
        slots,
        roles,
        teacher_masks=masks,
    )
    spatial_output, updated_slots = output[:2]

    torch.testing.assert_close(
        spatial_output,
        spatial,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(updated_slots, slots)
    assert output[3].shape == (2, 8, 12)
    assert output[4].shape == (2, 8, 16)
    assert output[5].all()
    assert not any(
        "teacher_" in name for name, _ in block.named_parameters()
    )
    updated_slots.square().mean().backward()
    assert spatial.grad is not None
    assert slots.grad is not None


def test_body_slot_model_replaces_stripes_with_an_rgb_only_1536d_descriptor():
    model = _body_slot_model()
    trainer = _loss_trainer()
    targets = _targets()
    teacher_masks = trainer._body_slot_teacher_masks(
        targets,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert teacher_masks.shape == (4, 8, 96, 32)
    assert teacher_masks[:, 6].sum() > 0

    model.train()
    logits, features = model(
        torch.randn(4, 3, 96, 32),
        anatomical_query_masks=teacher_masks,
    )
    assert len(logits) == 9
    assert features["raw_concat"].shape == (4, 1536)
    assert len(features["_body_slot_stage_slots"]) == 3
    assert [
        attention.shape[-1]
        for attention in features["_body_slot_stage_attentions"]
    ] == [48, 12, 12]
    assert len(features["_body_slot_teacher_slots"]) == 3

    model.eval()
    with torch.no_grad():
        descriptor = model(torch.randn(2, 3, 96, 32))
    assert descriptor.shape == (2, 1536)
    torch.testing.assert_close(
        descriptor.norm(dim=1),
        torch.ones(2),
        rtol=1e-5,
        atol=1e-5,
    )


def test_recurrent_read_slots_preserve_the_v8_global_feature_path_exactly():
    common = {
        "num_classes": 8,
        "loss": "triplet",
        "pretrained": False,
        "img_size": (96, 32),
        "feature_fusion": (
            "global_final_parts_stage0_semantic_fine"
        ),
        "spatial_conv_mode": "depthwise_separable",
        "attention_window_layout": "rect",
        "attention_mask": True,
        "interpolate_pretrained_attention_bias": True,
        "head_pool": "gelu_gem",
        "head_parts": (1, 2, 4),
        "inference_feature": "norm_concat_bn",
        "scale_balanced_branches": True,
    }
    stripe_model = csl_tinyvit_11m(**common)
    body_slot_model = _body_slot_model()
    body_state = body_slot_model.state_dict()
    shared_backbone = {
        name: value
        for name, value in stripe_model.state_dict().items()
        if not name.startswith("head.")
        and name in body_state
        and body_state[name].shape == value.shape
    }
    body_slot_model.load_state_dict(shared_backbone, strict=False)
    body_slot_model.head.global_pool.load_state_dict(
        stripe_model.head.global_pool.state_dict()
    )
    body_slot_model.head.global_neck.load_state_dict(
        stripe_model.head.bn_global.state_dict()
    )

    stripe_model.eval()
    body_slot_model.eval()
    images = torch.randn(2, 3, 96, 32)
    with torch.no_grad():
        stripe_global = stripe_model.forward_features(images)[0]
        body_slot_global = body_slot_model.forward_features(
            images
        ).global_map
        stripe_descriptor = stripe_model.head.bn_global(
            stripe_model.head.global_pool(stripe_global)
        )[0]
        body_slot_descriptor = body_slot_model.head.global_neck(
            body_slot_model.head.global_pool(body_slot_global)
        )[0]
    torch.testing.assert_close(
        body_slot_global,
        stripe_global,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        body_slot_descriptor,
        stripe_descriptor,
        rtol=0,
        atol=0,
    )


def test_body_slot_privileged_loss_is_finite_and_reaches_recurrent_slots():
    model = _body_slot_model()
    trainer = _loss_trainer()
    targets = _targets()
    teacher_masks = trainer._body_slot_teacher_masks(
        targets,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model.train()
    _, features = model(
        torch.randn(4, 3, 96, 32),
        anatomical_query_masks=teacher_masks,
    )
    loss, components = trainer._body_slot_privileged_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        epoch=None,
        return_components=True,
    )

    assert torch.isfinite(loss)
    assert loss > 0
    assert components["distill"] > 0
    assert components["attention"] > 0
    assert components["visibility"] > 0
    assert components["query_diversity"] > 0
    assert components["cross_scale"] > 0
    assert components["part_triplet"] >= 0
    loss.backward()
    assert model.body_slot_seed.grad is not None
    assert (
        model.body_slot_modules[0].memory_projection.weight.grad
        is not None
    )
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_body_slot_parameter_budget_and_tier_c_zero_gates():
    common = {
        "num_classes": 751,
        "loss": "triplet",
        "pretrained": False,
        "img_size": (96, 32),
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "spatial_conv_mode": "depthwise_separable",
        "attention_window_layout": "rect",
        "attention_mask": True,
        "head_pool": "gelu_gem",
        "head_parts": (1, 2, 4),
        "inference_feature": "norm_concat_bn",
        "scale_balanced_branches": True,
    }
    stripe = csl_tinyvit_11m(**common)
    tier_b = _body_slot_model(num_classes=751)
    tier_c = _body_slot_model(
        mode="recurrent_read_write",
        num_classes=751,
    )

    stripe_count = sum(parameter.numel() for parameter in stripe.parameters())
    tier_b_count = sum(parameter.numel() for parameter in tier_b.parameters())
    tier_c_count = sum(parameter.numel() for parameter in tier_c.parameters())
    assert tier_b_count > stripe_count
    assert tier_c_count > tier_b_count
    assert tier_b_count - stripe_count < 700_000
    assert tier_c_count - tier_b_count < 700_000
    assert all(
        module.broadcast_gate.item() == 0.0
        for module in tier_c.body_slot_modules
    )
    assert all(
        not module.memory_norm.elementwise_affine
        and module.memory_norm.weight is None
        and module.memory_norm.bias is None
        and not module.attention_memory_norm.elementwise_affine
        and module.attention_memory_norm.weight is None
        and module.attention_memory_norm.bias is None
        for module in tier_b.body_slot_modules
    )
    for name in (
        "body_slot_seed",
        "body_slot_roles",
        "body_slot_modules.0.memory_projection.weight",
    ):
        assert ReIDTrainer._is_reid_adaptation_param(name)
        assert ReIDTrainer._is_head_or_neck_param(name)
