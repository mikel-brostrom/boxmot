"""Tests for model-agnostic human-centric encoder pretraining primitives."""

from __future__ import annotations

import pytest
import torch

from boxmot.reid.backbones.families.csl_tinyvit import csl_tinyvit_7m_v20
from boxmot.reid.backbones.families.csl_tinyvit import pretrained as csl_tinyvit_pretrained
from boxmot.reid.training.human_pretraining import (
    export_tinyvit_backbone_checkpoint,
    foreground_aware_patch_target_weights,
    normalize_part_maps,
    pose_parser_guided_whole_part_mask,
    semantic_teacher_feature_reconstruction_loss,
    two_view_masked_consistency_loss,
)


def test_part_map_normalization_is_finite_and_keeps_empty_targets_zero():
    maps = torch.tensor(
        [
            [
                [[1.0, 0.0], [float("nan"), 2.0]],
                [[1.0, 3.0], [float("inf"), -2.0]],
            ],
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ],
        dtype=torch.float16,
    )
    foreground = torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
        ]
    )

    normalized = normalize_part_maps(maps, foreground)

    assert normalized.dtype == torch.float16
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized[0, :, 0, 0].float(), torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(normalized[0, :, 0, 1].float(), torch.tensor([0.0, 1.0]))
    assert normalized[0, :, 1].count_nonzero() == 0
    assert normalized[1].count_nonzero() == 0

    spatial = normalize_part_maps(maps.float(), normalization="spatial")
    torch.testing.assert_close(spatial[0, 0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(spatial[0, 1].sum(), torch.tensor(1.0))
    assert spatial[1].count_nonzero() == 0


def test_whole_part_mask_is_generator_deterministic_and_skips_empty_samples():
    part_maps = torch.zeros(2, 3, 3, 6)
    for part_index in range(3):
        start = 2 * part_index
        part_maps[0, part_index, :, start : start + 2] = 1.0

    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)
    first = pose_parser_guided_whole_part_mask(
        part_maps,
        mask_ratio=0.5,
        generator=generator_a,
    )
    torch.manual_seed(999)
    second = pose_parser_guided_whole_part_mask(
        part_maps,
        mask_ratio=0.5,
        generator=generator_b,
    )

    assert first is not None and second is not None
    assert torch.equal(first.selected_parts, second.selected_parts)
    assert torch.equal(first.pixel_mask, second.pixel_mask)
    assert first.selected_parts[0].sum() == 2
    assert first.pixel_mask[0].sum() == 12
    assert first.valid_samples.tolist() == [True, False]
    assert first.pixel_mask[1].count_nonzero() == 0

    masked = first.apply(torch.ones(2, 3, 6, 12), fill_value=-1.0)
    resized_mask = torch.nn.functional.interpolate(first.pixel_mask.float(), size=(6, 12), mode="nearest").bool()
    assert torch.all(masked.masked_select(resized_mask.expand_as(masked)) == -1)
    assert torch.all(masked.masked_select(~resized_mask.expand_as(masked)) == 1)


def test_whole_part_mask_missing_target_policy_is_explicit():
    assert (
        pose_parser_guided_whole_part_mask(
            None,
            mask_ratio=0.5,
            missing_target="skip",
        )
        is None
    )
    with pytest.raises(ValueError, match="part maps are required"):
        pose_parser_guided_whole_part_mask(
            None,
            mask_ratio=0.5,
            missing_target="error",
        )


def test_foreground_patch_weights_track_coverage_and_handle_missing_masks():
    foreground = torch.zeros(1, 1, 4, 4)
    foreground[:, :, :, :2] = 1.0

    weights = foreground_aware_patch_target_weights(
        foreground,
        (2, 2),
        foreground_weight=3.0,
        background_weight=1.0,
    )

    torch.testing.assert_close(
        weights,
        torch.tensor([[[1.5, 0.5], [1.5, 0.5]]]),
    )
    uniform = foreground_aware_patch_target_weights(
        None,
        (2, 3),
        batch_size=2,
        missing_target="uniform",
    )
    skipped = foreground_aware_patch_target_weights(
        None,
        (2, 3),
        batch_size=2,
        missing_target="skip",
    )
    assert torch.equal(uniform, torch.ones(2, 2, 3))
    assert skipped.count_nonzero() == 0
    with pytest.raises(ValueError, match="foreground_mask is required"):
        foreground_aware_patch_target_weights(
            None,
            (2, 3),
            batch_size=2,
            missing_target="error",
        )


def test_two_view_consistency_uses_only_masked_tokens_and_backpropagates():
    view_a = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        requires_grad=True,
    )
    view_b = view_a.detach().clone()
    view_b[:, 0] = torch.tensor([0.0, 1.0])
    view_b.requires_grad_()

    selected_loss = two_view_masked_consistency_loss(
        view_a,
        view_b,
        torch.tensor([[1.0, 0.0, 0.0]]),
    )
    ignored_loss = two_view_masked_consistency_loss(
        view_a,
        view_b,
        torch.tensor([[0.0, 1.0, 1.0]]),
    )

    assert selected_loss.item() == pytest.approx(1.0)
    assert ignored_loss.item() == pytest.approx(0.0, abs=1e-6)
    selected_loss.backward()
    assert view_a.grad is not None and torch.isfinite(view_a.grad).all()
    assert view_b.grad is not None and torch.isfinite(view_b.grad).all()

    empty_loss = two_view_masked_consistency_loss(
        view_a,
        view_b,
        torch.zeros(1, 3),
    )
    assert empty_loss.item() == 0.0
    assert torch.isfinite(empty_loss)


def test_semantic_teacher_reconstruction_detaches_teacher_and_combines_weights():
    student = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        requires_grad=True,
    )
    teacher = student.detach().clone()
    teacher[:, 0] = torch.tensor([0.0, 1.0])
    teacher.requires_grad_()

    loss = semantic_teacher_feature_reconstruction_loss(
        student,
        teacher,
        torch.tensor([[1.0, 1.0, 0.0]]),
        target_weights=torch.tensor([[2.0, 0.0, 10.0]]),
    )

    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None

    missing = semantic_teacher_feature_reconstruction_loss(
        student,
        None,
        missing_target="skip",
    )
    assert missing.item() == 0.0
    with pytest.raises(ValueError, match="teacher features are required"):
        semantic_teacher_feature_reconstruction_loss(
            student,
            None,
            missing_target="error",
        )


def test_semantic_reconstruction_pools_spatial_masks_to_feature_grid():
    student = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], requires_grad=True)
    teacher = student.detach().clone()
    teacher[:, :, :, 0] = torch.tensor([[[0.0], [1.0]]])
    high_resolution_mask = torch.zeros(1, 1, 2, 4)
    high_resolution_mask[:, :, :, :2] = 1.0

    loss = semantic_teacher_feature_reconstruction_loss(
        student,
        teacher,
        high_resolution_mask,
        channel_dim=1,
    )

    assert loss.item() == pytest.approx(1.0)


def test_backbone_export_is_loadable_by_existing_tinyvit_loader(tmp_path, monkeypatch):
    source = csl_tinyvit_7m_v20(
        num_classes=4,
        loss="triplet",
        pretrained=False,
    )
    wrapped_state = {f"encoder.{key}": value for key, value in source.state_dict().items()}
    wrapped_state["decoder.weight"] = torch.randn(2, 2)
    output = tmp_path / "human_pretrained_tinyvit.pt"

    returned = export_tinyvit_backbone_checkpoint(
        {"model_state_dict": wrapped_state},
        output,
        source_prefix="encoder.",
        metadata={"objective": "human-centric-masked-pretraining"},
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)

    assert returned == output
    assert checkpoint["format"] == "boxmot-tinyvit-backbone-v1"
    assert checkpoint["metadata"]["objective"] == "human-centric-masked-pretraining"
    assert checkpoint["state_dict"]
    assert all(key.startswith(("patch_embed.", "layers.")) for key in checkpoint["state_dict"])
    assert not any("head" in key or "decoder" in key for key in checkpoint["state_dict"])

    monkeypatch.setattr(
        csl_tinyvit_pretrained,
        "load_hub_checkpoint",
        lambda *args, **kwargs: checkpoint["state_dict"],
    )
    target = csl_tinyvit_7m_v20(
        num_classes=4,
        loss="triplet",
        pretrained=False,
    )
    csl_tinyvit_pretrained.load_pretrained_tinyvit(target, "test://human-pretrained")

    assert target.pretrained_backbone_tensor_coverage == 1.0
    assert target.pretrained_backbone_numel_coverage == 1.0
    required = csl_tinyvit_pretrained._required_pretrained_keys(target.state_dict())
    source_state = source.state_dict()
    target_state = target.state_dict()
    assert required
    for key in required:
        torch.testing.assert_close(target_state[key], source_state[key])


def test_backbone_export_rejects_incomplete_source(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        export_tinyvit_backbone_checkpoint(
            {"state_dict": {"patch_embed.weight": torch.ones(1)}},
            tmp_path / "incomplete.pt",
        )
