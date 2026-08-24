"""Checks for training-only hierarchical classifier suppression."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

import boxmot.reid.backbones.families.csl_tinyvit.multilevel_suppression as suppression_module
from boxmot.reid.backbones.families.csl_tinyvit import csl_tinyvit_7m
from boxmot.reid.backbones.families.csl_tinyvit.heads import MultiBranchHead
from boxmot.reid.backbones.families.csl_tinyvit.multilevel_suppression import (
    MultilevelClassifierSuppression,
    stripe_top_quantile_mask,
)


def _head(*, enabled: bool) -> MultiBranchHead:
    return MultiBranchHead(
        8,
        feat_dim=8,
        num_classes=5,
        metric_feature="raw_mean",
        inference_feature="norm_concat_bn",
        head_pool="gelu_gem",
        scale_balanced_branches=True,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        hierarchical_scales=True,
        return_auxiliary_features=True,
        multilevel_suppression=enabled,
        multilevel_suppression_ratio=0.15,
    )


def _maps() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    return (
        torch.randn(4, 8, 8, 4, generator=generator),
        torch.randn(4, 8, 8, 4, generator=generator),
        torch.randn(4, 8, 16, 8, generator=generator),
    )


def _v20_kwargs() -> dict[str, object]:
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


def test_stripe_mask_erases_exact_top_count_and_preserves_constant_cam() -> None:
    saliency = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    saliency = torch.cat((saliency, torch.ones_like(saliency)), dim=0)

    mask, active = stripe_top_quantile_mask(
        saliency,
        num_stripes=2,
        ratio=0.25,
    )

    assert torch.count_nonzero(mask[0] == 0).item() == 4
    assert torch.count_nonzero(mask[0, :, :2] == 0).item() == 2
    assert torch.count_nonzero(mask[0, :, 2:] == 0).item() == 2
    assert torch.count_nonzero(mask[1] == 0).item() == 0
    assert torch.equal(
        active,
        torch.tensor([[True, True], [False, False]]),
    )
    assert mask.requires_grad is False
    assert active.requires_grad is False


def test_clean_outputs_and_eval_descriptor_are_exact_with_auxiliary_active() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        baseline = _head(enabled=False)
        torch.manual_seed(27)
        treatment = _head(enabled=True)

    baseline_state = baseline.state_dict()
    treatment_state = treatment.state_dict()
    assert baseline_state.keys() == treatment_state.keys()
    assert all(
        torch.equal(value, treatment_state[key])
        for key, value in baseline_state.items()
    )

    baseline.train()
    treatment.train()
    treatment.set_multilevel_suppression_progress(1.0)
    maps = _maps()
    baseline_logits, baseline_features = baseline(maps)
    treatment_logits, treatment_features = treatment(
        tuple(feature.detach().requires_grad_(True) for feature in maps),
        pids=torch.tensor([0, 1, 2, 3]),
    )

    assert len(baseline_logits) == len(treatment_logits) == 7
    for baseline_logit, treatment_logit in zip(
        baseline_logits,
        treatment_logits,
        strict=True,
    ):
        torch.testing.assert_close(treatment_logit, baseline_logit, rtol=0, atol=0)
    torch.testing.assert_close(
        treatment_features["norm_concat_bn"],
        baseline_features["norm_concat_bn"],
        rtol=0,
        atol=0,
    )
    aux_logits = treatment_features["_multilevel_suppression_logits"]
    assert len(aux_logits["coarse"]) == 2
    assert len(aux_logits["fine"]) == 4
    assert all(logit.shape == (4, 5) for logit in (*aux_logits["coarse"], *aux_logits["fine"]))
    active = treatment_features["_multilevel_suppression_active"]
    assert active["coarse"].shape == (4, 2)
    assert active["fine"].shape == (4, 4)
    assert active["coarse"].dtype == active["fine"].dtype == torch.bool
    assert not active["coarse"].requires_grad
    assert not active["fine"].requires_grad
    diagnostics = treatment_features[
        "_multilevel_suppression_diagnostics"
    ]
    assert diagnostics["effective_ratio"].item() == pytest.approx(0.15)
    assert 0 <= diagnostics["coarse_erased_fraction"].item() <= 0.15 + 1 / 16
    assert 0 <= diagnostics["fine_erased_fraction"].item() <= 0.15 + 1 / 32
    torch.testing.assert_close(
        diagnostics["global_cam_active_fraction"],
        active["coarse"].float().mean(),
    )
    torch.testing.assert_close(
        diagnostics["coarse_cam_active_fraction"],
        active["fine"].float().mean(),
    )

    # The auxiliary functional BN must not introduce extra running-stat
    # updates beyond the one shared clean forward.
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(treatment.state_dict()[key], value, rtol=0, atol=0)

    baseline.eval()
    treatment.eval()
    with torch.no_grad():
        baseline_descriptor = baseline(maps)
        treatment_descriptor = treatment(maps)
        treatment_descriptor_with_pids = treatment(
            maps,
            pids=torch.tensor([0, 1, 2, 3]),
        )
    torch.testing.assert_close(treatment_descriptor, baseline_descriptor, rtol=0, atol=0)
    torch.testing.assert_close(
        treatment_descriptor_with_pids,
        baseline_descriptor,
        rtol=0,
        atol=0,
    )


def test_auxiliary_loss_updates_only_receiving_feature_maps() -> None:
    torch.manual_seed(41)
    head = _head(enabled=True)
    head.train()
    head.set_multilevel_suppression_progress(1.0)
    global_map, coarse_map, fine_map = (
        feature.requires_grad_(True) for feature in _maps()
    )
    _, features = head(
        (global_map, coarse_map, fine_map),
        pids=torch.tensor([0, 1, 2, 3]),
    )
    aux = features["_multilevel_suppression_logits"]
    targets = torch.tensor([0, 1, 2, 3])
    auxiliary_loss = torch.stack(
        [
            F.cross_entropy(logit, targets)
            for logit in (*aux["coarse"], *aux["fine"])
        ]
    ).mean()
    auxiliary_loss.backward()

    assert global_map.grad is None
    assert coarse_map.grad is not None
    assert torch.count_nonzero(coarse_map.grad).item() > 0
    assert fine_map.grad is not None
    assert torch.count_nonzero(fine_map.grad).item() > 0
    assert all(parameter.grad is None for parameter in head.parameters())


def test_inactive_view_cannot_couple_into_active_auxiliary_logits_or_gradients() -> None:
    torch.manual_seed(59)
    head_a = _head(enabled=True).train()
    head_b = copy.deepcopy(head_a).train()
    head_a.set_multilevel_suppression_progress(1.0)
    head_b.set_multilevel_suppression_progress(1.0)

    base_maps = _maps()
    maps_a_values = [feature.clone() for feature in base_maps]
    maps_b_values = [feature.clone() for feature in base_maps]
    # Spatially constant global/coarse teachers yield no ranked evidence for
    # sample zero. Its receiving maps deliberately differ strongly across the
    # two otherwise identical batches.
    maps_a_values[0][0].zero_()
    maps_a_values[1][0].zero_()
    maps_a_values[2][0].zero_()
    maps_b_values[0][0].fill_(2.0)
    maps_b_values[1][0].fill_(-3.0)
    maps_b_values[2][0].fill_(7.0)
    maps_a = tuple(feature.requires_grad_(True) for feature in maps_a_values)
    maps_b = tuple(feature.requires_grad_(True) for feature in maps_b_values)
    pids = torch.tensor([0, 1, 2, 3])

    _, features_a = head_a(maps_a, pids=pids)
    _, features_b = head_b(maps_b, pids=pids)
    active_a = features_a["_multilevel_suppression_active"]
    active_b = features_b["_multilevel_suppression_active"]
    assert not active_a["coarse"][0].any()
    assert not active_a["fine"][0].any()
    assert not active_b["coarse"][0].any()
    assert not active_b["fine"][0].any()
    assert active_a["coarse"][1:].any()
    assert active_a["fine"][1:].any()

    auxiliary_a = features_a["_multilevel_suppression_logits"]
    auxiliary_b = features_b["_multilevel_suppression_logits"]
    for scale in ("coarse", "fine"):
        torch.testing.assert_close(
            active_a[scale][1:],
            active_b[scale][1:],
            rtol=0,
            atol=0,
        )
        for logits_a, logits_b in zip(
            auxiliary_a[scale],
            auxiliary_b[scale],
            strict=True,
        ):
            torch.testing.assert_close(
                logits_a[1:],
                logits_b[1:],
                rtol=0,
                atol=0,
            )

    def masked_scale_loss(scale: str) -> torch.Tensor:
        per_sample = torch.stack(
            tuple(
                F.cross_entropy(logits, pids, reduction="none")
                for logits in auxiliary_a[scale]
            ),
            dim=1,
        )
        active = active_a[scale].to(per_sample.dtype)
        return (per_sample * active).sum() / active.sum().clamp_min(1.0)

    auxiliary_loss = 0.5 * (
        masked_scale_loss("coarse") + masked_scale_loss("fine")
    )
    auxiliary_loss.backward()

    coarse_grad = maps_a[1].grad
    fine_grad = maps_a[2].grad
    assert coarse_grad is not None
    assert fine_grad is not None
    assert torch.count_nonzero(coarse_grad[0]).item() == 0
    assert torch.count_nonzero(fine_grad[0]).item() == 0
    assert torch.count_nonzero(coarse_grad[1:]).item() > 0
    assert torch.count_nonzero(fine_grad[1:]).item() > 0


def test_target_masks_are_independent_of_cobatch_pids_and_features(
    monkeypatch,
) -> None:
    """The same view must not inherit another sample's target evidence."""
    torch.manual_seed(67)
    head_a = _head(enabled=True).train()
    head_b = copy.deepcopy(head_a).train()
    head_a.set_multilevel_suppression_progress(1.0)
    head_b.set_multilevel_suppression_progress(1.0)

    maps_a = tuple(feature.requires_grad_(True) for feature in _maps())
    maps_b = tuple(
        torch.cat(
            (
                feature[:1],
                feature[1:].flip(0).mul(3.0).add(5.0),
            ),
            dim=0,
        ).requires_grad_(True)
        for feature in _maps()
    )
    captured: list[torch.Tensor] = []
    original_mask = suppression_module.stripe_top_quantile_mask

    def capture_mask(*args, **kwargs):
        mask, active = original_mask(*args, **kwargs)
        captured.append(mask.detach().clone())
        return mask, active

    monkeypatch.setattr(
        suppression_module,
        "stripe_top_quantile_mask",
        capture_mask,
    )
    head_a(maps_a, pids=torch.tensor([0, 1, 2, 3]))
    masks_a = tuple(captured)
    captured.clear()
    head_b(maps_b, pids=torch.tensor([0, 4, 4, 4]))
    masks_b = tuple(captured)

    assert len(masks_a) == len(masks_b) == 2
    for mask_a, mask_b in zip(masks_a, masks_b, strict=True):
        torch.testing.assert_close(mask_a[0], mask_b[0], rtol=0, atol=0)


def test_coarse_classifier_cams_are_stitched_into_their_own_stripes() -> None:
    controller = MultilevelClassifierSuppression(0.15)
    pids = torch.zeros(2, dtype=torch.long)
    generator = torch.Generator().manual_seed(73)
    base = torch.randn(2, 3, 6, 4, generator=generator)

    def branch_logits(
        source: torch.Tensor,
        *,
        bottom_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_score = (
            source[:, 0, :3].sum(dim=(1, 2))
            + 0.25 * source[:, 1, :3].sum(dim=(1, 2))
        )
        bottom_score = bottom_scale * (
            source[:, 2, 3:].sum(dim=(1, 2))
            - 0.5 * source[:, 0, 3:].sum(dim=(1, 2))
        )
        return (
            torch.stack((top_score, -top_score), dim=1),
            torch.stack((bottom_score, -bottom_score), dim=1),
        )

    expected_source = base.clone().requires_grad_(True)
    expected_logits = branch_logits(expected_source)
    top_cam = controller._target_gradcam(
        expected_source,
        (expected_logits[0],),
        pids,
        retain_graph=True,
    )
    bottom_cam = controller._target_gradcam(
        expected_source,
        (expected_logits[1],),
        pids,
    )
    expected = torch.cat((top_cam[:, :, :3], bottom_cam[:, :, 3:]), dim=2)

    stitched_source = base.clone().requires_grad_(True)
    stitched = controller._stitched_target_gradcam(
        stitched_source,
        branch_logits(stitched_source),
        pids,
    )
    torch.testing.assert_close(stitched, expected, rtol=0, atol=0)

    changed_source = base.clone().requires_grad_(True)
    changed_bottom = controller._stitched_target_gradcam(
        changed_source,
        branch_logits(changed_source, bottom_scale=-7.0),
        pids,
    )
    torch.testing.assert_close(
        changed_bottom[:, :, :3],
        stitched[:, :, :3],
        rtol=0,
        atol=0,
    )

    output_size = (13, 7)
    resized = controller._resize_striped_saliency(
        stitched,
        output_size=output_size,
        num_stripes=2,
    )
    changed_resized = controller._resize_striped_saliency(
        changed_bottom,
        output_size=output_size,
        num_stripes=2,
    )
    upper_parent_end = output_size[0] // 2
    torch.testing.assert_close(
        changed_resized[:, :, :upper_parent_end],
        resized[:, :, :upper_parent_end],
        rtol=0,
        atol=0,
    )

    # A single whole-map bilinear resize leaks the changed lower parent CAM
    # into the last row of the upper parent when heights do not divide evenly.
    naive = F.interpolate(
        stitched,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    changed_naive = F.interpolate(
        changed_bottom,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    assert not torch.equal(
        changed_naive[:, :, :upper_parent_end],
        naive[:, :, :upper_parent_end],
    )

    fine_mask, _ = stripe_top_quantile_mask(
        resized,
        num_stripes=4,
        ratio=0.25,
    )
    changed_fine_mask, _ = stripe_top_quantile_mask(
        changed_resized,
        num_stripes=4,
        ratio=0.25,
    )
    for upper_child in (0, 1):
        start = upper_child * output_size[0] // 4
        end = (upper_child + 1) * output_size[0] // 4
        torch.testing.assert_close(
            changed_fine_mask[:, :, start:end],
            fine_mask[:, :, start:end],
            rtol=0,
            atol=0,
        )


def test_progress_validation_and_pid_free_secondary_forward_bypasses_aux() -> None:
    controller = MultilevelClassifierSuppression(0.15)
    with pytest.raises(ValueError, match="progress"):
        controller.set_progress(1.01)
    with pytest.raises(ValueError, match="ratio"):
        MultilevelClassifierSuppression(0.0)

    head = _head(enabled=True).train()
    head.set_multilevel_suppression_progress(1.0)
    logits, features = head(
        tuple(feature.requires_grad_(True) for feature in _maps())
    )
    assert len(logits) == 7
    assert "_multilevel_suppression_logits" not in features

    with pytest.raises(ValueError, match="scale-balanced"):
        MultiBranchHead(
            8,
            feat_dim=8,
            num_classes=5,
            head_pool="avg",
            head_parts=(1, 2, 4),
            part_pooling="stripes",
            hierarchical_scales=True,
            scale_balanced_branches=False,
            multilevel_suppression=True,
        )


def test_amp_diagnostics_are_detached_float32_scalars() -> None:
    head = _head(enabled=True).half().train()
    head.set_multilevel_suppression_progress(0.5)
    _, features = head(
        tuple(
            feature.half().requires_grad_(True)
            for feature in _maps()
        ),
        pids=torch.tensor([0, 1, 2, 3]),
    )

    diagnostics = features["_multilevel_suppression_diagnostics"]
    assert diagnostics["effective_ratio"].item() == pytest.approx(0.075)
    assert all(value.dtype == torch.float32 for value in diagnostics.values())
    assert all(value.ndim == 0 for value in diagnostics.values())
    assert all(not value.requires_grad for value in diagnostics.values())


def test_v20_treatment_adds_no_parameters_state_or_descriptor_width() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        baseline = csl_tinyvit_7m(**_v20_kwargs())
        torch.manual_seed(53)
        treatment = csl_tinyvit_7m(
            **_v20_kwargs(),
            multilevel_suppression=True,
            multilevel_suppression_ratio=0.15,
        )

    assert sum(parameter.numel() for parameter in treatment.parameters()) == 7_165_011
    assert treatment.state_dict().keys() == baseline.state_dict().keys()
    assert all(
        torch.equal(value, treatment.state_dict()[key])
        for key, value in baseline.state_dict().items()
    )
    branch_width = sum(
        getattr(treatment.head, treatment.head._bn_attr(key)).reduction.out_channels
        for key, _, _ in treatment.head.branch_specs
    )
    assert branch_width == 1152
    assert treatment.multilevel_suppression_enabled is True
    assert treatment.multilevel_suppression_ratio == pytest.approx(0.15)

    invalid_kwargs = _v20_kwargs()
    invalid_kwargs["feature_fusion"] = "final"
    with pytest.raises(ValueError, match="Stage-0 semantic-fine"):
        csl_tinyvit_7m(
            **invalid_kwargs,
            multilevel_suppression=True,
        )
