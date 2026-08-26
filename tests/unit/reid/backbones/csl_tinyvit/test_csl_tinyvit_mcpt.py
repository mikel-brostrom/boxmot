"""Regression checks for monotonic canonical part transport."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.reid.backbones.families.csl_tinyvit import (
    MonotonicCanonicalPartTransport,
    csl_tinyvit_7m,
    csl_tinyvit_11m,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.training.config import ReIDTrainConfig, trainer_kwargs_from_args
from boxmot.reid.training.resume import contract_differences
from boxmot.reid.training.trainer import ReIDTrainer


def _trainer(tmp_path, **kwargs) -> ReIDTrainer:
    values = {
        "model_name": "csl_tinyvit_7m",
        "dataset_name": "market1501",
        "data_dir": str(tmp_path),
        "device": "cpu",
        "epochs": 200,
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "spatial_conv_mode": "depthwise_separable",
        "feat_dim": 384,
        "neck_dim": 384,
        "head_parts": (1, 2, 4),
        "head_type": "standard",
        "part_pooling": "stripes",
        "scale_balanced_branches": True,
        "metric_feature": "raw_concat",
        "inference_feature": "norm_concat_bn",
        "anatomical_auxiliary": False,
        "mcpt_mode": "shared_multiscale",
        "mcpt_disabled_eval": True,
    }
    values.update(kwargs)
    return ReIDTrainer(**values)


def _build_7m(mode: str, *, anatomy: bool = False):
    model = csl_tinyvit_7m(
        num_classes=751,
        loss="triplet",
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        feat_dim=384,
        neck_dim=384,
        head_pool="gelu_gem",
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=anatomy,
        anatomical_token_dim=96,
        anatomical_multiscale=anatomy,
        anatomical_target_type="learned_pose_concat_ema",
        attention_window_layout="rect",
        attention_bias="absolute",
        interpolate_pretrained_attention_bias=True,
        attention_mask=True,
        mcpt_mode=mode,
    )
    if mode != "none":
        model.head.set_mcpt_epoch(0)
    return model.eval()


def _build_11m(mode: str):
    model = csl_tinyvit_11m(
        num_classes=751,
        loss="triplet",
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        feat_dim=512,
        neck_dim=512,
        head_pool="gelu_gem",
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=False,
        attention_window_layout="rect",
        attention_bias="absolute",
        interpolate_pretrained_attention_bias=True,
        attention_mask=True,
        mcpt_mode=mode,
    )
    if mode != "none":
        model.head.set_mcpt_epoch(0)
    return model.eval()


@pytest.mark.parametrize(
    ("mode", "fine_is_warped"),
    [
        ("dataset_boundaries", True),
        ("foreground_aware_shared_multiscale", True),
        ("per_image_stage2", False),
        ("shared_multiscale", True),
    ],
)
def test_mcpt_is_exact_identity_through_start_epoch(mode, fine_is_warped):
    module = MonotonicCanonicalPartTransport(32, mode=mode)
    local = torch.randn(2, 32, 24, 8)
    fine = torch.randn(2, 16, 48, 16)

    transported_local, transported_fine, diagnostics = module(local, fine)

    assert torch.equal(transported_local, local)
    assert torch.equal(transported_fine, fine)
    assert torch.allclose(
        diagnostics.boundary_mean,
        torch.tensor([0.25, 0.5, 0.75]),
        atol=1e-6,
        rtol=0,
    )
    assert diagnostics.local_gate.item() == 0
    assert diagnostics.fine_gate.item() == 0
    assert module.applies_to_fine is fine_is_warped


def test_mcpt_owns_tinyvit_predictor_initialization_across_backbones():
    module = MonotonicCanonicalPartTransport(
        384,
        fine_channels=384,
        mode="foreground_aware_shared_multiscale",
        hidden_dim=64,
    )

    for projection in (module.row_projection, module.fine_row_projection):
        assert projection.weight.mean().abs().item() < 0.002
        assert projection.weight.std().item() == pytest.approx(0.02, abs=0.002)
    assert torch.equal(module.row_norm.weight, torch.ones_like(module.row_norm.weight))
    assert torch.equal(module.row_norm.bias, torch.zeros_like(module.row_norm.bias))


def test_mcpt_coordinates_remain_ordered_and_capped():
    module = MonotonicCanonicalPartTransport(
        16,
        mode="dataset_boundaries",
        max_displacement=0.15,
    )
    with torch.no_grad():
        module.dataset_row_logits.copy_(
            torch.tensor([-1e6, 1e6, -1e6, 1e6])
        )
    local = torch.randn(3, 16, 24, 8)

    edges, uniform, _ = module._source_edges(local)
    displacement = edges - uniform

    assert torch.all(edges[:, 1:] > edges[:, :-1])
    assert displacement.abs().max().item() <= 0.150001
    assert torch.equal(edges[:, 0], torch.zeros(3))
    assert torch.equal(edges[:, -1], torch.ones(3))
    assert torch.isfinite(edges).all()


def test_mcpt_manual_interpolation_matches_grid_sample_reference():
    module = MonotonicCanonicalPartTransport(8, mode="shared_multiscale")
    feature = torch.randn(2, 8, 24, 7)
    logits = torch.randn(2, 24)
    lengths = F.softplus(logits)
    lengths = lengths / lengths.sum(dim=1, keepdim=True)
    edges = F.pad(lengths.cumsum(dim=1), (1, 0), value=0.0)

    actual = module._warp_y(feature, edges)
    source_y = module._sample_positions(edges, feature.shape[2])
    source_x = (
        torch.arange(feature.shape[3], dtype=feature.dtype) + 0.5
    ) / feature.shape[3]
    grid = torch.stack(
        (
            (2 * source_x - 1).view(1, 1, -1).expand(2, 24, -1),
            (2 * source_y - 1).unsqueeze(-1).expand(-1, -1, 7),
        ),
        dim=-1,
    )
    expected = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_mcpt_memory_efficient_residual_matches_direct_blend():
    module = MonotonicCanonicalPartTransport(8, mode="shared_multiscale")
    feature = torch.randn(2, 8, 24, 7)
    logits = torch.randn(2, 24)
    lengths = F.softplus(logits)
    lengths = lengths / lengths.sum(dim=1, keepdim=True)
    edges = F.pad(lengths.cumsum(dim=1), (1, 0), value=0.0)
    gate = torch.tensor(0.37)

    actual = module._transport_y(feature, edges, gate)
    warped = module._warp_y(feature, edges)
    expected = feature + gate * (warped - feature)

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_mcpt_off_schedule_has_no_parameter_gradient():
    module = MonotonicCanonicalPartTransport(16, mode="shared_multiscale")
    local = torch.randn(2, 16, 24, 8, requires_grad=True)
    fine = torch.randn(2, 8, 48, 16, requires_grad=True)

    transported_local, transported_fine, diagnostics = module(local, fine)
    (
        transported_local.square().mean()
        + transported_fine.square().mean()
        + diagnostics.smoothness
        + diagnostics.identity
    ).backward()

    assert module.row_output.weight.grad is None
    assert module.local_gate_delta.grad is None
    assert local.grad is not None
    assert fine.grad is not None


def test_mcpt_standalone_eval_uses_full_deployment_schedule():
    module = MonotonicCanonicalPartTransport(16, mode="shared_multiscale")
    assert module._schedule_scale == 0

    module.eval()
    assert module._schedule_scale == 1

    module.set_epoch(0)
    module.eval()
    assert module._schedule_scale == 0


def test_mcpt_schedule_has_live_predictor_gradient_and_disable_control():
    module = MonotonicCanonicalPartTransport(16, mode="shared_multiscale")
    module.set_epoch(20)
    local = torch.randn(2, 16, 24, 8, requires_grad=True)
    fine = torch.randn(2, 8, 48, 16, requires_grad=True)

    transported_local, transported_fine, diagnostics = module(local, fine)
    (transported_local.square().mean() + transported_fine.square().mean()).backward()

    assert 0 < diagnostics.local_gate.item() < 1
    assert 0 < diagnostics.fine_gate.item() < 1
    assert module.row_output.weight.grad is not None
    assert module.row_output.weight.grad.abs().sum().item() > 0

    with torch.no_grad():
        module.row_output.bias.fill_(1.0)
        module.row_output.weight.normal_(0, 0.1)
    active_local, active_fine, _ = module(local.detach(), fine.detach())
    module.set_force_disabled(True)
    disabled_local, disabled_fine, disabled_diagnostics = module(
        local.detach(), fine.detach()
    )
    assert not torch.equal(active_local, local.detach())
    assert not torch.equal(active_fine, fine.detach())
    assert torch.equal(disabled_local, local.detach())
    assert torch.equal(disabled_fine, fine.detach())
    assert disabled_diagnostics.local_gate.item() == 0


def test_foreground_aware_mcpt_starts_from_uniform_local_predictor():
    module = MonotonicCanonicalPartTransport(
        16,
        fine_channels=8,
        mode="foreground_aware_shared_multiscale",
    )
    local = torch.randn(2, 16, 24, 8)
    fine = torch.randn(2, 8, 48, 16)

    local_tokens = module._foreground_weighted_row_tokens(
        local,
        module.local_foreground_attention,
    )
    fine_tokens = module._foreground_weighted_row_tokens(
        fine,
        module.fine_foreground_attention,
    )

    assert torch.allclose(
        local_tokens,
        local.mean(dim=-1).transpose(1, 2),
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(
        fine_tokens,
        fine.mean(dim=-1).transpose(1, 2),
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.count_nonzero(module.fine_fusion_gate_delta) == 0


def test_foreground_aware_mcpt_has_live_multiscale_gradients():
    module = MonotonicCanonicalPartTransport(
        16,
        fine_channels=8,
        mode="foreground_aware_shared_multiscale",
    )
    module.set_epoch(40)
    with torch.no_grad():
        module.row_output.weight.normal_(0, 0.1)
        module.fine_fusion_gate_delta.fill_(0.1)
    local = torch.randn(2, 16, 24, 8, requires_grad=True)
    fine = torch.randn(2, 8, 48, 16, requires_grad=True)

    transported_local, transported_fine, _ = module(local, fine)
    (transported_local.square().mean() + transported_fine.square().mean()).backward()

    learned_parameters = (
        module.local_foreground_attention.weight,
        module.fine_foreground_attention.weight,
        module.fine_row_projection.weight,
        module.fine_fusion_gate_delta,
    )
    assert all(parameter.grad is not None for parameter in learned_parameters)
    assert all(
        parameter.grad.abs().sum().item() > 0
        for parameter in learned_parameters
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
@pytest.mark.parametrize(
    "mode",
    ["shared_multiscale", "foreground_aware_shared_multiscale"],
)
def test_mcpt_backward_is_supported_on_mps_with_determinism(mode):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        module = MonotonicCanonicalPartTransport(
            16,
            fine_channels=8,
            mode=mode,
        ).to("mps")
        module.set_epoch(20)
        if mode == "foreground_aware_shared_multiscale":
            with torch.no_grad():
                module.row_output.weight.normal_(0, 0.1)
                module.fine_fusion_gate_delta.fill_(0.1)
        local = torch.randn(2, 16, 24, 8, device="mps", requires_grad=True)
        fine = torch.randn(2, 8, 48, 16, device="mps", requires_grad=True)
        transported_local, transported_fine, diagnostics = module(local, fine)
        (
            transported_local.square().mean()
            + transported_fine.square().mean()
            + 0.01 * diagnostics.smoothness
            + 0.02 * diagnostics.identity
        ).backward()
        torch.mps.synchronize()
        assert module.row_output.weight.grad is not None
        assert module.row_output.weight.grad.abs().sum().item() > 0
        if mode == "foreground_aware_shared_multiscale":
            assert module.fine_row_projection.weight.grad is not None
            assert module.fine_row_projection.weight.grad.abs().sum().item() > 0
    finally:
        torch.use_deterministic_algorithms(previous)


def test_mcpt_visualization_capture_is_bounded_and_clearable(tmp_path):
    module = MonotonicCanonicalPartTransport(16, mode="shared_multiscale")
    module.set_epoch(40)
    module.enable_visualization_capture(limit=3)
    local = torch.randn(2, 16, 24, 8)
    fine = torch.randn(2, 8, 48, 16)

    module(local, fine)
    module(local, fine)
    captured = module.pop_visualization_capture()

    assert captured is not None
    assert captured["local_before"].shape == (3, 24, 8)
    assert captured["local_after"].shape == (3, 24, 8)
    assert captured["fine_before"].shape == (3, 48, 16)
    assert captured["fine_after"].shape == (3, 48, 16)
    ReIDTrainer._save_mcpt_energy_maps(tmp_path, 20, captured)
    assert (tmp_path / "mcpt_energy_epoch_0020.png").is_file()
    assert module.pop_visualization_capture() is None


def test_mcpt_preserves_7m_descriptor_and_parameter_budget_at_epoch_zero():
    torch.manual_seed(71)
    control = _build_7m("none")
    torch.manual_seed(71)
    treatment = _build_7m("shared_multiscale")
    probe = torch.randn(1, 3, 384, 128)

    with torch.inference_mode():
        control_descriptor = control(probe)
        treatment_descriptor = treatment(probe)

    assert torch.equal(treatment_descriptor, control_descriptor)
    assert treatment_descriptor.shape == (1, 1152)
    assert sum(parameter.numel() for parameter in control.parameters()) == 6_937_893
    assert sum(parameter.numel() for parameter in treatment.parameters()) == 6_963_688

    with torch.no_grad():
        treatment.head.mcpt.row_output.weight.normal_(0, 0.1)
        treatment.head.mcpt.row_output.bias.fill_(0.1)
    treatment.head.set_mcpt_epoch(40)
    with torch.inference_mode():
        active_descriptor = treatment(probe)
    assert torch.allclose(
        active_descriptor[:, :384],
        control_descriptor[:, :384],
        atol=1e-6,
        rtol=1e-6,
    )
    assert not torch.equal(active_descriptor[:, 384:], control_descriptor[:, 384:])


def test_mcpt_and_v8_pose_teacher_preserve_the_7m_deployment_descriptor():
    torch.manual_seed(72)
    pose_teacher = _build_7m("none", anatomy=True)
    torch.manual_seed(72)
    combined = _build_7m("shared_multiscale", anatomy=True)
    probe = torch.randn(1, 3, 384, 128)

    with torch.inference_mode():
        expected = pose_teacher(probe)
        actual = combined(probe)

    assert torch.equal(actual, expected)
    assert actual.shape == (1, 1152)
    assert sum(parameter.numel() for parameter in combined.parameters()) == 7_190_806
    assert combined.head.mcpt is not None
    assert combined.head.anatomical_auxiliary_pool is not None


def test_mcpt_preserves_11m_descriptor_at_epoch_zero():
    torch.manual_seed(73)
    control = _build_11m("none")
    torch.manual_seed(73)
    treatment = _build_11m("shared_multiscale")
    probe = torch.randn(1, 3, 384, 128)

    with torch.inference_mode():
        control_descriptor = control(probe)
        treatment_descriptor = treatment(probe)

    assert torch.equal(treatment_descriptor, control_descriptor)
    assert treatment_descriptor.shape == (1, 1536)
    assert sum(parameter.numel() for parameter in treatment.parameters()) > sum(
        parameter.numel() for parameter in control.parameters()
    )


def test_foreground_aware_mcpt_preserves_7m_descriptor_and_parameter_budget():
    torch.manual_seed(83)
    control = _build_7m("none")
    torch.manual_seed(83)
    treatment = _build_7m("foreground_aware_shared_multiscale")
    probe = torch.randn(1, 3, 384, 128)

    with torch.inference_mode():
        control_descriptor = control(probe)
        treatment_descriptor = treatment(probe)

    assert torch.equal(treatment_descriptor, control_descriptor)
    assert treatment_descriptor.shape == (1, 1152)
    assert sum(parameter.numel() for parameter in treatment.parameters()) == 6_989_866


def test_mcpt_trainer_schedule_lr_and_resume_contract(tmp_path):
    trainer = _trainer(tmp_path)
    assert trainer._mcpt_identity_weight_for_epoch(10) == pytest.approx(0.02)
    assert trainer._mcpt_identity_weight_for_epoch(35) == pytest.approx(0.01)
    assert trainer._mcpt_identity_weight_for_epoch(60) == 0
    assert trainer._vit_lr_scale_for_param("head.mcpt.row_output.weight", 4) == 2.0

    control = _trainer(tmp_path, mcpt_mode="none", mcpt_disabled_eval=False)
    differences = contract_differences(
        control._resume_contract(),
        trainer._resume_contract(),
    )
    assert "model.mcpt_mode: saved='none', requested='shared_multiscale'" in differences


def test_mcpt_optimizer_groups_apply_lr_multiplier_and_no_decay_coordinates(tmp_path):
    trainer = _trainer(tmp_path)
    model = _build_7m("shared_multiscale")
    groups = trainer._build_vit_param_groups(model)

    parameter_groups = {
        id(parameter): group
        for group in groups
        for parameter in group["params"]
    }
    predictor_group = parameter_groups[id(model.head.mcpt.row_output.weight)]
    gate_group = parameter_groups[id(model.head.mcpt.local_gate_delta)]
    assert predictor_group["lr"] == pytest.approx(
        trainer.lr * trainer.mcpt_lr_multiplier
    )
    assert predictor_group["weight_decay"] == pytest.approx(
        trainer.weight_decay
    )
    assert gate_group["lr"] == pytest.approx(
        trainer.lr * trainer.mcpt_lr_multiplier
    )
    assert gate_group["weight_decay"] == 0

    dataset_model = _build_7m("dataset_boundaries")
    dataset_groups = trainer._build_vit_param_groups(dataset_model)
    dataset_parameter_groups = {
        id(parameter): group
        for group in dataset_groups
        for parameter in group["params"]
    }
    logits_group = dataset_parameter_groups[
        id(dataset_model.head.mcpt.dataset_row_logits)
    ]
    assert logits_group["weight_decay"] == 0

    foreground_model = _build_7m("foreground_aware_shared_multiscale")
    foreground_groups = trainer._build_vit_param_groups(foreground_model)
    foreground_parameter_groups = {
        id(parameter): group
        for group in foreground_groups
        for parameter in group["params"]
    }
    foreground_weight_group = foreground_parameter_groups[
        id(foreground_model.head.mcpt.fine_row_projection.weight)
    ]
    foreground_gate_group = foreground_parameter_groups[
        id(foreground_model.head.mcpt.fine_fusion_gate_delta)
    ]
    assert foreground_weight_group["lr"] == pytest.approx(
        trainer.lr * trainer.mcpt_lr_multiplier
    )
    assert foreground_weight_group["weight_decay"] == pytest.approx(
        trainer.weight_decay
    )
    assert foreground_gate_group["weight_decay"] == 0


def test_mcpt_checkpoint_metadata_reconstructs_deployed_architecture(tmp_path):
    weights = tmp_path / "mcpt.pt"
    torch.save(
        {
            "model_name": "csl_tinyvit_7m",
            "mcpt_mode": "shared_multiscale",
            "mcpt_hidden_dim": 64,
            "mcpt_max_displacement": 0.15,
            "mcpt_start_epoch": 10,
            "mcpt_ramp_end_epoch": 40,
        },
        weights,
    )

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)

    assert kwargs == {
        "mcpt_mode": "shared_multiscale",
        "mcpt_hidden_dim": 64,
        "mcpt_max_displacement": 0.15,
        "mcpt_start_epoch": 10,
        "mcpt_ramp_end_epoch": 40,
    }


def test_mcpt_checkpoint_round_trip_keeps_active_inference(tmp_path):
    trainer = _trainer(
        tmp_path,
        head_pool="gelu_gem",
        attention_window_layout="rect",
        attention_bias="absolute",
        interpolate_pretrained_attention_bias=True,
        attention_mask=True,
        mcpt_mode="foreground_aware_shared_multiscale",
    )
    model = _build_7m("foreground_aware_shared_multiscale")
    with torch.no_grad():
        model.head.mcpt.row_output.weight.normal_(0, 0.1)
        model.head.mcpt.row_output.bias.fill_(0.2)
        model.head.mcpt.local_foreground_attention.weight.normal_(0, 0.1)
        model.head.mcpt.fine_foreground_attention.weight.normal_(0, 0.1)
        model.head.mcpt.fine_fusion_gate_delta.fill_(0.1)
    model.head.set_mcpt_epoch(40)
    model.eval()
    probe = torch.randn(1, 3, 384, 128)
    with torch.inference_mode():
        expected = model(probe)

    weights = tmp_path / "mcpt_round_trip.pt"
    checkpoint = {
        **trainer._checkpoint_metadata(model),
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, weights)
    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)
    restored = ReIDModelRegistry.build_model(
        "csl_tinyvit_7m",
        weights=weights,
        num_classes=751,
        loss="triplet",
        pretrained=False,
        use_gpu=False,
        **kwargs,
    )
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    restored.eval()
    with torch.inference_mode():
        actual = restored(probe)

    assert restored.head.mcpt is not None
    assert restored.head.mcpt._schedule_scale == 1
    assert torch.equal(actual, expected)


def test_mcpt_cli_recipe_resolves_rgb_only_treatment(monkeypatch):
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
            "train-reid",
            "--recipe",
            "csl_tinyvit_7m_v20",
            "--data-dir",
            ".",
            "--no-anatomical-auxiliary",
            "--no-anatomical-multiscale",
            "--mcpt-mode",
            "foreground_aware_shared_multiscale",
            "--mcpt-disabled-eval",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == "csl_tinyvit_7m_v20"
    assert args.anatomical_auxiliary is False
    assert args.mcpt_mode == "foreground_aware_shared_multiscale"
    assert args.mcpt_hidden_dim == 64
    assert args.mcpt_lr_multiplier == 2.0
    assert args.mcpt_disabled_eval is True


def test_mcpt_cli_11m_market_recipe_reaches_trainer(monkeypatch, tmp_path):
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
            "train-reid",
            "--recipe",
            "csl_tinyvit_11m",
            "--model",
            "csl_tinyvit_11m",
            "--dataset",
            "market1501",
            "--data-dir",
            str(tmp_path),
            "--no-anatomical-auxiliary",
            "--no-anatomical-multiscale",
            "--mcpt-mode",
            "shared_multiscale",
            "--mcpt-disabled-eval",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = trainer_kwargs_from_args(captured["args"], {})
    config = ReIDTrainConfig.from_flat_kwargs(**kwargs)
    trainer = ReIDTrainer.from_config(config)
    assert trainer.model_name == "csl_tinyvit_11m"
    assert trainer.dataset_name == "market1501"
    assert trainer.feat_dim == 512
    assert trainer.neck_dim == 512
    assert trainer.mcpt_mode == "shared_multiscale"
    assert trainer.anatomical_auxiliary is False


def test_mcpt_accepts_proven_11m_configuration(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        feat_dim=512,
        neck_dim=512,
    )

    assert trainer.model_name == "csl_tinyvit_11m"
    assert trainer.mcpt_mode == "shared_multiscale"


def test_mcpt_accepts_7m_v8_pose_teacher_and_rejects_other_combinations(
    tmp_path,
):
    treatment = _trainer(
        tmp_path,
        anatomical_auxiliary=True,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_metadata_dir=str(tmp_path),
        anatomical_accessory_query=False,
        anatomical_deployment=False,
    )

    assert treatment.mcpt_mode == "shared_multiscale"
    assert treatment.anatomical_auxiliary is True

    with pytest.raises(ValueError, match="multiscale V8 pose teacher"):
        _trainer(tmp_path, anatomical_auxiliary=True)
    with pytest.raises(ValueError, match="multiscale V8 pose teacher"):
        _trainer(
            tmp_path,
            model_name="csl_tinyvit_11m",
            feat_dim=512,
            neck_dim=512,
            anatomical_auxiliary=True,
            anatomical_multiscale=True,
            anatomical_target_type="learned_pose_concat_ema",
        )
    with pytest.raises(ValueError, match="feat_dim=neck_dim=384"):
        _trainer(tmp_path, feat_dim=512, neck_dim=512)


def test_mcpt_accepts_7m_v20_v8_pose_teacher(tmp_path):
    treatment = _trainer(
        tmp_path,
        model_name="csl_tinyvit_7m_v20",
        anatomical_auxiliary=True,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_metadata_dir=str(tmp_path),
        anatomical_accessory_query=False,
        anatomical_deployment=False,
    )

    assert treatment.model_name == "csl_tinyvit_7m_v20"
    assert treatment.mcpt_mode == "shared_multiscale"
    assert treatment.anatomical_auxiliary is True
