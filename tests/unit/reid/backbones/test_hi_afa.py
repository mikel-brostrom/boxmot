"""Focused regression tests for the Hi-AFA paper implementation."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from boxmot.engine.config import load_training_recipe
from boxmot.engine.reid.export import _default_export_img_size
from boxmot.reid.backbones import build_backbone, get_backbone_spec
from boxmot.reid.backbones.hi_afa import (
    DropBlock2d,
    FeatureSuppression,
    HiAFA,
    HorizontalStripePool,
    LightweightDualAttention,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.training.losses import MultiSimilarityLoss
from boxmot.reid.training.trainer import ModelBundle, ReIDTrainer


def _small_model(*, num_classes: int = 7) -> HiAFA:
    return HiAFA(
        num_classes=num_classes,
        loss="ms",
        pretrained=False,
        img_size=(128, 64),
    )


def test_hi_afa_is_registered_with_paper_geometry():
    spec = get_backbone_spec("hi_afa")
    model = build_backbone(
        "hi_afa",
        num_classes=3,
        loss="ms",
        pretrained=False,
        img_size=(128, 64),
    )

    assert spec.default_img_size == (384, 128)
    assert spec.pretrained_source == "imagenet"
    assert spec.accepts_model_kwargs is True
    assert isinstance(model, HiAFA)
    assert [len(model.stage3), len(model.stage4), len(model.stage5)] == [2, 3, 4]
    assert [len(model.attention3), len(model.attention4)] == [2, 3]


def test_core_registry_forwards_and_restores_hi_afa_model_kwargs(tmp_path):
    model = ReIDModelRegistry.build_model(
        "hi_afa",
        weights=tmp_path / "unused.pt",
        num_classes=3,
        loss="ms",
        pretrained=False,
        img_size=(128, 64),
        attention_gamma=0.5,
        suppression_tau=0.6,
        dropblock_prob=0.2,
        dropblock_size=5,
    )
    checkpoint = tmp_path / "hi_afa.pt"
    torch.save(
        {
            "model_name": "hi_afa",
            "model": {"reproduction_contract": model.reproduction_contract},
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(checkpoint)
    restored = ReIDModelRegistry.build_model(
        "hi_afa",
        weights=checkpoint,
        num_classes=3,
        loss="ms",
        pretrained=False,
        **kwargs,
    )
    saved_state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    restored.load_state_dict(saved_state, strict=True)

    assert kwargs["img_size"] == (128, 64)
    assert kwargs["attention_gamma"] == 0.5
    assert kwargs["suppression_tau"] == 0.6
    assert kwargs["dropblock_prob"] == 0.2
    assert kwargs["dropblock_size"] == 5
    assert model.reproduction_contract["implementation_version"] == 2
    assert restored.reproduction_contract == model.reproduction_contract
    assert model.reproduction_contract["architecture"]["attention"]["gamma_init"] == 0.5
    assert model.reproduction_contract["architecture"]["attention"]["gamma_trainable"] is True
    assert (
        model.reproduction_contract["architecture"]["attention"]["spatial_residual"]
        == "elementwise_feature_attention"
    )


def test_market1501_profile_records_the_stabilized_reproduction_contract():
    profile = load_training_recipe("hi_afa_market1501")

    assert profile["model"] == "hi_afa"
    assert profile["imgsz"] == [384, 128]
    assert profile["loss"] == "ms"
    assert profile["epochs"] == 200
    assert (profile["p_ids"], profile["k_instances"], profile["batch_size"]) == (8, 8, 64)
    assert profile["lr"] == profile["eta_min"] == 8e-4
    assert profile["weight_decay"] == 5e-4
    assert profile["id_loss_weight"] == profile["metric_loss_weight"] == 0.5
    assert profile["center_loss_weight"] == 0.0
    assert profile["project"] == "runs/hi_afa_market1501"
    assert profile["name"] == "stable_seed0"


def test_export_uses_registered_hi_afa_crop(tmp_path):
    assert _default_export_img_size(tmp_path / "hi_afa_market1501.pt", "hi_afa") == (384, 128)


def test_hi_afa_torchscript_preserves_v2_descriptor_contract():
    model = _small_model(num_classes=2).eval()
    example = torch.randn(1, 3, 128, 64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(model, example, check_trace=False)
    output = traced(torch.randn(2, 3, 128, 64))

    assert output.shape == (2, 8192)
    torch.testing.assert_close(output.norm(dim=1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_feature_suppression_uses_per_sample_minmax_and_strict_threshold():
    suppression = FeatureSuppression(tau=0.7)
    values = torch.tensor([[[[0.0, 0.7], [0.71, 1.0]]]])

    mask = suppression.mask(values)

    assert torch.equal(mask, torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]))
    assert torch.equal(suppression.mask(torch.ones_like(values)), torch.ones_like(values))


class _Offset(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.value


class _Scale(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.value


def test_hi_afa_forward_follows_the_paper_upper_triangle_recurrence():
    model = _small_model(num_classes=2)
    model.stem = _Offset(1)
    model.stage2 = _Offset(2)
    model.attention2 = _Offset(100)
    model.stage3 = nn.ModuleList((_Offset(3), _Offset(30)))
    model.attention3 = nn.ModuleList((_Offset(1_000), _Offset(2_000)))
    model.stage4 = nn.ModuleList((_Offset(4), _Offset(40), _Offset(400)))
    model.attention4 = nn.ModuleList(
        (_Offset(10_000), _Offset(20_000), _Offset(30_000))
    )
    model.stage5 = nn.ModuleList(
        (_Offset(5), _Offset(50), _Offset(500), _Offset(5_000))
    )
    model.suppression = _Scale(10)

    outputs = model.forward_features(torch.zeros(1, 3, 128, 64))

    expected = (11_115, 145_310, 403_100, 615_000)
    assert tuple(float(output[0, 0, 0, 0]) for output in outputs) == expected


def test_hi_afa_branch_stages_clone_weights_without_sharing_storage():
    model = _small_model(num_classes=2)

    for copies in (model.stage3, model.stage4, model.stage5):
        reference = next(copies[0].parameters())
        for branch in copies[1:]:
            candidate = next(branch.parameters())
            assert torch.equal(candidate, reference)
            assert candidate.data_ptr() != reference.data_ptr()


def test_ldam_attention_is_normalized_identity_safe_and_learnable():
    module = LightweightDualAttention(
        channels=16,
        spatial_size=(4, 2),
        reduction=4,
        groups=4,
    )
    inputs = (torch.rand(2, 16, 4, 2) + 0.1).requires_grad_()

    spatial = module.spatial.attention(inputs)
    channel = module.channel.attention(module.spatial(inputs))
    output = module(inputs)
    output.mean().backward()

    assert spatial.shape == (2, 1, 4, 2)
    assert channel.shape == (2, 16, 1, 1)
    assert torch.allclose(spatial.sum(dim=(2, 3)), torch.ones(2, 1))
    assert torch.allclose(channel.sum(dim=1), torch.ones(2, 1, 1))
    assert isinstance(module.spatial.gamma, nn.Parameter)
    assert isinstance(module.channel.gamma, nn.Parameter)
    assert module.spatial.gamma.item() == 0.0
    assert module.channel.gamma.item() == 0.0
    assert torch.equal(output, inputs)
    assert output.shape == inputs.shape
    assert inputs.grad is not None
    assert module.spatial.gamma.grad is not None
    assert module.channel.gamma.grad is not None


def test_spatial_attention_modulates_each_channel_elementwise():
    module = LightweightDualAttention(
        channels=2,
        spatial_size=(2, 2),
        reduction=1,
        groups=1,
    )
    inputs = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]],
    )
    attention = module.spatial.attention(inputs)
    with torch.no_grad():
        module.spatial.gamma.fill_(1.0)

    actual = module.spatial(inputs)

    torch.testing.assert_close(actual, inputs + inputs * attention)
    assert not torch.allclose(
        actual,
        inputs + inputs.sum(dim=1, keepdim=True) * attention,
    )


def test_hi_afa_default_ldam_gates_are_zero_initialized_parameters():
    model = _small_model(num_classes=2)
    attention_modules = [model.attention2, *model.attention3, *model.attention4]
    gates = [gate for module in attention_modules for gate in (module.spatial.gamma, module.channel.gamma)]

    assert len(gates) == 12
    assert all(isinstance(gate, nn.Parameter) for gate in gates)
    assert all(gate.requires_grad for gate in gates)
    assert all(gate.item() == 0.0 for gate in gates)
    assert model.reproduction_contract["architecture"]["attention"]["gamma_init"] == 0.0


@pytest.mark.parametrize("parts", (5, 3, 2))
def test_horizontal_stripe_pool_matches_adaptive_average_pool(parts):
    inputs = torch.randn(2, 7, 24, 8)

    actual = HorizontalStripePool(parts)(inputs)
    expected = nn.functional.adaptive_avg_pool2d(inputs, (parts, 1))

    assert torch.allclose(actual, expected, atol=1e-7)


def test_dropblock_is_contiguous_rescaled_and_train_only():
    torch.manual_seed(3)
    dropblock = DropBlock2d(drop_prob=0.5, block_size=3).train()
    inputs = torch.ones(4, 3, 12, 8)

    output = dropblock(inputs)

    assert (output == 0).any()
    assert output.mean() == pytest.approx(1.0, abs=1e-6)
    dropblock.eval()
    assert torch.equal(dropblock(inputs), inputs)


def test_hi_afa_emits_exact_paper_stream_counts_and_inference_descriptor():
    torch.manual_seed(7)
    model = _small_model(num_classes=7)
    inputs = torch.randn(2, 3, 128, 64)

    model.train()
    with torch.no_grad():
        feature_maps = model.forward_features(inputs)
        logits, packet = model.forward_head(feature_maps)

    assert [tuple(feature.shape) for feature in feature_maps] == [(2, 512, 8, 4)] * 4
    assert len(logits) == 17
    assert all(logit.shape == (2, 7) for logit in logits)
    assert len(packet["_metric_features"]) == 5
    assert len(packet["_center_features"]) == 22
    assert all(feature.shape == (2, 512) for feature in packet["_center_features"])
    assert packet["_classification_loss_aggregation"] == "sum"
    assert packet["_metric_loss_aggregation"] == "sum"
    assert packet["_center_loss_scale"] == 11.0

    model.eval()
    with torch.no_grad():
        raw_streams = model._raw_pooled_streams(feature_maps)
        descriptor = model.forward_head(feature_maps)
        expected = nn.functional.normalize(
            torch.cat(
                [nn.functional.normalize(stream, dim=1) for stream in raw_streams],
                dim=1,
            ),
            dim=1,
        )

    assert len(raw_streams) == 16
    assert descriptor.shape == (2, 16 * 512)
    assert torch.isfinite(descriptor).all()
    torch.testing.assert_close(descriptor, expected)
    torch.testing.assert_close(descriptor.norm(dim=1), torch.ones(2))
    assert model.reproduction_contract["objective"] == {
        "identity_streams": 17,
        "ranking_streams": 5,
        "center_streams": 22,
        "branch_reduction": "sum",
        "multi_similarity": {
            "alpha": 2.0,
            "beta": 40.0,
            "thresh": 0.5,
            "mining_margin": 0.1,
        },
        "center_loss_internal_scale": 11.0,
    }
    assert model.feature_dim == 8192
    assert model.reproduction_contract["inference"] == {
        "descriptor": "balanced_unique_raw_pooled_streams",
        "streams": 16,
        "dimension": 8192,
        "normalization": "per_stream_l2_then_concat_final_l2",
        "training_only_streams": ["dropped_global"],
    }


class _SquaredFeatureLoss(nn.Module):
    def forward(self, features: torch.Tensor, _targets: torch.Tensor) -> torch.Tensor:
        return features.square().sum(dim=1).mean()


def test_hi_afa_packet_drives_summed_ce_ms_and_center_inputs(tmp_path):
    model = _small_model(num_classes=4).train()
    inputs = torch.randn(4, 3, 128, 64)
    pids = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        logits, packet = model(inputs)

    trainer = ReIDTrainer(
        model_name="hi_afa",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        loss_type="ms",
        img_size=(128, 64),
        pretrained=False,
        metric_loss_weight=0.5,
        center_loss_weight=5e-4,
        device="cpu",
    )
    identity_loss = trainer._classification_loss_for_logits(
        nn.CrossEntropyLoss(),
        logits,
        pids,
        epoch=1,
        features=packet,
    )
    expected_identity = sum(nn.functional.cross_entropy(logit, pids) for logit in logits)
    metric_loss = trainer._metric_loss_for_features(_SquaredFeatureLoss(), packet, pids)
    center_features, center_pids, center_scale = trainer._center_loss_inputs(packet, pids)

    assert torch.allclose(identity_loss, expected_identity)
    assert metric_loss == pytest.approx(5.0)
    assert center_features.shape == (22 * len(pids), 512)
    assert torch.equal(center_pids, pids.repeat(22))
    assert center_scale == 11.0


def test_hi_afa_builds_paper_ms_loss_and_keeps_center_enabled(tmp_path):
    model = _small_model(num_classes=4)
    trainer = ReIDTrainer(
        model_name="hi_afa",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        loss_type="ms",
        img_size=(128, 64),
        pretrained=False,
        metric_loss_weight=0.5,
        center_loss_weight=5e-4,
        device="cpu",
    )
    bundle = ModelBundle(
        model=model,
        ema_model=None,
        val_model=model,
        is_transformer=False,
        training_family="cnn",
    )

    losses = trainer._build_loss_bundle(bundle, num_classes=4)

    assert isinstance(losses.criterion_metric, MultiSimilarityLoss)
    assert losses.criterion_metric.alpha == 2.0
    assert losses.criterion_metric.beta == 40.0
    assert losses.criterion_metric.thresh == 0.5
    assert losses.criterion_metric.mining_margin == 0.1
    assert losses.metric_dim == 512
    assert trainer.center_loss_weight == 5e-4
