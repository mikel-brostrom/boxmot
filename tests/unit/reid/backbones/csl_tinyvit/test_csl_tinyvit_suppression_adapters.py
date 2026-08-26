"""Focused checks for suppression-guided CSL-TinyViT ReID adapters."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from boxmot.reid.backbones.families.csl_tinyvit import (
    RMSFeatureSuppression,
    csl_tinyvit_7m,
)
from boxmot.reid.backbones.families.csl_tinyvit.blocks import (
    ReIDResidualAdapter,
)
from boxmot.reid.core.registry import ReIDModelRegistry


def test_rms_feature_suppression_does_not_cancel_opposite_channels() -> None:
    suppression = RMSFeatureSuppression(tau=0.7)
    features = torch.tensor(
        [[[[3.0, 1.0, 0.0]], [[-3.0, 1.0, 0.0]]]],
    )

    mask = suppression.mask(features)

    torch.testing.assert_close(
        mask,
        torch.tensor([[[[0.0, 1.0, 1.0]]]]),
    )
    # The most energetic location has a signed channel mean of zero, but RMS
    # still identifies and suppresses it.
    assert features[:, :, :, 0].mean().item() == 0.0


def test_rms_feature_suppression_keeps_constant_energy_maps() -> None:
    suppression = RMSFeatureSuppression(tau=0.7)
    features = torch.full((2, 3, 2, 2), 4.0)

    torch.testing.assert_close(
        suppression.mask(features),
        torch.ones((2, 1, 2, 2)),
    )


def test_reid_adapter_suppression_is_lateral_only_and_zero_gated() -> None:
    adapter = ReIDResidualAdapter(
        dim=2,
        reduction_ratio=1,
        suppression_tau=0.7,
    )
    spatial = torch.tensor(
        [[[[3.0, 1.0, 0.0]], [[-3.0, 1.0, 0.0]]]],
    )
    tokens = spatial.flatten(2).transpose(1, 2)

    torch.testing.assert_close(adapter(tokens, (1, 3)), tokens)

    adapter.adapter = nn.Identity()
    with torch.no_grad():
        adapter.gamma.fill_(1.0)

    expected_lateral = adapter.suppression(spatial).flatten(2).transpose(1, 2)
    torch.testing.assert_close(adapter(tokens, (1, 3)), tokens + expected_lateral)
    # Suppression never erases the main residual stream.
    torch.testing.assert_close(adapter(tokens, (1, 3))[:, 0], tokens[:, 0])


def test_stage3_reduction4_model_uses_suppression_guided_adapters_only() -> None:
    model = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        reid_adapter_stages=(3,),
        reid_adapter_reduction=4,
        reid_adapter_suppression_tau=0.7,
    )

    assert all(
        len(getattr(layer, "reid_adapters", ())) == 0
        for layer in model.layers[:3]
    )
    adapters = model.layers[3].reid_adapters
    assert len(adapters) == len(model.layers[3].blocks) == 2
    assert all(isinstance(adapter.suppression, RMSFeatureSuppression) for adapter in adapters)
    assert all(adapter.suppression.tau == pytest.approx(0.7) for adapter in adapters)
    assert all(adapter.adapter[0].out_channels == 80 for adapter in adapters)
    assert all(adapter.gamma.item() == 0.0 for adapter in adapters)
    assert sum(parameter.numel() for adapter in adapters for parameter in adapter.parameters()) == 103_362


@pytest.mark.parametrize("tau", [-0.01, 1.01])
def test_model_rejects_invalid_adapter_suppression_tau(tau: float) -> None:
    with pytest.raises(ValueError, match="reid_adapter_suppression_tau"):
        csl_tinyvit_7m(
            num_classes=4,
            pretrained=False,
            reid_adapter_stages=(3,),
            reid_adapter_suppression_tau=tau,
        )


def test_checkpoint_reconstructs_suppression_guided_adapter_kwargs(tmp_path) -> None:
    weights = tmp_path / "hi_afa_lite.pt"
    torch.save(
        {
            "reid_adapter_stages": [3],
            "reid_adapter_reduction": 4,
            "reid_adapter_suppression_tau": 0.7,
        },
        weights,
    )

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)

    assert kwargs["reid_adapter_stages"] == (3,)
    assert kwargs["reid_adapter_reduction"] == 4
    assert kwargs["reid_adapter_suppression_tau"] == pytest.approx(0.7)
