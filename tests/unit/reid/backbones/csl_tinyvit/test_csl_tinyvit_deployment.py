from __future__ import annotations

import pytest
import torch
from torch import nn

from boxmot.reid.backbones.families.csl_tinyvit.attention import Attention
from boxmot.reid.backbones.families.csl_tinyvit.blocks import Conv2d_BN, fuse_conv2d_bn_eval_
from boxmot.reid.backbones.families.csl_tinyvit.deployment import (
    FoldedBNNeck,
    optimize_csl_tinyvit_for_inference,
)
from boxmot.reid.backbones.families.csl_tinyvit.variants import csl_tinyvit_7m_v20
from boxmot.reid.backbones.heads.bnneck import BNNeck3


def _manual_attention(
    attention: Attention,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch, tokens, _ = x.shape
    normalized = attention.norm(x)
    qkv = attention.qkv(normalized)
    q, k, v = qkv.view(batch, tokens, attention.num_heads, -1).split(
        [attention.key_dim, attention.key_dim, attention.d],
        dim=3,
    )
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    logits = (q @ k.transpose(-2, -1)) * attention.scale
    bias = attention._attention_bias() if attention.training else attention.ab
    logits = logits + bias
    if attn_mask is not None:
        logits = logits.masked_fill(
            ~attn_mask[:, None, :, :],
            torch.finfo(logits.dtype).min,
        )
    attended = (logits.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, tokens, attention.dh)
    return attention.proj(attended)


def test_conv2d_bn_fusion_is_equivalent_and_idempotent() -> None:
    torch.manual_seed(0)
    block = Conv2d_BN(4, 6, ks=3, pad=1)
    with torch.no_grad():
        block.bn.running_mean.copy_(torch.randn(6))
        block.bn.running_var.copy_(torch.rand(6) + 0.1)
        block.bn.weight.copy_(torch.randn(6))
        block.bn.bias.copy_(torch.randn(6))
    model = nn.Sequential(block).eval()
    inputs = torch.randn(2, 4, 9, 5)

    expected = model(inputs)
    assert fuse_conv2d_bn_eval_(model) == 1
    actual = model(inputs)

    assert isinstance(model[0], nn.Conv2d)
    assert model[0].bias is not None
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    assert fuse_conv2d_bn_eval_(model) == 0


def test_conv2d_bn_fusion_rejects_training_mode() -> None:
    with pytest.raises(RuntimeError, match="eval mode"):
        fuse_conv2d_bn_eval_(nn.Sequential(Conv2d_BN(2, 2)))


def test_bnneck3_folded_projection_preserves_eval_outputs() -> None:
    torch.manual_seed(1)
    neck = BNNeck3(input_dim=8, class_num=3, feat_dim=5, return_f=True)
    with torch.no_grad():
        neck.bn.running_mean.copy_(torch.randn(5))
        neck.bn.running_var.copy_(torch.rand(5) + 0.1)
        neck.bn.weight.copy_(torch.randn(5))
        neck.bn.bias.copy_(torch.randn(5))
    neck.eval()
    inputs = torch.randn(4, 8, 1, 1)

    expected = neck(inputs)[0]
    neck.prepare_for_inference()
    actual = neck.forward_inference(inputs)

    assert neck._inference_weight is not None
    assert neck._inference_bias is not None
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(neck(inputs)[0], expected)
    cached_weight = neck._inference_weight
    neck.prepare_for_inference()
    assert neck._inference_weight is cached_weight
    assert not any("_inference_" in key for key in neck.state_dict())

    neck.train()
    assert neck._inference_weight is None
    assert neck._inference_bias is None


def test_attention_reload_refreshes_eval_bias_cache() -> None:
    attention = Attention(dim=16, key_dim=4, num_heads=2, resolution=(3, 2)).eval()
    original = attention.ab.clone()
    state = attention.state_dict()
    state["attention_biases"] = torch.full_like(state["attention_biases"], 2.0)

    attention.load_state_dict(state)

    assert not torch.equal(attention.ab, original)
    torch.testing.assert_close(attention.ab, attention._attention_bias())


def test_bnneck_reload_invalidates_folded_cache() -> None:
    neck = BNNeck3(input_dim=8, class_num=3, feat_dim=5, return_f=True).eval()
    neck.prepare_for_inference()
    state = neck.state_dict()
    state["reduction.weight"] = torch.randn_like(state["reduction.weight"])

    neck.load_state_dict(state)

    assert neck._inference_weight is None
    assert neck._inference_bias is None


@pytest.mark.parametrize("bias_mode", ["absolute", "signed_factorized"])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("masked", [True, False])
def test_sdpa_matches_explicit_attention(
    bias_mode: str,
    training: bool,
    masked: bool,
) -> None:
    torch.manual_seed(2)
    attention = Attention(
        dim=16,
        key_dim=4,
        num_heads=2,
        attn_ratio=2,
        resolution=(3, 2),
        bias_mode=bias_mode,
    )
    assert attention.train(training) is attention
    tokens = 6
    inputs = torch.randn(2, tokens, 16, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    attn_mask = None
    if masked:
        attn_mask = torch.ones(2, tokens, tokens, dtype=torch.bool)
        attn_mask[:, :, -1] = False
        attn_mask[:, -1, -1] = True

    actual = attention(inputs, attn_mask)
    expected = _manual_attention(attention, reference_inputs, attn_mask)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=5e-5, atol=5e-6)


def test_deployment_optimization_only_caches_necks_inside_csl_model() -> None:
    generic_neck = BNNeck3(input_dim=4, class_num=2, feat_dim=3, return_f=True).eval()
    optimize_csl_tinyvit_for_inference(generic_neck)
    assert generic_neck._inference_weight is None

    csl_like = nn.Sequential(
        Conv2d_BN(4, 4),
        BNNeck3(input_dim=4, class_num=2, feat_dim=3, return_f=True),
    ).eval()
    optimize_csl_tinyvit_for_inference(csl_like)
    assert isinstance(csl_like[0], nn.Conv2d)
    assert csl_like[1]._inference_weight is not None


def test_v20_deployment_prunes_training_modules_and_preserves_descriptor() -> None:
    torch.manual_seed(7)
    model = csl_tinyvit_7m_v20(num_classes=4, pretrained=False).eval()
    inputs = torch.randn(2, 3, 384, 128)
    with torch.inference_mode():
        expected = model(inputs)
    before_parameters = sum(parameter.numel() for parameter in model.parameters())

    optimize_csl_tinyvit_for_inference(model)

    with torch.inference_mode():
        actual = model(inputs)
    assert sum(isinstance(module, FoldedBNNeck) for module in model.modules()) == 7
    assert not any(isinstance(module, BNNeck3) for module in model.modules())
    assert not any("classifier" in name for name, _ in model.named_parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) < before_parameters
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
