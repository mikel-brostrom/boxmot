"""Batched EdgeTAM memories preserve the official per-object spatial resampler."""

from __future__ import annotations

from copy import deepcopy
from types import MethodType

import pytest
import torch

from boxmot.segmentors.propagation.perceiver import _forward_2d


def _perceiver(*, global_latents: int = 3, dropout: float = 0.0) -> torch.nn.Module:
    """Instantiate small official layers without a checkpoint or model download."""
    upstream = pytest.importorskip("sam2.modeling.perceiver")
    encoding = pytest.importorskip("sam2.modeling.position_encoding")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(196)
        return upstream.PerceiverResampler(
            dim=8,
            depth=2,
            dim_head=4,
            heads=2,
            num_latents=global_latents,
            num_latents_2d=4,
            hidden_dropout_p=dropout,
            attention_dropout_p=dropout,
            pos_enc_at_key_value=True,
            concat_kv_latents=False,
            position_encoding=encoding.PositionEmbeddingSine(num_pos_feats=8, normalize=True),
            use_self_attn=True,
        ).eval()


def _features(batch: int, *, noncontiguous: bool = False, device: str = "cpu") -> torch.Tensor:
    """Use different features for every object so incorrect batch ordering is visible."""
    values = torch.arange(batch * 8 * 8 * 8, device=device, dtype=torch.float32)
    values = torch.sin(values * 0.019).reshape(batch, 8, 8, 8)
    return values.transpose(2, 3) if noncontiguous else values


def _independent_outputs(perceiver: torch.nn.Module, features: torch.Tensor, with_positions: bool) -> tuple:
    """Use untouched upstream singleton execution as the numerical reference."""
    outputs = [perceiver(value, value.cos() if with_positions else None) for value in features.split(1)]
    latents = torch.cat([output[0] for output in outputs], dim=0)
    positions = torch.cat([output[1] for output in outputs], dim=0) if with_positions else None
    return latents, positions


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("global_latents", [0, 3])
@pytest.mark.parametrize("with_positions", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_batch_matches_independent_objects(batch, global_latents, with_positions, noncontiguous) -> None:
    reference = _perceiver(global_latents=global_latents)
    adapted = deepcopy(reference)
    adapted.forward_2d = MethodType(_forward_2d, adapted)
    features = _features(batch, noncontiguous=noncontiguous)
    original = features.clone()
    with torch.inference_mode():
        expected, expected_positions = _independent_outputs(reference, features, with_positions)
        actual, positions = adapted(features, features.cos() if with_positions else None)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    if with_positions:
        torch.testing.assert_close(positions, expected_positions, rtol=0, atol=0)
    else:
        assert positions is None
    assert torch.equal(features, original)


def test_binding_preserves_parameters_state_and_training_dropout() -> None:
    adapted = _perceiver(dropout=0.2).train()
    reference = deepcopy(adapted)
    original_method = type(adapted).forward_2d
    modules = dict(adapted.named_modules())
    parameters = dict(adapted.named_parameters())
    state = {name: value.clone() for name, value in adapted.state_dict().items()}
    adapted.forward_2d = MethodType(_forward_2d, adapted)
    features = _features(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        expected = reference(features, features.cos())
        torch.manual_seed(73)
        actual = adapted(features, features.cos())
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, rtol=0, atol=0)
    assert dict(adapted.named_modules()) == modules
    assert all(dict(adapted.named_parameters())[name] is parameter for name, parameter in parameters.items())
    assert all(module.training for module in adapted.modules())
    assert set(adapted.state_dict()) == set(state)
    assert all(torch.equal(adapted.state_dict()[name], value) for name, value in state.items())
    assert type(adapted).forward_2d is original_method
    assert reference.forward_2d.__func__ is original_method


def test_batch_preserves_input_and_parameter_gradients() -> None:
    reference = _perceiver().train()
    adapted = deepcopy(reference)
    adapted.forward_2d = MethodType(_forward_2d, adapted)
    expected_features = _features(4).requires_grad_()
    actual_features = expected_features.detach().clone().requires_grad_()
    expected, _ = _independent_outputs(reference, expected_features, True)
    actual, _ = adapted(actual_features, actual_features.cos())
    weights = torch.arange(expected.numel(), dtype=expected.dtype).reshape_as(expected) / expected.numel()
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(actual_features.grad, expected_features.grad, rtol=2e-4, atol=2e-6)
    expected_parameters = dict(reference.named_parameters())
    for name, parameter in adapted.named_parameters():
        torch.testing.assert_close(parameter.grad, expected_parameters[name].grad, rtol=2e-4, atol=2e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable in this process")
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_mps_fp16_batch_matches_independent_objects(batch) -> None:
    reference = _perceiver().to(device="mps", dtype=torch.float16)
    adapted = deepcopy(reference)
    adapted.forward_2d = MethodType(_forward_2d, adapted)
    features = _features(batch, device="mps").half()
    with torch.inference_mode(), torch.autocast("mps", dtype=torch.float16):
        expected, expected_positions = _independent_outputs(reference, features, True)
        actual, positions = adapted(features, features.cos())
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(positions, expected_positions, rtol=0, atol=0)
    # MPS autocast keeps LayerNorm output in FP32, including upstream's
    # singleton path. Check model precision without forcing its output dtype.
    assert {parameter.dtype for parameter in adapted.parameters()} == {torch.float16}
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
