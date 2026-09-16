"""MPS prompt selection preserves official embeddings and model ownership."""

from __future__ import annotations

from copy import deepcopy
from types import MethodType, SimpleNamespace

import pytest
import torch

from boxmot.segmentors.propagation import model
from boxmot.segmentors.propagation.prompt import _embed_points


@pytest.fixture
def encoder() -> torch.nn.Module:
    """Use the installed official implementation as the numerical reference."""
    upstream = pytest.importorskip("sam2.modeling.sam.prompt_encoder")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(417)
        return upstream.PromptEncoder(256, (64, 64), (1024, 1024), 16).eval()


def _prompts(values: tuple[int, ...], batch: int, dtype: torch.dtype, device: str = "cpu") -> tuple:
    """Include distinct subpixel coordinates without changing global RNG state."""
    points = torch.arange(batch * len(values) * 2, dtype=torch.float32, device=device).reshape(batch, len(values), 2)
    points = points * 3.125 - 4.5
    labels = torch.tensor(values, dtype=dtype, device=device).expand(batch, -1).clone()
    return points, labels


@pytest.mark.parametrize("values", [(), (-1,), (0,), (1,), (2,), (3,), (-2, 4, -1, 0, 1, 2, 3)])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("label_dtype", [torch.int64, torch.float32])
def test_point_embeddings_match_upstream_without_mutating_inputs(encoder, values, batch, pad, label_dtype) -> None:
    points, labels = _prompts(values, batch, label_dtype)
    original_points, original_labels = points.clone(), labels.clone()
    with torch.inference_mode():
        expected = encoder._embed_points(points, labels, pad)
        actual = _embed_points(encoder, points, labels, pad)
    assert actual.shape == (batch, len(values) + int(pad), 256)
    assert torch.equal(actual, expected)
    assert torch.equal(points, original_points)
    assert torch.equal(labels, original_labels)


def test_instance_binding_preserves_modules_parameters_and_full_prompt_forward(encoder) -> None:
    reference = deepcopy(encoder)
    upstream_method = type(encoder)._embed_points
    modules = dict(encoder.named_modules())
    parameters = dict(encoder.named_parameters())
    state = {key: tensor.clone() for key, tensor in encoder.state_dict().items()}
    encoder._embed_points = MethodType(_embed_points, encoder)
    points, labels = _prompts((-1, 0, 1, 2, 3, 4), 4, torch.int64)
    with torch.inference_mode():
        expected_sparse, expected_dense = reference((points, labels), boxes=None, masks=None)
        actual_sparse, actual_dense = encoder((points, labels), boxes=None, masks=None)
    assert torch.equal(actual_sparse, expected_sparse)
    assert torch.equal(actual_dense, expected_dense)
    assert dict(encoder.named_modules()) == modules
    assert all(dict(encoder.named_parameters())[name] is parameter for name, parameter in parameters.items())
    assert set(encoder.state_dict()) == set(state)
    assert all(torch.equal(encoder.state_dict()[key], tensor) for key, tensor in state.items())
    assert type(encoder)._embed_points is upstream_method
    assert reference._embed_points.__func__ is upstream_method


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:2"])
def test_builder_binds_only_the_selected_mps_encoder_instance(tmp_path, monkeypatch, device) -> None:
    """Exercise device routing without a GPU, checkpoint download, or model package."""

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(1, 8)

        def _embed_points(self, points, labels, pad):
            return points

    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.touch()
    upstream_method = Encoder._embed_points
    original = Encoder()
    untouched = Encoder()
    predictor = SimpleNamespace(sam_prompt_encoder=original, spatial_perceiver=SimpleNamespace())
    state = {name: value.clone() for name, value in original.state_dict().items()}
    calls = []

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return predictor

    monkeypatch.setattr(model, "resolve_device", lambda value: torch.device(value))
    monkeypatch.setattr(model, "find_spec", lambda name: object())
    monkeypatch.setattr(model, "_register_model_config", lambda: "fixture_config")
    monkeypatch.setattr(model, "postprocessing_metadata", lambda selected: {"effective_fill_hole_area": 0})
    monkeypatch.setattr(model, "import_module", lambda name: SimpleNamespace(build_sam2_video_predictor=build))
    assert model.build_edgetam_predictor(checkpoint, device, precision="fp32") is predictor
    assert calls[0][1]["device"] == device
    assert original._embed_points.__func__ is (_embed_points if device == "mps" else upstream_method)
    assert original._embed_points.__self__ is original
    assert untouched._embed_points.__func__ is upstream_method
    assert Encoder._embed_points is upstream_method
    assert set(original.state_dict()) == set(state)
    assert all(torch.equal(original.state_dict()[name], value) for name, value in state.items())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable in this process")
@pytest.mark.parametrize("values", [(), (-2, -1, 0, 1, 2, 3, 4)])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("pad", [False, True])
def test_mps_point_embeddings_are_bitwise_identical(encoder, values, batch, pad) -> None:
    encoder = encoder.to("mps")
    points, labels = _prompts(values, batch, torch.int64, device="mps")
    with torch.inference_mode():
        expected = encoder._embed_points(points, labels, pad)
        actual = _embed_points(encoder, points, labels, pad)
    assert torch.equal(actual, expected)
