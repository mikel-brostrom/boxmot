from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from boxmot.reid.backends.coreml_backend import CoreMLBackend
from boxmot.reid.exporters.coreml_exporter import (
    parse_coreml_buckets,
    prepare_coreml_export_model,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1,8,16,32", (1, 8, 16, 32)),
        ("8;1;8", (1, 8)),
        ([16, 1], (1, 16)),
    ],
)
def test_parse_coreml_buckets(value, expected):
    assert parse_coreml_buckets(value) == expected


@pytest.mark.parametrize("value", ["", "0,8", "1,64", "nope"])
def test_parse_coreml_buckets_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        parse_coreml_buckets(value)


class _CoreMLRewriteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn2d = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn1d = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.bn2d(self.conv(x))).flatten(1)
        return self.bn1d(x)


def test_coreml_export_rewrites_preserve_frozen_batch_norm_and_global_max_pool() -> None:
    torch.manual_seed(0)
    model = _CoreMLRewriteModel().eval()
    with torch.no_grad():
        for batch_norm in (model.bn2d, model.bn1d):
            batch_norm.running_mean.copy_(torch.randn(4))
            batch_norm.running_var.copy_(torch.rand(4) + 0.1)
            batch_norm.weight.copy_(torch.randn(4))
            batch_norm.bias.copy_(torch.randn(4))
    inputs = torch.randn(2, 3, 7, 5)
    expected = model(inputs)

    assert prepare_coreml_export_model(model) is model
    actual = model(inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
    assert not any(isinstance(module, nn.AdaptiveMaxPool2d) for module in model.modules())

    program = torch.export.export(model, (inputs,), strict=False).run_decompositions({})
    targets = {str(node.target) for node in program.graph_module.graph.nodes}
    assert "aten._native_batch_norm_legit_no_training.default" not in targets
    assert "aten.adaptive_max_pool2d.default" not in targets


def test_coreml_export_rewrites_require_eval_mode() -> None:
    with pytest.raises(RuntimeError, match="eval mode"):
        prepare_coreml_export_model(_CoreMLRewriteModel())


class _FakeMLModel:
    def __init__(self, calls: list[int]) -> None:
        self.calls = calls

    def predict(self, feed):
        batch = next(iter(feed.values()))
        self.calls.append(batch.shape[0])
        values = batch[:, 0, 0, 0]
        return {"features": np.stack((values, values + 1), axis=1)}


def _fake_backend() -> tuple[CoreMLBackend, list[int]]:
    backend = CoreMLBackend.__new__(CoreMLBackend)
    backend._crop_shape = (3, 2, 2)
    backend._buckets = (1, 8, 16, 32)
    backend._input_name = "images"
    backend._output_name = "features"
    backend._output_width = 2
    backend._pad_buffers = {
        batch: np.zeros((batch, *backend._crop_shape), dtype=np.float32) for batch in backend._buckets
    }
    calls: list[int] = []
    model = _FakeMLModel(calls)
    backend._load_bucket = lambda batch: model
    return backend, calls


def test_coreml_backend_chunks_and_pads_arbitrary_batches():
    backend, calls = _fake_backend()
    inputs = torch.zeros((37, 3, 2, 2), dtype=torch.float32)
    inputs[:, 0, 0, 0] = torch.arange(37)

    output = backend.forward(inputs)

    assert calls == [32, 32]
    assert output.shape == (37, 2)
    np.testing.assert_array_equal(output[:, 0], np.arange(37, dtype=np.float32))


def test_coreml_backend_uses_smallest_fitting_bucket():
    backend, calls = _fake_backend()

    output = backend.forward(torch.zeros((7, 3, 2, 2)))

    assert calls == [8]
    assert output.shape == (7, 2)


def test_coreml_backend_handles_empty_batch_without_loading_package():
    backend, calls = _fake_backend()

    output = backend.forward(torch.empty((0, 3, 2, 2)))

    assert calls == []
    assert output.shape == (0, 2)
