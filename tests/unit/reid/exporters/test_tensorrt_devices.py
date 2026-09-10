"""TensorRT export device requirements and CUDA context ownership."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import boxmot.reid.exporters.backends.tensorrt as module


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_tensorrt_export_rejects_non_cuda_before_loading_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
) -> None:
    def unexpected_requirements(*args: object) -> None:
        raise AssertionError("Reject unsupported devices before loading TensorRT")

    monkeypatch.setattr(module, "require_reid_backend_requirements", unexpected_requirements)
    exporter = module.EngineExporter(
        model=None, im=SimpleNamespace(device=torch.device(device)), file=tmp_path / "model.pt", verbose=False
    )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        exporter.export()


def test_tensorrt_builder_uses_selected_cuda_device_and_restores_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = torch.device("cuda:0")
    seen = []

    @contextmanager
    def device_context(device: torch.device):
        nonlocal current
        previous = current
        current = device
        try:
            yield
        finally:
            current = previous

    class FakeLogger:
        INFO = 1

        def __init__(self, level: int) -> None:
            pass

    def builder(logger: FakeLogger) -> None:
        seen.append(current)
        raise RuntimeError("builder failure")

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(module, "require_reid_backend_requirements", lambda *args: None)
    monkeypatch.setitem(
        sys.modules, "tensorrt", SimpleNamespace(__version__="10.0", Logger=FakeLogger, Builder=builder)
    )
    graph = tmp_path / "model.onnx"
    graph.touch()
    exporter = module.EngineExporter(
        model=None, im=SimpleNamespace(device=torch.device("cuda:1")), file=tmp_path / "model.pt", verbose=False
    )
    monkeypatch.setattr(exporter, "export_onnx", lambda: graph)

    with pytest.raises(RuntimeError, match="builder failure"):
        exporter.export()

    assert seen == [torch.device("cuda:1")]
    assert current == torch.device("cuda:0")
