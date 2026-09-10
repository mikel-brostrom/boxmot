"""ReID validates logical devices before creating any backend."""

import os
from types import SimpleNamespace

import pytest
import torch

from boxmot.reid.core.runtime import ReID


@pytest.mark.parametrize("device", ("1", "cuda:1", torch.device("cuda:1")))
def test_reid_preserves_logical_cuda_index_and_visibility(monkeypatch, device) -> None:
    """String and Torch selectors must reach the backend on the same GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(ReID, "get_backend", lambda self: SimpleNamespace(device=self.device))

    runtime = ReID(weights="model.pt", device=device)

    assert runtime.device == torch.device("cuda:1")
    assert runtime.model.device == torch.device("cuda:1")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,7"


def test_reid_cpu_preserves_cuda_visibility(monkeypatch) -> None:
    """Creating a CPU encoder must leave later CUDA components available."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    monkeypatch.setattr(ReID, "get_backend", lambda self: SimpleNamespace(device=self.device))

    runtime = ReID(weights="model.pt", device="cpu")

    assert runtime.device == torch.device("cpu")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,7"


@pytest.mark.parametrize("device", ("1", "cuda:1", torch.device("cuda:1")))
def test_reid_rejects_unavailable_device_before_backend_loading(monkeypatch, device) -> None:
    """A Torch device object must receive the same bounds check as a string."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(ReID, "get_backend", lambda self: pytest.fail("backend must not load"))

    with pytest.raises(RuntimeError, match="cuda:1"):
        ReID(weights="model.pt", device=device)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
