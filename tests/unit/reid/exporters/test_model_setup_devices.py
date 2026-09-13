"""Export model and warm-up input use the requested logical device."""

import os
from types import SimpleNamespace

import pytest
import torch

import boxmot.reid.exporters.model_setup as model_setup


@pytest.mark.parametrize("device", ("cpu", "1", "cuda:1", torch.device("cuda:1")))
def test_export_model_and_warmup_share_device_without_remapping(monkeypatch, tmp_path, device) -> None:
    """Record placement without allocating GPU memory or loading a checkpoint."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    requested_devices = []
    model = torch.nn.Identity()

    def make_runtime(*, weights, device, half):
        requested_devices.append(device)
        return SimpleNamespace(model=SimpleNamespace(model=model))

    real_empty = torch.empty

    def make_input(*shape, device, dtype):
        requested_devices.append(device)
        return real_empty(*shape, device="cpu", dtype=dtype)

    monkeypatch.setattr(model_setup, "ReID", make_runtime)
    monkeypatch.setattr(model_setup.torch, "empty", make_input)
    monkeypatch.setattr(model_setup.ReIDModelRegistry, "get_model_name", lambda _weights: "model")
    monkeypatch.setattr(model_setup, "default_export_img_size", lambda *_args: (8, 4))
    args = SimpleNamespace(
        weights=tmp_path / "model.pt", device=device, half=False, optimize=False, batch_size=2
    )

    exported_model, dummy_input = model_setup.prepare_export_model(args)

    expected_device = torch.device("cpu" if device == "cpu" else "cuda:1")
    assert args.device == expected_device
    assert requested_devices == [expected_device, expected_device]
    assert exported_model is model
    assert dummy_input.shape == (2, 3, 8, 4)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,7"
