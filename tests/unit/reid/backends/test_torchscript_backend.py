"""TorchScript ReID must honor the runtime device and evaluation mode."""

from pathlib import Path

import pytest
import torch

from boxmot.reid.backends.torchscript_backend import TorchscriptBackend


@pytest.mark.parametrize("device", ("cpu", torch.device("cuda:2")))
@pytest.mark.parametrize("half", (False, True))
def test_torchscript_maps_saved_tensors_to_selected_device(monkeypatch, device, half) -> None:
    calls = []

    class Model:
        def eval(self):
            calls.append("eval")
            return self

        def half(self):
            calls.append("half")
            return self

        def float(self):
            calls.append("float")
            return self

    def load(path, *, map_location):
        assert path == "model.torchscript"
        assert map_location == device
        return Model()

    monkeypatch.setattr(torch.jit, "load", load)
    backend = TorchscriptBackend.__new__(TorchscriptBackend)
    backend.device = device
    backend.half = half

    backend.load_model("model.torchscript")

    assert calls == ["eval", "half" if half else "float"]


def test_torchscript_loaded_in_training_mode_is_switched_to_evaluation(tmp_path: Path) -> None:
    model = torch.jit.script(torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.9)).train())
    path = tmp_path / "model.torchscript"
    torch.jit.save(model, path)
    backend = TorchscriptBackend.__new__(TorchscriptBackend)
    backend.device = torch.device("cpu")
    backend.half = False

    backend.load_model(path)

    assert not backend.model.training
    assert all(not module.training for module in backend.model.modules())
    assert all(parameter.device.type == "cpu" for parameter in backend.model.parameters())
    inputs = torch.ones((2, 3))
    torch.testing.assert_close(backend.forward(inputs), backend.forward(inputs))
