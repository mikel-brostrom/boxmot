import builtins
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import boxmot.reid.backends.tensorrt_backend as tensorrt_backend_module
import boxmot.reid.exporters.backends.tensorrt as tensorrt_exporter_module
from boxmot.reid.backends.tensorrt_backend import TensorRTBackend
from boxmot.reid.exporters.backends.tensorrt import EngineExporter


def _force_tensorrt_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorrt":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_tensorrt_backend_validates_runtime_before_import_failure(monkeypatch):
    calls = []
    backend = object.__new__(TensorRTBackend)
    backend.device = SimpleNamespace(type="cuda")

    monkeypatch.setattr(
        tensorrt_backend_module,
        "require_reid_backend_requirements",
        calls.append,
    )
    _force_tensorrt_import_error(monkeypatch)

    with pytest.raises(ImportError, match="TensorRT is installed"):
        backend.load_model("model.engine")

    assert calls == ["tensorrt"]


def test_tensorrt_exporter_validates_runtime_before_import_failure(monkeypatch, tmp_path):
    calls = []
    _mock_cuda_context(monkeypatch)
    exporter = EngineExporter(
        model=object(),
        im=SimpleNamespace(device=torch.device("cuda:0")),
        file=tmp_path / "model.pt",
    )

    monkeypatch.setattr(
        tensorrt_exporter_module,
        "require_reid_backend_requirements",
        calls.append,
    )
    _force_tensorrt_import_error(monkeypatch)

    with pytest.raises(ImportError, match="TensorRT is installed"):
        exporter.export()

    assert calls == ["tensorrt"]


def _mock_cuda_context(monkeypatch):
    """Observe scoped GPU changes without initializing CUDA."""
    state = {"device": torch.device("cuda:0")}

    @contextmanager
    def selected_device(device):
        previous = state["device"]
        state["device"] = device
        try:
            yield
        finally:
            state["device"] = previous

    monkeypatch.setattr(torch.cuda, "device", selected_device)
    return state


def test_tensorrt_loads_engine_context_and_bindings_on_selected_gpu(monkeypatch, tmp_path):
    """Engine allocation and tensor bindings must share the requested GPU."""
    state = _mock_cuda_context(monkeypatch)
    backend = object.__new__(TensorRTBackend)
    backend.device = torch.device("cuda:1")
    events = []

    def record(stage):
        events.append(stage)
        assert state["device"] == backend.device

    class Engine:
        num_bindings = 2

        def create_execution_context(self):
            record("context")
            return SimpleNamespace(get_binding_shape=lambda _index: (2, 3))

        def get_binding_name(self, index):
            return ("images", "output")[index]

        def get_binding_dtype(self, _index):
            return np.float32

        def binding_is_input(self, index):
            return index == 0

        def get_binding_shape(self, _index):
            return (2, 3)

    class Runtime:
        def __init__(self, _logger):
            record("runtime")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def deserialize_cuda_engine(self, _data):
            record("engine")
            return Engine()

    class Tensor:
        def to(self, device):
            assert device == backend.device
            record("binding")
            return self

        def data_ptr(self):
            return id(self)

    class Logger:
        INFO = 1

        def __init__(self, _level):
            pass

    monkeypatch.setitem(
        sys.modules, "tensorrt", SimpleNamespace(Runtime=Runtime, Logger=Logger, nptype=lambda dtype: dtype)
    )
    monkeypatch.setattr(tensorrt_backend_module, "require_reid_backend_requirements", lambda _backend: None)
    monkeypatch.setattr(torch, "from_numpy", lambda _array: Tensor())
    weights = tmp_path / "model.engine"
    weights.write_bytes(b"engine")

    backend.load_model(weights)

    assert events == ["runtime", "engine", "context", "binding", "binding"]
    assert state["device"] == torch.device("cuda:0")


@pytest.mark.parametrize("fails", (False, True))
def test_tensorrt_execution_scopes_selected_gpu_and_restores_previous(monkeypatch, fails):
    """Changing the calling thread's current GPU must not redirect execution."""
    state = _mock_cuda_context(monkeypatch)
    backend = object.__new__(TensorRTBackend)
    backend.device = torch.device("cuda:1")
    backend.input_name = "images"
    backend.output_name = "output"
    output = torch.zeros((2, 3))
    backend.bindings = {
        "images": SimpleNamespace(shape=(2, 3)),
        "output": SimpleNamespace(shape=(2, 3), data=output),
    }
    backend.binding_addrs = {"images": 0, "output": output.data_ptr()}

    def execute(_addresses):
        assert state["device"] == backend.device
        if fails:
            raise RuntimeError("engine failed")
        output.fill_(7)

    backend.context = SimpleNamespace(execute_v2=execute)

    if fails:
        with pytest.raises(RuntimeError, match="engine failed"):
            backend.forward(torch.ones((2, 3)))
    else:
        result = backend.forward(torch.ones((2, 3)))
        torch.testing.assert_close(result, torch.full((2, 3), 7.0))

    assert state["device"] == torch.device("cuda:0")


@pytest.mark.parametrize("device", ("cpu", "mps"))
def test_tensorrt_rejects_non_cuda_device_before_loading_dependencies(monkeypatch, device):
    """An explicit CPU/MPS request must not silently run on CUDA instead."""
    backend = object.__new__(TensorRTBackend)
    backend.device = torch.device(device)
    monkeypatch.setattr(
        tensorrt_backend_module,
        "require_reid_backend_requirements",
        lambda _backend: pytest.fail("dependencies must not be loaded"),
    )

    with pytest.raises(ValueError, match="requires a CUDA device"):
        backend.load_model("model.engine")
