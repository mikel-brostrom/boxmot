import sys
import types
from types import SimpleNamespace

import pytest

import boxmot.reid.backends.onnx_backend as onnx_backend_module
from boxmot.reid.backends.onnx_backend import ONNXBackend


def _make_backend(device_type: str = "cpu"):
    install_calls: list[tuple[str, ...]] = []
    backend = object.__new__(ONNXBackend)
    backend.device = SimpleNamespace(type="cpu")
    backend._requested_device = SimpleNamespace(type=device_type)
    backend.checker = SimpleNamespace(
        check_packages=lambda requirements: install_calls.append(tuple(requirements))
    )
    return backend, install_calls


@pytest.mark.parametrize(
    ("system_name", "device_type", "available_providers", "expected"),
    [
        (
            "Darwin",
            "cpu",
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ),
        (
            "Darwin",
            "mps",
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            "Windows",
            "cpu",
            ["DmlExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ),
        (
            "Linux",
            "cuda",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            "Linux",
            "cpu",
            ["CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ),
    ],
)
def test_select_execution_providers_prefers_system_accelerators(
    monkeypatch,
    system_name,
    device_type,
    available_providers,
    expected,
):
    backend, _ = _make_backend(device_type)
    monkeypatch.setattr(onnx_backend_module.platform, "system", lambda: system_name)

    assert backend._select_execution_providers(available_providers) == expected


def test_ensure_onnxruntime_installed_accepts_silicon_package_on_macos(monkeypatch):
    backend, install_calls = _make_backend("cpu")
    monkeypatch.setattr(onnx_backend_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        ONNXBackend,
        "_requirement_satisfied",
        staticmethod(lambda requirement: requirement.startswith("onnxruntime-silicon")),
    )

    backend._ensure_onnxruntime_installed()

    assert install_calls == []


def test_load_model_uses_selected_execution_providers(monkeypatch, tmp_path):
    backend, _ = _make_backend("cpu")
    requested: dict[str, object] = {}

    class FakeInputs:
        name = "images"
        shape = ["batch", 3, 384, 128]
        type = "tensor(float)"

    class FakeOutputs:
        name = "output0"

    class FakeSessionOptions:
        def __init__(self):
            self.graph_optimization_level = None
            self._overrides: dict[str, int] = {}

        def add_free_dimension_override_by_name(self, name, value):
            self._overrides[name] = value

    class FakeSession:
        def __init__(self, model_path, sess_options=None, providers=None):
            requested["model_path"] = model_path
            requested["providers"] = providers
            requested["sess_options"] = sess_options

        def get_inputs(self):
            return [FakeInputs()]

        def get_outputs(self):
            return [FakeOutputs()]

        def run(self, *_args, **_kwargs):
            return [None]

    fake_onnxruntime = types.SimpleNamespace(
        get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=FakeSession,
        SessionOptions=FakeSessionOptions,
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=1),
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setattr(onnx_backend_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ONNXBackend, "_requirement_satisfied", staticmethod(lambda _requirement: True))

    backend.input_shape = (384, 128)
    model_path = tmp_path / "model.onnx"
    backend.load_model(model_path)

    assert requested["model_path"] == str(model_path)
    # device=cpu → CPU EP only on macOS
    assert requested["providers"] == ["CPUExecutionProvider"]
    assert backend.providers == ["CPUExecutionProvider"]