import types

import numpy as np
import torch

import boxmot.reid.backends.tflite_backend as tflite_backend_module
from boxmot.reid.backends.tflite_backend import TFLiteBackend


class DummyChecker:
    def __init__(self):
        self.calls = []

    def check_packages(self, requirements):
        self.calls.append(tuple(requirements))


def make_backend() -> TFLiteBackend:
    backend = TFLiteBackend.__new__(TFLiteBackend)
    backend.checker = DummyChecker()
    return backend


def test_tflite_backend_prefers_litert_interpreter(monkeypatch):
    backend = make_backend()
    litert_interpreter = type("LiteRTInterpreter", (), {})

    def fake_import_module(name):
        if name == "ai_edge_litert.interpreter":
            return types.SimpleNamespace(Interpreter=litert_interpreter)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(tflite_backend_module, "import_module", fake_import_module)

    interpreter_class = backend._get_interpreter_class()

    assert interpreter_class is litert_interpreter
    assert backend.checker.calls == []


def test_tflite_backend_installs_litert_when_no_runtime_is_available(monkeypatch):
    backend = make_backend()
    litert_interpreter = type("LiteRTInterpreter", (), {})

    calls = []

    def fake_import_module(name):
        calls.append(name)
        if name != "ai_edge_litert.interpreter":
            raise AssertionError(f"Unexpected import: {name}")
        if len(calls) == 1:
            raise ModuleNotFoundError(name)
        return types.SimpleNamespace(Interpreter=litert_interpreter)

    monkeypatch.setattr(tflite_backend_module, "import_module", fake_import_module)
    monkeypatch.setattr(
        tflite_backend_module,
        "ensure_reid_backend_requirements",
        lambda checker, _backend: checker.check_packages(("ai-edge-litert>=2.1.0",)),
    )

    interpreter_class = backend._get_interpreter_class()

    assert interpreter_class is litert_interpreter
    assert backend.checker.calls == [("ai-edge-litert>=2.1.0",)]
    assert calls == ["ai_edge_litert.interpreter", "ai_edge_litert.interpreter"]


def test_tflite_backend_resizes_using_model_input_shape(monkeypatch):
    backend = make_backend()
    backend.nhwc = True

    class FakeInterpreter:
        def __init__(self, model_path):
            self.input_shape = np.array([1, 3, 384, 128], dtype=np.int32)
            self.output = np.zeros((1, 1), dtype=np.float32)
            self.resize_calls = []
            self.tensor = None

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{"index": 0, "shape": self.input_shape, "dtype": np.float32}]

        def get_output_details(self):
            return [{"index": 1}]

        def resize_tensor_input(self, index, shape):
            self.resize_calls.append((index, shape))
            self.input_shape = np.array(shape, dtype=np.int32)

        def set_tensor(self, index, value):
            self.tensor = value
            self.output = np.zeros((value.shape[0], 4), dtype=np.float32)

        def invoke(self):
            pass

        def get_tensor(self, index):
            return self.output

    fake_interpreter = None

    def fake_interpreter_class(model_path):
        nonlocal fake_interpreter
        fake_interpreter = FakeInterpreter(model_path)
        return fake_interpreter

    monkeypatch.setattr(backend, "_get_interpreter_class", lambda: fake_interpreter_class)

    backend.load_model("model.tflite")
    out = backend.forward(torch.zeros(2, 3, 384, 128))

    assert backend.nhwc is False
    assert fake_interpreter.resize_calls == [(0, [2, 3, 384, 128])]
    assert fake_interpreter.tensor.shape == (2, 3, 384, 128)
    assert out.shape == (2, 4)


def test_tflite_backend_transposes_nhwc_models(monkeypatch):
    backend = make_backend()
    backend.nhwc = False

    class FakeInterpreter:
        def __init__(self, model_path):
            self.input_shape = np.array([1, 384, 128, 3], dtype=np.int32)
            self.tensor = None

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{"index": 0, "shape": self.input_shape, "dtype": np.float32}]

        def get_output_details(self):
            return [{"index": 1}]

        def set_tensor(self, index, value):
            self.tensor = value

        def invoke(self):
            pass

        def get_tensor(self, index):
            return np.zeros((self.tensor.shape[0], 4), dtype=np.float32)

    fake_interpreter = None

    def fake_interpreter_class(model_path):
        nonlocal fake_interpreter
        fake_interpreter = FakeInterpreter(model_path)
        return fake_interpreter

    monkeypatch.setattr(backend, "_get_interpreter_class", lambda: fake_interpreter_class)

    backend.load_model("model.tflite")
    backend.forward(torch.zeros(1, 3, 384, 128))

    assert backend.nhwc is True
    assert fake_interpreter.tensor.shape == (1, 384, 128, 3)


def test_tflite_backend_quantizes_input_and_dequantizes_output(monkeypatch):
    backend = make_backend()
    backend.nhwc = False

    class FakeInterpreter:
        def __init__(self, model_path):
            self.input_shape = np.array([1, 3, 2, 2], dtype=np.int32)
            self.tensor = None

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{
                "index": 0,
                "shape": self.input_shape,
                "dtype": np.int8,
                "quantization": (0.5, -1),
            }]

        def get_output_details(self):
            return [{
                "index": 1,
                "dtype": np.int8,
                "quantization": (0.25, 2),
            }]

        def set_tensor(self, index, value):
            self.tensor = value

        def invoke(self):
            pass

        def get_tensor(self, index):
            return np.array([[2, 6]], dtype=np.int8)

    fake_interpreter = None

    def fake_interpreter_class(model_path):
        nonlocal fake_interpreter
        fake_interpreter = FakeInterpreter(model_path)
        return fake_interpreter

    monkeypatch.setattr(backend, "_get_interpreter_class", lambda: fake_interpreter_class)

    backend.load_model("model.tflite")
    out = backend.forward(torch.full((1, 3, 2, 2), 1.0))

    assert fake_interpreter.tensor.dtype == np.int8
    assert np.all(fake_interpreter.tensor == 1)
    np.testing.assert_allclose(out, np.array([[0.0, 1.0]], dtype=np.float32))
