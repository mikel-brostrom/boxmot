import types

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

    interpreter_class = backend._get_interpreter_class()

    assert interpreter_class is litert_interpreter
    assert backend.checker.calls == [("ai-edge-litert",)]
    assert calls == ["ai_edge_litert.interpreter", "ai_edge_litert.interpreter"]
