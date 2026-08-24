import builtins
from types import SimpleNamespace

import pytest

import boxmot.reid.backends.tensorrt_backend as tensorrt_backend_module
import boxmot.reid.exporters.tensorrt_exporter as tensorrt_exporter_module
from boxmot.reid.backends.tensorrt_backend import TensorRTBackend
from boxmot.reid.exporters.tensorrt_exporter import EngineExporter


def _force_tensorrt_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorrt":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_tensorrt_backend_installs_runtime_before_import_failure(monkeypatch):
    calls = []
    backend = object.__new__(TensorRTBackend)
    backend.checker = object()
    backend.device = SimpleNamespace(type="cuda")

    monkeypatch.setattr(
        tensorrt_backend_module,
        "ensure_reid_backend_requirements",
        lambda checker, backend_name: calls.append((checker, backend_name)),
    )
    _force_tensorrt_import_error(monkeypatch)

    with pytest.raises(ImportError, match="TensorRT auto-install completed"):
        backend.load_model("model.engine")

    assert calls == [(backend.checker, "tensorrt")]


def test_tensorrt_exporter_installs_runtime_before_import_failure(monkeypatch, tmp_path):
    calls = []
    exporter = EngineExporter(
        model=object(),
        im=SimpleNamespace(device=SimpleNamespace(type="cuda")),
        file=tmp_path / "model.pt",
    )

    monkeypatch.setattr(
        tensorrt_exporter_module,
        "ensure_reid_backend_requirements",
        lambda checker, backend_name: calls.append((checker, backend_name)),
    )
    _force_tensorrt_import_error(monkeypatch)

    with pytest.raises(ImportError, match="TensorRT auto-install completed"):
        exporter.export()

    assert calls == [(exporter.checker, "tensorrt")]
