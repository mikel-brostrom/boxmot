import types
import sys
from pathlib import Path

import pytest
import torch

from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.reid.exporters.onnx_exporter import ONNXExporter
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import RequirementsChecker


def _load_existing_osnet_model_and_input(batch_size=2):
    candidates = [
        ROOT / "osnet_x0_25_msmt17.pt",
        WEIGHTS / "osnet_x0_25_msmt17.pt",
    ]
    weights = next((p for p in candidates if p.exists()), None)
    if weights is None:
        pytest.skip("Missing osnet_x0_25_msmt17.pt in repository root or engine/weights.")

    backend = ReidAutoBackend(weights=weights, device="cpu", half=False)
    model = backend.model.model.eval()
    im = torch.randn(batch_size, 3, 256, 128)
    return model, im


def _install_fake_onnx(monkeypatch):
    fake_onnx_model = types.SimpleNamespace(ir_version=9)
    fake_onnx = types.SimpleNamespace(
        __version__="0.0-test",
        defs=types.SimpleNamespace(onnx_opset_version=lambda: 18),
        checker=types.SimpleNamespace(check_model=lambda _m: None),
        load=lambda _p: fake_onnx_model,
        save=lambda _m, _p: None,
    )
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)


def _disable_dep_sync(monkeypatch):
    monkeypatch.setattr(RequirementsChecker, "sync_extra", lambda *args, **kwargs: None)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_onnx_export_dynamic_uses_dynamic_shapes(monkeypatch, tmp_path, batch_size):
    _disable_dep_sync(monkeypatch)
    _install_fake_onnx(monkeypatch)

    calls = []

    def fake_export(model, args, f, **kwargs):
        calls.append((args, kwargs))
        Path(f).touch()

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    model, im = _load_existing_osnet_model_and_input(batch_size=batch_size)
    out_file = tmp_path / "osnet_x0_25_msmt17.pt"

    exporter = ONNXExporter(model, im, out_file, opset=17, dynamic=True, half=False, simplify=False)
    exported = exporter.export()

    assert exported == out_file.with_suffix(".onnx")
    assert len(calls) == 1
    export_args, export_kwargs = calls[0]
    assert export_args[0].shape[0] == batch_size
    assert "dynamic_shapes" in export_kwargs
    assert "dynamic_axes" not in export_kwargs


@pytest.mark.parametrize("batch_size", [1, 3])
def test_onnx_export_dynamic_fallback_uses_dynamic_axes(monkeypatch, tmp_path, batch_size):
    _disable_dep_sync(monkeypatch)
    _install_fake_onnx(monkeypatch)

    calls = []

    def fake_export(model, args, f, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise RuntimeError("force dynamic fallback")
        Path(f).touch()

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    model, im = _load_existing_osnet_model_and_input(batch_size=batch_size)
    out_file = tmp_path / "osnet_x0_25_msmt17.pt"

    exporter = ONNXExporter(model, im, out_file, opset=17, dynamic=True, half=False, simplify=False)
    exported = exporter.export()

    assert exported == out_file.with_suffix(".onnx")
    assert len(calls) == 2
    first_args, first_kwargs = calls[0]
    second_args, second_kwargs = calls[1]
    assert first_args[0].shape[0] == batch_size
    assert second_args[0].shape[0] == batch_size
    assert "dynamic_shapes" in first_kwargs
    assert "dynamic_axes" in second_kwargs


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_onnx_export_static_has_no_dynamic_shapes(monkeypatch, tmp_path, batch_size):
    _disable_dep_sync(monkeypatch)
    _install_fake_onnx(monkeypatch)

    calls = []

    def fake_export(model, args, f, **kwargs):
        calls.append((args, kwargs))
        Path(f).touch()

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    model, im = _load_existing_osnet_model_and_input(batch_size=batch_size)
    out_file = tmp_path / "osnet_x0_25_msmt17.pt"

    exporter = ONNXExporter(model, im, out_file, opset=17, dynamic=False, half=False, simplify=False)
    exported = exporter.export()

    assert exported == out_file.with_suffix(".onnx")
    assert len(calls) == 1
    export_args, export_kwargs = calls[0]
    assert export_args[0].shape[0] == batch_size
    assert "dynamic_shapes" not in export_kwargs
    assert "dynamic_axes" not in export_kwargs
