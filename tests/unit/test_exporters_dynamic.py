import types
import sys
from pathlib import Path

import pytest
import torch

from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.reid.exporters.onnx_exporter import ONNXExporter
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import RequirementsChecker


def _load_existing_osnet_model_and_input():
    candidates = [
        ROOT / "osnet_x0_25_msmt17.pt",
        WEIGHTS / "osnet_x0_25_msmt17.pt",
    ]
    weights = next((p for p in candidates if p.exists()), None)
    if weights is None:
        pytest.skip("Missing osnet_x0_25_msmt17.pt in repository root or engine/weights.")

    backend = ReidAutoBackend(weights=weights, device="cpu", half=False)
    model = backend.model.model.eval()
    im = torch.randn(2, 3, 256, 128)
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


def test_onnx_export_dynamic_uses_dynamic_shapes(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)
    _install_fake_onnx(monkeypatch)

    calls = []

    def fake_export(model, args, f, **kwargs):
        calls.append(kwargs)
        Path(f).touch()

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    model, im = _load_existing_osnet_model_and_input()
    out_file = tmp_path / "osnet_x0_25_msmt17.pt"

    exporter = ONNXExporter(model, im, out_file, opset=17, dynamic=True, half=False, simplify=False)
    exported = exporter.export()

    assert exported == out_file.with_suffix(".onnx")
    assert len(calls) == 1
    assert "dynamic_shapes" in calls[0]
    assert "dynamic_axes" not in calls[0]


def test_onnx_export_dynamic_fallback_uses_dynamic_axes(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)
    _install_fake_onnx(monkeypatch)

    calls = []

    def fake_export(model, args, f, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("force dynamic fallback")
        Path(f).touch()

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    model, im = _load_existing_osnet_model_and_input()
    out_file = tmp_path / "osnet_x0_25_msmt17.pt"

    exporter = ONNXExporter(model, im, out_file, opset=17, dynamic=True, half=False, simplify=False)
    exported = exporter.export()

    assert exported == out_file.with_suffix(".onnx")
    assert len(calls) == 2
    assert "dynamic_shapes" in calls[0]
    assert "dynamic_axes" in calls[1]


def test_onnx_export_static_has_no_dynamic_shapes(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)
    _install_fake_onnx(monkeypatch)

    calls = []

    def fake_export(model, args, f, **kwargs):
        calls.append(kwargs)
        Path(f).touch()

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    model, im = _load_existing_osnet_model_and_input()
    out_file = tmp_path / "osnet_x0_25_msmt17.pt"

    exporter = ONNXExporter(model, im, out_file, opset=17, dynamic=False, half=False, simplify=False)
    exported = exporter.export()

    assert exported == out_file.with_suffix(".onnx")
    assert len(calls) == 1
    assert "dynamic_shapes" not in calls[0]
    assert "dynamic_axes" not in calls[0]
