import inspect
import os
import sys
import types
from pathlib import Path

import pytest
import torch

import boxmot.reid.exporters.onnx_exporter as onnx_exporter_module
import boxmot.reid.exporters.openvino_exporter as openvino_exporter_module
import boxmot.reid.exporters.tensorrt_exporter as tensorrt_exporter_module
from boxmot.reid.core import ReID
from boxmot.reid.exporters.onnx_exporter import ONNXExporter, ensure_onnx_export
from boxmot.reid.exporters.openvino_exporter import OpenVINOExporter
from boxmot.reid.exporters.tensorrt_exporter import EngineExporter
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

    backend = ReID(weights=weights, device="cpu", half=False)
    model = backend.model.model.eval()
    im = torch.randn(batch_size, 3, 256, 128)
    return model, im


def _install_fake_onnx(monkeypatch):
    fake_onnx_model = types.SimpleNamespace(ir_version=9)
    fake_onnx = types.SimpleNamespace(
        __version__="0.0-test",
        defs=types.SimpleNamespace(onnx_opset_version=lambda: 18),
        checker=types.SimpleNamespace(check_model=lambda _m: None),
        load=lambda _p, **_kwargs: fake_onnx_model,
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


def test_onnx_export_quiet_mode_uses_legacy_dynamic_axes(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)
    _install_fake_onnx(monkeypatch)

    calls = []

    def fake_export(model, args, f, **kwargs):
        calls.append((args, kwargs))
        Path(f).touch()

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    model, im = _load_existing_osnet_model_and_input(batch_size=2)
    out_file = tmp_path / "osnet_x0_25_msmt17.pt"

    exporter = ONNXExporter(model, im, out_file, opset=17, dynamic=True, half=False, simplify=False, verbose=False)
    exported = exporter.export()

    assert exported == out_file.with_suffix(".onnx")
    assert len(calls) == 1
    export_args, export_kwargs = calls[0]
    assert export_args[0].shape[0] == 2
    assert "dynamic_axes" in export_kwargs
    assert "dynamic_shapes" not in export_kwargs


def test_onnx_export_traces_single_input_inference_wrapper(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)
    _install_fake_onnx(monkeypatch)

    class OptionalFeatureMapModel(torch.nn.Module):
        def forward(self, x, return_featuremaps: bool = False):
            if return_featuremaps:
                return x
            return x + 1

    calls = []

    def fake_export(model, args, f, **kwargs):
        calls.append((model, args, kwargs))
        assert list(inspect.signature(model.forward).parameters) == ["x"]
        torch.testing.assert_close(model(args[0]), args[0] + 1)
        Path(f).touch()

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    image = torch.randn(2, 3, 8, 4)
    exporter = ONNXExporter(
        OptionalFeatureMapModel(),
        image,
        tmp_path / "feature_flag.pt",
        opset=17,
        dynamic=False,
        half=False,
        simplify=False,
        verbose=False,
    )

    assert exporter.export() == tmp_path / "feature_flag.onnx"
    assert len(calls) == 1


def test_onnx_export_static_fallback_uses_legacy_exporter(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)
    _install_fake_onnx(monkeypatch)

    calls = []

    def fake_export(model, args, f, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise RuntimeError("unsupported torch.export op")
        Path(f).touch()

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    model, im = _load_existing_osnet_model_and_input(batch_size=2)
    out_file = tmp_path / "osnet_x0_25_msmt17.pt"

    exporter = ONNXExporter(model, im, out_file, opset=17, dynamic=False, half=False, simplify=False)
    exported = exporter.export()

    assert exported == out_file.with_suffix(".onnx")
    assert len(calls) == 2
    first_args, first_kwargs = calls[0]
    second_args, second_kwargs = calls[1]
    assert first_args[0].shape[0] == 2
    assert second_args[0].shape[0] == 2
    if "dynamo" in second_kwargs:
        assert second_kwargs["dynamo"] is False
    assert "dynamic_shapes" not in second_kwargs
    assert "dynamic_axes" not in second_kwargs


def test_ensure_onnx_export_reuses_current_file(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    onnx_file = tmp_path / "model.onnx"
    weights.write_bytes(b"weights")
    onnx_file.write_bytes(b"onnx")
    os.utime(weights, (1, 1))
    os.utime(onnx_file, (2, 2))

    class FailingExporter:
        def __init__(self, **_kwargs):
            raise AssertionError("should reuse current ONNX export")

    monkeypatch.setattr(onnx_exporter_module, "ONNXExporter", FailingExporter)

    assert ensure_onnx_export(object(), object(), weights, verbose=False) == onnx_file


def test_ensure_onnx_export_refreshes_static_file_when_dynamic_requested(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    onnx_file = tmp_path / "model.onnx"
    weights.write_bytes(b"weights")
    onnx_file.write_bytes(b"onnx")
    os.utime(weights, (1, 1))
    os.utime(onnx_file, (2, 2))
    calls = []

    static_batch_dim = types.SimpleNamespace(dim_value=1, dim_param="")
    fake_input = types.SimpleNamespace(
        type=types.SimpleNamespace(
            tensor_type=types.SimpleNamespace(
                shape=types.SimpleNamespace(dim=[static_batch_dim, 3, 384, 128])
            )
        )
    )
    fake_onnx_model = types.SimpleNamespace(graph=types.SimpleNamespace(input=[fake_input]))
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        types.SimpleNamespace(load=lambda _p, **_kwargs: fake_onnx_model),
    )

    class FakeExporter:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def export(self):
            onnx_file.write_bytes(b"dynamic")
            return onnx_file

    monkeypatch.setattr(onnx_exporter_module, "ONNXExporter", FakeExporter)

    result = ensure_onnx_export("model", "image", weights, dynamic=True, verbose=False)

    assert result == onnx_file
    assert calls == [
        {
            "model": "model",
            "im": "image",
            "file": weights,
            "opset": None,
            "dynamic": True,
            "half": False,
            "simplify": False,
            "verbose": False,
        }
    ]


def test_ensure_onnx_export_refreshes_stale_file(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"stale")
    weights.write_bytes(b"weights")
    os.utime(onnx_file, (1, 1))
    os.utime(weights, (2, 2))
    calls = []

    class FakeExporter:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def export(self):
            onnx_file.write_bytes(b"fresh")
            return onnx_file

    monkeypatch.setattr(onnx_exporter_module, "ONNXExporter", FakeExporter)

    result = ensure_onnx_export(
        model="model",
        im="image",
        file=weights,
        opset=18,
        dynamic=True,
        half=True,
        simplify=False,
        verbose=False,
    )

    assert result == onnx_file
    assert calls == [
        {
            "model": "model",
            "im": "image",
            "file": weights,
            "opset": 18,
            "dynamic": True,
            "half": True,
            "simplify": False,
            "verbose": False,
        }
    ]


def test_tensorrt_export_onnx_forwards_export_settings(monkeypatch, tmp_path):
    calls = []
    onnx_file = tmp_path / "model.onnx"

    def fake_ensure_onnx_export(**kwargs):
        calls.append(kwargs)
        return onnx_file

    monkeypatch.setattr(tensorrt_exporter_module, "ensure_onnx_export", fake_ensure_onnx_export)

    exporter = EngineExporter(
        model="model",
        im="image",
        file=tmp_path / "model.pt",
        opset=18,
        dynamic=True,
        half=True,
        simplify=False,
        verbose=False,
        workspace=8,
    )

    assert exporter.export_onnx() == onnx_file
    assert exporter.workspace == 8
    assert calls == [
        {
            "model": "model",
            "im": "image",
            "file": tmp_path / "model.pt",
            "opset": 18,
            "dynamic": True,
            "half": True,
            "simplify": False,
            "verbose": False,
        }
    ]


def test_openvino_export_creates_onnx_intermediate(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)
    calls = {}
    onnx_file = tmp_path / "model.onnx"

    def fake_ensure_onnx_export(**kwargs):
        calls["ensure"] = kwargs
        onnx_file.write_bytes(b"onnx")
        return onnx_file

    def fake_convert_model(path, input):
        calls["convert"] = (path, input)
        return types.SimpleNamespace(
            inputs=[types.SimpleNamespace(partial_shape="[-1, 3, 384, 128]")]
        )

    def fake_save_model(model, path, compress_to_fp16=False):
        calls["save"] = (model, Path(path), compress_to_fp16)
        Path(path).touch()

    monkeypatch.setattr(openvino_exporter_module, "ensure_onnx_export", fake_ensure_onnx_export)
    monkeypatch.setitem(
        sys.modules,
        "openvino",
        types.SimpleNamespace(
            __version__="0.0-test",
            convert_model=fake_convert_model,
            save_model=fake_save_model,
        ),
    )

    image = torch.randn(2, 3, 384, 128)
    weights = tmp_path / "model.pt"
    exporter = OpenVINOExporter(
        model="model",
        im=image,
        file=weights,
        opset=18,
        dynamic=True,
        half=True,
        simplify=False,
        verbose=False,
    )

    exported = exporter.export()

    assert Path(exported) == tmp_path / "model_openvino_model" / "model.xml"
    ensure_kwargs = calls["ensure"]
    assert ensure_kwargs["model"] == "model"
    assert ensure_kwargs["im"] is image
    assert {key: value for key, value in ensure_kwargs.items() if key != "im"} == {
        "model": "model",
        "file": weights,
        "opset": 18,
        "dynamic": True,
        "half": True,
        "simplify": False,
        "verbose": False,
    }
    assert calls["convert"] == (str(onnx_file), [("images", [-1, 3, 384, 128])])
    assert calls["save"][1] == Path(exported)
    assert calls["save"][2] is True
