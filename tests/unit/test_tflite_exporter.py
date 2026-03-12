import os
import sys
import types
from pathlib import Path

import torch

import boxmot.reid.exporters.tflite_exporter as tflite_exporter_module
from boxmot.engine.export import create_export_tasks
from boxmot.reid.exporters.tflite_exporter import TFLiteExporter
from boxmot.utils.checks import RequirementsChecker


def _disable_dep_sync(monkeypatch):
    monkeypatch.setattr(RequirementsChecker, "sync_extra", lambda *args, **kwargs: None)


def _install_fake_tflite_stack(monkeypatch, convert_impl):
    monkeypatch.setitem(
        sys.modules,
        "onnx2tf",
        types.SimpleNamespace(__version__="2.4.0", convert=convert_impl),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", types.SimpleNamespace(__version__="2.19.0"))
    monkeypatch.setattr(tflite_exporter_module.sys, "platform", "linux")


def test_tflite_export_uses_flatbuffer_direct_and_prefers_float32(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)

    weights = tmp_path / "osnet_x0_25_msmt17.pt"
    weights.with_suffix(".onnx").touch()
    calls = []

    def fake_convert(
        *,
        input_onnx_file_path,
        output_folder_path,
        tflite_backend=None,
        verbosity=None,
        output_float16_tflite=None,
    ):
        calls.append(
            {
                "input_onnx_file_path": input_onnx_file_path,
                "output_folder_path": output_folder_path,
                "tflite_backend": tflite_backend,
                "verbosity": verbosity,
                "output_float16_tflite": output_float16_tflite,
            }
        )
        output_dir = Path(output_folder_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "osnet_x0_25_msmt17_float16.tflite").touch()
        (output_dir / "osnet_x0_25_msmt17_float32.tflite").touch()

    _install_fake_tflite_stack(monkeypatch, fake_convert)

    exporter = TFLiteExporter(None, None, weights, opset=18, dynamic=True, half=False, simplify=True)
    exported = exporter.export()

    assert Path(exported).name == "osnet_x0_25_msmt17_float32.tflite"
    assert len(calls) == 1
    assert calls[0]["input_onnx_file_path"] == str(weights.with_suffix(".onnx"))
    assert calls[0]["output_folder_path"].endswith(os.sep)
    assert calls[0]["tflite_backend"] == "flatbuffer_direct"
    assert calls[0]["verbosity"] == "info"
    assert calls[0]["output_float16_tflite"] is False


def test_tflite_export_generates_onnx_when_missing(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)

    weights = tmp_path / "osnet_x0_25_msmt17.pt"
    onnx_calls = []

    class FakeONNXExporter:
        def __init__(self, model, im, file, opset=None, dynamic=False, half=False, simplify=False):
            onnx_calls.append(
                {
                    "model": model,
                    "im": im,
                    "file": file,
                    "opset": opset,
                    "dynamic": dynamic,
                    "half": half,
                    "simplify": simplify,
                }
            )
            self.file = Path(file)

        def export(self):
            onnx_path = self.file.with_suffix(".onnx")
            onnx_path.touch()
            return onnx_path

    def fake_convert(
        *,
        input_onnx_file_path,
        output_folder_path,
        tflite_backend=None,
        verbosity=None,
        output_float16_tflite=None,
    ):
        output_dir = Path(output_folder_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "osnet_x0_25_msmt17_float16.tflite").touch()

    monkeypatch.setattr(tflite_exporter_module, "ONNXExporter", FakeONNXExporter)
    _install_fake_tflite_stack(monkeypatch, fake_convert)

    image = torch.randn(1, 3, 256, 128)
    model = object()
    exporter = TFLiteExporter(model, image, weights, opset=17, dynamic=True, half=True, simplify=False)
    exported = exporter.export()

    assert Path(exported).name == "osnet_x0_25_msmt17_float16.tflite"
    assert len(onnx_calls) == 1
    assert onnx_calls[0]["model"] is model
    assert onnx_calls[0]["im"] is image
    assert onnx_calls[0]["file"] == weights
    assert onnx_calls[0]["opset"] == 17
    assert onnx_calls[0]["dynamic"] is True
    assert onnx_calls[0]["half"] is True
    assert onnx_calls[0]["simplify"] is False


def test_create_export_tasks_passes_tflite_export_settings():
    args = types.SimpleNamespace(
        include=("tflite",),
        weights=Path("models/osnet_x0_25_msmt17.pt"),
        opset=18,
        dynamic=True,
        half=True,
        simplify=False,
        optimize=False,
        verbose=False,
    )
    model = object()
    dummy_input = torch.randn(2, 3, 256, 128)

    tasks = create_export_tasks(args, model, dummy_input)

    flag, exporter_class, exp_args = tasks["tflite"]
    assert flag is True
    assert exporter_class is TFLiteExporter
    assert exp_args[0] is model
    assert exp_args[1] is dummy_input
    assert exp_args[2] == args.weights
    assert exp_args[3:] == (18, True, True, False)
