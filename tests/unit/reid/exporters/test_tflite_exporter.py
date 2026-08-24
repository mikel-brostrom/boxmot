import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from boxmot.engine.reid.export import (
    ExportTask,
    _resolve_export_weights,
    _run_tflite_for_parity,
    _verify_export_parity,
    create_export_tasks,
    perform_exports,
    setup_model,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.exporters.onnx_exporter import ONNXExporter
from boxmot.reid.exporters.openvino_exporter import OpenVINOExporter
from boxmot.reid.exporters.tensorrt_exporter import EngineExporter
from boxmot.reid.exporters.tflite_exporter import TFLiteExporter
from boxmot.utils.checks import RequirementsChecker


def _disable_dep_sync(monkeypatch):
    monkeypatch.setattr(RequirementsChecker, "sync_extra", lambda *args, **kwargs: None)


def _install_fake_litert_torch(monkeypatch, convert_impl):
    monkeypatch.setitem(
        sys.modules,
        "litert_torch",
        types.SimpleNamespace(__version__="0.9.0-test", convert=convert_impl),
    )


def test_export_keeps_existing_explicit_weight_path(tmp_path):
    weights = tmp_path / "best.pt"
    weights.touch()

    assert _resolve_export_weights(weights) == weights


def test_export_setup_uses_reid_crop_shape_for_mobilenetv4(monkeypatch, tmp_path):
    import boxmot.engine.reid.export as export_module

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return x.mean(dim=(2, 3))

    class FakeReID:
        def __init__(self, weights, device, half):
            self.model = types.SimpleNamespace(model=FakeModel())

    weights = tmp_path / "mobilenetv4_conv_small_market1501.pt"
    args = types.SimpleNamespace(
        weights=weights,
        device="cpu",
        half=False,
        optimize=False,
        batch_size=2,
    )
    monkeypatch.setattr(export_module, "ReID", FakeReID)
    monkeypatch.setattr(ReIDModelRegistry, "get_model_name", lambda _weights: "mobilenetv4_conv_small")

    model, dummy_input = setup_model(args)

    assert isinstance(model, FakeModel)
    assert args.imgsz == (384, 128)
    assert tuple(dummy_input.shape) == (2, 3, 384, 128)


def test_export_setup_allows_cpu_onnx_fp16_graph_conversion(monkeypatch, tmp_path):
    import boxmot.engine.reid.export as export_module

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return x.mean(dim=(2, 3))

    class FakeReID:
        def __init__(self, weights, device, half):
            assert half is False
            self.model = types.SimpleNamespace(model=FakeModel())

    weights = tmp_path / "mobilenetv4_conv_small_market1501.pt"
    args = types.SimpleNamespace(
        weights=weights,
        include=("onnx",),
        device="cpu",
        half=True,
        optimize=False,
        batch_size=2,
    )
    monkeypatch.setattr(export_module, "ReID", FakeReID)
    monkeypatch.setattr(ReIDModelRegistry, "get_model_name", lambda _weights: "mobilenetv4_conv_small")

    model, dummy_input = setup_model(args)

    assert isinstance(model, FakeModel)
    assert next(model.parameters()).dtype == torch.float32
    assert dummy_input.dtype == torch.float32


def test_export_setup_rejects_cpu_tensorrt_fp16(tmp_path):
    args = types.SimpleNamespace(
        weights=tmp_path / "mobilenetv4_conv_small_market1501.pt",
        include=("engine",),
        device="cpu",
        half=True,
        optimize=False,
        batch_size=2,
    )

    with pytest.raises(AssertionError, match="TensorRT export requires GPU"):
        setup_model(args)


def test_tflite_export_uses_litert_torch_direct_api(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)

    weights = tmp_path / "osnet_x0_25_msmt17.pt"
    calls = []

    class FakeEdgeModel:
        def export(self, path=None):
            assert path is not None
            calls[-1]["export_path"] = path
            Path(path).touch()

    def fake_convert(model, sample_inputs):
        calls.append({"model": model, "sample_inputs": sample_inputs})
        return FakeEdgeModel()

    _install_fake_litert_torch(monkeypatch, fake_convert)

    model = torch.nn.Linear(4, 2).eval()
    image = torch.randn(1, 3, 256, 128)
    exporter = TFLiteExporter(model, image, weights, opset=18, dynamic=True, half=False, simplify=True)
    exported = exporter.export()

    assert Path(exported) == weights.with_suffix(".tflite")
    assert len(calls) == 1
    assert calls[0]["model"] is model
    assert len(calls[0]["sample_inputs"]) == 1
    assert calls[0]["sample_inputs"][0] is image
    assert calls[0]["export_path"] == str(weights.with_suffix(".tflite"))


def test_tflite_export_accepts_tuple_sample_inputs(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)

    weights = tmp_path / "resnet18.pt"
    image = torch.randn(1, 3, 224, 224)
    calls = []

    class FakeEdgeModel:
        def export(self, path):
            Path(path).touch()

    def fake_convert(model, sample_inputs):
        calls.append(sample_inputs)
        return FakeEdgeModel()

    _install_fake_litert_torch(monkeypatch, fake_convert)

    exporter = TFLiteExporter(torch.nn.Identity(), (image,), weights)
    exported = exporter.export()

    assert Path(exported) == weights.with_suffix(".tflite")
    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert calls[0][0] is image


def test_tflite_export_replaces_static_adaptive_max_pool2d(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)

    weights = tmp_path / "adaptive_pool.pt"
    image = torch.randn(1, 3, 8, 4)
    calls = []

    class AdaptivePoolModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))

        def forward(self, x):
            return self.pool(x).flatten(1)

    class FakeEdgeModel:
        def export(self, path):
            Path(path).touch()

    original = AdaptivePoolModel().eval()
    expected = original(image)

    def fake_convert(model, sample_inputs):
        calls.append(model)
        assert isinstance(model.pool, torch.nn.MaxPool2d)
        assert model.pool.kernel_size == (8, 4)
        torch.testing.assert_close(model(*sample_inputs), expected)
        return FakeEdgeModel()

    _install_fake_litert_torch(monkeypatch, fake_convert)

    exporter = TFLiteExporter(original, image, weights)
    exported = exporter.export()

    assert Path(exported) == weights.with_suffix(".tflite")
    assert len(calls) == 1
    assert isinstance(original.pool, torch.nn.AdaptiveMaxPool2d)


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
        tflite_quantize="static",
        tflite_calibration_data=Path("calibration"),
        tflite_calibration_samples=64,
        tflite_calibration_preprocess="resize_pad",
        tflite_calibration_seed=7,
        tflite_calibration_update="moving_average",
        tflite_static_activation_bits=8,
    )
    model = object()
    dummy_input = torch.randn(2, 3, 256, 128)

    tasks = create_export_tasks(args, model, dummy_input)

    task = tasks["tflite"]
    assert task.exporter_class is TFLiteExporter
    assert task.report is True
    assert task.kwargs == {
        "model": model,
        "im": dummy_input,
        "file": args.weights,
        "opset": 18,
        "dynamic": True,
        "half": True,
        "simplify": False,
        "quantize": "static",
        "calibration_data": Path("calibration"),
        "calibration_samples": 64,
        "calibration_preprocess": "resize_pad",
        "calibration_seed": 7,
        "calibration_update": "moving_average",
        "static_activation_bits": 8,
        "verbose": False,
    }


def test_create_export_tasks_passes_onnx_dependent_export_settings():
    args = types.SimpleNamespace(
        include=("engine", "openvino"),
        weights=Path("models/osnet_x0_25_msmt17.pt"),
        opset=18,
        dynamic=True,
        half=True,
        simplify=False,
        optimize=False,
        verbose=False,
        workspace=6,
    )
    model = object()
    dummy_input = torch.randn(2, 3, 256, 128)

    tasks = create_export_tasks(args, model, dummy_input)

    onnx_task = tasks["onnx"]
    assert onnx_task.exporter_class is ONNXExporter
    assert onnx_task.report is False
    assert onnx_task.kwargs == {
        "model": model,
        "im": dummy_input,
        "file": args.weights,
        "opset": 18,
        "dynamic": True,
        "half": True,
        "simplify": False,
        "verbose": False,
    }

    engine_task = tasks["engine"]
    assert engine_task.exporter_class is EngineExporter
    assert engine_task.report is True
    assert engine_task.kwargs == {
        "model": model,
        "im": dummy_input,
        "file": args.weights,
        "opset": 18,
        "dynamic": True,
        "half": True,
        "simplify": False,
        "verbose": False,
        "workspace": 6,
    }

    openvino_task = tasks["openvino"]
    assert openvino_task.exporter_class is OpenVINOExporter
    assert openvino_task.report is True
    assert openvino_task.kwargs == {
        "model": model,
        "im": dummy_input,
        "file": args.weights,
        "opset": 18,
        "dynamic": True,
        "half": True,
        "simplify": False,
        "verbose": False,
    }


def test_perform_exports_runs_hidden_dependency_without_reporting():
    calls = []

    class FakeExporter:
        def __init__(self, path):
            self.path = path

        def export(self):
            calls.append(self.path)
            return self.path

    exported = perform_exports(
        {
            "onnx": ExportTask(FakeExporter, {"path": Path("model.onnx")}, report=False),
            "engine": ExportTask(FakeExporter, {"path": Path("model.engine")}),
        }
    )

    assert calls == [Path("model.onnx"), Path("model.engine")]
    assert exported == {"engine": Path("model.engine")}


def test_tflite_export_quantizes_and_removes_float_intermediate(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)

    weights = tmp_path / "osnet_x0_25_msmt17.pt"
    image = torch.randn(1, 3, 256, 128)
    calls = {}

    class FakeEdgeModel:
        def export(self, path):
            calls["float_export_path"] = Path(path)
            Path(path).write_bytes(b"float")

    def fake_convert(model, sample_inputs):
        return FakeEdgeModel()

    class FakeQuantizationResult:
        def export_model(self, path, overwrite=False):
            calls["quant_export_path"] = Path(path)
            calls["overwrite"] = overwrite
            Path(path).write_bytes(b"quant")

    class FakeQuantizer:
        def __init__(self, float_path):
            calls["quantizer_float_path"] = Path(float_path)

        def add_weight_only_config(self, regex, operation_name, num_bits):
            calls["quant_config"] = ("weight", regex, operation_name, num_bits)

        def quantize(self):
            return FakeQuantizationResult()

    fake_qtyping = types.SimpleNamespace(
        TFLOperationName=types.SimpleNamespace(ALL_SUPPORTED="*")
    )
    fake_quantizer_module = types.SimpleNamespace(Quantizer=FakeQuantizer)

    _install_fake_litert_torch(monkeypatch, fake_convert)
    monkeypatch.setitem(
        sys.modules,
        "ai_edge_quantizer",
        types.SimpleNamespace(qtyping=fake_qtyping, quantizer=fake_quantizer_module),
    )

    exporter = TFLiteExporter(torch.nn.Identity(), image, weights, quantize="weight")
    exported = exporter.export()

    assert Path(exported) == weights.with_suffix(".tflite")
    assert calls["float_export_path"] == weights.with_suffix(".float.tflite")
    assert calls["quantizer_float_path"] == weights.with_suffix(".float.tflite")
    assert calls["quant_export_path"] == weights.with_suffix(".tflite")
    assert calls["overwrite"] is True
    assert calls["quant_config"] == ("weight", ".*", "*", 8)
    assert not weights.with_suffix(".float.tflite").exists()
    assert weights.with_suffix(".tflite").read_bytes() == b"quant"


def test_tflite_static_quantization_uses_calibration_images(monkeypatch, tmp_path):
    _disable_dep_sync(monkeypatch)

    import cv2

    weights = tmp_path / "osnet_x0_25_msmt17.pt"
    image = torch.randn(2, 3, 4, 2)
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    for index in range(3):
        cv2.imwrite(
            str(calibration_dir / f"{index}.jpg"),
            np.full((8, 4, 3), index * 32, dtype=np.uint8),
        )
    calls = {}
    calibration_result = {"ranges": {"tensor": {"min": -1.0, "max": 1.0}}}

    class FakeEdgeModel:
        def export(self, path):
            calls["float_export_path"] = Path(path)
            Path(path).write_bytes(b"float")

    def fake_convert(model, sample_inputs):
        return FakeEdgeModel()

    class FakeRunner:
        def get_input_details(self):
            return {
                "args_0": {
                    "shape": np.array([2, 3, 4, 2]),
                    "dtype": np.float32,
                }
            }

    class FakeInterpreter:
        def __init__(self, model_path):
            calls["calibration_model_path"] = Path(model_path)

        def allocate_tensors(self):
            pass

        def get_signature_list(self):
            return {"serving_default": {"inputs": ["args_0"], "outputs": ["output_0"]}}

        def get_signature_runner(self, signature_key):
            calls["signature_key"] = signature_key
            return FakeRunner()

    class FakeQuantizationResult:
        def export_model(self, path, overwrite=False):
            calls["quant_export_path"] = Path(path)
            calls["overwrite"] = overwrite
            Path(path).write_bytes(b"quant")

    class FakeQuantizer:
        def __init__(self, float_path):
            calls["quantizer_float_path"] = Path(float_path)
            calls["quant_config"] = []

        def add_static_config(self, regex, operation_name, activation_num_bits, weight_num_bits):
            calls["quant_config"].append(
                ("static", regex, operation_name, activation_num_bits, weight_num_bits)
            )

        def calibrate(self, calibration_data):
            calls["calibration_data"] = calibration_data
            return calibration_result

        def quantize(self, result=None):
            calls["calibration_result"] = result
            return FakeQuantizationResult()

    fake_ops = types.SimpleNamespace(
        ALL_SUPPORTED="*",
        CONV_2D="CONV_2D",
        DEPTHWISE_CONV_2D="DEPTHWISE_CONV_2D",
        FULLY_CONNECTED="FULLY_CONNECTED",
        BATCH_MATMUL="BATCH_MATMUL",
    )
    fake_qtyping = types.SimpleNamespace(TFLOperationName=fake_ops)
    fake_quantizer_module = types.SimpleNamespace(Quantizer=FakeQuantizer)

    _install_fake_litert_torch(monkeypatch, fake_convert)
    monkeypatch.setitem(
        sys.modules,
        "ai_edge_litert.interpreter",
        types.SimpleNamespace(Interpreter=FakeInterpreter),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_edge_quantizer",
        types.SimpleNamespace(qtyping=fake_qtyping, quantizer=fake_quantizer_module),
    )

    exporter = TFLiteExporter(
        torch.nn.Identity(),
        image,
        weights,
        quantize="static",
        calibration_data=calibration_dir,
        calibration_samples=3,
        calibration_update="moving_average",
    )
    exported = exporter.export()

    batches = calls["calibration_data"]["serving_default"]
    assert Path(exported) == weights.with_suffix(".tflite")
    assert calls["calibration_model_path"] == weights.with_suffix(".float.tflite")
    assert calls["signature_key"] == "serving_default"
    assert calls["quant_config"] == [
        ("static", ".*", "CONV_2D", 16, 8),
        ("static", ".*", "DEPTHWISE_CONV_2D", 16, 8),
        ("static", ".*", "FULLY_CONNECTED", 16, 8),
        ("static", ".*", "BATCH_MATMUL", 16, 8),
    ]
    assert calls["calibration_result"] is calibration_result
    assert len(batches) == 2
    assert batches[0]["args_0"].shape == (2, 3, 4, 2)
    assert batches[1]["args_0"].shape == (2, 3, 4, 2)
    assert batches[0]["args_0"].dtype == np.float32
    assert calls["overwrite"] is True
    assert not weights.with_suffix(".float.tflite").exists()


def test_tflite_calibration_directory_sampling_is_nested(tmp_path):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    for index in range(10):
        (calibration_dir / f"{index:02d}.jpg").write_bytes(b"image")

    first = TFLiteExporter(
        torch.nn.Identity(),
        torch.randn(1, 3, 4, 2),
        tmp_path / "model.pt",
        quantize="static",
        calibration_data=calibration_dir,
        calibration_samples=4,
        calibration_seed=123,
    )
    second = TFLiteExporter(
        torch.nn.Identity(),
        torch.randn(1, 3, 4, 2),
        tmp_path / "model.pt",
        quantize="static",
        calibration_data=calibration_dir,
        calibration_samples=7,
        calibration_seed=123,
    )

    first_names = [path.name for path in first._calibration_image_paths()]
    second_names = [path.name for path in second._calibration_image_paths()]

    assert second_names[:len(first_names)] == first_names


def test_tflite_export_parity_uses_litert_interpreter(monkeypatch, tmp_path):
    captured = {}

    class FakeInterpreter:
        def __init__(self, model_path):
            captured["model_path"] = model_path

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{"index": 0, "shape": np.array([1, 3, 4, 2]), "dtype": np.float32}]

        def get_output_details(self):
            return [{"index": 1}]

        def set_tensor(self, index, value):
            captured["input_index"] = index
            captured["input"] = value

        def invoke(self):
            pass

        def get_tensor(self, index):
            captured["output_index"] = index
            return captured["input"].reshape(captured["input"].shape[0], -1)

    monkeypatch.setitem(
        sys.modules,
        "ai_edge_litert.interpreter",
        types.SimpleNamespace(Interpreter=FakeInterpreter),
    )

    model = torch.nn.Flatten().eval()
    dummy_input = torch.empty(1, 3, 4, 2)
    exported = tmp_path / "flatten.tflite"
    args = types.SimpleNamespace(half=False)

    report = _verify_export_parity(args, model, dummy_input, {"tflite": str(exported)})

    assert captured["model_path"] == str(exported)
    assert captured["input_index"] == 0
    assert captured["output_index"] == 1
    assert report["tflite"]["ok"] is True
    assert report["tflite"]["parity_ok"] is True


def test_tflite_export_parity_handles_quantized_io(monkeypatch, tmp_path):
    captured = {}

    class FakeInterpreter:
        def __init__(self, model_path):
            captured["model_path"] = model_path

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{
                "index": 0,
                "shape": np.array([1, 3, 1, 2]),
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
            captured["input"] = value

        def invoke(self):
            pass

        def get_tensor(self, index):
            return np.array([[2, 6]], dtype=np.int8)

    monkeypatch.setitem(
        sys.modules,
        "ai_edge_litert.interpreter",
        types.SimpleNamespace(Interpreter=FakeInterpreter),
    )

    output = _run_tflite_for_parity(
        tmp_path / "quantized.tflite",
        torch.full((1, 3, 1, 2), 1.0),
    )

    assert captured["input"].dtype == np.int8
    assert np.all(captured["input"] == 1)
    np.testing.assert_allclose(output, np.array([[0.0, 1.0]], dtype=np.float32))
