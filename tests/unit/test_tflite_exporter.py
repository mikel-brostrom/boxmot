import sys
import types
from pathlib import Path

import numpy as np
import torch

from boxmot.engine.export import _verify_export_parity, create_export_tasks
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
