import sys
import types
from types import SimpleNamespace

import pytest
import torch

import boxmot.reid.backends.onnx_backend as onnx_backend_module
from boxmot.reid.backends.onnx_backend import ONNXBackend
from boxmot.reid.core.artifacts import write_artifact_metadata
from boxmot.reid.core.registry import ReIDModelRegistry


def _make_backend(device_type: str = "cpu"):
    install_calls: list[tuple[str, ...]] = []
    backend = object.__new__(ONNXBackend)
    backend.device = SimpleNamespace(type="cpu")
    backend._requested_device = SimpleNamespace(type=device_type)
    backend.checker = SimpleNamespace(check_packages=lambda requirements: install_calls.append(tuple(requirements)))
    return backend, install_calls


def test_onnx_init_uses_metadata_without_building_source_model(monkeypatch, tmp_path):
    weights = tmp_path / "csl_tinyvit_7m_v20.onnx"
    weights.write_bytes(b"onnx")
    write_artifact_metadata(
        weights,
        {
            "model_name": "csl_tinyvit_7m_v20",
            "num_classes": 751,
            "model_kwargs_schema_version": 1,
            "model_kwargs": {"img_size": [320, 96]},
        },
    )
    loaded = []
    monkeypatch.setattr(
        ReIDModelRegistry,
        "build_model",
        lambda *args, **kwargs: pytest.fail("ONNX must not build a source PyTorch model"),
    )
    monkeypatch.setattr(ONNXBackend, "load_model", lambda self, path: loaded.append(path))

    backend = ONNXBackend(weights, torch.device("cpu"), half=False)

    assert backend.model is None
    assert backend.model_name == "csl_tinyvit_7m_v20"
    assert backend.input_shape == (320, 96)
    assert loaded == [weights]


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
            "Windows",
            "cpu",
            ["DmlExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ),
        (
            "Linux",
            "cuda",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider"],
        ),
        (
            "Linux",
            "cpu",
            ["CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ),
    ],
)
def test_select_execution_providers_honours_explicit_device(
    monkeypatch,
    system_name,
    device_type,
    available_providers,
    expected,
):
    backend, _ = _make_backend(device_type)
    monkeypatch.setattr(onnx_backend_module.platform, "system", lambda: system_name)

    assert backend._select_execution_providers(available_providers) == expected


def test_mps_device_rejects_onnx_runtime_coreml_alias(monkeypatch):
    backend, _ = _make_backend("mps")
    monkeypatch.setattr(onnx_backend_module.platform, "system", lambda: "Darwin")

    with pytest.raises(ValueError, match="has no MPS execution provider"):
        backend._select_execution_providers(["CoreMLExecutionProvider", "CPUExecutionProvider"])


def test_explicit_coreml_device_selects_only_coreml_provider():
    backend, _ = _make_backend("coreml")

    assert backend._select_execution_providers(["CoreMLExecutionProvider", "CPUExecutionProvider"]) == [
        "CoreMLExecutionProvider"
    ]


@pytest.mark.parametrize(
    ("device_type", "provider"),
    [
        ("cpu", "CPUExecutionProvider"),
        ("cuda", "CUDAExecutionProvider"),
        ("coreml", "CoreMLExecutionProvider"),
    ],
)
def test_explicit_device_rejects_unavailable_provider(device_type, provider):
    backend, _ = _make_backend(device_type)

    with pytest.raises(RuntimeError, match=rf"{provider} was explicitly requested"):
        backend._select_execution_providers(["SomeOtherExecutionProvider"])


def test_auto_device_uses_available_provider_fallback_order(monkeypatch):
    backend, _ = _make_backend("auto")
    monkeypatch.setattr(onnx_backend_module.platform, "system", lambda: "Windows")

    assert backend._select_execution_providers(["DmlExecutionProvider", "CPUExecutionProvider"]) == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]


@pytest.mark.parametrize(("explicit", "expected"), [(True, "1"), (False, None)])
def test_session_disables_cpu_fallback_only_for_explicit_accelerator(monkeypatch, explicit, expected):
    backend, _ = _make_backend("cuda")
    backend._provider_selection_is_explicit = explicit
    captured = {}

    class FakeSessionOptions:
        def __init__(self):
            self.graph_optimization_level = None
            self.config = {}

        def add_session_config_entry(self, name, value):
            self.config[name] = value

    def fake_session(_weights, sess_options, providers):
        captured["config"] = sess_options.config
        captured["providers"] = providers
        return object()

    fake_onnxruntime = types.SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=1),
        InferenceSession=fake_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    backend._make_session("model.onnx", ["CUDAExecutionProvider"], None)

    assert captured["config"].get("session.disable_cpu_ep_fallback") == expected
    assert captured["providers"] == ["CUDAExecutionProvider"]


@pytest.mark.parametrize("value", ["ort", "cv", "dnn", "opencv_dnn", "ONNXRUNTIME", ""])
def test_runtime_backend_rejects_aliases_and_unknown_values(monkeypatch, value):
    monkeypatch.setenv("BOXMOT_REID_BACKEND", value)

    with pytest.raises(ValueError, match="Invalid BOXMOT_REID_BACKEND"):
        ONNXBackend._select_runtime_backend()


@pytest.mark.parametrize("value", ["auto", "onnxruntime", "opencv"])
def test_runtime_backend_accepts_only_canonical_values(monkeypatch, value):
    monkeypatch.setenv("BOXMOT_REID_BACKEND", value)

    assert ONNXBackend._select_runtime_backend() == value


def test_explicit_opencv_backend_rejects_accelerator(monkeypatch):
    backend, _ = _make_backend("cuda")
    monkeypatch.setenv("BOXMOT_REID_BACKEND", "opencv")

    with pytest.raises(ValueError, match="OpenCV DNN ReID supports only device=cpu"):
        backend.load_model("model.onnx")


def test_explicit_onnxruntime_backend_does_not_fall_back(monkeypatch):
    backend, _ = _make_backend("cpu")
    monkeypatch.setenv("BOXMOT_REID_BACKEND", "onnxruntime")
    monkeypatch.setattr(
        backend,
        "_ensure_onnxruntime_installed",
        lambda: (_ for _ in ()).throw(ImportError("unavailable")),
    )

    with pytest.raises(RuntimeError, match="explicitly requested"):
        backend.load_model("model.onnx")


def test_auto_backend_may_fallback_to_opencv(monkeypatch):
    backend, _ = _make_backend("cpu")
    loaded = []
    monkeypatch.setenv("BOXMOT_REID_BACKEND", "auto")
    monkeypatch.setattr(
        backend,
        "_ensure_onnxruntime_installed",
        lambda: (_ for _ in ()).throw(ImportError("unavailable")),
    )
    monkeypatch.setattr(backend, "_load_opencv_dnn", loaded.append)

    backend.load_model("model.onnx")

    assert backend._backend == "opencv"
    assert loaded == ["model.onnx"]


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


def test_ensure_onnxruntime_installed_uses_shared_cpu_requirement(monkeypatch):
    backend, install_calls = _make_backend("cpu")
    monkeypatch.setattr(onnx_backend_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ONNXBackend, "_requirement_satisfied", staticmethod(lambda _requirement: False))

    backend._ensure_onnxruntime_installed()

    assert install_calls == [("onnxruntime==1.24.3",)]


def test_load_model_uses_selected_execution_providers(monkeypatch, tmp_path):
    backend, _ = _make_backend("cpu")
    requested: dict[str, object] = {}

    class FakeInputs:
        name = "images"
        shape = ["batch", 3, 20, 10]
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
    monkeypatch.setenv("BOXMOT_REID_ORT_BUCKETS", "2")

    backend.input_shape = (384, 128)
    model_path = tmp_path / "model.onnx"
    backend.load_model(model_path)

    assert requested["model_path"] == str(model_path)
    # device=cpu → CPU EP only on macOS
    assert requested["providers"] == ["CPUExecutionProvider"]
    assert backend.providers == ["CPUExecutionProvider"]
    assert backend.input_shape == (20, 10)
    assert backend._pad_buffers[2].shape == (2, 3, 20, 10)
