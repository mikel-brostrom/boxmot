from types import SimpleNamespace

import boxmot.reid.backends.dependencies as deps


def test_reid_backend_requirements_selects_onnx_gpu_runtime():
    requirements = deps.reid_backend_requirements("onnx", device=SimpleNamespace(type="cuda"))

    assert requirements == ("onnxruntime-gpu>=1.18.1",)


def test_reid_backend_requirements_accepts_macos_onnx_runtime_alternatives():
    requirements = deps.reid_backend_requirements(
        "onnx",
        device=SimpleNamespace(type="cpu"),
        system_name="Darwin",
    )

    assert requirements == ("onnxruntime==1.24.3", "onnxruntime-silicon>=1.18.1")


def test_require_reid_backend_requirements_accepts_any_matching_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(
        deps,
        "requirement_satisfied",
        lambda requirement: requirement == "onnxruntime-silicon>=1.18.1",
    )

    monkeypatch.setattr(deps, "require_packages", lambda *args, **kwargs: calls.append((args, kwargs)))

    deps.require_reid_backend_requirements(
        "onnx",
        requirements=("onnxruntime==1.24.3", "onnxruntime-silicon>=1.18.1"),
    )

    assert calls == []


def test_require_reid_backend_requirements_explains_tensorrt_nvidia_index(monkeypatch):
    calls = []
    monkeypatch.setattr(deps, "requirement_satisfied", lambda _requirement: False)
    monkeypatch.setattr(
        deps,
        "require_packages",
        lambda requirements, *, purpose, extra_args: calls.append((requirements, purpose, extra_args)),
    )

    deps.require_reid_backend_requirements("tensorrt")

    assert calls == [
        (("nvidia-tensorrt",), "tensorrt ReID runtime", ("--extra-index-url", "https://pypi.ngc.nvidia.com")),
    ]
