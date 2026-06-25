from types import SimpleNamespace

import boxmot.reid.backends.dependencies as deps


class DummyChecker:
    def __init__(self):
        self.calls = []

    def check_packages(self, requirements, extra_args=None):
        self.calls.append((tuple(requirements), tuple(extra_args or ())))


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


def test_ensure_reid_backend_requirements_skips_install_when_any_runtime_matches(monkeypatch):
    checker = DummyChecker()
    monkeypatch.setattr(
        deps,
        "requirement_satisfied",
        lambda requirement: requirement == "onnxruntime-silicon>=1.18.1",
    )

    deps.ensure_reid_backend_requirements(
        checker,
        "onnx",
        requirements=("onnxruntime==1.24.3", "onnxruntime-silicon>=1.18.1"),
    )

    assert checker.calls == []


def test_ensure_reid_backend_requirements_installs_tensorrt_with_nvidia_index(monkeypatch):
    checker = DummyChecker()
    monkeypatch.setattr(deps, "requirement_satisfied", lambda _requirement: False)

    deps.ensure_reid_backend_requirements(checker, "tensorrt")

    assert checker.calls == [
        (("nvidia-tensorrt",), ("--extra-index-url", "https://pypi.ngc.nvidia.com")),
    ]
