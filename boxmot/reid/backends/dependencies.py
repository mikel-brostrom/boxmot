from __future__ import annotations

import platform
from typing import Any, Iterable, Sequence

from boxmot.utils.checks import requirement_satisfied

ONNX_RUNTIME_VERSION = "==1.24.3"
ONNX_RUNTIME_MIN_VERSION = ">=1.18.1"
TENSORRT_INDEX_ARGS = ("--extra-index-url", "https://pypi.ngc.nvidia.com")

REID_BACKEND_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "onnx": (f"onnxruntime{ONNX_RUNTIME_VERSION}",),
    "onnx-cuda": (f"onnxruntime-gpu{ONNX_RUNTIME_MIN_VERSION}",),
    "onnx-darwin": (
        f"onnxruntime{ONNX_RUNTIME_VERSION}",
        f"onnxruntime-silicon{ONNX_RUNTIME_MIN_VERSION}",
    ),
    "openvino": ("openvino>=2025.2.0",),
    "tensorrt": ("nvidia-tensorrt",),
    "tflite": ("ai-edge-litert>=2.1.0",),
}

REID_BACKEND_INSTALL_ARGS: dict[str, tuple[str, ...]] = {
    "tensorrt": TENSORRT_INDEX_ARGS,
}

def device_type(device: Any) -> str:
    """Normalize a torch/string device object to its type name."""
    return str(getattr(device, "type", device))


def reid_backend_requirements(
    backend: str,
    *,
    device: Any | None = None,
    system_name: str | None = None,
) -> tuple[str, ...]:
    """Return runtime requirements for a ReID backend.

    ONNX has platform/device-specific runtime wheels. On macOS either
    ``onnxruntime`` or ``onnxruntime-silicon`` is acceptable.
    """
    key = backend.strip().lower()
    if key == "onnx":
        if device is not None and device_type(device) == "cuda":
            key = "onnx-cuda"
        elif (system_name or platform.system()) == "Darwin":
            key = "onnx-darwin"
    return REID_BACKEND_REQUIREMENTS[key]


def reid_backend_install_args(backend: str) -> tuple[str, ...]:
    """Return installer flags needed for a ReID backend runtime package."""
    return REID_BACKEND_INSTALL_ARGS.get(backend.strip().lower(), ())


def ensure_reid_backend_requirements(
    checker: Any,
    backend: str,
    *,
    device: Any | None = None,
    requirements: Iterable[str] | None = None,
    extra_args: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Install the first acceptable runtime package when no option is present."""
    runtime_requirements = tuple(requirements or reid_backend_requirements(backend, device=device))
    if any(requirement_satisfied(requirement) for requirement in runtime_requirements):
        return runtime_requirements

    install_args = tuple(extra_args) if extra_args is not None else reid_backend_install_args(backend)
    checker.check_packages((runtime_requirements[0],), extra_args=install_args or None)
    return runtime_requirements
