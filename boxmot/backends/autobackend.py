from pathlib import Path
import torch

# TODO: Move all backends into __init__.py
from boxmot.backends import (
    Backend,
    PyTorchBackend,
    ONNXBackend,
    TensorRTBackend,
    TFLiteBackend,
    OpenVinoBackend
)


class AutoBackend:
    _EXT_MAP = {
        "pt":   PyTorchBackend,
        "pth":  PyTorchBackend,
        "jit":  PyTorchBackend,
        "torchscript": PyTorchBackend,
        "onnx": ONNXBackend,
        "engine": TensorRTBackend,
        "tflite": TFLiteBackend,
        "xml":  OpenVinoBackend,
    }

    def __new__(cls,
                identifier: str | Path,
                device: torch.device = torch.device("cpu"),
                half: bool = False) -> Backend:
        """
        Create and return the right backend instance based on weights suffix.
        """
        # TODO: Add GRPC Backend
        # if isinstance(identifier, str) and identifier.startswith("grpc://"):
        #     return GRPCBackend(grpc_url=identifier, *args, **kwargs)
        
        weights = Path(identifier)
        ext = weights.suffix.lstrip(".").lower()

        # find backend class
        backend_cls = cls._EXT_MAP.get(ext)
        if backend_cls is None:
            raise ValueError(f"No backend for extension '.{ext}'")

        # instantiate and return it directly
        return backend_cls(weights=weights, device=device, half=half)

    def __init__(self, *args, **kwargs):
        pass
