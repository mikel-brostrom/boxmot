import platform
from importlib.metadata import PackageNotFoundError, version

from packaging.requirements import Requirement

from boxmot.reid.backends.base_backend import BaseModelBackend


class ONNXBackend(BaseModelBackend):
    _CUDA_RUNTIME_REQUIREMENTS = ("onnxruntime-gpu>=1.18.1",)
    _DARWIN_RUNTIME_REQUIREMENTS = (
        "onnxruntime>=1.18.1",
        "onnxruntime-silicon>=1.18.1",
    )
    _DEFAULT_RUNTIME_REQUIREMENTS = ("onnxruntime>=1.18.1",)
    _DEVICE_PROVIDER_ORDER = {
        "cuda": ("CUDAExecutionProvider",),
        "mps": ("CoreMLExecutionProvider", "MPSExecutionProvider"),
    }
    _SYSTEM_PROVIDER_ORDER = {
        "Darwin": ("CoreMLExecutionProvider", "MPSExecutionProvider"),
        "Windows": ("DmlExecutionProvider",),
    }

    def __init__(self, weights, device, half, preprocess=None):
        super().__init__(weights, device, half, preprocess=preprocess)
        self.nhwc = False
        self.half = half

    @staticmethod
    def _device_type(device) -> str:
        return str(getattr(device, "type", device))

    @staticmethod
    def _requirement_satisfied(requirement: str) -> bool:
        parsed = Requirement(requirement)
        try:
            installed_version = version(parsed.name)
        except PackageNotFoundError:
            return False
        return not parsed.specifier or parsed.specifier.contains(installed_version, prereleases=True)

    def _runtime_requirements(self) -> tuple[str, ...]:
        if self._device_type(self.device) == "cuda":
            return self._CUDA_RUNTIME_REQUIREMENTS
        if platform.system() == "Darwin":
            return self._DARWIN_RUNTIME_REQUIREMENTS
        return self._DEFAULT_RUNTIME_REQUIREMENTS

    def _ensure_onnxruntime_installed(self) -> None:
        requirements = self._runtime_requirements()
        if any(self._requirement_satisfied(requirement) for requirement in requirements):
            return
        self.checker.check_packages((requirements[0],))

    def _select_execution_providers(self, available_providers) -> list[str]:
        device_type = self._device_type(self.device)
        preferred = list(self._DEVICE_PROVIDER_ORDER.get(device_type, ()))
        if device_type != "cuda":
            preferred.extend(self._SYSTEM_PROVIDER_ORDER.get(platform.system(), ()))
        preferred.append("CPUExecutionProvider")

        providers: list[str] = []
        for provider in preferred:
            if provider in available_providers and provider not in providers:
                providers.append(provider)

        if providers:
            return providers
        return list(available_providers) if available_providers else ["CPUExecutionProvider"]

    def load_model(self, w):
        self._ensure_onnxruntime_installed()
        import onnxruntime

        available_providers = onnxruntime.get_available_providers()
        providers = self._select_execution_providers(available_providers)
        self.providers = providers
        self.session = onnxruntime.InferenceSession(str(w), providers=providers)

    def forward(self, im_batch):
        # Convert torch tensor to numpy (onnxruntime expects numpy arrays)
        im_batch = im_batch.cpu().numpy()

        # Run inference using ONNX session
        features = self.session.run(
            [self.session.get_outputs()[0].name],
            {self.session.get_inputs()[0].name: im_batch},
        )[0]

        return features
