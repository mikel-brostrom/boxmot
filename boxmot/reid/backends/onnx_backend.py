from __future__ import annotations

import os
import platform
from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.requirements import Requirement

from boxmot.reid.backends.base_backend import BaseModelBackend
from boxmot.utils import logger as LOGGER


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
        # ONNX Runtime always consumes plain numpy arrays, so any torch tensor
        # we build will be `.cpu().numpy()`-ed before forward(). Materialising
        # crops on MPS/CUDA only to immediately copy back to host wastes time
        # (measured ~+50% on get_crops on Apple Silicon). Remember the user's
        # requested device for execution-provider selection, but keep all torch
        # tensors on CPU.
        self._requested_device = device
        cpu_device = torch.device("cpu") if isinstance(device, torch.device) else "cpu"
        super().__init__(weights, cpu_device, half, preprocess=preprocess)
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
        if self._device_type(self._requested_device) == "cuda":
            return self._CUDA_RUNTIME_REQUIREMENTS
        if platform.system() == "Darwin":
            return self._DARWIN_RUNTIME_REQUIREMENTS
        return self._DEFAULT_RUNTIME_REQUIREMENTS

    def _ensure_onnxruntime_installed(self) -> None:
        requirements = self._runtime_requirements()
        if any(self._requirement_satisfied(requirement) for requirement in requirements):
            return
        self.checker.check_packages((requirements[0],))

    @staticmethod
    def _select_runtime_backend() -> str:
        """Pick the ONNX runtime backend.

        Honours ``BOXMOT_REID_BACKEND`` (also consulted by the native C++ ReID
        path), so users can swap between ``onnxruntime`` and OpenCV's DNN
        module without editing code:

        * ``ort`` / ``onnxruntime`` → ``onnxruntime`` (default)
        * ``opencv`` / ``cv`` / ``dnn`` → ``cv2.dnn.readNetFromONNX``
        * ``auto`` / unset → ``onnxruntime``
        """
        raw = os.environ.get("BOXMOT_REID_BACKEND", "").strip().lower()
        if raw in {"opencv", "cv", "dnn", "opencv_dnn"}:
            return "opencv"
        return "ort"

    def _select_execution_providers(self, available_providers) -> list[str]:
        device_type = self._device_type(self._requested_device)
        if device_type == "cpu":
            return ["CPUExecutionProvider"]
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

    _ORT_TYPE_TO_NUMPY = {
        "tensor(float)": "float32",
        "tensor(float16)": "float16",
        "tensor(double)": "float64",
    }

    # CoreML recompiles on every new input shape, so we keep a small set of
    # bucket sessions (each with a static batch dim) and dispatch per call.
    # Powers-of-two up to 16 cover the typical ReID per-frame det count well
    # without spending too much memory on warm sessions.
    _COREML_DEFAULT_BUCKETS = (1, 2, 4, 8, 16)

    @staticmethod
    def _parse_bucket_env(value: str) -> tuple[int, ...]:
        out: list[int] = []
        for part in value.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                n = int(part)
            except ValueError:
                continue
            if n > 0:
                out.append(n)
        return tuple(sorted(set(out))) if out else ()

    def _make_session(self, weights, providers, batch_size: int | None):
        import os

        import onnxruntime

        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Silence ONNX Runtime's chatty default warnings (e.g. CoreML / CUDA EP
        # node-assignment notices) so they don't tear up Rich live displays.
        # Override at runtime via BOXMOT_ORT_LOG_LEVEL in {0..4} (VERBOSE,
        # INFO, WARNING, ERROR, FATAL).
        log_level = 3
        raw_level = os.environ.get("BOXMOT_ORT_LOG_LEVEL", "").strip()
        if raw_level:
            try:
                parsed = int(raw_level)
                if 0 <= parsed <= 4:
                    log_level = parsed
            except ValueError:
                pass
        sess_opts.log_severity_level = log_level
        # Free shape symbols (e.g. 'batch') can be pinned to a concrete value to
        # avoid repeated CoreML graph recompilations across calls.
        if batch_size is not None:
            sess_opts.add_free_dimension_override_by_name("batch", batch_size)
        return onnxruntime.InferenceSession(str(weights), sess_options=sess_opts, providers=providers)

    def load_model(self, w):
        self._backend = self._select_runtime_backend()
        if self._backend == "opencv":
            self._load_opencv_dnn(w)
            return
        self._ensure_onnxruntime_installed()
        import numpy as np
        import onnxruntime

        available_providers = onnxruntime.get_available_providers()
        providers = self._select_execution_providers(available_providers)
        self.providers = providers

        active_provider = providers[0] if providers else ""

        # Resolve bucket configuration:
        # 1. honor explicit BOXMOT_REID_ORT_BATCH (single fixed batch)
        # 2. honor BOXMOT_REID_ORT_BUCKETS (comma list of batch sizes)
        # 3. otherwise use sensible defaults: multi-bucket on CoreML, dynamic elsewhere
        env_batch = os.environ.get("BOXMOT_REID_ORT_BATCH", "").strip()
        env_buckets = os.environ.get("BOXMOT_REID_ORT_BUCKETS", "").strip()

        buckets: tuple[int, ...] = ()
        if env_batch:
            try:
                n = int(env_batch)
                if n > 0:
                    buckets = (n,)
            except ValueError:
                pass
        elif env_buckets:
            buckets = self._parse_bucket_env(env_buckets)
        elif active_provider == "CoreMLExecutionProvider":
            buckets = self._COREML_DEFAULT_BUCKETS

        # Probe the model's declared batch dim once with a dynamic session so we
        # can honour models exported with a fixed batch (e.g. legacy exports).
        probe_session = self._make_session(w, providers, None)
        probe_input = probe_session.get_inputs()[0]
        probe_first = probe_input.shape[0] if probe_input.shape else None
        static_batch = probe_first if isinstance(probe_first, int) and probe_first > 0 else None
        if static_batch is not None:
            # Model only accepts this batch size; force a single bucket.
            buckets = (static_batch,)

        # Build sessions
        self._sessions: dict[int, "onnxruntime.InferenceSession"] = {}
        if buckets:
            for bs in buckets:
                if static_batch is not None:
                    # No need for free-dim override; the model dim is already pinned.
                    self._sessions[bs] = probe_session if bs == static_batch else self._make_session(w, providers, None)
                else:
                    self._sessions[bs] = self._make_session(w, providers, bs)
            base = self._sessions[buckets[0]]
        else:
            base = probe_session
            self._sessions[0] = base  # 0 == dynamic batch

        input_meta = base.get_inputs()[0]
        self._input_name = input_meta.name
        self._output_name = base.get_outputs()[0].name
        np_dtype_name = self._ORT_TYPE_TO_NUMPY.get(input_meta.type, "float32")
        self._input_np_dtype = np.dtype(np_dtype_name)
        self.session = base  # back-compat alias

        # Compatibility shim: tests still introspect `_fixed_batch_size`.
        self._fixed_batch_size = buckets[0] if len(buckets) == 1 else None
        self._buckets: tuple[int, ...] = buckets

        # Pre-allocate pad buffers per bucket so we don't reallocate per call.
        # Derive crop shape from the model's declared input dims (skip batch).
        meta_dims = list(getattr(input_meta, "shape", []) or [])
        crop_dims: list[int] = []
        for d in meta_dims[1:]:
            if isinstance(d, int) and d > 0:
                crop_dims.append(d)
            else:
                # Fall back to BaseModelBackend.input_shape when the model
                # leaves spatial dims symbolic.
                crop_dims = []
                break
        if not crop_dims:
            input_shape = tuple(getattr(self, "input_shape", (384, 128)))
            crop_dims = [3, *input_shape]
        crop_shape = tuple(crop_dims)
        self._pad_buffers: dict[int, np.ndarray] = {
            bs: np.zeros((bs,) + crop_shape, dtype=self._input_np_dtype)
            for bs in buckets
        }

        # Warm each session so first inference doesn't pay the CoreML compile cost.
        for bs in buckets:
            self._sessions[bs].run([self._output_name], {self._input_name: self._pad_buffers[bs]})

        LOGGER.info(
            f"ONNXRuntime ReID provider={active_provider or '?'} "
            f"input={getattr(input_meta, 'shape', None)} "
            f"dtype={np_dtype_name} buckets={list(buckets) if buckets else 'dynamic'}"
        )

    def forward(self, im_batch):
        import numpy as np

        # Convert torch tensor to numpy (onnxruntime expects numpy arrays)
        im_batch = im_batch.cpu().numpy()
        if self._backend == "opencv":
            return self._forward_opencv_dnn(im_batch, np)
        if im_batch.dtype != self._input_np_dtype:
            im_batch = im_batch.astype(self._input_np_dtype, copy=False)

        if not self._buckets:
            return self._sessions[0].run(
                [self._output_name],
                {self._input_name: im_batch},
            )[0]

        return self._forward_bucketed(im_batch, np)

    def _forward_bucketed(self, im_batch, np):
        n = im_batch.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=np.float32)

        buckets = self._buckets
        smallest = buckets[0]
        outputs: list[np.ndarray] = []
        i = 0
        while i < n:
            remaining = n - i
            # Pick the largest bucket that fits; otherwise pad with smallest.
            candidates = [b for b in buckets if b <= remaining]
            bs = max(candidates) if candidates else smallest
            pad = self._pad_buffers[bs]
            valid = min(bs, remaining)
            if valid == bs:
                chunk = im_batch[i : i + bs]
            else:
                pad[:valid] = im_batch[i : i + valid]
                chunk = pad
            out = self._sessions[bs].run(
                [self._output_name],
                {self._input_name: chunk},
            )[0]
            outputs.append(out[:valid])
            i += valid
        return np.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]

    # ------------------------------------------------------------------
    # OpenCV-DNN runtime path
    # ------------------------------------------------------------------

    def _load_opencv_dnn(self, w) -> None:
        """Load the ONNX graph through ``cv2.dnn`` instead of ONNX Runtime."""
        import cv2
        import numpy as np

        net = cv2.dnn.readNetFromONNX(str(w))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # OpenCV's DNN module ships only the CPU target on the wheels we
        # depend on. Anyone wanting OpenCL/Vulkan can override after init.
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self._cv_net = net
        self._input_np_dtype = np.dtype("float32")
        # Keep these symbols present so downstream code that introspects them
        # (tests, telemetry) keeps working with the OpenCV path.
        self._input_name = "images"
        self._output_name = "output0"
        self._sessions = {}
        self._buckets = ()
        self._fixed_batch_size = None
        self.providers = ["OpenCVDnn"]
        self.session = net  # back-compat alias

        LOGGER.info(
            f"OpenCV-DNN ReID model={os.path.basename(str(w))} "
            f"target=CPU (BOXMOT_REID_BACKEND=opencv)"
        )

    def _forward_opencv_dnn(self, im_batch, np):
        """Run a single forward pass through the loaded ``cv2.dnn.Net``."""
        if im_batch.dtype != self._input_np_dtype:
            im_batch = im_batch.astype(self._input_np_dtype, copy=False)
        if not im_batch.flags["C_CONTIGUOUS"]:
            im_batch = np.ascontiguousarray(im_batch)
        self._cv_net.setInput(im_batch, self._input_name)
        out = self._cv_net.forward()
        return out.astype(np.float32, copy=False)
