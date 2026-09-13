from __future__ import annotations

import os
import platform

import torch

from boxmot.reid.backends.base_backend import BaseModelBackend
from boxmot.reid.backends.dependencies import (
    reid_backend_requirements,
    require_reid_backend_requirements,
)
from boxmot.utils import logger as LOGGER
from boxmot.utils.dependencies import MissingDependencyError


class ONNXBackend(BaseModelBackend):
    # Export metadata supplies the architecture and preprocessing contract.
    # ONNX Runtime executes the exported graph directly, so constructing and
    # retaining a second random PyTorch model only adds startup latency and RAM.
    build_source_model = False

    _DEVICE_PROVIDER_ORDER = {
        "cuda": ("CUDAExecutionProvider",),
    }
    _SYSTEM_PROVIDER_ORDER = {
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

    def _runtime_requirements(self) -> tuple[str, ...]:
        return reid_backend_requirements(
            "onnx",
            device=self._requested_device,
            system_name=platform.system(),
        )

    def _require_onnxruntime(self) -> None:
        """Validate the runtime selected for the requested device and platform."""
        require_reid_backend_requirements("onnx", requirements=self._runtime_requirements())

    @staticmethod
    def _select_runtime_backend() -> str:
        """Return the exact runtime requested through ``BOXMOT_REID_BACKEND``.

        ``auto`` (also used when the variable is unset) may choose another
        available implementation. Explicit requests are never downgraded.
        """
        raw = os.environ.get("BOXMOT_REID_BACKEND")
        if raw is None:
            return "auto"
        if raw in {"auto", "onnxruntime", "opencv"}:
            return raw
        raise ValueError(f"Invalid BOXMOT_REID_BACKEND value {raw!r}. Expected one of: auto, onnxruntime, opencv.")

    def _validate_opencv_device(self) -> None:
        device_type = self._device_type(self._requested_device)
        if device_type not in {"auto", "cpu"}:
            raise ValueError(f"OpenCV DNN ReID supports only device=cpu; got device={device_type!r}.")

    def _select_execution_providers(self, available_providers) -> list[str]:
        device_type = self._device_type(self._requested_device)
        available = list(available_providers)
        self._provider_selection_is_explicit = device_type != "auto"

        if device_type == "mps":
            raise ValueError(
                "ONNX Runtime has no MPS execution provider. "
                "Use a Core ML artifact for device='mps' or select device='cpu'."
            )

        explicit_provider = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
        }.get(device_type)
        if explicit_provider is not None:
            if explicit_provider not in available:
                raise RuntimeError(
                    f"{explicit_provider} was explicitly requested for ONNX ReID but is unavailable. "
                    f"Available providers: {available or ['none']}."
                )
            return [explicit_provider]

        if device_type != "auto":
            raise ValueError(f"Unsupported ONNX ReID device {device_type!r}. Expected one of: auto, cpu, cuda, coreml.")

        preferred = list(self._DEVICE_PROVIDER_ORDER["cuda"])
        preferred.extend(self._SYSTEM_PROVIDER_ORDER.get(platform.system(), ()))
        preferred.append("CPUExecutionProvider")
        providers = [provider for provider in preferred if provider in available]
        if not providers:
            raise RuntimeError(
                "No supported ONNX Runtime execution provider is available. "
                f"Available providers: {available or ['none']}."
            )
        return providers

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
        if (
            getattr(self, "_provider_selection_is_explicit", False)
            and providers
            and providers[0] in {"CUDAExecutionProvider", "CoreMLExecutionProvider"}
        ):
            sess_opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
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
        session_kwargs = {}
        if self._device_type(self._requested_device) == "cuda" and "CUDAExecutionProvider" in providers:
            # CUDA indices are logical indices within the caller's existing
            # visibility mask, matching the Torch device used by the detector.
            cuda_index = self._requested_device.index if isinstance(self._requested_device, torch.device) else None
            session_kwargs["provider_options"] = [
                {"device_id": cuda_index or 0} if provider == "CUDAExecutionProvider" else {}
                for provider in providers
            ]
        return onnxruntime.InferenceSession(
            str(weights), sess_options=sess_opts, providers=providers, **session_kwargs
        )

    def load_model(self, w):
        backend_request = self._select_runtime_backend()
        self._backend = "onnxruntime" if backend_request == "auto" else backend_request
        if self._backend == "opencv":
            self._validate_opencv_device()
            self._load_opencv_dnn(w)
            return
        try:
            self._require_onnxruntime()
            import onnxruntime
        except ImportError as exc:
            if backend_request != "auto" or self._device_type(self._requested_device) not in {"auto", "cpu"}:
                if isinstance(exc, MissingDependencyError):
                    raise
                raise RuntimeError("ONNX Runtime was explicitly requested for ReID but is unavailable.") from exc
            self._validate_opencv_device()
            self._backend = "opencv"
            self._load_opencv_dnn(w)
            return
        import numpy as np

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
        self._buckets: tuple[int, ...] = buckets

        # Pre-allocate pad buffers per bucket so we don't reallocate per call.
        # The graph's static spatial dimensions are the execution contract and
        # therefore override stale/missing sidecar metadata. Symbolic spatial
        # dimensions retain the configured checkpoint/registry crop size.
        meta_dims = list(getattr(input_meta, "shape", []) or [])
        if len(meta_dims) != 4:
            raise ValueError(f"ONNX ReID input must be rank-4 NCHW, got shape {meta_dims}")
        channel_dim = meta_dims[1]
        if isinstance(channel_dim, int) and channel_dim > 0 and channel_dim != 3:
            raise ValueError(f"ONNX ReID input must have 3 channels, got shape {meta_dims}")

        configured_shape = tuple(int(dim) for dim in self.input_shape)
        graph_height = meta_dims[2] if isinstance(meta_dims[2], int) and meta_dims[2] > 0 else configured_shape[0]
        graph_width = meta_dims[3] if isinstance(meta_dims[3], int) and meta_dims[3] > 0 else configured_shape[1]
        graph_input_shape = (int(graph_height), int(graph_width))
        if graph_input_shape != configured_shape:
            LOGGER.warning(
                f"ONNX graph input size {graph_input_shape} overrides configured ReID crop size {configured_shape}"
            )
        self.input_shape = graph_input_shape
        crop_shape = (3, *self.input_shape)
        self._pad_buffers: dict[int, np.ndarray] = {
            bs: np.zeros((bs,) + crop_shape, dtype=self._input_np_dtype) for bs in buckets
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

        LOGGER.info(f"OpenCV-DNN ReID model={os.path.basename(str(w))} target=CPU (BOXMOT_REID_BACKEND=opencv)")

    def _forward_opencv_dnn(self, im_batch, np):
        """Run a single forward pass through the loaded ``cv2.dnn.Net``."""
        if im_batch.dtype != self._input_np_dtype:
            im_batch = im_batch.astype(self._input_np_dtype, copy=False)
        if not im_batch.flags["C_CONTIGUOUS"]:
            im_batch = np.ascontiguousarray(im_batch)
        self._cv_net.setInput(im_batch, self._input_name)
        out = self._cv_net.forward()
        return out.astype(np.float32, copy=False)
