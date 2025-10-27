import json
import sys
from collections import OrderedDict, namedtuple
from typing import List, Tuple

import numpy as np
import torch

from boxmot.appearance.backends.base_backend import BaseModelBackend
from boxmot.utils import logger as LOGGER


def _ver_tuple(v: str) -> Tuple[int, ...]:
    try:
        return tuple(int(p) for p in v.split(".") if p.isdigit())
    except Exception:
        return (0,)


class TensorRTBackend(BaseModelBackend):
    """
    TensorRT runtime backend with Ultralytics-like engine loading:

    - Accepts a raw .engine OR an engine preceded by a 4-byte little-endian metadata length and JSON metadata.
      If the metadata contains {"dla": N}, we set runtime.DLA_core=N before deserialization.
    - Works with TRT >=7, supports both legacy (<10) and TRT 10+ tensor APIs.
    - Auto-detects FP16 and dynamic shapes. Handles CPU->CUDA fallback.
    - Discovers input/output names; prefers 'images'/'output' if present.
    """

    def __init__(self, weights, device, half):
        # Initialize everything that load_model() uses BEFORE BaseModelBackend.__init__()
        self.is_trt10 = False
        self.nhwc = False
        self.half = bool(half)
        self.device = device
        self.weights = weights

        self.fp16 = False
        self.dynamic = False
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.bindings: "OrderedDict[str, tuple]" = OrderedDict()
        self.binding_addrs: "OrderedDict[str, int]" = OrderedDict()

        # BaseModelBackend will call self.load_model(self.weights)
        super().__init__(weights, device, half)

    # ------------------------------ Load ---------------------------------- #
    def load_model(self, w: str):
        LOGGER.info(f"Loading {w} for TensorRT inference...")
        # Ensure attributes exist even if BaseModelBackend calls us very early
        if not hasattr(self, "fp16"):
            self.fp16 = False
        if not hasattr(self, "dynamic"):
            self.dynamic = False

        self.checker.check_packages(("nvidia-tensorrt",))
        try:
            import tensorrt as trt  # TensorRT library
        except ImportError as e:
            raise ImportError("Please install 'nvidia-tensorrt' to use this backend.") from e

        trt_ver = getattr(trt, "__version__", "0")
        if _ver_tuple(trt_ver) < (7, 0, 0):
            raise RuntimeError(f"TensorRT >= 7.0.0 required, found {trt_ver}")
        if trt_ver.startswith("10.1.0"):
            LOGGER.warning("TensorRT 10.1.0 has known issues with some engines.")

        # CPU -> CUDA fallback
        if self.device.type == "cpu":
            if torch.cuda.is_available():
                LOGGER.info("No GPU device specified. Falling back to CUDA:0 for TensorRT.")
                self.device = torch.device("cuda:0")
            else:
                raise ValueError("CUDA device not available for TensorRT inference.")

        # Jetson numpy note for older Python
        if sys.platform.startswith("linux") and sys.version_info <= (3, 8, 10):
            try:
                _ = np.bool  # noqa: F401
            except Exception:
                LOGGER.warning(
                    "If running on Jetson with Python <= 3.8.10 and you hit numpy bool issues, "
                    "pin numpy==1.23.5."
                )

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)

        # -------------------- Deserialize engine with optional metadata -------------------- #
        with open(w, "rb") as f, trt.Runtime(logger) as runtime:
            # Try to read optional metadata header (len + JSON)
            try:
                meta_len_bytes = f.read(4)
                if len(meta_len_bytes) == 4:
                    meta_len = int.from_bytes(meta_len_bytes, byteorder="little", signed=True)
                    if 0 < meta_len < (64 * 1024):
                        meta_json = f.read(meta_len).decode("utf-8")
                        metadata = json.loads(meta_json)
                        dla = metadata.get("dla", None)
                        if dla is not None:
                            try:
                                runtime.DLA_core = int(dla)
                                LOGGER.info(f"Set TensorRT Runtime DLA_core={runtime.DLA_core} from metadata.")
                            except Exception as e:
                                LOGGER.warning(f"Failed to set DLA_core from metadata: {e}")
                    else:
                        # Not a valid metadata block; rewind to start
                        f.seek(0)
                else:
                    f.seek(0)
            except UnicodeDecodeError:
                f.seek(0)
            except Exception as e:
                LOGGER.warning(f"Engine metadata probe failed ({e}); proceeding without metadata.")
                f.seek(0)

            # Deserialize engine bytes
            try:
                self.model_ = runtime.deserialize_cuda_engine(f.read())
            except Exception as e:
                LOGGER.error(
                    f"Failed to deserialize TensorRT engine. "
                    f"(Potential version mismatch: engine built vs runtime {trt_ver})"
                )
                raise e

        # ---------------------------- Create context ----------------------------- #
        try:
            self.context = self.model_.create_execution_context()
        except Exception as e:
            LOGGER.error("Could not create TensorRT execution context.")
            raise e

        # ------------------------- Parse bindings/IO ----------------------------- #
        self.bindings = OrderedDict()
        self.is_trt10 = not hasattr(self.model_, "num_bindings")
        num_range = range(self.model_.num_io_tensors) if self.is_trt10 else range(self.model_.num_bindings)

        input_names: List[str] = []
        output_names: List[str] = []

        for idx in num_range:
            if self.is_trt10:
                name = self.model_.get_tensor_name(idx)
                dtype = trt.nptype(self.model_.get_tensor_dtype(name))
                is_input = self.model_.get_tensor_mode(name) == trt.TensorIOMode.INPUT

                if is_input:
                    shp = tuple(self.model_.get_tensor_shape(name))
                    if -1 in shp:
                        self.dynamic = True
                        opt_shape = tuple(self.model_.get_tensor_profile_shape(name, 0)[1])
                        self.context.set_input_shape(name, opt_shape)
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    output_names.append(name)

                shape = tuple(self.context.get_tensor_shape(name))
            else:
                name = self.model_.get_binding_name(idx)
                dtype = trt.nptype(self.model_.get_binding_dtype(idx))
                is_input = self.model_.binding_is_input(idx)

                if is_input:
                    shp = tuple(self.model_.get_binding_shape(idx))
                    if -1 in shp:
                        self.dynamic = True
                        opt_shape = tuple(self.model_.get_profile_shape(0, idx)[1])
                        self.context.set_binding_shape(idx, opt_shape)
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    output_names.append(name)

                shape = tuple(self.context.get_binding_shape(idx))

            if is_input:
                input_names.append(name)

            # Allocate device tensor
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

        # Build address list in the same enumeration order
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

        # Prefer conventional names if present
        self.input_name = "images" if "images" in input_names else (input_names[0] if input_names else None)
        self.output_name = "output" if "output" in output_names else (output_names[0] if output_names else None)

        if self.input_name is None or self.output_name is None:
            raise RuntimeError(
                f"Could not infer model IO names. Inputs found: {input_names}, outputs found: {output_names}"
            )

        fp16_flag = bool(getattr(self, "fp16", False))
        dyn_flag = bool(getattr(self, "dynamic", False))
        LOGGER.info(
            f"TensorRT ready | TRT {trt_ver} | FP16:{fp16_flag} | dynamic:{dyn_flag} | "
            f"input='{self.input_name}' {self.bindings[self.input_name].shape} | "
            f"output='{self.output_name}' {self.bindings[self.output_name].shape}"
        )

    # ------------------------------ Run ----------------------------------- #
    @torch.no_grad()
    def forward(self, im_batch: torch.Tensor) -> torch.Tensor:
        """
        im_batch: NCHW tensor on the same device as self.device.
        Handles sub-batching to respect engine's (opt) batch dimension.
        """
        assert isinstance(im_batch, torch.Tensor), "im_batch must be a torch.Tensor"
        assert im_batch.device.type == self.device.type, (
            f"Input tensor device {im_batch.device} != engine device {self.device}"
        )

        temp = im_batch.clone()
        batches: List[torch.Tensor] = []

        # Determine engine's current output batch capacity
        out_shape = self.bindings[self.output_name].shape
        out_bdim = int(out_shape[0]) if len(out_shape) > 0 else temp.shape[0]
        if out_bdim <= 0:
            out_bdim = int(temp.shape[0])

        # Split into chunks the engine can handle in one pass
        while temp.shape[0] > out_bdim:
            batches.append(temp[:out_bdim].contiguous())
            temp = temp[out_bdim:]
        if temp.shape[0] > 0:
            batches.append(temp.contiguous())

        outputs: List[torch.Tensor] = []

        for tb in batches:
            # Adjust dynamic shapes if needed
            if tuple(tb.shape) != tuple(self.bindings[self.input_name].shape):
                if self.is_trt10:
                    self.context.set_input_shape(self.input_name, tuple(tb.shape))
                    self.bindings[self.input_name] = self.bindings[self.input_name]._replace(shape=tuple(tb.shape))
                    out_shape = tuple(self.context.get_tensor_shape(self.output_name))
                    self.bindings[self.output_name].data.resize_(out_shape)
                else:
                    i_in = self.model_.get_binding_index(self.input_name)
                    i_out = self.model_.get_binding_index(self.output_name)
                    self.context.set_binding_shape(i_in, tuple(tb.shape))
                    self.bindings[self.input_name] = self.bindings[self.input_name]._replace(shape=tuple(tb.shape))
                    out_shape = tuple(self.context.get_binding_shape(i_out))
                    self.bindings[self.output_name].data.resize_(out_shape)

            exp_shape = self.bindings[self.input_name].shape
            assert tuple(tb.shape) == tuple(exp_shape), (
                f"Input size {tuple(tb.shape)} does not match engine-bound size {tuple(exp_shape)}"
            )

            # Set input address to current tb and run
            self.binding_addrs[self.input_name] = int(tb.data_ptr())

            # Execute (works for both TRT<10 and TRT>=10 when passing ptrs in binding order)
            self.context.execute_v2(list(self.binding_addrs.values()))

            # Clone to avoid overwrite on next iteration
            outputs.append(self.bindings[self.output_name].data.clone())

        if len(outputs) == 1:
            return outputs[0]
        out = torch.cat(outputs, dim=0)
        return out[: im_batch.shape[0]]
