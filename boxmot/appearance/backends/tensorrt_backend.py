import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict, namedtuple
from boxmot.utils import logger as LOGGER
from boxmot.appearance.backends.base_backend import BaseModelBackend

class TensorRTBackend(BaseModelBackend):
    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half
        self.device = device
        self.weights = weights
        self.fp16 = False  # Will be updated in load_model
        self.load_model(self.weights)

    def load_model(self, w):
        LOGGER.info(f"Loading {w} for TensorRT inference...")
        self.checker.check_packages(("nvidia-tensorrt",))
        try:
            import tensorrt as trt  # TensorRT library
        except ImportError:
            raise ImportError("Please install tensorrt to use this backend.")

        if self.device.type == "cpu":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                raise ValueError("CUDA device not available for TensorRT inference.")

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)

        # Deserialize the engine
        with open(w, "rb") as f, trt.Runtime(logger) as runtime:
            self.model_ = runtime.deserialize_cuda_engine(f.read())
        
        # Execution context
        self.context = self.model_.create_execution_context()
        self.bindings = OrderedDict()

        # Parse bindings
        for index in range(self.model_.num_bindings):
            name = self.model_.get_binding_name(index)
            dtype = trt.nptype(self.model_.get_binding_dtype(index))
            is_input = self.model_.binding_is_input(index)

            # Handle dynamic shapes
            if is_input and -1 in self.model_.get_binding_shape(index):
                profile_index = 0
                min_shape, opt_shape, max_shape = self.model_.get_profile_shape(profile_index, index)
                self.context.set_binding_shape(index, opt_shape)

            if is_input and dtype == np.float16:
                self.fp16 = True

            shape = tuple(self.context.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def forward(self, im_batch):
        # Adjust for dynamic shapes
        if im_batch.shape != self.bindings["images"].shape:
            i_in = self.model_.get_binding_index("images")
            i_out = self.model_.get_binding_index("output")
            self.context.set_binding_shape(i_in, im_batch.shape)
            self.bindings["images"] = self.bindings["images"]._replace(shape=im_batch.shape)
            output_shape = tuple(self.context.get_binding_shape(i_out))
            self.bindings["output"].data.resize_(output_shape)

        s = self.bindings["images"].shape
        assert im_batch.shape == s, f"Input size {im_batch.shape} does not match model size {s}"

        # Set input buffer
        self.binding_addrs["images"] = int(im_batch.data_ptr())

        # Execute inference
        self.context.execute_v2(list(self.binding_addrs.values()))
        features = self.bindings["output"].data
        return features
