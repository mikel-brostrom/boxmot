import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict, namedtuple
from boxmot.utils import logger as LOGGER
from boxmot.appearance.backends.base_backend import BaseModelBackend

class TensorRTBackend(BaseModelBackend):
    def __init__(self, weights, device, half):
        self.is_trt10 = False
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

        self.is_trt10 = not hasattr(self.model_, "num_bindings")
        num = range(self.model_.num_io_tensors) if self.is_trt10 else range(self.model_.num_bindings)

        # Parse bindings
        for index in num:
            if self.is_trt10:
                name = self.model_.get_tensor_name(index)
                dtype = trt.nptype(self.model_.get_tensor_dtype(name))
                is_input = self.model_.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if is_input and -1 in tuple(self.model_.get_tensor_shape(name)):
                        self.context.set_input_shape(name, tuple(self.model_.get_tensor_profile_shape(name, 0)[1]))
                if is_input and dtype == np.float16:
                    self.fp16 = True

                shape = tuple(self.context.get_tensor_shape(name))

            else:
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
        temp_im_batch = im_batch.clone()
        batch_array = []
        inp_batch = im_batch.shape[0]
        out_batch = self.bindings["output"].shape[0]
        resultant_features = []

        # Divide batch to sub batches
        while inp_batch > out_batch:
            batch_array.append(temp_im_batch[:out_batch])
            temp_im_batch = temp_im_batch[out_batch:]
            inp_batch = temp_im_batch.shape[0]
        if temp_im_batch.shape[0] > 0:
            batch_array.append(temp_im_batch)
        
        for temp_batch in batch_array:
            # Adjust for dynamic shapes
            if temp_batch.shape != self.bindings["images"].shape:
                if self.is_trt10:
                    
                    self.context.set_input_shape("images", temp_batch.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=temp_batch.shape)
                    self.bindings["output"].data.resize_(tuple(self.context.get_tensor_shape("output")))
                else:
                    i_in = self.model_.get_binding_index("images")
                    i_out = self.model_.get_binding_index("output")
                    self.context.set_binding_shape(i_in, temp_batch.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=temp_batch.shape)
                    output_shape = tuple(self.context.get_binding_shape(i_out))
                    self.bindings["output"].data.resize_(output_shape)

            s = self.bindings["images"].shape
            assert temp_batch.shape == s, f"Input size {temp_batch.shape} does not match model size {s}"

            self.binding_addrs["images"] = int(temp_batch.data_ptr())

            # Execute inference
            self.context.execute_v2(list(self.binding_addrs.values()))
            features = self.bindings["output"].data
            resultant_features.append(features.clone())

        if len(resultant_features)== 1:
            return resultant_features[0]
        else:
            rslt_features = torch.cat(resultant_features,dim=0)
            rslt_features= rslt_features[:im_batch.shape[0]]
            return rslt_features
