from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import torch

from boxmot.backends.backend import Backend
from boxmot.utils import logger as LOGGER


class TensorRTBackend(Backend):
    def __init__(self, 
                 weights: str | Path, 
                 device: str, 
                 half: bool, 
                 nhwc: bool = False,
                 numpy: bool = True):
        self.is_trt10 = False
        super().__init__(half=half, nhwc=nhwc, numpy=numpy)
        self.weights = weights
        self.device = device
        self.trt = self.import_trt()
        self.model = self.load()

        # Execution context
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

        self.is_trt10 = not hasattr(self.model, "num_bindings")
        num = range(self.model.num_io_tensors) if self.is_trt10 else range(self.model.num_bindings)

        # Parse bindings
        for index in num:
            if self.is_trt10:
                name = self.model.get_tensor_name(index)
                dtype = self.trt.nptype(self.model.get_tensor_dtype(name))
                is_input = self.model.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
                if is_input and -1 in tuple(self.model.get_tensor_shape(name)):
                        self.context.set_input_shape(name, tuple(self.model.get_tensor_profile_shape(name, 0)[1]))
                # if is_input and dtype == np.float16:
                #     self.fp16 = True

                shape = tuple(self.context.get_tensor_shape(name))

            else:
                name = self.model.get_binding_name(index)
                dtype = self.trt.nptype(self.model.get_binding_dtype(index))
                is_input = self.model.binding_is_input(index)

                # Handle dynamic shapes
                if is_input and -1 in self.model.get_binding_shape(index):
                    profile_index = 0
                    min_shape, opt_shape, max_shape = self.model.get_profile_shape(profile_index, index)
                    self.context.set_binding_shape(index, opt_shape)

                # if is_input and dtype == np.float16:
                #     self.fp16 = True

                shape = tuple(self.context.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())


    def import_trt(self):
        self.checker.check_packages(("nvidia-tensorrt",))
        try:
            import tensorrt as trt
            return trt
        except ImportError:
            raise ImportError("This backend requires `tensorrt`. Please install it to proceed.")


    def load(self):
        LOGGER.info(f"Loading {str(self.weights)} for TensorRT inference...")
        
        
        dev = torch.device(self.device)

        if dev.type == "cpu":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                raise ValueError("CUDA device not available for TensorRT inference.")

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = self.trt.Logger(self.trt.Logger.INFO)

        # Deserialize the engine
        with open(self.weights, "rb") as f, self.trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())

        return model
    
    def preprocess(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = super().preprocess(x)

        self._cur_batch = x.shape[0]
        temp_x_batch = x.clone()
        batch_array = []
        inp_batch = x.shape[0]
        out_batch = self.bindings["output"].shape[0]
        

        # Divide batch to sub batches
        while inp_batch > out_batch:
            batch_array.append(temp_x_batch[:out_batch])
            temp_x_batch = temp_x_batch[out_batch:]
            inp_batch = temp_x_batch.shape[0]
        if temp_x_batch.shape[0] > 0:
            batch_array.append(temp_x_batch)

        return batch_array


    def process(self, x: list[torch.Tensor]):
        resultant_outputs = []

        for temp_batch in x:
            # Adjust for dynamic shapes
            if temp_batch.shape != self.bindings["images"].shape:
                if self.is_trt10:

                    self.context.set_input_shape("images", temp_batch.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=temp_batch.shape)
                    self.bindings["output"].data.resize_(tuple(self.context.get_tensor_shape("output")))
                else:
                    i_in = self.model.get_binding_index("images")
                    i_out = self.model.get_binding_index("output")
                    self.context.set_binding_shape(i_in, temp_batch.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=temp_batch.shape)
                    output_shape = tuple(self.context.get_binding_shape(i_out))
                    self.bindings["output"].data.resize_(output_shape)

            s = self.bindings["images"].shape
            assert temp_batch.shape == s, f"Input size {temp_batch.shape} does not match model size {s}"

            self.binding_addrs["images"] = int(temp_batch.data_ptr())

            # Execute inference
            self.context.execute_v2(list(self.binding_addrs.values()))
            outputs = self.bindings["output"].data
            resultant_outputs.append(outputs.clone())

        if len(resultant_outputs) == 1:
            return resultant_outputs[0]
        else:
            rslt_outputs = torch.cat(resultant_outputs, dim=0)
            rslt_outputs = rslt_outputs[: self._cur_batch.shape[0]]
            return rslt_outputs
