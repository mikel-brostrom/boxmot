import torch
import numpy as np
from pathlib import Path
from boxmot.utils import logger as LOGGER

from boxmot.appearance.backends.base_backend import BaseModelBackend



class TensorRTBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):

        LOGGER.info(f"Loading {w} for TensorRT inference...")
        self.checker.check_packages(("nvidia-tensorrt",))
        import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

        if device.type == "cpu":
            device = torch.device("cuda:0")
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)
        with open(w, "rb") as f, trt.Runtime(logger) as runtime:
            self.model_ = runtime.deserialize_cuda_engine(f.read())
        self.context = self.model_.create_execution_context()
        self.bindings = OrderedDict()
        self.fp16 = False  # default updated below
        # dynamic = False
        for index in range(self.model_.num_bindings):
            name = self.model_.get_binding_name(index)
            dtype = trt.nptype(self.model_.get_binding_dtype(index))
            if self.model_.binding_is_input(index):
                if -1 in tuple(self.model_.get_binding_shape(index)):  # dynamic
                    # dynamic = True
                    self.context.set_binding_shape(
                        index, tuple(self.model_.get_profile_shape(0, index)[2])
                    )
                if dtype == np.float16:
                    self.fp16 = True
            shape = tuple(self.context.get_binding_shape(index))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(
                name, dtype, shape, im, int(im.data_ptr())
            )
        self.binding_addrs = OrderedDict(
            (n, d.ptr) for n, d in self.bindings.items()
        )
        # batch_size = self.bindings["images"].shape[
        #     0
        # ]  # if dynamic, this is instead max batch size

    def forward(self, im_batch):
        if True and im_batch.shape != self.bindings["images"].shape:
            i_in, i_out = (
                self.model_.get_binding_index(x) for x in ("images", "output")
            )
            self.context.set_binding_shape(
                i_in, im_batch.shape
            )  # reshape if dynamic
            self.bindings["images"] = self.bindings["images"]._replace(
                shape=im_batch.shape
            )
            self.bindings["output"].data.resize_(
                tuple(self.context.get_binding_shape(i_out))
            )
        s = self.bindings["images"].shape
        assert (
            im_batch.shape == s
        ), f"input size {im_batch.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs["images"] = int(im_batch.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        features = self.bindings["output"].data
        return features
