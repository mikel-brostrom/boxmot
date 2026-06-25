from pathlib import Path

import numpy as np

from boxmot.reid.backends.base_backend import BaseModelBackend
from boxmot.reid.backends.dependencies import ensure_reid_backend_requirements
from boxmot.utils import logger as LOGGER


class OpenVinoBackend(BaseModelBackend):

    def __init__(self, weights, device, half, preprocess=None):
        super().__init__(weights, device, half, preprocess=preprocess)
        self.nhwc = False
        self.half = half
        self._max_batch: int | None = None

    def load_model(self, w):
        ensure_reid_backend_requirements(self.checker, "openvino")

        LOGGER.info(f"Loading {w} for OpenVINO inference...")
        from openvino import Core, Layout

        ie = Core()
        w = Path(w)
        LOGGER.info(w)
        if w.suffix == '.bin':
            w = w.with_suffix('.xml')

        if not w.is_file():  # if not *.xml
            w = next(
                Path(w).glob("*.xml")
            )  # get *.xml file from *_openvino_model dir
        network = ie.read_model(model=w, weights=Path(w).with_suffix(".bin"))
        if network.get_parameters()[0].get_layout().empty:
            # PyTorch / ONNX export uses NCHW (batch, channel, height,
            # width). Labeling it as ``"NCWH"`` would tell OpenVINO that
            # the spatial dims are swapped, which silently corrupts
            # layout-aware optimization passes and shows up as numeric
            # drift versus the source PyTorch model.
            network.get_parameters()[0].set_layout(Layout("NCHW"))
        self.executable_network = ie.compile_model(
            network,
            device_name="CPU",
            # OpenVINO 2025+ defaults to BF16 inference on CPU plugins,
            # which truncates accumulators and causes ~5% output drift
            # versus the source PyTorch model. Pin FP32 inference and
            # ACCURACY execution mode for parity with the original
            # weights (disables additional approximate fusions).
            config={
                "INFERENCE_PRECISION_HINT": "f32",
                "EXECUTION_MODE_HINT": "ACCURACY",
            },
        )  # device_name="MYRIAD" for Intel NCS2
        self.output_layer = next(iter(self.executable_network.outputs))

        # Honor the model's batch-dim bounds (e.g. 1..80) by chunking inputs
        # in ``forward``; OpenVINO raises if the actual batch exceeds the
        # upper bound declared in the IR.
        try:
            batch_dim = self.executable_network.inputs[0].partial_shape[0]
            if batch_dim.is_dynamic and batch_dim.get_max_length() > 0:
                self._max_batch = int(batch_dim.get_max_length())
            elif batch_dim.is_static:
                self._max_batch = int(batch_dim.get_length())
        except Exception:
            self._max_batch = None

    def forward(self, im_batch):
        im_batch = im_batch.cpu().numpy()  # FP32
        max_batch = self._max_batch
        if max_batch is None or im_batch.shape[0] <= max_batch:
            return self.executable_network([im_batch])[self.output_layer]

        outputs = []
        for start in range(0, im_batch.shape[0], max_batch):
            chunk = im_batch[start : start + max_batch]
            outputs.append(self.executable_network([chunk])[self.output_layer])
        return np.concatenate(outputs, axis=0)
