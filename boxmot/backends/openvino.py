from pathlib import Path

import numpy as np
import torch

from boxmot.backends.backend import Backend
from boxmot.utils import logger as LOGGER


class OpenVinoBackend(Backend):

    def __init__(self, 
                 weights: str | Path, 
                 half: bool, 
                 nhwc: bool = False,
                 numpy: bool = True):
        super().__init__(half=half, nhwc=nhwc, numpy=numpy)
        self.weights = Path(weights)
        self.model = self.load()
        self.output_name = next(iter(self.model.outputs))

    def load(self):
        self.checker.check_packages(("openvino-dev>=2022.3",))

        LOGGER.info(f"Loading {self.weights} for OpenVINO inference...")
        try:
            # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout
        except ImportError:
            LOGGER.error(
                f"Running {self.__class__} with the specified OpenVINO weights\n{self.weights.name}\n"
                "requires openvino pip package to be installed!\n"
                "$ pip install openvino-dev>=2022.3\n"
            )
        ie = Core()

        if not self.weights.is_file():  # if not *.xml
            self.weights = next(
                self.weights.glob("*.xml")
            )  # get *.xml file from *_openvino_model dir

        network = ie.read_model(model=self.weights,
                                weights=self.weights.with_suffix(".bin"))

        if network.get_parameters()[0].get_layout().empty:
            network.get_parameters()[0].set_layout(Layout("NCWH"))

        model = ie.compile_model(
            network, device_name="CPU"
        )  # device_name="MYRIAD" for Intel NCS2

        return model

    def preprocess(self, x: torch.Tensor) -> np.ndarray:
        x = super().preprocess(x)
        return x.cpu().numpy()

    def process(self, x: np.ndarray):
        y = self.model([x])[self.output_name]
        return y
