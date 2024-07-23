import numpy as np
from pathlib import Path
from boxmot.utils import logger as LOGGER

from boxmot.appearance.backends.base_backend import BaseModelBackend


class OpenVinoBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):
        self.checker.check_packages(("openvino-dev>=2022.3",))

        LOGGER.info(f"Loading {w} for OpenVINO inference...")
        try:
            # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout
        except ImportError:
            LOGGER.error(
                f"Running {self.__class__} with the specified OpenVINO weights\n{w.name}\n"
                "requires openvino pip package to be installed!\n"
                "$ pip install openvino-dev>=2022.3\n"
            )
        ie = Core()
        if not Path(w).is_file():  # if not *.xml
            w = next(
                Path(w).glob("*.xml")
            )  # get *.xml file from *_openvino_model dir
        network = ie.read_model(model=w, weights=Path(w).with_suffix(".bin"))
        if network.get_parameters()[0].get_layout().empty:
            network.get_parameters()[0].set_layout(Layout("NCWH"))
        self.executable_network = ie.compile_model(
            network, device_name="CPU"
        )  # device_name="MYRIAD" for Intel NCS2
        self.output_layer = next(iter(self.executable_network.outputs))

    def forward(self, im_batch):
        im_batch = im_batch.cpu().numpy()  # FP32
        features = self.executable_network([im_batch])[self.output_layer]
        return features
