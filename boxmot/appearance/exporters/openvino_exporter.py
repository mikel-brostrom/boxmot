import os
from pathlib import Path
import openvino.runtime as ov
from openvino.tools import mo
from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class OpenVINOExporter(BaseExporter):
    required_packages = ("openvino-dev>=2023.0",)
    
    def export(self):

        f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")
        f_onnx = self.file.with_suffix(".onnx")
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        ov_model = mo.convert_model(
            f_onnx,
            model_name=self.file.with_suffix(".xml"),
            framework="onnx",
            compress_to_fp16=self.half,
        )
        ov.serialize(ov_model, f_ov)
        
        return f