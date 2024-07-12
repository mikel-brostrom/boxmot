import os
from pathlib import Path
import openvino.runtime as ov
from openvino.tools import mo
from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class OpenVINOExporter(BaseExporter):
    def export(self):
        self.checker.check_packages(
            ("openvino-dev>=2023.0",)
        )
        f = str(self.file).replace(self.file.suffix, f"_openvino_model{os.sep}")
        f_onnx = self.file.with_suffix(".onnx")
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)
        try:
            LOGGER.info(f"\nStarting export with openvino {ov.__version__}...")
            ov_model = mo.convert_model(
                f_onnx,
                model_name=self.file.with_suffix(".xml"),
                framework="onnx",
                compress_to_fp16=self.half,
            )
            ov.serialize(ov_model, f_ov)
        except Exception as e:
            LOGGER.error(f"Export failure: {e}")
        LOGGER.info(f"Export success, saved as {f_ov} ({self.file_size(f_ov):.1f} MB)")
        return f