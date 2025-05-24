import os
from pathlib import Path
import openvino as ov
from boxmot.appearance.exporters.base_exporter import BaseExporter

class OpenVINOExporter(BaseExporter):
    group = "openvino"

    def export(self):
        # 1. Take your .onnx name and make a dedicated output folder
        onnx_path = self.file.with_suffix(".onnx")
        export_dir = self.file.with_suffix("_openvino_model")
        export_dir = Path(str(export_dir))
        export_dir.mkdir(parents=True, exist_ok=True)

        # 2. Convert ONNX → ov.Model
        #    (no need for example_input here; ONNX contains the shapes)
        ov_model = ov.convert_model(input_model=onnx_path)
        # :contentReference[oaicite:0]{index=0}

        # 3. Save to IR (XML + BIN)
        xml_path = export_dir / self.file.with_suffix(".xml").name
        #    compress_to_fp16 defaults to True; disable with self.half=False
        ov.save_model(ov_model, xml_path, compress_to_fp16=self.half)
        # :contentReference[oaicite:1]{index=1}

        return str(export_dir)