import openvino as ov

from boxmot.appearance.exporters.base_exporter import BaseExporter


class OpenVINOExporter(BaseExporter):
    group = "openvino"

    def export(self):

        from openvino.tools import mo

    def export(self) -> str:
        # 1. Paths
        #    assume self.file is e.g. "model.onnx"
        onnx_path = self.file.with_suffix(".onnx")
        export_dir = self.file.parent / f"{self.file.stem}_openvino_model"
        export_dir.mkdir(parents=True, exist_ok=True)

        # 2. Convert ONNX → ov.Model
        ov_model = ov.convert_model(input_model=onnx_path)

        # 3. Save to IR (XML + BIN)
        xml_name = self.file.with_suffix(".xml").name
        xml_path = export_dir / xml_name
        ov.save_model(ov_model, xml_path, compress_to_fp16=self.half)

        # Return the actual XML file so that BaseExporter.with_suffix() works correctly
        return str(xml_path)