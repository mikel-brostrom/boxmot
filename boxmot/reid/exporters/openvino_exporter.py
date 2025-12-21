
from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class OpenVINOExporter(BaseExporter):
    group = "openvino"

    def export(self) -> str:
        import openvino as ov

        onnx_path = self.file.with_suffix(".onnx")
        export_dir = self.file.parent / f"{self.file.stem}_openvino_model"
        export_dir.mkdir(parents=True, exist_ok=True)
        xml_path = export_dir / self.file.with_suffix(".xml").name

        LOGGER.info(f"Exporting OpenVINO with openvino {ov.__version__}...")

        # Ensure ONNX exists (since you run --include onnx this should be true)
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX file for OpenVINO export: {onnx_path}")

        # Fixed CHW for your ReID crops; only batch should vary
        c, h, w = 3, 256, 128

        # Bounded dynamic batch (pick something that covers your worst case)
        max_batch = 80
        batch_dim = ov.Dimension(1, max_batch)

        if self.dynamic:
            shape = [batch_dim, c, h, w]
        else:
            n = int(self.im.shape[0])
            shape = [n, c, h, w]

        # Convert from ONNX and FORCE the input shape (name must match ONNX input: "images")
        ov_model = ov.convert_model(
            str(onnx_path),
            input=[("images", shape)],
        )

        LOGGER.info(f"OpenVINO input partial shape (final): {ov_model.inputs[0].partial_shape}")

        # Save IR (compress_to_fp16 only affects weights)
        try:
            ov.save_model(ov_model, str(xml_path), compress_to_fp16=bool(self.half))
        except TypeError:
            ov.save_model(ov_model, str(xml_path))

        return str(xml_path)
