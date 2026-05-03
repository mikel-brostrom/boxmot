
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

        # Derive CHW from the dummy input so model-specific resolutions
        # (e.g. lmbn uses 384x128, hacnn uses 160x64) are honored. Hardcoding
        # 256x128 here caused OpenVINO's ONNX frontend to fail conversion for
        # models whose spatial dims don't survive the network's pooling
        # stack at the wrong resolution.
        n_dummy, c, h, w = (int(d) for d in self.im.shape)

        if self.dynamic:
            # Truly dynamic batch: ``-1`` makes OpenVINO accept any batch
            # size at inference. A bounded range (e.g. 1..N) would still
            # raise on overflow, defeating the purpose of ``--dynamic``.
            shape = [-1, c, h, w]
        else:
            shape = [n_dummy, c, h, w]

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
