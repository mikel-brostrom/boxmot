from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.reid.exporters.onnx_exporter import ensure_onnx_export
from boxmot.utils import logger as LOGGER


class OpenVINOExporter(BaseExporter):
    group = "openvino"

    def __init__(
        self,
        model,
        im,
        file,
        opset: int | None = None,
        dynamic: bool = False,
        half: bool = False,
        simplify: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            model=model,
            im=im,
            file=file,
            optimize=False,
            dynamic=dynamic,
            half=half,
            simplify=simplify,
            verbose=verbose,
        )
        self.opset = opset

    def export(self) -> str:
        import openvino as ov

        onnx_path = ensure_onnx_export(
            model=self.model,
            im=self.im,
            file=self.file,
            opset=self.opset,
            dynamic=self.dynamic,
            half=self.half,
            simplify=self.simplify,
            verbose=self.verbose,
        )
        export_dir = self.file.parent / f"{self.file.stem}_openvino_model"
        export_dir.mkdir(parents=True, exist_ok=True)
        xml_path = export_dir / self.file.with_suffix(".xml").name

        LOGGER.info(f"Exporting OpenVINO with openvino {ov.__version__}...")

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
