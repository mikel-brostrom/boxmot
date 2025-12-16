import sys
import platform
import torch

from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class OpenVINOExporter(BaseExporter):
    group = "openvino"

    def export(self) -> str:
        import openvino as ov

        # 1) Paths
        onnx_path = self.file.with_suffix(".onnx")
        export_dir = self.file.parent / f"{self.file.stem}_openvino_model"
        export_dir.mkdir(parents=True, exist_ok=True)

        xml_name = self.file.with_suffix(".xml").name
        xml_path = export_dir / xml_name

        LOGGER.info(f"Exporting OpenVINO with openvino {ov.__version__}...")

        # 2) Convert → ov.Model (prefer PyTorch frontend; fallback to ONNX if present)
        ov_model = None
        convert_err = None

        # Prefer PyTorch -> OV
        try:
            self.model.eval()
            im = self.im
            if self.half:
                # keep input consistent with half export intent
                im = im.half()
                self.model = self.model.half()

            ov_model = ov.convert_model(
                self.model,
                input=None if self.dynamic else [im.shape],
                example_input=im,
            )
        except Exception as e:
            convert_err = e
            ov_model = None

        # Fallback: ONNX -> OV (only if ONNX exists)
        if ov_model is None and onnx_path.exists():
            LOGGER.warning(f"PyTorch→OpenVINO conversion failed ({convert_err}); falling back to ONNX at {onnx_path}")
            ov_model = ov.convert_model(input_model=str(onnx_path))

        if ov_model is None:
            raise RuntimeError(
                f"OpenVINO conversion failed. PyTorch frontend error: {convert_err}. "
                f"ONNX fallback not available (missing: {onnx_path})."
            )

        # 3) Optional: set lightweight RT info
        try:
            ov_model.set_rt_info("BoxMOT", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
        except Exception:
            pass  # RT info is optional

        # 4) Save IR (XML+BIN). Prefer ov.save_model (supports compress_to_fp16)
        try:
            ov.save_model(ov_model, str(xml_path), compress_to_fp16=bool(self.half))
        except TypeError:
            # Older OpenVINO API (no compress_to_fp16 kw)
            ov.save_model(ov_model, str(xml_path))
            if self.half:
                LOGGER.warning("OpenVINO save_model() does not support compress_to_fp16 in this version; saved as FP32.")

        return str(xml_path)
