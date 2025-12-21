import os
import shutil
from pathlib import Path


from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class TFLiteExporter(BaseExporter):
    group = "tflite"
    cmds = "--extra-index-url https://pypi.ngc.nvidia.com"

    def export(self) -> str:
        import onnx2tf
        import tensorflow as tf

        LOGGER.info(f"Exporting TFLite with tensorflow {tf.__version__}...")

        # 1) Paths
        onnx_path = self.file.with_suffix(".onnx")
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX file: {onnx_path}")

        saved_model_dir = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        # Match onnx2tf expectation of a trailing separator
        output_folder_path = str(saved_model_dir) + os.sep

        # Clean old export folder
        if saved_model_dir.is_dir():
            shutil.rmtree(saved_model_dir)

        # 2) Convert ONNX -> SavedModel (+ TFLite artifacts depending on onnx2tf version)
        # Prefer onnxslim pipeline
        import inspect

        kwargs = dict(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=output_folder_path,
            not_use_onnxsim=True,
            verbosity=True,
        )

        sig = inspect.signature(onnx2tf.convert)

        # Ask onnx2tf to emit FP16 tflite when available/desired
        if "output_float16_tflite" in sig.parameters:
            kwargs["output_float16_tflite"] = bool(self.half)

        # Keep behavior stable across versions (set optional flags only if supported)
        if "disable_group_convolution" in sig.parameters:
            kwargs["disable_group_convolution"] = False

        onnx2tf.convert(**kwargs)

        # 3) Pick the best produced .tflite (Ultralytics naming preference)
        tflites = sorted(saved_model_dir.rglob("*.tflite"))
        tflites = [p for p in tflites if "quant_with_int16_act.tflite" not in p.name]

        if not tflites:
            # Some onnx2tf builds only emit SavedModel; treat folder as the artifact.
            LOGGER.warning("No .tflite files found; returning SavedModel directory path.")
            return output_folder_path

        # Prefer float16 if half=True, else float32; fallback to first match
        preferred = None
        if self.half:
            preferred = next((p for p in tflites if "float16" in p.stem.lower() or "fp16" in p.stem.lower()), None)
        else:
            preferred = next((p for p in tflites if "float32" in p.stem.lower() or "fp32" in p.stem.lower()), None)

        tflite_path = preferred or tflites[0]

        return str(tflite_path)
