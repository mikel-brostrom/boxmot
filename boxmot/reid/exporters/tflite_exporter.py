import inspect
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.reid.exporters.onnx_exporter import ONNXExporter
from boxmot.utils import logger as LOGGER


class TFLiteExporter(BaseExporter):
    group = "tflite"

    def __init__(self, model, im, file, opset=None, dynamic=False, half=False, simplify=False):
        super().__init__(model, im, file, optimize=False, dynamic=dynamic, half=half, simplify=simplify)
        self.opset = opset

    def export(self) -> str:
        if sys.version_info < (3, 12):
            raise RuntimeError("TFLite export requires Python 3.12 for the pinned onnx2tf flatbuffer_direct stack.")
        if sys.platform not in {"linux", "win32"}:
            raise RuntimeError("TFLite export is only supported on Linux or Windows in this environment.")

        import onnx2tf
        import tensorflow as tf

        LOGGER.info(
            f"Exporting TFLite with tensorflow {tf.__version__} and onnx2tf {onnx2tf.__version__}..."
        )

        onnx_path = self._ensure_onnx_file()
        export_dir = self.file.parent / f"{self.file.stem}_saved_model"
        output_folder_path = str(export_dir) + os.sep

        if export_dir.is_dir():
            shutil.rmtree(export_dir)

        onnx2tf.convert(**self._build_convert_kwargs(onnx2tf, onnx_path, output_folder_path))

        tflite_path = self._select_tflite_artifact(export_dir)
        if tflite_path is None:
            raise RuntimeError(f"onnx2tf completed without producing a .tflite artifact in {export_dir}")

        return str(tflite_path)

    def _ensure_onnx_file(self) -> Path:
        onnx_path = self.file.with_suffix(".onnx")
        if onnx_path.exists():
            return onnx_path

        LOGGER.info("Missing ONNX export; generating an intermediate ONNX model for TFLite conversion...")
        exported = ONNXExporter(
            self.model,
            self.im,
            self.file,
            opset=self.opset,
            dynamic=self.dynamic,
            half=self.half,
            simplify=self.simplify,
        ).export()
        if not exported:
            raise RuntimeError("ONNX export failed; cannot continue with TFLite export.")

        return Path(exported)

    def _build_convert_kwargs(self, onnx2tf_module, onnx_path: Path, output_folder_path: str) -> dict:
        kwargs = {
            "input_onnx_file_path": str(onnx_path),
            "output_folder_path": output_folder_path,
        }

        try:
            sig = inspect.signature(onnx2tf_module.convert)
        except (TypeError, ValueError):
            sig = None

        self._set_if_supported(kwargs, sig, "tflite_backend", "flatbuffer_direct")
        self._set_if_supported(kwargs, sig, "not_use_onnxsim", True)
        self._set_if_supported(kwargs, sig, "verbosity", "info")
        self._set_if_supported(kwargs, sig, "output_float16_tflite", bool(self.half))

        return kwargs

    @staticmethod
    def _set_if_supported(kwargs: dict, sig, name: str, value) -> None:
        if sig is None or name in sig.parameters:
            kwargs[name] = value

    def _select_tflite_artifact(self, export_dir: Path) -> Optional[Path]:
        tflites = sorted(export_dir.rglob("*.tflite"))
        tflites = [p for p in tflites if "quant_with_int16_act.tflite" not in p.name]
        if not tflites:
            return None

        if self.half:
            preferred_tokens = ("float16", "fp16")
            fallback_tokens = ("float32", "fp32")
        else:
            preferred_tokens = ("float32", "fp32")
            fallback_tokens = ("float16", "fp16")

        preferred = next((p for p in tflites if self._matches_any_token(p, preferred_tokens)), None)
        fallback = next((p for p in tflites if self._matches_any_token(p, fallback_tokens)), None)

        return preferred or fallback or tflites[0]

    @staticmethod
    def _matches_any_token(path: Path, tokens: Tuple[str, ...]) -> bool:
        stem = path.stem.lower()
        return any(token in stem for token in tokens)
