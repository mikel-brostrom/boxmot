import inspect
import torch
from torch.export import Dim

from boxmot.reid.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class ONNXExporter(BaseExporter):
    group = "onnx"

    def __init__(self, model, im, file, opset=None, dynamic=False, half=False, simplify=False):
        # keep BaseExporter behavior (optimize handled elsewhere in boxmot)
        super().__init__(model, im, file, optimize=False, dynamic=dynamic, half=half, simplify=simplify)
        self.opset = opset  # None -> auto

    def export(self):
        import onnx

        f = self.file.with_suffix(".onnx")

        opset = self.opset or self._best_onnx_opset(onnx, cuda=torch.cuda.is_available())
        LOGGER.info(f"Exporting ONNX with onnx {onnx.__version__} opset {opset}...")

        # Determine output count for correct output_names length
        output_names = self._infer_output_names()

        # --- Export ---
        args = (self.im,)
        export_sig = inspect.signature(torch.onnx.export)
        has_dynamo_arg = "dynamo" in export_sig.parameters

        export_kwargs = {
            "opset_version": opset,
            "input_names": ["images"],
            "output_names": output_names,
        }

        if self.dynamic:
            # Constrain dynamic batch range to satisfy torch.export shape guards on CUDA.
            export_kwargs["dynamic_shapes"] = ({0: Dim("batch", min=1, max=65535)},)

        if has_dynamo_arg:
            export_kwargs["dynamo"] = True

        try:
            torch.onnx.export(
                self.model,
                args,
                str(f),
                **export_kwargs,
            )
        except Exception as e:
            if not self.dynamic:
                raise

            LOGGER.warning(
                f"Dynamic export via torch.export failed ({e}). "
                "Retrying with legacy dynamic_axes export..."
            )

            # Fallback for torch.export/dynamo dynamic shape guard failures.
            fallback_kwargs = {
                "opset_version": opset,
                "input_names": ["images"],
                "output_names": output_names,
                "dynamic_axes": self._build_dynamic_axes(output_names),
            }
            if has_dynamo_arg:
                fallback_kwargs["dynamo"] = False

            torch.onnx.export(
                self.model,
                args,
                str(f),
                **fallback_kwargs,
            )

        # --- Load + validate ---
        model_onnx = onnx.load(str(f))
        onnx.checker.check_model(model_onnx)

        # --- Simplify (onnxslim) ---
        if self.simplify:
            model_onnx = self.simplify_model(model_onnx)

        # --- IR version clamp for ONNXRuntime compatibility ---
        if getattr(model_onnx, "ir_version", 0) > 10:
            LOGGER.info(
                f"Limiting IR version {model_onnx.ir_version} -> 10 for ONNXRuntime compatibility..."
            )
            model_onnx.ir_version = 10

        # --- Optional FP16 conversion for CPU export ---
        # (If you already exported in FP16 on GPU, you typically don't need this.)
        if self.half and self.im.device.type == "cpu":
            model_onnx = self._try_fp16_convert_cpu(model_onnx)

        onnx.save(model_onnx, str(f))
        return f

    def simplify_model(self, model_onnx):
        try:
            import onnxslim

            LOGGER.info(f"Slimming with onnxslim {onnxslim.__version__}...")
            return onnxslim.slim(model_onnx)
        except Exception as e:
            LOGGER.warning(f"Simplifier failure: {e}")
            return model_onnx

    # -----------------
    # Helpers
    # -----------------
    def _best_onnx_opset(self, onnx, cuda: bool = False) -> int:
        """
        - If torch exposes ONNX_MAX_OPSET: use second-latest for safety, and reduce further on CUDA.
        - Else fallback by torch major.minor mapping.
        """
        # torch.onnx.utils._constants.ONNX_MAX_OPSET exists in newer torch; safest is "max-1"
        max_opset = getattr(getattr(torch.onnx.utils, "_constants", None), "ONNX_MAX_OPSET", None)
        if isinstance(max_opset, int) and max_opset > 0:
            opset = max_opset - 1
            if cuda:
                opset -= 2  # matches Ultralytics CUDA-quirk mitigation
        else:
            # Fallback mapping (Ultralytics-style)
            v = ".".join(torch.__version__.split(".")[:2])
            opset = {
                "1.8": 12,
                "1.9": 12,
                "1.10": 13,
                "1.11": 14,
                "1.12": 15,
                "1.13": 17,
                "2.0": 17,
                "2.1": 17,
                "2.2": 17,
                "2.3": 17,
                "2.4": 20,
                "2.5": 20,
                "2.6": 20,
                "2.7": 20,
                "2.8": 23,
            }.get(v, 12)

        return min(int(opset), int(onnx.defs.onnx_opset_version()))

    def _infer_output_names(self):
        # Ensure output_names matches the number of ONNX graph outputs.
        try:
            self.model.eval()
            with torch.no_grad():
                y = self.model(self.im)
            if isinstance(y, (tuple, list)):
                return [f"output{i}" for i in range(len(y))]
        except Exception:
            # If inference fails here, keep single output name (previous behavior)
            pass
        return ["output0"]

    def _build_dynamic_axes(self, output_names):
        # Ultralytics always makes images dynamic in batch/H/W when dynamic=True
        dyn = {"images": {0: "batch", 2: "height", 3: "width"}}

        # For outputs, always make batch dynamic; add extra dims only when obvious
        try:
            with torch.no_grad():
                y = self.model(self.im)
            ys = list(y) if isinstance(y, (tuple, list)) else [y]
            for name, t in zip(output_names, ys):
                if not isinstance(t, torch.Tensor):
                    dyn[name] = {0: "batch"}
                    continue
                if t.dim() == 4:
                    dyn[name] = {0: "batch", 2: f"{name}_h", 3: f"{name}_w"}
                elif t.dim() == 3:
                    dyn[name] = {0: "batch", 2: f"{name}_n"}
                else:
                    dyn[name] = {0: "batch"}
        except Exception:
            for name in output_names:
                dyn[name] = {0: "batch"}

        return dyn

    def _try_fp16_convert_cpu(self, model_onnx):
        try:
            from onnxruntime.transformers import float16

            LOGGER.info("Converting ONNX graph to FP16 (CPU export)...")
            return float16.convert_float_to_float16(model_onnx, keep_io_types=True)
        except Exception as e:
            LOGGER.warning(f"FP16 conversion failure: {e}")
            return model_onnx
