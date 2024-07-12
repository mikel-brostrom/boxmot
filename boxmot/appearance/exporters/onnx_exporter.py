import torch
import onnx
from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class ONNXExporter(BaseExporter):
    def export(self):
        try:
            self.checker.check_packages(("onnx==1.14.0",))
            f = self.file.with_suffix(".onnx")
            LOGGER.info(f"\nStarting export with onnx {onnx.__version__}...")

            dynamic = {"images": {0: "batch"}, "output": {0: "batch"}} if self.dynamic else None

            torch.onnx.export(
                self.model.cpu() if self.dynamic else self.model,
                self.im.cpu() if self.dynamic else self.im,
                f,
                verbose=False,
                opset_version=12,
                do_constant_folding=True,
                input_names=["images"],
                output_names=["output"],
                dynamic_axes=dynamic,
            )

            model_onnx = onnx.load(f)
            onnx.checker.check_model(model_onnx)
            onnx.save(model_onnx, f)

            if self.simplify:
                self.simplify_model(model_onnx, f)

            LOGGER.info(f"Export success, saved as {f} ({self.file_size(f):.1f} MB)")
            return f
        except Exception as e:
            LOGGER.error(f"Export failure: {e}")

    def simplify_model(self, model_onnx, f):
        try:
            cuda = torch.cuda.is_available()
            self.checker.check_packages(
                (
                    "onnxruntime-gpu" if cuda else "onnxruntime",
                    "onnx-simplifier>=0.4.1",
                )
            )
            import onnxsim

            LOGGER.info(
                f"Simplifying with onnx-simplifier {onnxsim.__version__}..."
            )
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.error(f"Simplifier failure: {e}")