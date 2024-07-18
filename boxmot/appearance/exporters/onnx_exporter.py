import torch
import onnx
from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class ONNXExporter(BaseExporter):
    required_packages = ("onnx>=1.16.1",)
    
    def export(self):

        f = self.file.with_suffix(".onnx")

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
            
        return f


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