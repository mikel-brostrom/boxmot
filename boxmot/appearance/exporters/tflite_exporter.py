import subprocess
from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class TFLiteExporter(BaseExporter):
    def export(self):
        try:
            self.checker.check_packages(
                ("onnx2tf>=1.15.4", "tensorflow", "onnx_graphsurgeon>=0.3.26", "sng4onnx>=1.0.1"),
                cmds='--extra-index-url https://pypi.ngc.nvidia.com'
            )
            import onnx2tf

            LOGGER.info(f"\nStarting {self.file} export with onnx2tf {onnx2tf.__version__}")
            f = str(self.file).replace(".onnx", f"_saved_model{os.sep}")
            cmd = f"onnx2tf -i {self.file} -o {f} -osd -coion --non_verbose"
            subprocess.check_output(cmd.split())
            LOGGER.info(f"Export success, results saved in {f} ({self.file_size(f):.1f} MB)")
            return f
        except Exception as e:
            LOGGER.error(f"\nExport failure: {e}")