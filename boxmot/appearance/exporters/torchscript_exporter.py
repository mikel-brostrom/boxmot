import torch
from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class TorchScriptExporter(BaseExporter):
    def export(self):
        try:
            LOGGER.info(f"\nStarting export with torch {torch.__version__}...")
            f = self.file.with_suffix(".torchscript")
            ts = torch.jit.trace(self.model, self.im, strict=False)
            if self.optimize:
                torch.utils.mobile_optimizer.optimize_for_mobile(ts)._save_for_lite_interpreter(str(f))
            else:
                ts.save(str(f))

            LOGGER.info(f"Export success, saved as {f} ({self.file_size(f):.1f} MB)")
            return f
        except Exception as e:
            LOGGER.error(f"Export failure: {e}")