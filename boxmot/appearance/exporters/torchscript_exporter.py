import torch
from boxmot.appearance.exporters.base_exporter import BaseExporter
from boxmot.utils import logger as LOGGER


class TorchScriptExporter(BaseExporter):
    def export(self):
        f = self.file.with_suffix(".torchscript")
        ts = torch.jit.trace(self.model, self.im, strict=False)
        if self.optimize:
            torch.utils.mobile_optimizer.optimize_for_mobile(ts)._save_for_lite_interpreter(str(f))
        else:
            ts.save(str(f))

        return f
