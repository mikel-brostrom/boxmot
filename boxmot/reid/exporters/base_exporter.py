from functools import wraps
from pathlib import Path

from torch import nn

from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker


class InferenceExportWrapper(nn.Module):
    """Single-input inference wrapper used by graph exporters.

    Several ReID backbones expose optional runtime flags on ``forward`` for
    feature map extraction. Legacy tracers can treat those optional flags as
    tensor inputs, which produces noisy and sometimes brittle graphs. Export
    inference embeddings through a narrow ``forward(x)`` contract instead.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def as_inference_export_model(model: nn.Module) -> nn.Module:
    if isinstance(model, InferenceExportWrapper):
        return model
    return InferenceExportWrapper(model).eval()


def export_decorator(export_func):
    @wraps(export_func)
    def wrapper(self, *args, **kwargs):
        # Exporters can declare one project extra to install before running.
        group = getattr(self, "group", None)
        extra = getattr(self, "extra", None)
        if group and extra:
            raise ValueError("Provide only one of `group` or `extra` in exporter.")
        dependency_extra = group or extra
        if dependency_extra:
            extra_args = getattr(self, "cmd", None) or getattr(self, "extra_args", None)
            self.checker.sync_extra(
                extra=dependency_extra,
                extra_args=extra_args,
                verbose=self.verbose,
            )

        if self.verbose:
            LOGGER.info(f"Starting {self.file} export with {self.__class__.__name__}...")
        result = export_func(self, *args, **kwargs)
        if result and self.verbose:
            LOGGER.info(
                f"Export success, saved as {result} ({self.file_size(result):.1f} MB)"
            )
        return result

    return wrapper


class BaseExporter:
    def __init__(self, model, im, file, optimize=True, dynamic=True, half=True, simplify=True, verbose=True):
        self.model = model
        self.im = im
        self.file = Path(file)
        self.optimize = optimize
        self.dynamic = dynamic
        self.half = half
        self.simplify = simplify
        self.verbose = bool(verbose)
        self.checker = RequirementsChecker()
        self.workspace = 4

    @staticmethod
    def file_size(path):
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / 1e6
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / 1e6
        else:
            return 0.0

    def export(self):
        raise NotImplementedError("Export method must be implemented in subclasses.")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "export" in cls.__dict__:
            cls.export = export_decorator(cls.export)
