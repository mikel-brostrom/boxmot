from pathlib import Path

from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker


def export_decorator(export_func):
    def wrapper(self, *args, **kwargs):
        try:
            # If a subclass defined a dependency bucket, install it now.
            if hasattr(self, "group") and self.group:
                # Optional: subclasses can define `cmd` or `extra_args` for installer flags
                extra_args = getattr(self, "cmd", None) or getattr(self, "extra_args", None)
                # Allow either a uv group or a project extra. If you want extras, set `self.extra`
                extra = getattr(self, "extra", None)
                if extra and self.group:
                    raise ValueError("Provide only one of `group` or `extra` in exporter.")
                if self.group:
                    self.checker.sync_extra(extra=self.group, extra_args=extra_args)
                elif extra:
                    self.checker.sync_extra(extra=extra, extra_args=extra_args)

            LOGGER.info(f"\nStarting {self.file} export with {self.__class__.__name__}...")
            result = export_func(self, *args, **kwargs)
            if result:
                LOGGER.info(
                    f"Export success, saved as {result} ({self.file_size(result):.1f} MB)"
                )
            return result
        except Exception as e:
            LOGGER.error(f"Export failure: {e}")
            return None

    return wrapper


class BaseExporter:
    def __init__(self, model, im, file, optimize=True, dynamic=True, half=True, simplify=True):
        self.model = model
        self.im = im
        self.file = Path(file)
        self.optimize = optimize
        self.dynamic = dynamic
        self.half = half
        self.simplify = simplify
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
