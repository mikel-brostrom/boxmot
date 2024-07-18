import logging
import torch
from pathlib import Path
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils import logger as LOGGER


def export_decorator(export_func):
    def wrapper(self, *args, **kwargs):
        try:
            if hasattr(self, 'required_packages'):
                if hasattr(self, 'cmd'):
                    self.checker.check_packages(self.required_packages, cmd=self.cmd)
                else:
                    self.checker.check_packages(self.required_packages)
                
            LOGGER.info(f"\nStarting {self.file} export with {self.__class__.__name__}...")
            result = export_func(self, *args, **kwargs)
            if result:
                LOGGER.info(f"Export success, saved as {result} ({self.file_size(result):.1f} MB)")
            return result
        except Exception as e:
            LOGGER.error(f"Export failure: {e}")
            return None
    return wrapper


class BaseExporter:
    def __init__(self, model, im, file, optimize=False, dynamic=False, half=False, simplify=False):
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
        if 'export' in cls.__dict__:
            cls.export = export_decorator(cls.export)