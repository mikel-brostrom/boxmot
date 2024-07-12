import logging
import torch
from pathlib import Path
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils import logger as LOGGER


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