from abc import abstractmethod
from pathlib import Path

from boxmot.utils.checks import RequirementsChecker


class Backend:
    def __init__(self):
        self.checker = RequirementsChecker()

    @abstractmethod
    def load(self):
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    def preprocess(self, x):
        return x

    @abstractmethod
    def process(self, x):
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    def postprocess(self, x):
        return x

    def __call__(self, x):
        x = self.preprocess(x)
        x = self.process(x)
        x = self.postprocess(x)
        return x
