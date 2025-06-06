from abc import abstractmethod
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch

from boxmot.utils.checks import RequirementsChecker

X = TypeVar("X")


def to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


class Backend:
    def __init__(self, half: bool, nhwc: bool, numpy: bool = True):
        """
        Initializes the Backend with given half, nhwc, and numpy flags.

        Args:
            half (bool): Should input be converted to half precision.
            nhwc (bool): Should input be converted to NHWC.
            numpy (bool): Should output be converted to numpy.
        """
        self.half = half
        self.nhwc = nhwc
        self.numpy = numpy
        self.checker = RequirementsChecker()

    @abstractmethod
    def load(self):
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    def preprocess(
            self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Universal preprocessing including:

        1) Converting to half if self.half is True
        2) Converting to NHWC if self.nhwc is True
        """

        # Convert to half
        if self.half:
            if isinstance(x, torch.Tensor):
                if x.dtype != torch.float16:
                    x = x.half()
            elif isinstance(x, np.ndarray):
                if x.dtype != np.float16:
                    x = x.astype(np.float16)

        # Convert from NCHW to NHWC
        if self.nhwc:
            if isinstance(x, torch.Tensor):
                x = x.permute(0, 2, 3, 1)
            elif isinstance(x, np.ndarray):
                x = np.transpose(x, (0, 2, 3, 1))

        return x

    @abstractmethod
    def process(self, x):
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    def postprocess(self, x: X) -> np.ndarray | list[np.ndarray] | X:
        if self.numpy:
            if isinstance(x, (list, tuple)):
                if len(x) == 1:
                    return to_numpy(x[0])
                else:
                    return [to_numpy(x_i) for x_i in x]
            else:
                return to_numpy(x)
        else:
            return x

    def __call__(self, x: X):
        x = self.preprocess(x)
        x = self.process(x)
        x = self.postprocess(x)
        return x
