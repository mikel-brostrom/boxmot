from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boxmot.reid.backends.base_backend import BaseModelBackend
from boxmot.utils import logger as LOGGER


class TFLiteBackend(BaseModelBackend):
    """
    A class to handle LiteRT inference for TFLite models with dynamic batch size support.

    Attributes:
        nhwc (bool): A flag indicating the order of dimensions.
        half (bool): A flag to indicate if half precision is used.
        interpreter: The LiteRT interpreter instance.
        current_allocated_batch_size (int): The current batch size allocated in the interpreter.
    """

    def __init__(self, weights: Path, device: str, half: bool, preprocess: str | None = None):
        """
        Initializes the TFLiteBackend with given weights, device, and precision flag.

        Args:
            weights (Path): Path to the TFLite model file.
            device (str): Device type (e.g., 'cpu', 'gpu').
            half (bool): Flag to indicate if half precision is used.
            preprocess (str | None): Name of preprocessing function from the registry.
        """
        self.nhwc = True  # Default; load_model may override based on actual model layout
        super().__init__(weights, device, half, preprocess=preprocess)
        self.half = False
        # self.interpreter: tf.lite.Interpreter = None
        # self.current_allocated_batch_size: int = None

    def _get_interpreter_class(self) -> type[Any]:
        """Resolve the LiteRT interpreter class, installing LiteRT when needed."""
        try:
            litert = import_module("ai_edge_litert.interpreter")
        except ModuleNotFoundError:
            self.checker.check_packages(("ai-edge-litert",))
            litert = import_module("ai_edge_litert.interpreter")
        return litert.Interpreter

    def load_model(self, w):
        """
        Loads the TFLite model and initializes the LiteRT interpreter.

        Args:
            w (str): Path to the TFLite model file.
        """
        interpreter_class = self._get_interpreter_class()
        LOGGER.info(f"Loading {str(w)} for LiteRT inference...")
        self.interpreter = interpreter_class(model_path=str(w))

        self.interpreter.allocate_tensors()  # allocate
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs
        self.current_allocated_batch_size = self.input_details[0]["shape"][0]
        shape = tuple(int(dim) for dim in self.input_details[0]["shape"])
        if len(shape) == 4:
            self.nhwc = shape[-1] == 3 and shape[1] != 3

    def forward(self, im_batch: torch.Tensor) -> np.ndarray:
        """
        Runs forward pass for the given image batch through the TFLite model.

        Args:
            im_batch (torch.Tensor): Input image batch tensor.

        Returns:
            np.ndarray: Output features from the TFLite model.
        """
        im_batch = im_batch.cpu().numpy()
        if self.nhwc:
            im_batch = np.transpose(im_batch, (0, 2, 3, 1))

        batch_size = im_batch.shape[0]

        # Resize tensors if the batch size differs from what's currently allocated
        if batch_size != self.current_allocated_batch_size:
            if not self._try_resize(batch_size):
                return self._forward_chunked(im_batch)

        self.interpreter.set_tensor(self.input_details[0]["index"], im_batch)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])

    def _try_resize(self, batch_size: int) -> bool:
        """Attempt to resize the interpreter to *batch_size*. Returns True on success."""
        original_bs = self.current_allocated_batch_size
        try:
            input_shape = list(self.input_details[0]["shape"])
            input_shape[0] = batch_size
            self.interpreter.resize_tensor_input(
                self.input_details[0]["index"], input_shape
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.current_allocated_batch_size = batch_size
            return True
        except RuntimeError:
            # Restore the original batch size so the interpreter is usable
            input_shape = list(self.input_details[0]["shape"])
            input_shape[0] = original_bs
            self.interpreter.resize_tensor_input(
                self.input_details[0]["index"], input_shape
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            return False

    def _forward_chunked(self, im_batch: np.ndarray) -> np.ndarray:
        """Process *im_batch* in chunks of the allocated batch size (XNNPACK fallback)."""
        alloc_bs = self.current_allocated_batch_size
        outputs = []
        for start in range(0, im_batch.shape[0], alloc_bs):
            chunk = im_batch[start: start + alloc_bs]
            actual = chunk.shape[0]
            if actual < alloc_bs:
                pad = np.zeros(
                    (alloc_bs - actual, *chunk.shape[1:]), dtype=chunk.dtype
                )
                chunk = np.concatenate([chunk, pad], axis=0)
            self.interpreter.set_tensor(self.input_details[0]["index"], chunk)
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(self.output_details[0]["index"])
            outputs.append(out[:actual])
        return np.concatenate(outputs, axis=0)
