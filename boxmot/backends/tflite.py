from pathlib import Path

import numpy as np
import torch

from boxmot.backends.backend import Backend
from boxmot.utils import logger as LOGGER


class TFLiteBackend(Backend):
    """
    A class to handle TensorFlow Lite model inference with dynamic batch size support.
    """

    def __init__(self, 
                 weights: str | Path, 
                 half: bool, 
                 nhwc: bool = False,
                 numpy: bool = True):
        """
        Initializes the TFLiteBackend with given weights, device, and precision flag.

        Args:
            weights (Path): Path to the TFLite model file.
        """
        super().__init__(half=half, nhwc=nhwc, numpy=numpy)
        self.weights = weights
        self.model = self.load()

        self.model.allocate_tensors()  # allocate
        self.input_details = self.model.get_input_details()  # inputs
        self.output_details = self.model.get_output_details()  # outputs
        self.current_allocated_batch_size = self.input_details[0]["shape"][0]

        self.nhwc = True # TODO add into preprocessing and remove here

    def load(self):
        """
        Loads the TensorFlow Lite model and initializes the interpreter.
        """
        self.checker.check_packages(("tensorflow",))

        LOGGER.info(f"Loading {str(self.weights)} for TensorFlow Lite inference...")

        import tensorflow as tf

        model = tf.lite.Interpreter(model_path=str(self.weights))

        return model
    
    def preprocess(self, x: torch.Tensor) -> np.ndarray:
        x = super().preprocess(x)
        return x.cpu().numpy()


    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Runs forward pass for the given image batch through the TFLite model.
        """
        # Extract batch size from im_batch
        batch_size = x.shape[0]

        # Resize tensors if the new batch size is different from the current allocated batch size
        if batch_size != self.current_allocated_batch_size:
            # print(f"Resizing tensor input to batch size {batch_size}")
            self.model.resize_tensor_input(
                self.input_details[0]["index"], [batch_size, 256, 128, 3]
            )
            self.model.allocate_tensors()
            self.current_allocated_batch_size = batch_size

        # Set the tensor to point to the input data
        self.model.set_tensor(self.input_details[0]["index"], x)

        # Run inference
        self.model.invoke()

        # Get the output data
        y = self.model.get_tensor(self.output_details[0]["index"])

        return y
