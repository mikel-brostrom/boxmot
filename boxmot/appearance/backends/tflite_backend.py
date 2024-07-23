import torch
import numpy as np
from pathlib import Path
from boxmot.utils import logger as LOGGER

from boxmot.appearance.backends.base_backend import BaseModelBackend


class TFLiteBackend(BaseModelBackend):
    """
    A class to handle TensorFlow Lite model inference with dynamic batch size support.

    Attributes:
        nhwc (bool): A flag indicating the order of dimensions.
        half (bool): A flag to indicate if half precision is used.
        interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter.
        current_allocated_batch_size (int): The current batch size allocated in the interpreter.
    """

    def __init__(self, weights: Path, device: str, half: bool):
        """
        Initializes the TFLiteBackend with given weights, device, and precision flag.

        Args:
            weights (Path): Path to the TFLite model file.
            device (str): Device type (e.g., 'cpu', 'gpu').
            half (bool): Flag to indicate if half precision is used.
        """
        super().__init__(weights, device, half)
        self.nhwc = True
        self.half = False
        # self.interpreter: tf.lite.Interpreter = None
        # self.current_allocated_batch_size: int = None

    def load_model(self, w):
        """
        Loads the TensorFlow Lite model and initializes the interpreter.

        Args:
            w (str): Path to the TFLite model file.
        """
        self.checker.check_packages(("tensorflow",))

        LOGGER.info(f"Loading {str(w)} for TensorFlow Lite inference...")

        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(w))


        self.interpreter.allocate_tensors()  # allocate
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs
        self.current_allocated_batch_size = self.input_details[0]['shape'][0]

    def forward(self, im_batch: torch.Tensor) -> np.ndarray:
        """
        Runs forward pass for the given image batch through the TFLite model.

        Args:
            im_batch (torch.Tensor): Input image batch tensor.

        Returns:
            np.ndarray: Output features from the TFLite model.
        """
        im_batch = im_batch.cpu().numpy()

        # Extract batch size from im_batch
        batch_size = im_batch.shape[0]

        # Resize tensors if the new batch size is different from the current allocated batch size
        if batch_size != self.current_allocated_batch_size:
            # print(f"Resizing tensor input to batch size {batch_size}")
            self.interpreter.resize_tensor_input(self.input_details[0]['index'], [batch_size, 256, 128, 3])
            self.interpreter.allocate_tensors()
            self.current_allocated_batch_size = batch_size

        # Set the tensor to point to the input data
        self.interpreter.set_tensor(self.input_details[0]['index'], im_batch)

        # Run inference
        self.interpreter.invoke()

        # Get the output data
        features = self.interpreter.get_tensor(self.output_details[0]['index'])

        return features