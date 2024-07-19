import torch
import numpy as np
from pathlib import Path
from boxmot.utils import logger as LOGGER

from boxmot.appearance.backends.base_backend import BaseModelBackend
from boxmot.utils.checks import RequirementsChecker

checker = RequirementsChecker()


class TFLiteBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half
        self.interpreter = None
        self.current_allocated_batch_size = None

    def load_model(self, w):
        checker.check_packages(("tensorflow",))

        LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
        
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=str(w))
        except Exception as e:
            LOGGER.error(f'{e}. If SignatureDef error. Export your model with the official onnx2tf docker')
            exit()
            
        self.interpreter.allocate_tensors()  # allocate
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs
        self.current_allocated_batch_size = self.input_details[0]['shape'][0]

    def forward(self, im_batch):
        im_batch = im_batch.cpu().numpy()
        # Extract batch size from im_batch
        batch_size = im_batch.shape[0]

        # Resize tensors if the new batch size is different from the current allocated batch size
        if batch_size != self.current_allocated_batch_size:
            print(f"Resizing tensor input to batch size {batch_size}")
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