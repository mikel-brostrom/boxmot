import torch
import numpy as np
from pathlib import Path
from boxmot.utils import logger as LOGGER

from boxmot.appearance.backends.base_backend import BaseModelBackend
from boxmot.utils.checks import TestRequirements

tr = TestRequirements()


class TFLiteBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):

        LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(w))
        try:
            self.tf_lite_model = interpreter.get_signature_runner()
        except Exception as e:
            LOGGER.error(f'{e}. If SignatureDef error. Export you model with the official onn2tf docker')
            exit()

    def forward(self, im_batch):
        im_batch = im_batch.cpu().numpy()
        inputs = {
            'images': im_batch,
        }
        tf_lite_output = self.tf_lite_model(**inputs)
        features = tf_lite_output['output']
        return features


im = (np.random.rand(800, 800, 3) * 255).astype(np.uint8)
xyxy = np.array(
    [[ 345,  246,  794,  498],
     [ 400,  400, 500, 500]]
)

b = TFLiteBackend(
    weights=Path("/home/mikel.brostrom/yolo_tracking/tracking/weights/osnet_x0_25_msmt17_saved_model/osnet_x0_25_msmt17_float16.tflite"),
    half=False,
    device="cpu"
)

f = b.get_features(xyxy, im)
print(f.shape)