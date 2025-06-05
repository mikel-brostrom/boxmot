import cv2
import numpy as np
from typing import Union


class Detector:
    def __init__(self, path: str):
        self.path = path
        self.model = self._load_model(path)


    def _load_model(self, path: str):
        return NotImplementedError()

    def preprocess(self, frame: np.ndarray, **kwargs):
        raise NotImplementedError()

    def process(self, frame, **kwargs):
        raise NotImplementedError()

    def default_postprocess(self, boxes, **kwargs):
        raise NotImplementedError()
        
    def __call__(self, image: Union[np.ndarray, str], **kwargs):
        image = resolve_image(image)
        
        frame = self.preprocess(frame, **kwargs)
        boxes = self.process(frame, **kwargs)
        boxes = self.postprocess(boxes, **kwargs)
        
        return boxes