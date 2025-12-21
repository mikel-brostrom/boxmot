from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch


def resolve_image(image: Union[np.ndarray, str]) -> np.ndarray:
    """
    Resolves an image input to a numpy array (cv2 BGR format).
    """
    if isinstance(image, str) or isinstance(image, Path):
        image_path = str(image)
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        return img
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def load_weights(path: str) -> Any:
    """
    Generic weight loader. By default uses torch.load
    """
    if isinstance(path, str) and not Path(path).exists():
         raise FileNotFoundError(f"Weights file not found: {path}")
         
    # This is a placeholder. Real models often need architecture init before loading weights.
    # But strictly following the user snippet:
    return torch.load(path, map_location='cpu') 

class Detector:
    def __init__(self, path: str):
        self.path = path
        self.model = self._load_model(path)

    def _load_model(self, path: str):
        return load_weights(path)

    def preprocess(self, frame: np.ndarray, **kwargs):
        raise NotImplementedError()

    def process(self, frame, **kwargs):
        raise NotImplementedError()

    def postprocess(self, boxes, **kwargs):
        raise NotImplementedError()
        
    def __call__(self, image: Union[np.ndarray, str], **kwargs):
        image = resolve_image(image)
        
        frame = self.preprocess(image, **kwargs)
        boxes = self.process(frame, **kwargs)
        boxes = self.postprocess(boxes, **kwargs)
        
        return boxes
