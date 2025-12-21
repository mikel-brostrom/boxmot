from pathlib import Path
from typing import Union

import numpy as np

from boxmot.reid.core.auto_backend import ReidAutoBackend


class ReID:
    def __init__(self, weights: Union[str, Path], device='cpu', half=False):
        self.weights = Path(weights)
        self.device = device
        self.half = half
        
        # Instantiate the backend
        # ReidAutoBackend detects model type from weights suffix automatically
        self.backend = ReidAutoBackend(weights=self.weights, device=device, half=half)
        self.model = self.backend.model
        
    def __call__(self, frame: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """
        Extract features for detections in a frame.
        
        Args:
            frame: (H, W, C) BGR image
            dets: (N, 6) detections (x1, y1, x2, y2, conf, cls) or similar. 
        
        Returns:
            embs: (N, D) embeddings.
        """
        if dets.shape[0] == 0:
            # We don't know embedding dimension D until we run, but typical is 2048 or 128.
            # If empty, return empty 2D array.
            return np.empty((0, 0)) 
            
        xyxy = dets[:, :4]
        
        # Delegate to the backend model
        embs = self.model.get_features(xyxy, frame)
        return embs

