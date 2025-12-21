
import numpy as np
from ultralytics import YOLO

from boxmot.detectors.detector import Detector


class UltralyticsYolo(Detector):
    def __init__(self, path: str, device='cpu', conf=0.25, iou=0.45, imgsz=640):
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        
        # Load Ultralytics model
        # path could be 'yolov8n.pt' or local path
        super().__init__(path)
        
    def _load_model(self, path: str):
        # We rely on ultralytics own loading mechanism
        return YOLO(path)
        
    def preprocess(self, image: np.ndarray, **kwargs):
        # Ultralytics handles preprocessing internally in __call__ usually,
        # but to adhere to strict pipeline if needed:
        # Here we just pass the image through, as model() call handles it.
        return image

    def process(self, frame, **kwargs):
        # frame is just the image here
        # Return results object
        results = self.model(
            frame, 
            conf=self.conf, 
            iou=self.iou, 
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            classes=kwargs.get('classes')
        )
        return results

    def postprocess(self, results, **kwargs):
        # results is a list of [Results] object (batch size 1 usually)
        result = results[0]
        
        # Extract boxes: x1, y1, x2, y2, conf, cls
        if result.boxes is None or len(result.boxes) == 0:
             return np.empty((0, 6))
             
        # boxes.data is often (N, 6) tensor: x1, y1, x2, y2, conf, cls
        dets = result.boxes.data.cpu().numpy()
        return dets
        
    def __call__(self, image, **kwargs):
        # Let ultralytics handle loading if passed to predict  
        return super().__call__(image, **kwargs)
