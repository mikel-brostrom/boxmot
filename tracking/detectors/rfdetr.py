# Mikel Broström 🔥 RFDETR Tracking 🧾 AGPL-3.0 license

import cv2
import numpy as np
import torch
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor

from boxmot.utils import logger as LOGGER
from tracking.detectors.yolo_interface import YoloInterface


class RFDETRStrategy(YoloInterface):
    pt = False
    stride = 32
    fp16 = False
    triton = False
    names = COCO_CLASSES

    def __init__(self, model, device, args):
        self.args = args
        LOGGER.info("Loading RFDETR model")
        self.model = RFDETRBase(device="cpu")

    @torch.no_grad()
    def __call__(self, im, augment, visualize, embed):
        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert frame BGR2RGB for RFDETR
        with torch.no_grad():
            detections = self.model.predict(image, threshold=self.args.conf)

        return torch.from_numpy(np.column_stack([
                detections.xyxy,
                detections.confidence[:, np.newaxis],
                detections.class_id[:, np.newaxis]
            ])).unsqueeze(0)

    def warmup(self, imgsz):
        pass
    
    def update_im_paths(self, predictor: DetectionPredictor):
        """
        This function saves image paths for the current batch,
        being passed as callback on_predict_batch_start
        """
        assert (isinstance(predictor, DetectionPredictor),
                "Only ultralytics predictors are supported")
        self.im_paths = predictor.batch[0]
    
    def preprocess(self, im) -> torch.Tensor:
        assert isinstance(im, list)
        return im[0]

    def postprocess(self, preds, im, im0s):
        results = []
        for i, pred in enumerate(preds):
            if pred is None or len(pred) == 0:
                continue
            im_path = self.im_paths[i] if len(self.im_paths) else ""
            if self.args.classes:
                pred = pred[torch.isin(pred[:, 5].cpu(), torch.as_tensor(self.args.classes))]
            results.append(Results(path=im_path, boxes=pred, orig_img=im0s[i], names=COCO_CLASSES))
        return results
