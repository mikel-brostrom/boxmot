# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from boxmot.detectors.detector import Detections, Detector
from boxmot.utils import logger as LOGGER


class RTDetrDetector(Detector):

    pt = False
    stride = 32
    fp16 = False
    triton = False
    ch = 3

    def __init__(self, model, device, args=None, imgsz=None):
        # args/imgsz accepted for a consistent constructor signature; RTDetr uses its own processor
        self.device = device

        model = Path(str(model)).name
        while model.endswith(".pt"):
            model = model[:-3]
        if not model.startswith("PekingU/"):
            model = f"PekingU/{model}"

        LOGGER.info(f"Loading RTDetr model: {model}")

        self.image_processor = RTDetrImageProcessor.from_pretrained(model)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model).to(device)
        self.names = self.model.config.id2label
        self._im0s = []

    def preprocess(self, images: list):
        """Convert BGR numpy images to PIL RGB and run image_processor."""
        assert isinstance(images, list)
        self._im0s = images
        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
        self._sizes = torch.tensor([(img.height, img.width) for img in pil_images], device=self.device)
        return self.image_processor(images=pil_images, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def process(self, preprocessed) -> torch.Tensor:
        """
        Run RT-DETR inference on preprocessed inputs.
        Returns raw detections tensor of shape (B, N, 6): [x1, y1, x2, y2, conf, cls].
        """
        outputs = self.model(**preprocessed)

        # threshold=0.0: collect all; conf filtering happens in postprocess
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=self._sizes,
            threshold=0.0,
        )

        detections = []
        for r in results:
            for box, score, label in zip(r["boxes"], r["scores"], r["labels"]):
                detections.append([*box.cpu().tolist(), score.item(), float(label.item())])

        if not detections:
            return torch.zeros((1, 0, 6), device=self.device)

        return torch.tensor(detections, device=self.device).unsqueeze(0)

    def postprocess(self, detections, conf, classes, **kwargs) -> list:
        results = []
        for i, det in enumerate(detections):
            orig_img = self._im0s[i] if i < len(self._im0s) else None

            if det is None or len(det) == 0:
                results.append(Detections(dets=np.empty((0, 6)), orig_img=orig_img, names=self.names))
                continue

            det_np = det.cpu().numpy() if isinstance(det, torch.Tensor) else det
            det_np = det_np[det_np[:, 4] >= conf]

            if classes:
                det_np = det_np[np.isin(det_np[:, 5].astype(int), classes)]

            results.append(Detections(dets=det_np, orig_img=orig_img, names=self.names))

        return results
