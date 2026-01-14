# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor

from boxmot.utils import logger as LOGGER


class RTDetrStrategy:
    pt = False
    stride = 32
    fp16 = False
    triton = False
    ch = 3

    def __init__(self, model, device, args):
        self.args = args
        self.device = device

        model = str(model)
        if model.endswith(".pt"):
            model = model[:-3]
        if not model.startswith("PekingU/"):
            model = f"PekingU/{model}"

        LOGGER.info(f"Loading RTDetr model: {model}")

        # Load model and processor from Hugging Face
        self.image_processor = RTDetrImageProcessor.from_pretrained(model)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model).to(device)

        # Get class names from model config
        self.names = self.model.config.id2label

    @torch.no_grad()
    def __call__(self, im, augment, visualize, embed):
        if isinstance(im, torch.Tensor):
            im = im.cpu().numpy()
            if im.ndim == 3 and im.shape[0] == 3:
                im = im.transpose(1, 2, 0)
            im = np.ascontiguousarray(im)

        # Convert numpy image (BGR) to PIL Image (RGB)
        image = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)], device=self.device),
            threshold=self.args.conf,
        )

        # Format results: [x1, y1, x2, y2, conf, cls]
        detections = []
        for result in results:
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                box = box.cpu().numpy()
                score = score.item()
                label = label.item()
                detections.append([*box, score, label])

        if not detections:
            return torch.zeros((1, 0, 6), device=self.device)

        return torch.tensor(detections, device=self.device).unsqueeze(0)

    def warmup(self, imgsz):
        pass

    def update_im_paths(self, predictor: DetectionPredictor):
        """
        This function saves image paths for the current batch,
        being passed as callback on_predict_batch_start
        """
        assert isinstance(predictor, DetectionPredictor), "Only ultralytics predictors are supported"
        self.im_paths = predictor.batch[0]

    def preprocess(self, im) -> torch.Tensor:
        # RTDetr expects PIL images or list of them, but here we just pass through
        # The actual preprocessing happens in __call__
        assert isinstance(im, list)
        return im[0]

    def postprocess(self, preds, im, im0s):
        results = []
        for i, pred in enumerate(preds):
            im_path = self.im_paths[i] if hasattr(self, "im_paths") and len(self.im_paths) else ""

            if pred is None or len(pred) == 0:
                pred = torch.empty((0, 6), device=self.device)
                results.append(Results(path=im_path, boxes=pred, orig_img=im0s[i], names=self.names))
                continue

            if self.args.classes:
                pred = pred[torch.isin(pred[:, 5].cpu(), torch.as_tensor(self.args.classes))]

            results.append(Results(path=im_path, boxes=pred, orig_img=im0s[i], names=self.names))
        return results
