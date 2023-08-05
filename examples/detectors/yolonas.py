# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch
from super_gradients.training import models
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from .yolo_interface import YoloInterface


class YoloNASStrategy(YoloInterface):
    pt = False
    stride = 32
    fp16 = False
    triton = False
    names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
        21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
        26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
        31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
        41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
        66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
        71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
        76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }

    def __init__(self, model, device, args):
        self.args = args

        self.model = models.get(
            str(model),
            pretrained_weights="coco"
        ).to(device)

        self.has_run = False

    def __call__(self, im, augment, visualize):

        preds = next(iter(
            self.model.predict(
                # (1, 3, h, w) norm --> (h, w, 3) un-norm
                im[0].permute(1, 2, 0).cpu().numpy() * 255,
                iou=self.args.iou,
                conf=self.args.conf
            )
        )).prediction  # Returns a generator of the batch, which here is 1
        preds = np.concatenate(
            [
                preds.bboxes_xyxy,
                preds.confidence[:, np.newaxis],
                preds.labels[:, np.newaxis]
            ], axis=1
        )

        return preds

    def warmup(self, imgsz):
        pass

    def postprocess(self, path, preds, im, im0s):
        preds = torch.from_numpy(preds).unsqueeze(0)
        results = []
        for i, pred in enumerate(preds):

            # scale from im to im0
            pred[:, :4] = ops.scale_boxes(im.shape[2:], pred[:, :4], im0s[i].shape)

            if self.args.classes:  # Filter boxes by classes
                pred = pred[np.isin(pred[:, 5], self.args.classes)]

            r = Results(
                path=path,
                boxes=pred,
                orig_img=im0s[i],
                names=self.names
            )

            results.append(r)

        return results
