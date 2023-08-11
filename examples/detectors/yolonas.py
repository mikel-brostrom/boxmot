# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from boxmot.utils import logger as LOGGER
from examples.detectors.yolo_interface import YoloInterface


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

        avail_models = [x.lower() for x in list(Models.__dict__.keys())]
        model_type = self.get_model_from_weigths(avail_models, model)

        LOGGER.info(f'Loading {model_type} with {str(model)}')
        if not model.exists() and model.stem == model_type:
            LOGGER.info('Downloading pretrained weights...')
            self.model = models.get(
                model_type,
                pretrained_weights="coco"
            ).to(device)
        else:
            self.model = models.get(
                model_type,
                num_classes=-1,  # set your num classes
                checkpoint_path=str(model)
            ).to(device)

        self.device = device

    @torch.no_grad()
    def __call__(self, im, augment, visualize):

        im = im[0].permute(1, 2, 0).cpu().numpy() * 255

        with torch.no_grad():
            preds = self.model.predict(
                im,
                iou=0.5,
                conf=0.7,
                fuse_model=False
            )[0].prediction

        preds = np.concatenate(
            [
                preds.bboxes_xyxy,
                preds.confidence[:, np.newaxis],
                preds.labels[:, np.newaxis]
            ], axis=1
        )

        preds = torch.from_numpy(preds).unsqueeze(0)

        return preds

    def warmup(self, imgsz):
        pass

    def postprocess(self, path, preds, im, im0s):

        results = []
        for i, pred in enumerate(preds):

            if pred is None:
                pred = torch.empty((0, 6))
                r = Results(
                    path=path,
                    boxes=pred,
                    orig_img=im0s[i],
                    names=self.names
                )
                results.append(r)
            else:

                pred[:, :4] = ops.scale_boxes(im.shape[2:], pred[:, :4], im0s[i].shape)

                r = Results(
                    path=path,
                    boxes=pred,
                    orig_img=im0s[i],
                    names=self.names
                )
            results.append(r)
        return results
