# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Results
from abc import ABC, abstractmethod


class YoloInterface(ABC):

    @abstractmethod
    def __call__(self, im):
        pass

    @abstractmethod
    def postprocess(self, preds):
        pass

    def get_scaling_factors(self, im, im0):

        # im to im0 factor for predictions
        im0_w = im0.shape[1]
        im0_h = im0.shape[0]
        im_w = im.shape[2]
        im_h = im.shape[1]
        w_r = im0_w / im_w
        h_r = im0_h / im_h

        return im_w, im_h, w_r, h_r

    def scale_and_clip(self, preds, im_w, im_h, w_r, h_r):
        # scale bboxes to original image
        preds[:, [0, 2]] = preds[:, [0, 2]] * self.w_r
        preds[:, [1, 3]] = preds[:, [1, 3]] * self.h_r

        if not isinstance(preds, (torch.Tensor)):
            preds = torch.from_numpy(preds)

        preds[:, [0, 2]] = torch.clip(preds[:, [0, 2]], min=0)  # max=im_w
        preds[:, [1, 3]] = torch.clip(preds[:, [1, 3]], min=0)  # max=im_h

        return preds

    def preds_to_yolov8_results(self, path, preds, im, im0s, names):
        return Results(
            path=path,
            boxes=preds,
            orig_img=im0s[0],
            names=names
        )

    def get_model_from_weigths(self, l, model):
        model_type = None
        for key in l:
            if Path(key).stem in str(model.name):
                model_type = str(Path(key).with_suffix(''))
                break
        return model_type
