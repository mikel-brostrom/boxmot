import torch
import numpy as np

from .yolo_strategy import YoloStrategy

from super_gradients.common.object_names import Models
from super_gradients.training import models

from ultralytics.yolo.engine.results import Results


class YoloNASStrategy(YoloStrategy):
    def __init__(self, model, device, args):
        self.args = args

        self.model = models.get(
            str(model),
            pretrained_weights="coco"
        ).to(device)

        self.has_run = False
        

    def inference(self, im):

        # (1, 3, h, w) norm --> (h, w, 3) un-norm
        im = im[0].permute(1, 2, 0).cpu().numpy() * 255

        preds = next(iter(
            self.model.predict(im,
                                iou=self.args.iou,
                                conf=self.args.conf)
            )
        ).prediction  # Returns a generator of the batch, which here is 1
        preds = np.concatenate(
            [
                preds.bboxes_xyxy,
                preds.confidence[:, np.newaxis],
                preds.labels[:, np.newaxis]
            ], axis=1
        )

        return preds

    def postprocess(self, path, preds, im, im0s, predictor):

        if not self.has_run:
            self.w_r, self.h_r = self.get_scaling_factors(im, im0s)
            self.has_run = True
        
        # scale bboxes to original image
        preds[:, [0, 2]] = preds[:, [0, 2]] * self.w_r
        preds[:, [1, 3]] = preds[:, [1, 3]] * self.h_r

        preds = torch.from_numpy(preds)

        # SG models can generate negative values
        preds = torch.clip(preds, min=0)

        if self.args.classes:  # Filter boxes by classes
            preds = preds[np.isin(preds[:, 5], self.args.classes)]
    
        # postprocess is embedded in inference for yolonas
        preds = self.preds_to_yolov8_results(path, preds, im, im0s, predictor)
        return preds

    def preds_to_yolov8_results(self, path, preds, im, im0s, predictor):
        predictor.results[0] = Results(
            path=path,
            boxes=preds,
            orig_img=im0s[0],
            names=predictor.model.names
        )
        return predictor.results
