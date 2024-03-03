# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from .yolo_interface import YoloInterface


class Yolov8Strategy(YoloInterface):

    def __init__(self, model, device, args):
        self.model = model

    def inference(self, im):
        preds = self.model(im, augment=False, visualize=False)
        return preds

    def postprocess(self, path, preds, im, im0s, predictor):
        postprocessed_preds = predictor.postprocess(preds, im, im0s)
        return postprocessed_preds
