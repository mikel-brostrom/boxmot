from .yolo_strategy import YoloStrategy
from boxmot.utils.checks import TestRequirements
tr = TestRequirements()


class Yolov8Strategy(YoloStrategy):
    def __init__(self, model, device, args):
        self.model = model

    def inference(self, im):
        preds = self.model(im, augment=False, visualize=False)
        return preds

    def postprocess(self, path, preds, im, im0s, predictor):
        postprocessed_preds = predictor.postprocess(preds, im, im0s)
        return postprocessed_preds

