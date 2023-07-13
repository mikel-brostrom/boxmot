class Yolo:
    def __init__(self, strategy):
        self.strategy = strategy

    def inference(self, im):
        return self.strategy.inference(im=im)

    def postprocess(self, path, preds, im, im0s, predictor):
        return self.strategy.postprocess(path, preds, im, im0s, predictor)

    def filter_results(self, i, predictor):
        return self.strategy.filter_results(i, predictor)

    def overwrite_results(self, i, im0_shape, predictor):
        return self.strategy.overwrite_results(i, im0_shape, predictor)