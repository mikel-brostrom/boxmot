import numpy as np
import torch
from ultralytics.yolo.engine.results import Boxes, Results


class YoloInterface:

    def inference(self, im):
        raise NotImplementedError('Subclasses must implement this method.')

    def postprocess(self, preds):
        raise NotImplementedError('Subclasses must implement this method.')

    def filter_results(self, i, predictor):
        if predictor.tracker_outputs[i].size != 0:
            # filter boxes masks and pose results by tracking results
            sorted_confs = predictor.tracker_outputs[i][:, 5].argsort()[::-1]
            predictor.tracker_outputs[i] = predictor.tracker_outputs[i][sorted_confs]
            yolo_confs = predictor.results[i].boxes.conf.cpu().numpy()
            tracker_confs = predictor.tracker_outputs[i][:, 5]
            mask = np.in1d(yolo_confs, tracker_confs)

            if predictor.results[i].masks is not None:
                predictor.results[i].masks = predictor.results[i].masks[mask]
                predictor.results[i].boxes = predictor.results[i].boxes[mask]
            elif predictor.results[i].keypoints is not None:
                predictor.results[i].boxes = predictor.results[i].boxes[mask]
                predictor.results[i].keypoints = predictor.results[i].keypoints[mask]
        else:
            pass

    def overwrite_results(self, i, im0_shape, predictor):
        # overwrite bbox results with tracker predictions
        if predictor.tracker_outputs[i].size != 0:
            predictor.results[i].boxes = Boxes(
                # xyxy, (track_id), conf, cls
                boxes=torch.from_numpy(predictor.tracker_outputs[i]).to(predictor.device),
                orig_shape=im0_shape,  # (height, width)
            )

    def get_scaling_factors(self, im, im0s):

        # im to im0 factor for predictions
        im0_w = im0s[0].shape[1]
        im0_h = im0s[0].shape[0]
        im_w = im[0].shape[2]
        im_h = im[0].shape[1]
        w_r = im0_w / im_w
        h_r = im0_h / im_h

        return w_r, h_r

    def preds_to_yolov8_results(self, path, preds, im, im0s, predictor):
        predictor.results[0] = Results(
            path=path,
            boxes=preds,
            orig_img=im0s[0],
            names=predictor.model.names
        )
        return predictor.results
