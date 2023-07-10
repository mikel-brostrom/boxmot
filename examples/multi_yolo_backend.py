from pathlib import Path
import numpy as np
import torch
import gdown

from boxmot.utils.checks import TestRequirements
from boxmot.utils import WEIGHTS

tr = TestRequirements()

from ultralytics.yolo.engine.results import Boxes, Results
from boxmot.utils import logger as LOGGER
from boxmot.utils.ops import xywh2xyxy


YOLOX_ZOO = {
    'yolox_n': 'https://drive.google.com/uc?id=1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX',
    'yolox_s': 'https://drive.google.com/uc?id=1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj',
    'yolox_m': 'https://drive.google.com/uc?id=11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun',
    'yolox_l': 'https://drive.google.com/uc?id=1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz',
    'yolox_x': 'https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5',
}


class MultiYolo():
    def __init__(self, model, device, args):
        self.args = args
        self.device = device
        if not (isinstance(model, str) or isinstance(model, Path)):
            self.model_name = 'yolov8'
            self.model = model
        else:
            self.model_name = str(model.stem).lower()

        if 'yolo_nas' in self.model_name:
            self.try_sg_import()
            from super_gradients.common.object_names import Models
            from super_gradients.training import models
            self.model_type = 'yolo_nas'
            self.model = models.get(
                self.model_name,
                pretrained_weights="coco"
            ).to(self.device)
        elif 'yolox' in self.model_name:
            self.try_yolox_import()
            from yolox.exp import get_exp
            self.model_type = 'yolox'
            if self.model_name == 'yolox_n':
                exp = get_exp(None, 'yolox_nano')
            else:
                exp = get_exp(None, self.model_name.replace("-", "_"))
            exp.num_classes = 1  # bytetrack yolox models
            self.model = exp.get_model()
            self.model.eval()
            gdown.download(
                url=YOLOX_ZOO[self.model_name.replace("-", "_")],
                output=str(WEIGHTS / (self.model_name.replace("-", "_") + '.pth')),
                quiet=False
            )

            ckpt = torch.load(str(WEIGHTS / (self.model_name.replace("-", "_") + '.pth')))
            
            self.model.load_state_dict(ckpt["model"])
            self.model.to(self.device)
        # already loaded
        elif 'yolov8' in self.model_name:
            self.model = model

    def try_sg_import(self):
        try:
            import super_gradients  # for linear_assignment
        except (ImportError, AssertionError, AttributeError):
            tr.check_packages(('super-gradients==3.1.1',))  # install

    def try_yolox_import(self):
        try:
            import yolox  # for linear_assignment
        except (ImportError, AssertionError, AttributeError):
            tr.check_packages(('yolox==0.3.0',))  # install

    def __call__(self, im, im0s):
        if 'yolo_nas' in self.model_name:
            prediction = next(iter(
                self.model.predict(im0s,
                                   iou=self.args.iou,
                                   conf=self.args.conf)
            )
            ).prediction  # Returns a generator of the batch, which here is 1
            preds = np.concatenate(
                [
                    prediction.bboxes_xyxy,
                    prediction.confidence[:, np.newaxis],
                    prediction.labels[:, np.newaxis]
                ], axis=1
            )
            if self.args.classes:  # Filter boxes by classes
                preds = preds[np.isin(preds[:, 5], self.args.classes)]
            preds = torch.from_numpy(preds)
            preds[:, 0:4] = preds[:, 0:4].int()
            # SG models can generate negative values
            preds = torch.clip(preds, min=0)
        elif 'yolox' in self.model_name:
            from yolox.utils import postprocess
            preds = self.model(im)
            preds = postprocess(
                preds, 1, conf_thre=self.args.conf,
                nms_thre=0.45, class_agnostic=True
            )[0]

            # (x, y, x, y, conf, obj, cls) --> (x, y, x, y, conf, cls)
            preds[:, 4] = preds[:, 4] * preds[:, 5]
            preds = preds[:, [0, 1, 2, 3, 4, 6]]

            # calculate factor for predictions
            im0_w = im0s[0].shape[1]
            im0_h = im0s[0].shape[0]
            im_w = im[0].shape[2]
            im_h = im[0].shape[1]
            w_r = im0_w / im_w
            h_r = im0_h / im_h

            # scale to original image
            preds[:, [0, 2]] = preds[:, [0, 2]] * w_r
            preds[:, [1, 3]] = preds[:, [1, 3]] * h_r

            preds = torch.clip(preds, min=0)
            preds.detach().cpu().numpy()

        elif 'yolov8' in self.model_name:
            preds = self.model(
                im,
                augment=False,
                visualize=False
            )
        else:
            LOGGER.error('The Yolo model you selected is not available')
            exit()

        return preds

    def overwrite_results(self, i, im0_shape, predictor):
        # overwrite bbox results with tracker predictions
        if predictor.tracker_outputs[i].size != 0:
            predictor.results[i].boxes = Boxes(
                # xyxy, (track_id), conf, cls
                boxes=torch.from_numpy(predictor.tracker_outputs[i]).to(self.device),
                orig_shape=im0_shape,  # (height, width)
            )

    def filter_results(self, i, predictor):
        if predictor.tracker_outputs[i].size != 0:
            # filter boxes masks and pose results by tracking results
            predictor.tracker_outputs[i] = predictor.tracker_outputs[i][predictor.tracker_outputs[i][:, 5].argsort()[::-1]]
            yolo_confs = predictor.results[i].boxes.conf.cpu().numpy()
            tracker_confs = predictor.tracker_outputs[i][:, 5]
            mask = np.in1d(yolo_confs, tracker_confs)

            if predictor.results[i].masks is not None:
                predictor.results[i].masks = predictor.results[i].masks[mask]
                predictor.results[i].boxes = predictor.results[i].boxes[mask]
            elif predictor.results[i].keypoints is not None:
                predictor.results[i].boxes = predictor.results[i].boxes[mask]
                predictor.results[i].keypoints = predictor.results[i].keypoints[mask]

    def postprocess(self, path, preds, im, im0s, predictor):
        if 'yolo_nas' in self.model_name or 'yolox' in self.model_name:
            predictor.results[0] = Results(
                path=path,
                boxes=preds,
                orig_img=im0s[0],
                names=predictor.model.names
            )
        else:
            predictor.results = predictor.postprocess(preds, im, im0s)
        return predictor.results


if __name__ == "__main__":
    yolo = MultiYolo(model='YOLO_NAS_S', device='cuda:0')
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    yolo(rgb, rgb)
