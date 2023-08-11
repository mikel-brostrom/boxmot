# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import gdown
import torch
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils.model_utils import fuse_model

from boxmot.utils import logger as LOGGER
from examples.detectors.yolo_interface import YoloInterface

# default model weigths for these model names
YOLOX_ZOO = {
    'yolox_n.pt': 'https://drive.google.com/uc?id=1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX',
    'yolox_s.pt': 'https://drive.google.com/uc?id=1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj',
    'yolox_m.pt': 'https://drive.google.com/uc?id=11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun',
    'yolox_l.pt': 'https://drive.google.com/uc?id=1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz',
    'yolox_x.pt': 'https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5',
}


class YoloXStrategy(YoloInterface):
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
        self.pt = False
        self.stride = 32  # max stride in YOLOX

        # model_type one of: 'yolox_n', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x'
        model_type = self.get_model_from_weigths(YOLOX_ZOO.keys(), model)

        if model_type == 'yolox_n':
            exp = get_exp(None, 'yolox_nano')
        else:
            exp = get_exp(None, model_type)

        LOGGER.info(f'Loading {model_type} with {str(model)}')

        # download crowdhuman bytetrack models
        if not model.exists() and model.stem == model_type:
            LOGGER.info('Downloading pretrained weights...')
            gdown.download(
                url=YOLOX_ZOO[model_type + '.pt'],
                output=str(model),
                quiet=False
            )
            # needed for bytetrack yolox people models
            # update with your custom model needs
            exp.num_classes = 1
        elif model.stem == model_type:
            exp.num_classes = 1

        ckpt = torch.load(
            str(model),
            map_location=torch.device('cpu')
        )

        self.model = exp.get_model()
        self.model.eval()
        self.model.load_state_dict(ckpt["model"])
        self.model = fuse_model(self.model)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, im, augment, visualize):
        preds = self.model(im)
        return preds

    def warmup(self, imgsz):
        pass

    def postprocess(self, path, preds, im, im0s):

        results = []
        for i, pred in enumerate(preds):

            pred = postprocess(
                pred.unsqueeze(0),  # YOLOX postprocessor expects 3D arary
                1,
                conf_thre=0.1,
                nms_thre=0.45,
                class_agnostic=True
            )[0]

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
                # (x, y, x, y, conf, obj, cls) --> (x, y, x, y, conf, cls)
                pred[:, 4] = pred[:, 4] * pred[:, 5]
                pred = pred[:, [0, 1, 2, 3, 4, 6]]

                pred[:, :4] = ops.scale_boxes(im.shape[2:], pred[:, :4], im0s[i].shape)

                # filter boxes by classes
                if self.args.classes:
                    pred = pred[torch.isin(pred[:, 5].cpu(), torch.as_tensor(self.args.classes))]

                r = Results(
                    path=path,
                    boxes=pred,
                    orig_img=im0s[i],
                    names=self.names
                )

            results.append(r)

        return results
