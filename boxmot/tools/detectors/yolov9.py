# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import torch
from ultralytics.engine.results import Results
from yolov9 import load

from boxmot.utils import logger as LOGGER
from examples.detectors.yolo_interface import YoloInterface

YOLOv9_ZOO = {
    "gelan-c.pt": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt",
    "gelan-e.pt": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt",
    "yolov9-c.pt": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt",
    "yolov9-e.pt": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt",
}


class Yolov9Strategy(YoloInterface):
    pt = False
    stride = 32
    fp16 = False
    triton = False
    names = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }

    def __init__(self, model, device, args):

        self.args = args
        self.pt = False
        self.stride = 32  # max stride in YOLOX

        # model_type one of: 'yolox_n', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x'
        model_type = self.get_model_from_weigths(YOLOv9_ZOO.keys(), model)

        LOGGER.info(f"Loading {model_type} with {str(model)}")

        # download crowdhuman bytetrack models
        if not model.exists() and model.stem == model_type:
            LOGGER.info("Downloading Yolov9 pretrained weights...")
            # download(
            #     url=YOLOv9_ZOO[model_type + '.pt'],
            #     dir="./weights",
            # )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = load(
            "/home/mikel.brostrom/yolo_tracking/examples/weights/yolov9-c.pt",
            device=device,
        )

        self.model.conf = args.conf
        self.model.iou = args.iou
        self.model.classes = args.classes

    @torch.no_grad()
    def __call__(self, im, augment, visualize):

        im = im[0].permute(1, 2, 0).cpu().numpy() * 255

        with torch.no_grad():
            results = self.model(im)
            preds = results.pred[0]

        preds = preds.unsqueeze(0)

        return preds

    def warmup(self, imgsz):
        pass

    def postprocess(self, path, preds, im, im0s):

        results = []
        for i, pred in enumerate(preds):

            if pred is None:
                pred = torch.empty((0, 6))
                r = Results(path=path, boxes=pred, orig_img=im0s[i], names=self.names)
            else:
                pred = self.clip(pred, im0s[i])
                r = Results(path=path, boxes=pred, orig_img=im0s[i], names=self.names)
            results.append(r)
        return results
