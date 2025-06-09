# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import cv2
import gdown
import numpy as np
import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils.model_utils import fuse_model

from boxmot.utils import logger as LOGGER
from boxmot.engine.detectors.yolo_interface import YoloInterface

# default model weigths for these model names
YOLOX_ZOO = {
    "yolox_n.pt": "https://drive.google.com/uc?id=1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX",
    "yolox_s.pt": "https://drive.google.com/uc?id=1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj",
    "yolox_m.pt": "https://drive.google.com/uc?id=11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun",
    "yolox_l.pt": "https://drive.google.com/uc?id=1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz",
    "yolox_x.pt": "https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
    "yolox_x_ablation.pt": "https://drive.google.com/uc?id=1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob",
}


class YoloXStrategy(YoloInterface):
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

        self.ch = 3
        self.args = args
        self.imgsz = args.imgsz
        self.pt = False
        self.stride = 32  # max stride in YOLOX

        # model_type one of: 'yolox_n', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x'
        model_type = self.get_model_from_weigths(YOLOX_ZOO.keys(), model)

        if model_type == "yolox_n":
            exp = get_exp(None, "yolox_nano")
        else:
            exp = get_exp(None, model_type)

        LOGGER.info(f"Loading {model_type} with {str(model)}")

        # download crowdhuman bytetrack models
        if not model.exists() and (
            model.stem == model_type or model.stem == "yolox_x_ablation"
        ):
            LOGGER.info("Downloading pretrained weights...")
            gdown.download(
                url=YOLOX_ZOO[model.stem + ".pt"], output=str(model), quiet=False
            )
            # needed for bytetrack yolox people models
            # update with your custom model needs
            exp.num_classes = 1
        elif model.stem.startswith(model_type):
            exp.num_classes = 1

        ckpt = torch.load(str(model), map_location=torch.device("cpu"))

        self.device = device
        self.model = exp.get_model()
        self.model.eval()
        self.model.load_state_dict(ckpt["model"])
        self.model = fuse_model(self.model)
        self.model.to(self.device)
        self.model.eval()
        self.im_paths = []
        self._preproc_data = []

    @torch.no_grad()
    def __call__(self, im, augment, visualize, embed):
        if isinstance(im, list):
            if len(im[0].shape) == 3:
                im = torch.stack(im)
            else:
                im = torch.vstack(im)

        if len(im.shape) == 3:
            im = im.unsqueeze(0)

        assert len(im.shape) == 4, f"Expected 4D tensor as input, got {im.shape}"

        preds = self.model(im)
        return preds

    def warmup(self, imgsz):
        pass

    def update_im_paths(self, predictor: DetectionPredictor):
        """
        This function saves image paths for the current batch,
        being passed as callback on_predict_batch_start
        """
        assert (
            isinstance(predictor, DetectionPredictor),
            "Only ultralytics predictors are supported",
        )
        self.im_paths = predictor.batch[0]

    # This preprocess differs from the current version of YOLOX preprocess, but ByteTrack uses it
    # https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/data/data_augment.py#L189
    def yolox_preprocess(
        self,
        image,
        input_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        swap=(2, 0, 1),
    ):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def preprocess(self, im) -> torch.Tensor:
        assert isinstance(im, list)
        im_preprocessed = []
        self._preproc_data = []
        for i, img in enumerate(im):
            img_pre, ratio = self.yolox_preprocess(img, input_size=self.imgsz)
            img_pre = torch.Tensor(img_pre).unsqueeze(0).to(self.device)

            im_preprocessed.append(img_pre)
            self._preproc_data.append(ratio)

        im_preprocessed = torch.vstack(im_preprocessed)

        return im_preprocessed

    def postprocess(self, preds, im, im0s):

        results = []
        for i, pred in enumerate(preds):
            im_path = self.im_paths[i] if len(self.im_paths) else ""

            pred = postprocess(
                pred.unsqueeze(0),  # YOLOX postprocessor expects 3D arary
                1,
                conf_thre=self.args.conf,
                nms_thre=self.args.iou,
                class_agnostic=self.args.agnostic_nms,
            )[0]

            if pred is None:
                pred = torch.empty((0, 6))
                r = Results(
                    path=im_path, boxes=pred, orig_img=im0s[i], names=self.names
                )
                results.append(r)
            else:
                ratio = self._preproc_data[i]
                pred[:, 0] = pred[:, 0] / ratio
                pred[:, 1] = pred[:, 1] / ratio
                pred[:, 2] = pred[:, 2] / ratio
                pred[:, 3] = pred[:, 3] / ratio
                pred[:, 4] *= pred[:, 5]
                pred = pred[:, [0, 1, 2, 3, 4, 6]]

                # filter boxes by classes
                if self.args.classes:
                    pred = pred[
                        torch.isin(pred[:, 5].cpu(), torch.as_tensor(self.args.classes))
                    ]

                r = Results(
                    path=im_path, boxes=pred, orig_img=im0s[i], names=self.names
                )

            results.append(r)

        return results
