# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import fnmatch

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils.model_utils import fuse_model

from boxmot.utils import logger as LOGGER

# default model weigths for these model names
YOLOX_ZOO = {
    "yolox_n.pt": "https://drive.google.com/uc?id=1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX",
    "yolox_s.pt": "https://drive.google.com/uc?id=1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj",
    "yolox_m.pt": "https://drive.google.com/uc?id=11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun",
    "yolox_l.pt": "https://drive.google.com/uc?id=1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz",
    "yolox_x.pt": "https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
    # the source for the models below is SparseTrack: https://github.com/hustvl/SparseTrack#model-zoo
    "yolox_x_MOT17_ablation.pt": "https://drive.google.com/uc?id=1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob",
    "yolox_x_MOT20_ablation.pt": "https://drive.google.com/uc?id=1H1BxOfinONCSdQKnjGq0XlRxVUo_4M8o",
    "yolox_x_dancetrack_ablation.pt": "https://drive.google.com/uc?id=1ZKpYmFYCsRdXuOL60NRuc7VXAFYRskXB",
    "yolox_x_visdrone.pt": "https://drive.google.com/uc?id=1ajehBs9enBHhuBqGIoQPGqkkzasE9d3o"
}


def _coerce_torch_dtype(dtype, fallback: torch.Tensor) -> torch.dtype:
    """Map YOLOX's dtype strings (e.g., 'torch.mps.FloatTensor') to real torch dtypes."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        lowered = dtype.lower()
        if "bfloat16" in lowered:
            return torch.bfloat16
        if "float16" in lowered or "half" in lowered:
            return torch.float16
    # Default to the fallback tensor's dtype or float32.
    return fallback.dtype if isinstance(fallback, torch.Tensor) else torch.float32


def _patch_yolox_head_decode_outputs_for_mps() -> None:
    """Monkeypatch YOLOXHead.decode_outputs to work on MPS (avoids .type with dtype strings)."""
    try:
        from yolox.models.yolo_head import YOLOXHead
        from yolox.utils import meshgrid
    except Exception:
        return

    if getattr(YOLOXHead, "_boxmot_mps_patched", False):
        return

    def decode_outputs(self, outputs, dtype):
        dtype = _coerce_torch_dtype(dtype, outputs)
        device = outputs.device
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([
                torch.arange(hsize, device=device),
                torch.arange(wsize, device=device),
            ])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride, device=device, dtype=grid.dtype))

        grids = torch.cat(grids, dim=1).to(device=device, dtype=dtype)
        strides = torch.cat(strides, dim=1).to(device=device, dtype=dtype)

        outputs = outputs.clone()
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    YOLOXHead.decode_outputs = decode_outputs
    YOLOXHead._boxmot_mps_patched = True


_patch_yolox_head_decode_outputs_for_mps()


class YoloXStrategy:
    """YOLOX strategy for use with Ultralytics predictor workflow."""
    
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
        raw = getattr(args, 'imgsz', None) or 640
        vals = raw if isinstance(raw, (list, tuple)) else (raw,)
        w, h = (vals * 2)[:2]
        self.imgsz = [w, h]
        self.pt = False
        self.stride = 32  # max stride in YOLOX

        # model_type one of: 'yolox_n', 'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x'
        model_type = self.get_model_from_weigths(YOLOX_ZOO.keys(), model)

        # Map model type to YOLOX experiment name
        # Custom trained models (e.g., yolox_x_MOT17_ablation) use the base architecture
        if model_type == "yolox_n":
            exp_name = "yolox_nano"
        elif "_MOT" in model_type or "_dancetrack" in model_type or "_visdrone" in model_type:
            # Extract base model: yolox_x_MOT17_ablation / yolox_x_visdrone -> yolox_x
            exp_name = (
                model_type.split("_MOT")[0]
                .split("_dancetrack")[0]
                .split("_visdrone")[0]
            )
        else:
            exp_name = model_type
        exp = get_exp(None, exp_name)

        LOGGER.info(f"Loading {model_type} with {str(model)}")

        # download crowdhuman bytetrack models
        if not model.exists() and (
            model.stem == model_type or fnmatch.fnmatch(model.stem, "yolox_x_*_ablation")
        ):
            LOGGER.info("Downloading pretrained weights...")
            from boxmot.utils.download import download_file
            download_file(
                url=YOLOX_ZOO[model.stem + ".pt"], dest=model, overwrite=False
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
        
        # folow official yolox loading procedure
        # https://github.com/Megvii-BaseDetection/YOLOX/blob/d872c71b/tools/eval.py#L148-L176
        self.model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(ckpt["model"])
        self.model = fuse_model(self.model)
        self.im_paths = []
        self._preproc_data = []

    def get_model_from_weigths(self, model_names, weight_path):
        for name in model_names:
            if name in str(weight_path):
                return name.split('.')[0]
        return "yolox_s"  # default

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
        assert isinstance(
            predictor, DetectionPredictor
        ), "Only ultralytics predictors are supported"
        self.im_paths = predictor.batch[0]

    # This preprocess differs from the current version of YOLOX preprocess, but ByteTrack uses it
    # https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/data/data_augment.py\#L189
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


# Alias for backward compatibility
YOLOX = YoloXStrategy
