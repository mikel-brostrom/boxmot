# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils.model_utils import fuse_model

from boxmot.detectors.detector import Detections, Detector
from boxmot.utils import BENCHMARK_CONFIGS, logger as LOGGER

# default model weights for generic YOLOX model names
YOLOX_ZOO = {
    "yolox_n.pt": "https://drive.google.com/uc?id=1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX",
    "yolox_s.pt": "https://drive.google.com/uc?id=1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj",
    "yolox_m.pt": "https://drive.google.com/uc?id=11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun",
    "yolox_l.pt": "https://drive.google.com/uc?id=1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz",
    "yolox_x.pt": "https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
}
YOLOX_BASE_MODELS = tuple(Path(name).stem for name in YOLOX_ZOO)


def _find_benchmark_model_url(model: Path) -> str | None:
    """Look up a detector download URL from benchmark configs by filename."""
    lowered_name = model.name.lower()
    for cfg_path in sorted(BENCHMARK_CONFIGS.glob("*.yaml")):
        try:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            continue

        detector_cfg = cfg.get("detector") or {}
        default_model = detector_cfg.get("default_model") or detector_cfg.get("model")
        model_url = detector_cfg.get("model_url") or detector_cfg.get("url")
        if not default_model or not model_url:
            continue
        if Path(default_model).name.lower() == lowered_name:
            return str(model_url)
    return None


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


class YoloXDetector(Detector):
    """YOLOX detector with standalone preprocess/process/postprocess pipeline."""

    pt = False
    stride = 32
    fp16 = False
    triton = False
    names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
        34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
        38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
        42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
        47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
        52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
        57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
        61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
        66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
        70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
        75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
        79: "toothbrush",
    }

    def __init__(self, model, device, args=None, imgsz=None):
        # args: accepted for backward compatibility but not stored
        # imgsz: explicit image size override; falls back to args.imgsz or 640
        raw = imgsz or (getattr(args, 'imgsz', None) if args is not None else None) or 640
        vals = raw if isinstance(raw, (list, tuple)) else (raw,)
        w, h = (vals * 2)[:2]
        self.imgsz = [w, h]

        model_type = self._get_model_type(YOLOX_BASE_MODELS, model)

        if model_type == "yolox_n":
            exp_name = "yolox_nano"
        elif "_MOT" in model_type or "_dancetrack" in model_type or "_visdrone" in model_type:
            exp_name = (
                model_type.split("_MOT")[0]
                .split("_dancetrack")[0]
                .split("_visdrone")[0]
            )
        else:
            exp_name = model_type
        exp = get_exp(None, exp_name)

        LOGGER.info(f"Loading {model_type} with {str(model)}")

        benchmark_model_url = _find_benchmark_model_url(model)
        if not model.exists() and model.stem == model_type:
            LOGGER.info("Downloading pretrained weights...")
            from boxmot.utils.download import download_file
            download_file(url=YOLOX_ZOO[model.stem + ".pt"], dest=model, overwrite=False)
            exp.num_classes = 1
        elif not model.exists() and benchmark_model_url:
            LOGGER.info("Downloading benchmark detector weights...")
            from boxmot.utils.download import download_file
            download_file(url=benchmark_model_url, dest=model, overwrite=False)
            exp.num_classes = 1
        elif model.stem.startswith(model_type):
            exp.num_classes = 1

        ckpt = torch.load(str(model), map_location=torch.device("cpu"))

        self.device = device
        self.model = exp.get_model()
        self.model.eval()
        self.model.to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model = fuse_model(self.model)
        self._preproc_data = []
        self._im0s = []

    def _get_model_type(self, model_names, weight_path):
        for name in model_names:
            if name in str(weight_path):
                return name.split('.')[0]
        return "yolox_s"

    # This preprocess matches ByteTrack's implementation:
    # https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/data/data_augment.py#L189
    def _letterbox(
        self,
        image,
        input_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        if len(image.shape) == 3:
            padded = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized
        padded = padded[:, :, ::-1] / 255.0
        padded = (padded - mean) / std
        padded = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
        return padded, r

    def preprocess(self, images: list) -> torch.Tensor:
        assert isinstance(images, list)
        self._im0s = images
        self._preproc_data = []
        tensors = []
        for img in images:
            t, ratio = self._letterbox(img, input_size=self.imgsz)
            tensors.append(torch.from_numpy(t).unsqueeze(0).to(self.device))
            self._preproc_data.append(ratio)
        return torch.vstack(tensors)

    @torch.no_grad()
    def process(self, preprocessed: torch.Tensor) -> torch.Tensor:
        if preprocessed.ndim == 3:
            preprocessed = preprocessed.unsqueeze(0)
        return self.model(preprocessed)

    def postprocess(self, detections, conf, iou, classes, agnostic_nms, **kwargs) -> list:
        results = []
        for i, det in enumerate(detections):
            orig_img = self._im0s[i] if i < len(self._im0s) else None

            filtered = postprocess(
                det.unsqueeze(0), 1,
                conf_thre=conf, nms_thre=iou, class_agnostic=agnostic_nms,
            )[0]

            if filtered is None:
                boxes = np.empty((0, 6))
            else:
                ratio = self._preproc_data[i]
                filtered[:, :4] /= ratio
                filtered[:, 4] *= filtered[:, 5]   # obj_conf * class_conf → final conf
                filtered = filtered[:, [0, 1, 2, 3, 4, 6]]  # drop class_conf column

                if classes:
                    mask = np.isin(filtered[:, 5].cpu().numpy().astype(int), classes)
                    filtered = filtered[torch.from_numpy(mask)]

                boxes = filtered.cpu().numpy()

            results.append(Detections(dets=boxes, orig_img=orig_img, names=self.names))

        return results
