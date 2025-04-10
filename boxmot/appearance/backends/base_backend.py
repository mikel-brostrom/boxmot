import cv2
import torch
import gdown
import numpy as np
from abc import ABC, abstractmethod
from boxmot.utils import logger as LOGGER
from boxmot.appearance.reid.registry import ReIDModelRegistry
from boxmot.utils.checks import RequirementsChecker


class BaseModelBackend:
    def __init__(self, weights, device, half):
        self.weights = weights[0] if isinstance(weights, list) else weights
        self.device = device
        self.half = half
        self.model = None
        self.cuda = torch.cuda.is_available() and self.device.type != "cpu"

        self.download_model(self.weights)
        self.model_name = ReIDModelRegistry.get_model_name(self.weights)

        self.model = ReIDModelRegistry.build_model(
            self.model_name,
            num_classes=ReIDModelRegistry.get_nr_classes(self.weights),
            pretrained=not (self.weights and self.weights.is_file()),
            use_gpu=device,
        )
        self.checker = RequirementsChecker()
        self.load_model(self.weights)


    def get_crops(self, xyxys, img):
        h, w = img.shape[:2]
        resize_dims = (128, 256)
        interpolation_method = cv2.INTER_LINEAR
        mean_array = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std_array = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        # Preallocate tensor for crops
        num_crops = len(xyxys)
        crops = torch.empty((num_crops, 3, resize_dims[1], resize_dims[0]), 
                            dtype=torch.half if self.half else torch.float, device=self.device)
        
        for i, box in enumerate(xyxys):
            x1, y1, x2, y2 = box.round().astype('int')
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2]
            
            # Resize and convert color in one step
            crop = cv2.resize(crop, resize_dims, interpolation=interpolation_method)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and normalize (convert to [0, 1] by dividing by 255 in batch later)
            crop = torch.from_numpy(crop).to(self.device, dtype=torch.half if self.half else torch.float)
            crops[i] = torch.permute(crop, (2, 0, 1))  # Change to (C, H, W)
        
        # Normalize the entire batch in one go
        crops = crops / 255.0

        # Standardize the batch
        crops = (crops - mean_array) / std_array
        
        return crops


    @torch.no_grad()
    def get_features(self, xyxys, img):
        if xyxys.size != 0:
            crops = self.get_crops(xyxys, img)
            crops = self.inference_preprocess(crops)
            features = self.forward(crops)
            features = self.inference_postprocess(features)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features

    def warmup(self, imgsz=[(256, 128, 3)]):
        # warmup model by running inference once
        if self.device.type != "cpu":
            im = np.random.randint(0, 255, *imgsz, dtype=np.uint8)
            crops = self.get_crops(xyxys=np.array(
                [[0, 0, 64, 64], [0, 0, 128, 128]]),
                img=im
            )
            crops = self.inference_preprocess(crops)
            self.forward(crops)  # warmup

    def to_numpy(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def inference_preprocess(self, x):
        if self.half:
            if isinstance(x, torch.Tensor):
                if x.dtype != torch.float16:
                    x = x.half()
            elif isinstance(x, np.ndarray):
                if x.dtype != np.float16:
                    x = x.astype(np.float16)

        if self.nhwc:
            if isinstance(x, torch.Tensor):
                x = x.permute(0, 2, 3, 1)  # Convert from NCHW to NHWC
            elif isinstance(x, np.ndarray):
                x = np.transpose(x, (0, 2, 3, 1))  # Convert from NCHW to NHWC
        return x
    
    def inference_postprocess(self, features):
        if isinstance(features, (list, tuple)):
            return (
                self.to_numpy(features[0]) if len(features) == 1 else [self.to_numpy(x) for x in features]
            )
        else:
            return self.to_numpy(features)

    @abstractmethod
    def forward(self, im_batch):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def load_model(self, w):
        raise NotImplementedError("This method should be implemented by subclasses.")


    def download_model(self, w):
        if w.suffix == ".pt":
            model_url = ReIDModelRegistry.get_model_url(w)
            if not w.exists() and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif not w.exists():
                LOGGER.error(
                    f"No URL associated with the chosen StrongSORT weights ({w}). Choose between:"
                )
                ReIDModelRegistry.show_downloadable_models()
                exit()