import cv2
import torch
import gdown
import numpy as np
from abc import ABC, abstractmethod
from boxmot.utils import logger as LOGGER
from boxmot.appearance.reid_model_factory import (
    get_model_name,
    get_model_url,
    build_model,
    get_nr_classes,
    show_downloadable_models
)
from boxmot.utils.checks import RequirementsChecker


class BaseModelBackend:
    def __new__(cls, weights: str, device: torch.device, half: bool):
        """
        Creates a new instance of the model and returns the initialized model directly.
        
        Args:
            weights (str): Path to the model weights file.
            device (torch.device): Device to load the model on ('cpu' or 'cuda').
            half (bool): Whether to use half precision.
        
        Returns:
            torch.nn.Module: Initialized model.
        """
        instance = super(BaseModelBackend, cls).__new__(cls)
        instance.__init__(weights, device, half)
        return instance.model

    def __init__(self, weights: str, device: torch.device, half: bool):
        """
        Initializes the model backend with the specified weights, device, and precision.
        
        Args:
            weights (str): Path to the model weights file.
            device (torch.device): Device to load the model on ('cpu' or 'cuda').
            half (bool): Whether to use half precision.
        """
        self.weights = weights[0] if isinstance(weights, list) else weights
        self.device = device
        self.half = half
        self.model = None
        self.cuda = torch.cuda.is_available() and self.device.type != "cpu"

        self.download_model(self.weights)
        self.model_name = get_model_name(self.weights)

        self.model = build_model(
            self.model_name,
            num_classes=get_nr_classes(self.weights),
            pretrained=not (self.weights and self.weights.is_file()),
            use_gpu=device,
        )
        self.checker = RequirementsChecker()
        self.load_model(self.weights)

    def get_crops(self, xyxys: np.ndarray, img: np.ndarray) -> torch.Tensor:
        """
        Extracts and preprocesses crops from the input image based on bounding boxes.

        Args:
            xyxys (np.ndarray): Bounding boxes in the format [x1, y1, x2, y2].
            img (np.ndarray): The input image.

        Returns:
            torch.Tensor: Preprocessed crops as a batch of tensors.
        """
        crops = []
        h, w = img.shape[:2]
        resize_dims = (128, 256)
        interpolation_method = cv2.INTER_LINEAR
        mean_array = np.array([0.485, 0.456, 0.406])
        std_array = np.array([0.229, 0.224, 0.225])

        for box in xyxys:
            x1, y1, x2, y2 = box.astype('int')
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, resize_dims, interpolation=interpolation_method)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = torch.from_numpy(crop).float()
            crops.append(crop)

        crops = torch.stack(crops, dim=0)
        crops = crops / 255.0
        crops = (crops - mean_array) / std_array
        crops = torch.permute(crops, (0, 3, 1, 2))
        crops = crops.to(dtype=torch.half if self.half else torch.float, device=self.device)

        return crops

    @torch.no_grad()
    def get_features(self, xyxys: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Extracts feature vectors for the input bounding boxes and image.

        Args:
            xyxys (np.ndarray): Bounding boxes in the format [x1, y1, x2, y2].
            img (np.ndarray): The input image.

        Returns:
            np.ndarray: Normalized feature vectors.
        """
        if xyxys.size != 0:
            crops = self.get_crops(xyxys, img)
            crops = self.inference_preprocess(crops)
            features = self.forward(crops)
            features = self.inference_postprocess(features)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features)
        return features

    def warmup(self, imgsz: list = [(256, 128, 3)]):
        """
        Warms up the model by performing a dummy forward pass.

        Args:
            imgsz (list): List of image size dimensions.
        """
        if self.device.type != "cpu":
            im = np.random.randint(0, 255, *imgsz, dtype=np.uint8)
            crops = self.get_crops(np.array([[0, 0, 64, 64], [0, 0, 128, 128]]), img=im)
            crops = self.inference_preprocess(crops)
            self.forward(crops)

    def to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """
        Converts a torch tensor to a numpy array.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            np.ndarray: Numpy array.
        """
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def inference_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input tensor for inference.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Preprocessed tensor.
        """
        if self.half:
            x = x.half() if isinstance(x, torch.Tensor) and x.dtype != torch.float16 else x

        if self.nhwc:
            x = x.permute(0, 2, 3, 1) if isinstance(x, torch.Tensor) else np.transpose(x, (0, 2, 3, 1))
        return x

    def inference_postprocess(self, features) -> np.ndarray:
        """
        Postprocesses the feature output after inference.

        Args:
            features: Feature output from the model.

        Returns:
            np.ndarray: Postprocessed features as numpy arrays.
        """
        if isinstance(features, (list, tuple)):
            return self.to_numpy(features[0]) if len(features) == 1 else [self.to_numpy(x) for x in features]
        else:
            return self.to_numpy(features)

    @abstractmethod
    def forward(self, im_batch: torch.Tensor):
        """
        Forward pass through the model.

        Args:
            im_batch (torch.Tensor): Batch of input images.

        Raises:
            NotImplementedError: Needs to be implemented by subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def load_model(self, w: str):
        """
        Loads the model weights.

        Args:
            w (str): Path to model weights.

        Raises:
            NotImplementedError: Needs to be implemented by subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def download_model(self, w: str):
        """
        Downloads the model weights if not available locally.

        Args:
            w (str): Path to model weights.
        """
        if w.suffix == ".pt":
            model_url = get_model_url(w)
            if not w.exists() and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif not w.exists():
                LOGGER.error(f"No URL associated with the chosen StrongSORT weights ({w}).")
                show_downloadable_models()
                exit()
