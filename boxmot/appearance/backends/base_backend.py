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
from boxmot.utils.iou import iou_batch


class BaseModelBackend:
    def __init__(self, weights, device, half):
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
        self.iou_threshold = 0.15
        self.ars_threshold=0.6

    def get_crops(self, xyxys, img):
        crops = []
        h, w = img.shape[:2]
        resize_dims = (128, 256)
        interpolation_method = cv2.INTER_LINEAR
        mean_array = np.array([0.485, 0.456, 0.406])
        std_array = np.array([0.229, 0.224, 0.225])
        # dets are of different sizes so batch preprocessing is not possible
        for box in xyxys:
            x1, y1, x2, y2 = box.astype('int')
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            crop = img[y1:y2, x1:x2]
            # resize
            crop = cv2.resize(
                crop,
                resize_dims,  # from (x, y) to (128, 256) | (w, h)
                interpolation=interpolation_method,
            )

            # (cv2) BGR 2 (PIL) RGB. The ReID models have been trained with this channel order
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crop = torch.from_numpy(crop).float()
            crops.append(crop)

        # List of torch tensor crops to unified torch tensor
        crops = torch.stack(crops, dim=0)

        # Normalize the batch
        crops = crops / 255.0

        # Standardize the batch
        crops = (crops - mean_array) / std_array

        crops = torch.permute(crops, (0, 3, 1, 2))
        crops = crops.to(dtype=torch.half if self.half else torch.float, device=self.device)

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
        features = features / np.linalg.norm(features)
        return features
    
    def aspect_ratio_similarity(self, box1, box2):
        # Ensure both boxes are 1D arrays of length 4
        if len(box1.shape) == 2 and box1.shape[0] == 1:
            box1 = box1.squeeze(0)  # Convert (1, 4) to (4,)
        if len(box2.shape) == 2 and box2.shape[0] == 1:
            box2 = box2.squeeze(0)  # Convert (1, 4) to (4,)
        # Calculate width and height for box1
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        # Calculate width and height for box2
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        
        # Calculate aspect ratios
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2
        
        # Compute aspect ratio similarity using the formula provided
        similarity = 4 / (np.pi ** 2) * (np.arctan(aspect_ratio1) - np.arctan(aspect_ratio2)) ** 2
        return similarity

    @torch.no_grad()
    def get_features_fast(self, xyxy, img, active_tracks, embs):

        risky_detections = []
        non_risky_matches = {}
        for i, det in enumerate(xyxy):
            matching_tracks = []
            for at in active_tracks:
                iou = iou_batch(det.reshape(1, -1), at.to_tlbr().reshape(1, -1))[0][0]
                if iou > self.iou_threshold:
                    matching_tracks.append((at, iou))

            if len(matching_tracks) == 1:
                track, iou = matching_tracks[0]
                ars = self.aspect_ratio_similarity(det, track.to_tlbr())
                v = ars
                alpha = v / ((1 - iou) + v)
                if alpha <= self.ars_threshold:
                    # Non-risky detection, use track's features
                    non_risky_matches[i] = track
                    continue

            # Risky detection, needs feature extraction
            risky_detections.append(i)
            
        # Extract features only for risky detections otherwise use last feature
        if embs is not None:
            features = embs[risky_detections]
        else:
            features = self.get_features(xyxy[risky_detections], img)

        # Prepare detections
        feats = []
        for i, _ in enumerate(xyxy):
            if i in risky_detections:
                print('riskyyy!')
                feat = features[risky_detections.index(i)]
            else:
                # For non-risky detections, use the matching track's features
                feat = non_risky_matches[i].features[-1]  # Use the latest feature from the matching track
            feats.append(feat)
            
        # Check if the total number of features matches the number of detections
        if len(feats) != len(xyxy):
            raise ValueError(f"Mismatch between number of detections ({len(xyxy)}) and number of features ({len(feats)}).")
    
        feats = torch.tensor(feats, dtype=torch.float32)
            
        return feats

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
            model_url = get_model_url(w)
            if not w.exists() and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif not w.exists():
                LOGGER.error(
                    f"No URL associated with the chosen StrongSORT weights ({w}). Choose between:"
                )
                show_downloadable_models()
                exit()