import os
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from boxmot.reid.backbones import get_backbone_spec
from boxmot.reid.core.crops import build_crop_batch
from boxmot.reid.core.preprocessing import get_preprocess_fn
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.misc import resolve_model_path


class BaseModelBackend:
    build_source_model = True

    def __init__(self, weights, device, half, preprocess=None):
        self.weights = weights[0] if isinstance(weights, list) else weights
        if isinstance(self.weights, str):
            self.weights = Path(self.weights)
        self.weights = resolve_model_path(self.weights)
        LOGGER.info(self.weights)
        self.device = device
        self.half = half
        self.model = None
        self.cuda = torch.cuda.is_available() and self.device.type != "cpu"

        self.download_model(self.weights)
        # Metadata inspection and the final state load all read the same
        # checkpoint. Scope their stat-keyed cache to this construction so the
        # serialized tensors are released as soon as the backend owns them.
        with ReIDModelRegistry.checkpoint_load_scope():
            self.model_name = ReIDModelRegistry.get_model_name(self.weights)
            checkpoint_model_kwargs = {}
            if self.weights and self.weights.exists():
                checkpoint_model_kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(self.weights)
            self.model_kwargs = ReIDModelRegistry.deployment_model_kwargs(
                self.model_name,
                checkpoint_model_kwargs,
            )
            # Resolve the configured crop contract before loading the runtime.
            # Compiled backends can then validate it against their graph input and
            # make a more specific graph-declared shape authoritative.
            if self.model_kwargs.get("img_size"):
                self.input_shape = tuple(self.model_kwargs["img_size"])
            elif "vehicleid" in self.weights.name or "veri" in self.weights.name:
                self.input_shape = (256, 256)
            else:
                try:
                    self.input_shape = get_backbone_spec(self.model_name).default_img_size
                except KeyError:
                    self.input_shape = (256, 128)
            if self.build_source_model:
                num_classes = ReIDModelRegistry.get_nr_classes(self.weights)
                if str(self.model_name or "").startswith("csl_tinyvit"):
                    # Classification layers are never traversed by a deployed CSL
                    # descriptor. One-output placeholders avoid materializing up
                    # to a million random parameters before deployment pruning,
                    # while remaining valid for standard torch initializers.
                    num_classes = 1
                self.model = ReIDModelRegistry.build_model(
                    self.model_name,
                    self.weights,
                    num_classes=num_classes,
                    pretrained=not (self.weights and self.weights.exists()),
                    use_gpu=device,
                    **self.model_kwargs,
                )
            self.checker = RequirementsChecker()
            self._preprocess_name = preprocess
            self.preprocess_fn = get_preprocess_fn(preprocess)
            self.load_model(self.weights)

        self.mean_array = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std_array = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def get_crops(self, xyxys, img):
        return build_crop_batch(
            xyxys,
            img,
            input_shape=self.input_shape,
            device=self.device,
            half=self.half,
            preprocess_fn=self.preprocess_fn,
            mean=self.mean_array,
            std=self.std_array,
        )

    @torch.no_grad()
    def get_features(self, xyxys, img):
        xyxys = np.asarray(xyxys)
        if xyxys.size != 0:
            crops = self.get_crops(xyxys, img)
            crops = self.inference_preprocess(crops)
            features = self.forward(crops)
            features = self.inference_postprocess(features)
        else:
            features = np.array([])
        features = np.asarray(features)
        if features.size == 0:
            return features
        features = np.where(np.isfinite(features), features, 0.0)
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        return features / safe_norms

    def warmup(self, imgsz=[(256, 128, 3)]):
        # warmup model by running inference once
        if self.device.type != "cpu":
            im = np.random.randint(0, 255, *imgsz, dtype=np.uint8)
            crops = self.get_crops(
                xyxys=np.array([[0, 0, 64, 64], [0, 0, 128, 128]]), img=im
            )
            crops = self.inference_preprocess(crops)
            self.forward(crops)  # warmup

    def to_numpy(self, x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

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
        if isinstance(w, str):
            w = Path(w)
        w = resolve_model_path(w)

        if w.suffix != ".pt":
            return

        # A local checkpoint is already complete and needs no synchronization.
        # Check it before constructing/acquiring the cross-process download lock:
        # another process may legitimately hold that lock for minutes, but it
        # must not delay startup for callers that already have usable weights.
        # The same condition is checked again under the lock below to preserve
        # the missing-file download race guarantee.
        if w.exists() or "openvino" in w.name:
            LOGGER.info(f"[PID {os.getpid()}] Found existing ReID weights at {w}; skipping download.")
            return

        w.parent.mkdir(parents=True, exist_ok=True)

        model_url = ReIDModelRegistry.get_model_url(w)
        # Use a temp directory for lock files to avoid "no space left" errors
        # when the local disk is full but the model already exists.
        import tempfile

        lock_path = Path(tempfile.gettempdir()) / (w.name + ".lock")
        # FileLock uses an OS-level lock, so a crashed downloader releases
        # ownership even if its marker file remains on disk.
        lock = FileLock(str(lock_path), timeout=300)  # Wait up to 5 minutes

        with lock:
            # A peer may have completed the download while this process waited
            # for the lock, so keep the authoritative check inside the critical
            # section as well as the lock-free startup fast path above.
            if w.exists() or "openvino" in w.name:
                LOGGER.info(f"[PID {os.getpid()}] Found existing ReID weights at {w}; skipping download.")
                return

            if model_url:
                LOGGER.info(f"[PID {os.getpid()}] Downloading ReID weights from {model_url} → {w}")
                # Always route through download_file: it handles both the
                # Google Drive confirm-token flow (via gdown) and direct
                # HTTP(S) downloads, and integrates with an active Rich
                # workflow's status callback so the progress is rendered
                # inside the panel instead of leaking raw tqdm output.
                from boxmot.utils.download import download_file

                download_file(model_url, w)
            else:
                LOGGER.error(
                    f"No URL associated with the chosen ReID weights ({w}).\n"
                    f"Choose one of the following:"
                )
                ReIDModelRegistry.show_downloadable_models()
