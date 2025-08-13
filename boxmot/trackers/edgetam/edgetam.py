"""EdgeTAM tracker wrapper with streaming support.

This module exposes a minimal tracker interface compatible with the rest of
BoxMOT trackers. The underlying EdgeTAM project performs detection and
tracking in a single step. The class below loads the predictor if the
optional ``samtam`` dependencies are available and processes frames one by one
using an in-memory frame loader.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from boxmot.utils import logger as LOGGER, ROOT

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2_video_predictor


class RealTimeFrameLoader:
    """Frame loader for streaming frames to EdgeTAM."""

    def __init__(self, image_size: int, device: torch.device) -> None:
        self.image_size = image_size
        self.device = device
        self.images: list[torch.Tensor] = []
        self.video_height: int | None = None
        self.video_width: int | None = None
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[
            :, None, None
        ]
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[
            :, None, None
        ]

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        """Append a new frame in BGR format."""

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.video_height is None:
            self.video_height, self.video_width = frame_rgb.shape[:2]
        img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        img -= self.img_mean
        img /= self.img_std
        self.images.append(img.to(self.device))

    def __getitem__(self, index: int) -> torch.Tensor:  # pragma: no cover - simple
        return self.images[index]

    def __len__(self) -> int:  # pragma: no cover - simple
        return len(self.images)


class EdgeTAM:
    """Simple EdgeTAM tracker."""

    def __init__(
        self,
        config_path: str | Path = "edgetam.yaml",
        checkpoint_path: str | Path = "edgetam.pt",
        device: str = "cpu",
        **_: dict,
    ) -> None:
        """Initialise the tracker.

        Parameters
        ----------
        config_path:
            Path to the ``edgetam.yaml`` configuration file.
        checkpoint_path:
            Path to the model checkpoint (``.pt``).
        device:
            Device on which the predictor will run, e.g. ``"cpu"`` or ``"cuda"``.
        **_:
            Additional keyword arguments are accepted for API compatibility and
            ignored.
        """

        self.device = device
        self.predictor = None
        self.inference_state: Optional[dict] = None
        self.frame_loader: Optional[RealTimeFrameLoader] = None
        print('edgetam!!!!')

        cfg_path = Path(config_path)
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(ROOT / str(cfg_path.parent)), version_base=None):
            self.predictor = build_sam2_video_predictor(
                cfg_path.stem,
                str(checkpoint_path),
                add_all_frames_to_correct_as_cond=True,
                device=device,
            )
        # Create an empty frame loader so update() can be called safely
        self.frame_loader = RealTimeFrameLoader(
            image_size=self.predictor.image_size,
            device=self.predictor.device,
        )
        print('init sic')


    # ------------------------------------------------------------------
    def _init_state(self) -> dict:
        """Create an empty inference state for streaming mode."""

        assert self.frame_loader is not None
        assert self.predictor is not None
        return {
            "images": self.frame_loader,
            "num_frames": len(self.frame_loader),
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": self.frame_loader.video_height,
            "video_width": self.frame_loader.video_width,
            "device": self.predictor.device,
            "storage_device": self.predictor.device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),
                "non_cond_frame_outputs": set(),
            },
            "tracking_has_started": False,
            "frames_already_tracked": {},
        }

    # ------------------------------------------------------------------
    def initialize(
        self,
        first_frame: np.ndarray,
        ann_frame_idxs: list[int],
        ann_obj_ids: list[int],
        boxes: list[np.ndarray],
    ) -> None:
        """Prepare the predictor with initial annotations.

        Parameters
        ----------
        first_frame:
            The first frame of the video in BGR format.
        ann_frame_idxs:
            Zero-based frame indices for the first appearance of each object.
        ann_obj_ids:
            Object identifiers matching ``ann_frame_idxs``.
        boxes:
            Bounding boxes in ``xyxy`` format corresponding to the objects.
        """

        if self.predictor is None:
            return

        if self.frame_loader is None:
            self.frame_loader = RealTimeFrameLoader(
                self.predictor.image_size, self.predictor.device
            )
        else:
            # Reset any previous streaming state
            self.frame_loader.images.clear()
            self.frame_loader.video_height = None
            self.frame_loader.video_width = None

        self.frame_loader.add_frame(first_frame)
        self.inference_state = self._init_state()

        for fidx, oid, box in zip(ann_frame_idxs, ann_obj_ids, boxes):
            try:  # pragma: no cover - heavy predictor call
                _ = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=fidx,
                    obj_id=oid,
                    box=box,
                )
            except Exception as err:
                LOGGER.warning(f"Failed to add initial box for id {oid}: {err}")

    # ------------------------------------------------------------------
    def update(
        self,
        dets: Optional[np.ndarray] = None,
        img: Optional[np.ndarray] = None,
        embs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process a frame and return tracks.

        The EdgeTAM predictor works directly on the image and performs detection
        and tracking internally. The ``dets`` and ``embs`` arguments are accepted
        for interface compatibility and ignored.

        Parameters
        ----------
        dets:
            Ignored detections from external detectors.
        img:
            Image array in ``H×W×C`` BGR format.
        embs:
            Ignored embeddings.

        Returns
        -------
        np.ndarray
            Tracking results shaped ``(N, 8)`` in ``xyxy`` format with columns
            ``[x1, y1, x2, y2, id, conf, cls, -1]``. When the predictor is not
            available an empty array is returned.
        """

        # if (
        #     self.predictor is None
        #     or img is None
        #     or self.inference_state is None
        #     or self.frame_loader is None
        # ):
        #     return np.empty((0, 8), dtype=np.float32)

        self.frame_loader.add_frame(img)
        self.inference_state["num_frames"] = len(self.frame_loader)
        frame_idx = len(self.frame_loader) - 1

        try:  # pragma: no cover - predictor heavy
            _, pred_masks = self.predictor._run_single_frame_inference(
                inference_state=self.inference_state,
                output_dict=self.inference_state["output_dict"],
                frame_idx=frame_idx,
                batch_size=len(self.inference_state["obj_ids"]),
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
        except Exception as err:
            LOGGER.warning(f"EdgeTAM inference failed: {err}")
            return np.empty((0, 8), dtype=np.float32)

        tracks: list[list[float]] = []
        for oid, mask in zip(self.inference_state["obj_ids"], pred_masks):
            mask_np = (mask > 0.0).cpu().numpy().squeeze()
            ys, xs = np.where(mask_np)
            if not ys.size or not xs.size:
                continue
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            tracks.append([x0, y0, x1, y1, int(oid), 1.0, 0, -1])

        return np.asarray(tracks, dtype=np.float32)

