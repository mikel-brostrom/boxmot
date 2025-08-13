"""EdgeTAM tracker wrapper.

This module exposes a minimal tracker interface compatible with the rest of
BoxMOT trackers.  The underlying EdgeTAM project performs detection and
tracking in a single step.  The class below loads the predictor if the
`samtam` dependencies are available.  When the dependencies are missing the
tracker gracefully falls back to returning no tracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from boxmot.utils import logger as LOGGER

try:  # pragma: no cover - heavy optional dependency
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2_video_predictor
    _HAS_SAM2 = True
except Exception as err:  # pragma: no cover - executed when sam2 not installed
    LOGGER.warning(f"EdgeTAM is unavailable: {err}")
    _HAS_SAM2 = False


class EdgeTAM:
    """Simple EdgeTAM tracker.

    Parameters
    ----------
    config_path:
        Path to the ``edgetam.yaml`` configuration file.
    checkpoint_path:
        Path to the model checkpoint (``.pt``).
    device:
        Device on which the predictor will run, e.g. ``"cpu"`` or ``"cuda"``.
    **_: dict
        Additional keyword arguments are accepted for API compatibility and
        ignored.
    """

    def __init__(
        self,
        config_path: str | Path = "edgetam.yaml",
        checkpoint_path: str | Path = "edgetam.pt",
        device: str = "cpu",
        **_: dict,
    ) -> None:
        self.device = device
        self.predictor = None
        self.inference_state = None

        if _HAS_SAM2:
            try:
                cfg_path = Path(config_path)
                GlobalHydra.instance().clear()
                with initialize_config_dir(config_dir=str(cfg_path.parent), version_base=None):
                    self.predictor = build_sam2_video_predictor(
                        cfg_path.stem,
                        str(checkpoint_path),
                        add_all_frames_to_correct_as_cond=True,
                        device=device,
                    )
            except Exception as err:  # pragma: no cover - failing to init predictor
                LOGGER.warning(f"Failed to initialise EdgeTAM predictor: {err}")
                self.predictor = None

    def initialize(
        self,
        video_dir: str | Path,
        ann_frame_idxs: list[int],
        ann_obj_ids: list[int],
        boxes: list[np.ndarray],
    ) -> None:
        """Prepare the predictor with initial annotations.

        Parameters
        ----------
        video_dir:
            Directory containing the video frames.
        ann_frame_idxs:
            Zero-based frame indices for the first appearance of each object.
        ann_obj_ids:
            Object identifiers matching ``ann_frame_idxs``.
        boxes:
            Bounding boxes in ``xyxy`` format corresponding to the objects.
        """
        if self.predictor is None:
            return

        self.inference_state = self.predictor.init_state(video_path=str(video_dir))

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

    def update(
        self,
        dets: Optional[np.ndarray] = None,
        img: Optional[np.ndarray] = None,
        embs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process a frame and return tracks.

        The EdgeTAM predictor works directly on the image and performs
        detection and tracking internally.  The ``dets`` and ``embs`` arguments
        are accepted for interface compatibility and ignored.

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
            ``[x1, y1, x2, y2, id, conf, cls, -1]``.  When the predictor is not
            available an empty array is returned.
        """
        if self.predictor is None or img is None:
            return np.empty((0, 8), dtype=np.float32)

        if self.inference_state is None:
            # Initialise streaming state for a new video.
            self.inference_state = self.predictor.init_state()

        # Run the EdgeTAM predictor.  The predictor yields masks per object;
        # we convert them to bounding boxes.  If the predictor fails the
        # method returns an empty array.
        try:  # pragma: no cover - predictor heavy
            frame_idx, out_ids, out_logits = next(
                self.predictor.propagate_in_video(self.inference_state, [img])
            )
        except Exception as err:
            LOGGER.warning(f"EdgeTAM inference failed: {err}")
            return np.empty((0, 8), dtype=np.float32)

        tracks = []
        for oid, logit in zip(out_ids, out_logits):
            mask = (logit > 0.0).cpu().numpy().astype(bool)
            ys, xs = np.where(mask.squeeze())
            if not ys.size or not xs.size:
                continue
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            tracks.append([x0, y0, x1, y1, int(oid), 1.0, 0, -1])

        return np.asarray(tracks, dtype=np.float32)
