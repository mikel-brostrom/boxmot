from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from boxmot.trackers.base import BaseTracker
from boxmot.trackers.common.appearance import (
    confidence_aware_alpha,
    resolve_batch_embeddings,
)
from boxmot.trackers.common.association.boost import (
    MhDist_similarity,
    associate,
    iou_batch,
    shape_similarity,
    soft_biou_batch,
)
from boxmot.trackers.common.geometry.obb import xywha_to_xyxy
from boxmot.trackers.common.motion import MotionModelKind, create_motion_model
from boxmot.trackers.common.motion.cmc import create_cmc
from boxmot.trackers.common.track_models.boosttrack import KalmanBoxTracker


class BoostTrack(BaseTracker):
    """Initialize the BoostTrack tracker.

    Args:
        reid_model: Pre-built ReID backend model (e.g. ``ReID(...).model``).
        use_cmc (bool): Whether to enable camera-motion compensation.
        min_box_area (int): Minimum detection area.
        aspect_ratio_thresh (float): Maximum accepted aspect ratio.
        cmc_method (str): Camera-motion compensation method.
        lambda_iou (float): Weight applied to IoU association.
        lambda_mhd (float): Weight applied to Mahalanobis association.
        lambda_shape (float): Weight applied to shape similarity.
        use_dlo_boost (bool): Whether to enable DLO boosting.
        use_duo_boost (bool): Whether to enable DUO boosting.
        dlo_boost_coef (float): Coefficient used by DLO boosting.
        s_sim_corr (bool): Whether to enable shape-similarity correction.
        use_rich_s (bool): Whether to enable rich shape features.
        use_sb (bool): Whether to enable soft-BIoU.
        use_vt (bool): Whether to enable visual tracking cues.
        with_reid (bool): Whether to enable ReID features.
        reid_model: Pre-built ReID backend model (e.g. ``ReID(...).model``).
        **kwargs: Base tracker settings forwarded to :class:`BaseTracker`.

    Attributes:
        frame_count (int): Number of processed frames.
        active_tracks (list): Currently active tracks.
        trackers (list[KalmanBoxTracker]): Internal Kalman trackers.
        cmc: Camera-motion compensation method when enabled.
        reid_model: ReID model used for appearance extraction when enabled.
    """

    supports_obb = True

    def __init__(
        self,
        reid_model: Any | None = None,
        # BoostTrack-specific parameters
        use_cmc: bool = True,
        min_box_area: int = 10,
        aspect_ratio_thresh: float = 1.6,
        cmc_method: str = "ecc",
        lambda_iou: float = 0.5,
        lambda_mhd: float = 0.25,
        lambda_shape: float = 0.25,
        use_dlo_boost: bool = True,
        use_duo_boost: bool = True,
        dlo_boost_coef: float = 0.65,
        s_sim_corr: bool = False,
        use_rich_s: bool = False,
        use_sb: bool = False,
        use_vt: bool = False,
        with_reid: bool = False,
        adaptive_kf: bool = False,
        **kwargs: Any,  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        super().__init__(**init_args, _tracker_name="BoostTrack", **kwargs)

        self.active_tracks = []
        self.frame_count = 0
        self.trackers: List[KalmanBoxTracker] = []

        # Parameters for BoostTrack (these can be tuned as needed)
        self.use_cmc = use_cmc  # use camera motion compensation
        self.min_box_area = min_box_area  # minimum box area for detections
        self.aspect_ratio_thresh = aspect_ratio_thresh  # aspect ratio threshold for detections
        self.cmc_method = cmc_method

        self.lambda_iou = lambda_iou
        self.lambda_mhd = lambda_mhd
        self.lambda_shape = lambda_shape
        self.use_dlo_boost = use_dlo_boost
        self.use_duo_boost = use_duo_boost
        self.dlo_boost_coef = dlo_boost_coef
        self.s_sim_corr = s_sim_corr

        self.use_rich_s = use_rich_s
        self.use_sb = use_sb
        self.use_vt = use_vt

        self.with_reid = bool(with_reid)
        self.reid_model = reid_model
        self.adaptive_kf = bool(adaptive_kf)

        self.cmc = create_cmc(cmc_method, enabled=self.use_cmc)

    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: Optional[np.ndarray] = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        """
        Update the tracker with detections and an image.

        Args:
          dets (np.ndarray): Detection boxes in the format [[x1,y1,x2,y2,score], ...]
          img (np.ndarray): The current image frame.
          embs (Optional[np.ndarray]): Optional precomputed embeddings.

        Returns:
          np.ndarray: Tracked objects in the format
                      [x1, y1, x2, y2, id, confidence, cls, det_ind]
                      (with cls and det_ind set to -1 if unused)
        """
        self.check_inputs(dets=dets, embs=embs, img=img)
        batch = self.make_detection_batch(dets, embs=embs, masks=masks)
        indexed_dets = batch.as_indexed_detections(dtype=dets.dtype)

        self.frame_count += 1

        if self.cmc is not None:
            self.apply_cmc(img, indexed_dets, self.trackers)

        trks = []
        confs = []

        for trk in self.trackers:
            pos = trk.predict()[0]
            conf = trk.get_confidence()
            confs.append(conf)
            assoc_pos = xywha_to_xyxy(pos.reshape(1, 5))[0] if self.is_obb else pos[:4]
            trks.append(np.concatenate([assoc_pos, [conf]]))
        trks_np = np.vstack(trks) if len(trks) > 0 else np.empty((0, 5))

        assoc_dets = self.aabb_detections_for_association(indexed_dets).copy()
        if self.use_dlo_boost:
            assoc_dets = self.dlo_confidence_boost(assoc_dets)
        if self.use_duo_boost and not self.is_obb:
            assoc_dets = self.duo_confidence_boost(assoc_dets)

        keep = assoc_dets[:, 4] >= self.det_thresh
        assoc_dets = assoc_dets[keep]
        batch = batch.select(keep).with_confs(assoc_dets[:, 4])
        dets = batch.as_indexed_detections(dtype=dets.dtype)
        scores = batch.confs

        dets_embs = resolve_batch_embeddings(
            batch,
            img,
            model=self.reid_model,
            enabled=self.with_reid,
            boxes=assoc_dets[:, :4],
            placeholder_value=1.0,
        )

        if self.with_reid and len(self.trackers) > 0:
            tracker_embs = np.array([trk.get_emb() for trk in self.trackers])
            if dets_embs.shape[0] == 0:
                emb_cost = np.empty((0, tracker_embs.shape[0]))
            else:
                emb_cost = (
                    dets_embs.reshape(dets_embs.shape[0], -1) @ tracker_embs.reshape((tracker_embs.shape[0], -1)).T
                )
        else:
            emb_cost = None

        mh_dist_matrix = self.get_mh_dist_matrix(dets)

        matched, unmatched_dets, unmatched_trks, _ = associate(
            assoc_dets,
            trks_np,
            self.iou_threshold,
            mahalanobis_distance=mh_dist_matrix,
            track_confidence=np.array(confs).reshape(-1, 1),
            detection_confidence=scores,
            emb_cost=emb_cost,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape,
            s_sim_corr=self.s_sim_corr,
        )

        dets_alpha = confidence_aware_alpha(batch.confs, self.det_thresh)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        for i in unmatched_dets:
            if batch.confs[i] >= self.det_thresh:
                self.trackers.append(
                    KalmanBoxTracker(
                        dets[i, :],
                        max_obs=self.max_obs,
                        emb=dets_embs[i],
                        is_obb=self.is_obb,
                        adaptive_kf=self.adaptive_kf,
                        id_allocator=self.id_allocator,
                    )
                )

        outputs = []
        self.active_tracks = []
        for trk in self.trackers:
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                outputs.append(self.format_output_row(d, trk.id, trk.conf, trk.cls, trk.det_ind))
                self.active_tracks.append(trk)

        self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]

        outputs = self.format_output_rows(outputs, dtype=np.float32)
        return self.filter_outputs(outputs)

    def filter_outputs(self, outputs: np.ndarray) -> np.ndarray:
        return self.filter_outputs_by_geometry(
            outputs,
            min_box_area=self.min_box_area,
            max_aspect_ratio=self.aspect_ratio_thresh,
        )

    def reset(self) -> None:
        self._reset_common_state()

    def get_iou_matrix(self, detections: np.ndarray, buffered: bool = False) -> np.ndarray:
        trackers = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            assoc_pos = xywha_to_xyxy(pos.reshape(1, 5))[0] if self.is_obb else pos[:4]
            trk[:] = [
                assoc_pos[0],
                assoc_pos[1],
                assoc_pos[2],
                assoc_pos[3],
                self.trackers[t].get_confidence(),
            ]

        return iou_batch(detections, trackers) if not buffered else soft_biou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int | None = None) -> np.ndarray:
        if len(self.trackers) == 0:
            return np.zeros((0, 0))
        n_dims = self.detection_layout.box_cols if n_dims is None else n_dims
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(self.trackers), n_dims), dtype=float)
        sigma_inv = np.zeros((len(self.trackers), n_dims), dtype=float)
        motion_model = create_motion_model(MotionModelKind.XYHR, is_obb=self.is_obb)

        for i in range(len(detections)):
            if self.is_obb:
                z[i, :n_dims] = motion_model.to_measurement(detections[i, :5], column=False)[:n_dims]
            else:
                z[i, :n_dims] = motion_model.to_measurement(detections[i, :4], column=False)[:n_dims]
        for i, trk in enumerate(self.trackers):
            x[i] = trk.kf.x[:n_dims]
            sigma_inv[i] = np.reciprocal(np.diag(trk.kf.covariance[:n_dims, :n_dims]))
        return (
            (z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))
        ).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return detections

        n_dims = 4
        limit = 13.2767
        mh_dist = self.get_mh_dist_matrix(detections, n_dims)

        # If there are no existing trackers, bail out immediately
        if mh_dist.size == 0:
            return detections

        min_dists = mh_dist.min(1)
        mask = (min_dists > limit) & (detections[:, 4] < self.det_thresh)
        boost_inds = np.where(mask)[0]
        iou_limit = 0.3
        if len(boost_inds) == 0:
            return detections

        bdiou = iou_batch(detections[boost_inds], detections[boost_inds]) - np.eye(len(boost_inds))
        bdiou_max = bdiou.max(axis=1)
        remaining = boost_inds[bdiou_max <= iou_limit]
        args = np.where(bdiou_max > iou_limit)[0]
        for i in range(len(args)):
            bi = args[i]
            tmp = np.where(bdiou[bi] > iou_limit)[0]
            args_tmp = np.append(np.intersect1d(boost_inds[args], boost_inds[tmp]), boost_inds[bi])
            conf_max = np.max(detections[args_tmp, 4])
            if detections[boost_inds[bi], 4] == conf_max:
                remaining = np.concatenate([remaining, [boost_inds[bi]]])

        mask_boost = np.zeros_like(detections[:, 4], dtype=bool)
        mask_boost[remaining] = True
        detections[:, 4] = np.where(mask_boost, self.det_thresh + 1e-4, detections[:, 4])
        return detections

    def dlo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return detections

        sbiou_matrix = self.get_iou_matrix(detections, True)
        if sbiou_matrix.size == 0:
            return detections

        trackers = np.zeros((len(self.trackers), 6))
        for t, trk in enumerate(self.trackers):
            pos = trk.get_state()[0]
            trackers[t] = [pos[0], pos[1], pos[2], pos[3], 0, trk.time_since_update - 1]

        if self.use_rich_s:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections), 1)
            shape_sim = shape_similarity(detections, trackers, self.s_sim_corr)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, False)

        if not self.use_sb and not self.use_vt:
            max_s = S.max(1)
            detections[:, 4] = np.maximum(detections[:, 4], max_s * self.dlo_boost_coef)
            return detections

        if self.use_sb:
            max_s = S.max(1)
            alpha = 0.65
            detections[:, 4] = np.maximum(detections[:, 4], alpha * detections[:, 4] + (1 - alpha) * max_s**1.5)
        if self.use_vt:
            threshold_s = 0.95
            threshold_e = 0.8
            tmp = (
                S
                > np.maximum(threshold_s - np.array([trk.time_since_update - 1 for trk in self.trackers]), threshold_e)
            ).max(1)
            scores = detections[:, 4].copy()
            scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)
            detections[:, 4] = scores
        return detections
