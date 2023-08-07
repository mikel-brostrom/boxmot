# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import cv2
import numpy as np
import torch

from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from boxmot.trackers.strongsort.sort.detection import Detection
from boxmot.trackers.strongsort.sort.tracker import Tracker
from boxmot.utils.matching import NearestNeighborDistanceMetric
from boxmot.utils.ops import tlwh2xyxy, xywh2tlwh, xyxy2xywh


class StrongSORT(object):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        max_dist=0.2,
        max_iou_dist=0.7,
        max_age=30,
        n_init=0,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
    ):
        self.model = ReIDDetectMultiBackend(
            weights=model_weights,
            device=device,
            fp16=fp16
        )
        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )

    def update(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.tracker.camera_update(curr_img=img)
        self.previous_img = img

        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        xywhs = xyxy2xywh(xyxys)
        self.height, self.width = img.shape[:2]

        # generate detections
        features = self._get_features(xyxys, img)
        bbox_tlwh = xywh2tlwh(xywhs)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confs)
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            tlwh = track.to_tlwh()
            x1, y1, x2, y2 = tlwh2xyxy(tlwh)

            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            # queue = track.q
            outputs.append(
                np.array([x1, y1, x2, y2, track_id, conf, class_id], dtype=np.float64)
            )
        outputs = np.asarray(outputs)
        return outputs

    def increment_ages(self):
        self.tracker.increment_ages()

    @torch.no_grad()
    def _get_features(self, xyxys, img):
        im_crops = []
        for box in xyxys:
            x1, y1, x2, y2 = box.astype('int')
            im = img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features

    def trajectory(self, im0, q, color):
        # Add rectangle to image (PIL-only)
        for i, p in enumerate(q):
            thickness = int(np.sqrt(float(i + 1)) * 1.5)
            if p[0] == "observationupdate":
                cv2.circle(im0, p[1], 2, color=color, thickness=thickness)
            else:
                cv2.circle(im0, p[1], 2, color=(255, 255, 255), thickness=thickness)
