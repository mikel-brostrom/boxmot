import numpy as np
import torch
import sys
import cv2
import gdown
from os.path import exists as file_exists, join
import torchvision.transforms as transforms

from sort.nn_matching import NearestNeighborDistanceMetric
from sort.detection import Detection
from sort.tracker import Tracker

from reid_multibackend import ReIDDetectMultiBackend

from yolov5.utils.general import xyxy2xywh


class StrongSORT(object):
    def __init__(self, 
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70,
                 max_unmatched_preds=7,
                 n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9
                ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init, max_unmatched_preds=max_unmatched_preds)

    def update(self, ori_img, **kwargs):

        dets = kwargs['det']
        
        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        
        classes = clss.numpy()
        xywhs = xyxy2xywh(xyxys.numpy())
        
        kwargs['det'] = xywhs
        
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]
        
        # generate detections
        if 'det' in kwargs and 'masks' in kwargs:
            features = self._get_refined_features(ori_img, **kwargs)
        else:
            features = self._get_features(ori_img, **kwargs)
            
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confs)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            queue = track.q
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf, queue]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_refined_features(self, ori_img, det, masks):
        
        bbox_xywhs = det
        masks = masks.numpy()

        crops = []
        for box, mask in zip(bbox_xywhs, masks):
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            m = mask[y1:y2, x1:x2]
            crop = ori_img[y1:y2, x1:x2]
            # grey out everything that is not the detected object's mask
            masked_crop = np.where(m[...,None] == 0, (114, 114, 114), crop).astype(np.uint8)
            crops.append(masked_crop)
        if crops:
            features = self.model(crops)
        else:
            features = np.array([])
        return features
    
    def _get_features(self, ori_img, det):

        crops = []
        for box in det:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            crop = ori_img[y1:y2, x1:x2]
            # grey out everything that is not the detected object's mask
            crops.append(crop)
        if crops:
            features = self.model(crops)
        else:
            features = np.array([])
        return features
    
    def trajectory(self, im0, q, color):
        # Add rectangle to image (PIL-only)
        for i, p in enumerate(q):
            thickness = int(np.sqrt(float (i + 1)) * 1.5)
            if p[0] == 'observationupdate': 
                cv2.circle(im0, p[1], 2, color=color, thickness=thickness)
            else:
                cv2.circle(im0, p[1], 2, color=(255,255,255), thickness=thickness)
