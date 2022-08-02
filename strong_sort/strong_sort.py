import numpy as np
import torch
import sys
import cv2
import gdown
from os.path import exists as file_exists, join
import torchvision.transforms as transforms

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
from .deep.reid_model_factory import show_downloadeable_models, get_model_url, get_model_name

from torchreid.utils import FeatureExtractor
from torchreid.utils.tools import download_url

import numpy as np
import tensorflow as tf

__all__ = ['StrongSORT']


class StrongSORT(object):
    def __init__(self, 
                 model_weights,
                 device, max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70, n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9
                ):
        model_name = get_model_name(model_weights)
        model_url = get_model_url(model_weights)

        if not file_exists(model_weights) and model_url is not None:
            gdown.download(model_url, str(model_weights), quiet=False)
        elif file_exists(model_weights):
            pass
        elif model_url is None:
            print('No URL associated to the chosen DeepSort weights. Choose between:')
            show_downloadeable_models()
            exit()

        # self.extractor = FeatureExtractor(
        #     # get rid of dataset information DeepSort model name
        #     model_name=model_name,
        #     model_path=model_weights,
        #     device=str(device)
        # )
        
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.size = (256, 128)
        
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path="/home/mikel.brostrom/Yolov5_StrongSORT_OSNet/resnet50_msmt17_tflite_model/resnet50_msmt17.tflite")
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        print(self.input_details)
        self.output_details = self.interpreter.get_output_details()
        
        # Test model on random input data.
        input_data = np.array(np.random.random_sample((1,256,128,3)), dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        print(output_data.shape)

        
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

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
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
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

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            def _resize(im, size):
                return cv2.resize(im.astype(np.float32)/255., size)

            im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
            # NCHW --> NHWC
            im_batch = torch.transpose(im_batch, 1, 3)
            print('im_batch shape', im_batch.shape)
            #images = torch.stack(images, dim=0)
            #im_batch = im_batch.to(self.device)
            print(len(im_crops))
            print(type(im_crops[0]))
            print(im_crops[0].shape)
            
        
            self.interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            features = []
            for i in range(0, im_batch.shape[0]):
                input = np.array(im_batch[i].unsqueeze(0), dtype=np.float32)
                print(f'input {i}:',  input.shape)
                self.interpreter.set_tensor(self.input_details[0]['index'], input)
                feature = torch.tensor(self.interpreter.get_tensor(self.output_details[0]['index']))
                # NHWC -->  NCHW 
                print('feature_output', feature.shape)
                features.append(feature.squeeze())
            #features = torch.tensor(features)
        else:
            features = np.array([])
        return features
