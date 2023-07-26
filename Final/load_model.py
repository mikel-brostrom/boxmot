import torch
import os
import cv2

from ultralytics import YOLO
from module.boxmot import OCSORT, DeepOCSORT
from pathlib import Path

class load_model():
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model_yolov8 = YOLO("./ckpt_saved_model/yolov8/yolov8s.engine") 

    #yolov5
    # model_yolov5 = torch.hub.load('./module/yolov5', 'custom', './ckpt_saved_model/yolov5/yolov5n.pt', force_reload=True, source='local',trust_repo=True)

    #yolov7 
    # model_yolov7 = torch.hub.load('./module/yolov7', 'custom', './ckpt_saved_model/yolov7/yolov7.pt', force_reload=True, source='local',trust_repo=True)

    #Tracker
    # tracker = DeepOCSORT(
    # model_weights=Path('./ckpt_saved_model/tracking/mobilenetv2_x1_4_dukemtmcreid.pt'), # which ReID model to use
    # device='cuda:0',
    # fp16=True,
    # )
    tracker = OCSORT()