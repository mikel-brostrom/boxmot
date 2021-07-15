import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import (
  LoadImages, 
  LoadStreams, 
  letterbox,
)
from yolov5.utils.general import (
  check_img_size, 
  non_max_suppression, 
  scale_coords,
  check_imshow,
  
)
from yolov5.utils.plots import (
  plot_one_box,
)
from yolov5.utils.torch_utils import (
  select_device, 
  time_synchronized,
)
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn




import pandas as pd 
import numpy as np 

import torch 
import cv2


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)



import typing


import dataclasses
@dataclasses.dataclass 
class Options():
  yolo_weights: str = 'yolov5/weights/yolov5s.pt'
  deep_sort_weights: str = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
  img_size: int = 640
  conf_thres: float = 0.4
  iou_thres: float = 0.5
  device: str = ''
  show_vid: bool = False 
  save_vid: bool = False
  save_txt: bool = False
  classes: str = None
  agnostic_nms: bool = False 
  augment: bool = False 
  evaluate: bool = False
  config_deepsort: str = 'deep_sort_pytorch/configs/deep_sort.yaml'
  size_fixed: bool = True




from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import (
  select_device, 
  time_synchronized,
)
from yolov5.utils.general import (
  check_img_size, 
  non_max_suppression, 
  scale_coords,
  check_imshow,
)


import torch.backends.cudnn as cudnn
import numpy as np


class Yolo():

  def __init__(
    self,
    opt: Options,
  ) -> typing.NoReturn:
    device = select_device(opt.device)
    model = attempt_load(opt.yolo_weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    half = device.type != 'cpu'
    if half:
      model.half()  # to FP16

    if opt.size_fixed:
        cudnn.benchmark = True 
  
    if half:
      model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    self.__device = device
    self.__half = half 
    self.__opt = opt
    self.__model = model
    self.__imgsz = imgsz
    self.__names = names
  

  def get_names(
    self,
  ):
    return self.__names 


  def __to_yolo_input(
    self,
    img0: np.array,
  ) -> torch.Tensor:
    img = letterbox(
      img0,
      new_shape=self.__imgsz,
      stride=32,
    )[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(
      img,
    ).to(self.__device)
    img = (
      img.half() if self.__half
      else img.float()
    ) 
    img /= 255.0 
    if img.ndimension() == 3:
      img = img.unsqueeze(0)
    return img 


  def __predict(
    self, 
    img: torch.Tensor,
  ) -> torch.Tensor:
    cfg = self.__opt
    pred = self.__model(
      img,
      augment=cfg.augment,
    )[0]
    pred = non_max_suppression(
      pred,
      cfg.conf_thres,
      cfg.iou_thres,
      classes=cfg.classes,
      agnostic=(
        cfg.agnostic_nms,
      ),
    ) # filter

    return pred 


  def detect(
    self,
    img0: np.ndarray,
  ) -> typing.Optional[
    torch.Tensor
  ]:
    with torch.no_grad():
      img = self.__to_yolo_input(img0)
      pred = self.__predict(img)
      dets = pred[0]
      if dets is None:
        return
      # convert to original scale
      dets[:, :4] = scale_coords(
        img.shape[2:],
        dets[:, :4],
        img0.shape[:2],
      ).round() 
      return dets 



def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h



class Deepsort():
  def __init__(
    self,
    opt: Options,
  ) -> typing.NoReturn:
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(
      opt.deep_sort_weights, 
      repo='mikel-brostrom/Yolov5_DeepSort_Pytorch',
    )
    model = DeepSort(
      cfg.DEEPSORT.REID_CKPT,
      max_dist=cfg.DEEPSORT.MAX_DIST, 
      min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
      nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
      max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
      max_age=cfg.DEEPSORT.MAX_AGE, 
      n_init=cfg.DEEPSORT.N_INIT, 
      nn_budget=cfg.DEEPSORT.NN_BUDGET,
      use_cuda=True,
    )
    self.__model = model
  

  def __call__(
    self,
    dets: typing.Optional[
      torch.Tensor
    ],
    img0: np.array,
  ) -> typing.Optional[
    typing.List[torch.Tensor]
  ]:
    if dets is None:
      return
    with torch.no_grad():
      bbox_xywh = []
      confs = []
      for (
        *xyxy, conf, cls_,
      ) in dets:
        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
        bbox_xywh.append(xywh_obj)
        confs.append([conf.item()])

      xywhs = torch.Tensor(bbox_xywh)
      confs = torch.Tensor(confs)

      outputs = self.__model.update(
        xywhs,
        confs,
        img0,
      )
      return outputs


import cv2 

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
  """
  Simple function that adds fixed color depending on the class
  """
  color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
  return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
  for i, box in enumerate(bbox):
    x1, y1, x2, y2 = [int(i) for i in box]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]
    # box text and bar
    id = int(identities[i]) if identities is not None else 0
    color = compute_color_for_labels(id)
    label = '{}{:d}'.format("", id)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(
      img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
    cv2.putText(
      img, label, 
      (x1, y1 + t_size[1] + 4), 
      cv2.FONT_HERSHEY_PLAIN, 
      2, [255, 255, 255], 2)
  return img


from tqdm import (
  trange,
)

def main():
  opt = Options()
  yolo = Yolo(opt)
  deepsort = Deepsort(opt)
  img = cv2.imread('test.jpeg')
  cap = cv2.VideoCapture(
    'test.mp4',
  )
  n = cap.get(
    cv2.CAP_PROP_FRAME_COUNT,
  )
  fps = cap.get(
    cv2.CAP_PROP_FPS,
  )
  w = cap.get(
    cv2.CAP_PROP_FRAME_WIDTH,
  )
  h = cap.get(
    cv2.CAP_PROP_FRAME_HEIGHT,
  )
  writer = cv2.VideoWriter(
    'result.mp4',
    cv2.VideoWriter_fourcc(
      *'MP4V',
    ),
    fps,
    frameSize=(
      int(w),
      int(h),
    ),
  )
  n = int(n)
  for i in trange(n):
    success, img = cap.read()

    if not success:
      continue 
    dets = yolo.detect(img)
    out = deepsort(dets, img)
    if len(out) == 0:
      # cv2.imshow('./tmp.png', img)
      writer.write(img)
      continue
    bbox_xyxy = out[:, :4]
    identities = out[:, -1]
    draw_boxes(img, bbox_xyxy, identities)
    # cv2.imshow('./tmp.png', img)
    writer.write(img)

  cap.release()
  writer.release()
  ... 


if __name__ == '__main__':
  main()