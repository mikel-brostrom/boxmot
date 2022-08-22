#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import os.path as osp
import sys
import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.evaler import Evaler
from yolov6.utils.events import LOGGER
from yolov6.utils.general import increment_name


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Evalating', add_help=add_help)
    parser.add_argument('--data', type=str, default='./data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='./weights/yolov6s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='val, or speed')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', default=False, action='store_true', help='whether to use fp16 infer')
    parser.add_argument('--save_dir', type=str, default='runs/val/', help='evaluation save dir')
    parser.add_argument('--name', type=str, default='exp', help='save evaluation results to save_dir/name')
    args = parser.parse_args()
    LOGGER.info(args)
    return args


@torch.no_grad()
def run(data,
        weights=None,
        batch_size=32,
        img_size=640,
        conf_thres=0.001,
        iou_thres=0.65,
        task='val',
        device='',
        half=False,
        model=None,
        dataloader=None,
        save_dir='',
        name = ''
        ):
    """ Run the evaluation process

    This function is the main process of evaluataion, supporting image file and dir containing images.
    It has tasks of 'val', 'train' and 'speed'. Task 'train' processes the evaluation during training phase.
    Task 'val' processes the evaluation purely and return the mAP of model.pt. Task 'speed' precesses the
    evaluation of inference speed of model.pt.

    """

     # task
    Evaler.check_task(task)
    if task == 'train':
        save_dir = save_dir
    else:
        save_dir = str(increment_name(osp.join(save_dir, name)))
        os.makedirs(save_dir, exist_ok=True)

    # reload thres/device/half/data according task
    conf_thres, iou_thres = Evaler.reload_thres(conf_thres, iou_thres, task)
    device = Evaler.reload_device(device, model, task)
    half = device.type != 'cpu' and half
    data = Evaler.reload_dataset(data) if isinstance(data, str) else data

    # init
    val = Evaler(data, batch_size, img_size, conf_thres, \
                iou_thres, device, half, save_dir)
    model = val.init_model(model, weights, task)
    dataloader = val.init_data(dataloader, task)

    # eval
    model.eval()
    pred_result, vis_outputs, vis_paths = val.predict_model(model, dataloader, task)
    eval_result = val.eval_model(pred_result, model, dataloader, task)
    return eval_result, vis_outputs, vis_paths


def main(args):
    run(**vars(args))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
