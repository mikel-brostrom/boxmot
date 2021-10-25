import sys

sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from multiprocessing import Process, Manager
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import warnings
import subprocess
from reid import REID
import collections
import copy
import numpy as np
import operator
import cv2
import multiprocessing as mp
import queue as Queue
import re_id as re
import frameget as fg


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


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


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def detect(opt, dataset_list, return_dict, ids_per_frame_list, string, video_get, coor_get):
    out, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.img_size, opt.evaluate
    time_init = time.time()
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    # Initialize
    device = select_device(opt.device)
    """
    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    """
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    count = 0
    while True:
        print(string + 'start')
        while (dataset_list.empty()):
            time.sleep(1)
        start_time = time.time()
        dataset = dataset_list.get()
        coor_get_list = list()
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        track_cnt = dict()
        # print('time (init) : {}'.format(time.time() - time_init))
        t0 = time.time()
        frame_cnt = 1
        images_by_id = dict()
        ids_per_frame = []
        drawimage = []
        for im0s in dataset:
            img = letterbox(im0s, 640, stride=32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x 416 x 416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # if webcam:  # batch_size >= 1
                #    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                # else:
                s, im0 = '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                # save_path = str(Path(out) / Path(p).name)
                coor_dict = dict()
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    xywh_bboxs = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        # to deep sort format
                        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)

                    # pass detections to deepsort
                    outputs, images_by_id = deepsort.update(xywhs, confss, im0, images_by_id, ids_per_frame, track_cnt,
                                                            frame_cnt)

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            coor_dict[output[4]] = [output[2], (output[1] + output[3]) / 2]
                    # print(len(coor_dict))


                else:
                    deepsort.increment_ages()
                coor_get_list.append(coor_dict)
                # Print time (inference + NMS)
                #print('{}, {}/{} {}Done. ({}s)'.format(string, frame_cnt, len(dataset), s, t2 - t1))

            drawimage.append(im0)
            frame_cnt += 1
        coor_get.put(coor_get_list)
        if args.save_vid:
            video_get.put(drawimage)
            video_get.put(track_cnt)
        return_dict.put(images_by_id)
        ids_per_frame_list.put(ids_per_frame)
        print(string + ' Tracking Done')
        count += 1
        if count == opt.limit:
            break


def pstart(frame_get, frame_get2, count):
    if count != 0:
        cnt = 0

        while (cnt < count):
            p1 = Process(target=fg.get_frame, args=(0, frame_get), daemon=True)
            p2 = Process(target=fg.get_frame, args=(1, frame_get2), daemon=True)
            p1.start()
            p2.start()
            p1.join()
            p2.join()
            cnt += 1
    else:
        while True:
            p1 = Process(target=fg.get_frame, args=(0, frame_get), daemon=True)
            p2 = Process(target=fg.get_frame, args=(1, frame_get2), daemon=True)
            p1.start()
            p2.start()
            p1.join()
            p2.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5l.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                        help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--model', type=str, default='osnet_x1_0', help='select reid model')
    parser.add_argument('--modelpth', type=str, default='model_data/models/model.pth.tar-80', help='select reid model.pth')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', default='0', type=int,
                        help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--realtime", type=int, default=1)
    parser.add_argument("--matrix", type=str, default='None')
    parser.add_argument("--num_video", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--background", type=str, default='calliberation/rest_plan.jpg')
    parser.add_argument("--heatmap", type=str, default=1)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--second", type=int, default=15)
    parser.add_argument("--threshold", type=int, default=320)
    parser.add_argument("--video", type=str, default='None')
    parser.add_argument("--heatmapsec", type=int, default=60)

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    credential_path = "atsm-202107-50b0c3dc3869.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # reid = REID()
    video1 = ['calliberation/in1_Trim.mp4']  # 엘리베이터
    video2 = ['calliberation/in2_Trim.mp4']  # 입구
    videos = [['calliberation/sample_video/ele.mp4'], ['calliberation/sample_video/en.mp4'], ['calliberation/sample_video/in.mp4']]  # 엘리베이터, 입구, 내부\
    str_video = ['ele', 'en', 'in']

    try:
        from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy

        IS_CYTHON_AVAI = True
    except ImportError:
        IS_CYTHON_AVAI = False
        stdout = subprocess.run(['python ./torchreid/metrics/rank_cylib/setup.py build_ext --inplace'], shell=True)
        warnings.warn(
            'Cython does not work, will run cython'
        )

    with torch.no_grad():
        # mp.set_start_method('spawn')
        if args.realtime == 0 and args.video != 'None':
            x = args.video.split(',')
            video1 = videos[int(x[0])]
            if args.num_video == 2:
                video2 = videos[int(x[1])]
            args.matrix = 'coor_' + str_video[int(x[0])]
            if args.num_video == 2:
                args.matrix += ' coor_' + str_video[int(x[1])]

        frame_get1 = Manager().Queue()
        frame_get2 = Manager().Queue()
        if args.realtime:
            p0 = Process(target=pstart, args=(frame_get1, frame_get2, args.limit))
            p0.start()
        else:
            p1 = Process(target=fg.get_frame_video, args=(video1, frame_get1, args.frame, args.second))
            p1.start()
            if args.num_video == 2:
                p2 = Process(target=fg.get_frame_video, args=(video2, frame_get2, args.frame, args.second))
                p2.start()
                p2.join()
            p1.join()
            args.limit = frame_get1.qsize()
            if args.num_video == 2:
                size2 = frame_get2.qsize()
                if args.limit > size2:
                    args.limit = size2

        ids_per_frame1 = Manager().Queue()
        ids_per_frame2 = Manager().Queue()
        return_dict1 = Manager().Queue()
        return_dict2 = Manager().Queue()
        video_get1 = Manager().Queue()
        video_get2 = Manager().Queue()
        coor_get1 = Manager().Queue()
        coor_get2 = Manager().Queue()
        p5 = mp.Process(target=detect,
                        args=(args, frame_get1, return_dict1, ids_per_frame1, 'Video1', video_get1, coor_get1),
                        daemon=True)
        if args.realtime == 1 or args.num_video == 2:
            p6 = mp.Process(target=detect,
                            args=(args, frame_get2, return_dict2, ids_per_frame2, 'Video2', video_get2, coor_get2),
                            daemon=True)
        p7 = mp.Process(target=re.re_identification,
                        args=(args, return_dict1, return_dict2, ids_per_frame1, ids_per_frame2,
                              video_get1, video_get2, coor_get1, coor_get2), daemon=True)
        p5.start()
        if args.num_video == 2:
            p6.start()
        p7.start()

        p7.join()


