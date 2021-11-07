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
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import copy
import numpy as np


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
                            coor_dict[output[4]] = [(output[0] + output[2])/2, (output[1] + output[3]) / 2]
                    # print(len(coor_dict))


                else:
                    deepsort.increment_ages()
                coor_get_list.append(coor_dict)
                # Print time (inference + NMS)
                #print('{}, {}/{} {}Done. ({}s)'.format(string, frame_cnt, len(dataset), s, t2 - t1))

            drawimage.append(im0)
            frame_cnt += 1
        coor_get.put(coor_get_list)
        if opt.save_vid:
            video_get.put(drawimage)
            video_get.put(track_cnt)
        return_dict.put(images_by_id)
        ids_per_frame_list.put(ids_per_frame)
        print(string + ' Tracking Done')
        count += 1
        if count == opt.limit:
            print(string + 'Done')
            break

