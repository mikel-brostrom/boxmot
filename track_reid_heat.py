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
from reid import REID
import collections
import copy
import numpy as np
import operator
import cv2
import multiprocessing as mp
import queue as Queue
from heatmappy import Heatmapper
from PIL import Image
import matplotlib.pyplot as plt

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

class LoadVideo:  # for inference
    def __init__(self, path, img_size=(640, 480)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        print('Length of {}: {:d} frames'.format(path, self.vn))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh

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
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


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
        drawimage=[]
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

                    coor_dict = dict()
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            coor_dict[output[4]] = [output[2], (output[1] + output[3])/2]
                    #print(len(coor_dict))
                    coor_get_list.append(coor_dict)
                else:
                    deepsort.increment_ages()
                # Print time (inference + NMS)
                print('{}, {}/{} {}Done. ({}s)'.format(string, frame_cnt, len(dataset), s, t2 - t1))

            drawimage.append(im0)
            frame_cnt += 1
        coor_get.put(coor_get_list)
        video_get.put(drawimage)
        video_get.put(track_cnt)
        return_dict.put(images_by_id)
        ids_per_frame_list.put(ids_per_frame)
        print(string + ' Tracking Done')
        break

def re_identification(return_dict1, return_dict2, ids_per_frame1_list, ids_per_frame2_list, video_get1, video_get2, coor_get1, coor_get2):
    reid = REID()
    M2 = np.load("calliberation/transformation_matrix2.npy")
    M2 = np.array(M2, np.float32)
    f2 = open('calliberation/coor2.txt', 'r')
    line2 = f2.readline()
    coor2 = line2.split(' ')
    coor2_x = coor2[0]
    coor2_y = coor2[1]
    f2.close()
    M1 = np.load("calliberation/transformation_matrix1.npy")
    M1 = np.array(M1, np.float32)
    f1 = open('calliberation/coor1.txt', 'r')
    line1 = f1.readline()
    coor1 = line1.split(' ')
    coor1_x = coor1[0]
    coor1_y = coor1[1]
    f1.close()
    count = 0
    while True:
        while (return_dict1.empty()) or (return_dict2.empty()) or (ids_per_frame1_list.empty()) or ids_per_frame2_list.empty():
                time.sleep(1)
        start_time = time.time()
        return_list = return_dict1.get()
        return_list2 = return_dict2.get()


        ids_per_frame1 = ids_per_frame1_list.get()
        ids_per_frame2 = ids_per_frame2_list.get()
        threshold = 320
        exist_ids = set()
        final_fuse_id = dict()
        ids_per_frame = []
        ids_per_frame22 = []
        images_by_id = dict()
        feats = dict()
        size = len(return_list)
        for key, value in return_list2.items():
            return_list[key + size] = return_list2[key]
        images_by_id = copy.deepcopy(return_list)
        print(len(images_by_id))

        for i in ids_per_frame2:
            d = set()
            for k in i:
                k += size
                d.add(k)
            ids_per_frame22.append(d)

        ids_per_frame = copy.deepcopy(ids_per_frame1)
        for k in ids_per_frame22:
            ids_per_frame.append(k)

        for i in images_by_id:
            feats[i] = reid._features(images_by_id[i])
        reid_dict = dict()
        for f in ids_per_frame:
            if f:
                if len(exist_ids) == 0:
                    for i in f:
                        final_fuse_id[i] = [i]
                    exist_ids = exist_ids or f
                else:
                    new_ids = f - exist_ids
                    for nid in new_ids:
                        dis = []
                        if len(images_by_id[nid]) < 10:
                            exist_ids.add(nid)
                            continue
                        unpickable = []
                        for i in f:
                            for key, item in final_fuse_id.items():
                                if i in item:
                                    unpickable += final_fuse_id[key]
                        print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                        for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                            tmp = np.mean(reid.compute_distance(feats[nid], feats[oid]))
                            print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                            dis.append([oid, tmp])
                        exist_ids.add(nid)
                        if not dis:
                            final_fuse_id[nid] = [nid]
                            continue
                        dis.sort(key=operator.itemgetter(1))
                        if dis[0][1] < threshold:
                            combined_id = dis[0][0]
                            images_by_id[combined_id] += images_by_id[nid]
                            final_fuse_id[combined_id].append(nid)
                            reid_dict[nid] = combined_id
                        else:
                            final_fuse_id[nid] = [nid]

        print('Final ids and their sub-ids:', final_fuse_id)
        print('people : ', len(final_fuse_id))
        print(reid_dict)
        drawimage = video_get1.get()  # list
        size2 = len(drawimage)
        track_cnt1 = video_get1.get()  # dict plus id 해야됨
        imag2 = video_get2.get()
        track_cnt2 = video_get2.get()
        for a in imag2:
            drawimage.append(a)
        for key, value in track_cnt2.items():
            for a in range(len(track_cnt2[key])):
                track_cnt2[key][a][0] +=size2
            track_cnt1[key + size] = track_cnt2[key]

        output = str(count) + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        shape = drawimage[0].shape[:2]
        size_output = (shape[1], shape[0])
        out = cv2.VideoWriter(output, fourcc, 7.5, size_output)

        for frame in range(len(drawimage)):
            img = drawimage[frame]
            for idx in final_fuse_id:
                  for i in final_fuse_id[idx]: #i = id
                     for f in track_cnt1[i]:
                        if frame == f[0]:
                            text_scale, text_thickness, line_thickness = get_FrameLabels(img)
                            cv2_addBox(idx, img, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
            out.write(img)
        out.release()
        example_img_path = './struct2.JPG'
        example_img = Image.open(example_img_path)

        example_points = []  # 히트맵 중심 좌표 설정

        # 히트맵 그리기
        heatmapper = Heatmapper(
            point_diameter=50,  # the size of each point to be drawn
            point_strength=0.5,  # the strength, between 0 and 1, of each point to be drawn
            opacity=0.5,  # the opacity of the heatmap layer
            colours='default',  # 'default' or 'reveal'
            # OR a matplotlib LinearSegmentedColorMap object
            # OR the path to a horizontal scale image
            grey_heatmapper='PIL'  # The object responsible for drawing the points
            # Pillow used by default, 'PySide' option available if installed
        )
        video1_coor = coor_get1.get()
        video2_coor = coor_get2.get()
        heatmaplist1 = list()
        heatmaplist2 = list()
        print('Video1')
        reid_set_list = list()
        for heatframelist in video1_coor:
            temp_list = list()
            temp_set = set()
            #print(len(heatframelist))
            if len(heatframelist) > 0:
                for key, value in heatframelist.items():
                    temp_list.append(value)
                    temp_set.add(key)
            #print(len(temp_list))
            heatmaplist1.append(temp_list)
            reid_set_list.append(temp_set)
        print('Video2')
        index = 0
        for heatframelist in video2_coor:
            temp_list = list()
            if len(heatframelist) > 0:
                for key, value in heatframelist.items():
                    key_size = key + size
                    if key_size not in reid_dict:
                        temp_list.append(value)
                        reid_set_list[index].add(key_size)
                    elif reid_dict[key_size] not in reid_set_list[index]:
                        temp_list.append(value)
                        reid_set_list[index].add(key_size)
            #print(len(temp_list))
            heatmaplist2.append(temp_list)
            index += 1
        save_path = 'test'
        for i in range(len(heatmaplist1)):
            background_image = cv2.imread("calliberation/struct2.JPG")
            pts1 = heatmaplist1.pop(0)
            pts2 = heatmaplist2.pop(0)
            if len(pts1) > 0:
                pts1_p = cv2.perspectiveTransform(
                    np.array(pts1, dtype=np.float32, ).reshape(1, -1, 2), M1,
                )
                for point in pts1_p[0]:
                    a, b = tuple(point)
                    x = (int(a) + int(coor1_x), int(b) + int(coor1_y))
                    example_points.append(x)
                    cv2.circle(background_image, x, 10, (0, 255, 0), -1)
            if len(pts2) > 0:
                pts2_p = cv2.perspectiveTransform(
                    np.array(pts2, dtype=np.float32, ).reshape(1, -1, 2), M2,
                )
                for point in pts2_p[0]:
                    a, b = tuple(point)
                    x = (int(a) + int(coor2_x), int(b) + int(coor2_y))
                    example_points.append(x)
                    cv2.circle(background_image, x, 10, (0, 255, 0), -1)
            name = str(i) + "_heat.jpg"
            if i % 20 == 0:
                cv2.imwrite(os.path.join(save_path, name), background_image)
        heatmap = heatmapper.heatmap_on_img(example_points, example_img)
        count = count + 1
        heatmap.save('./a.png')
        print("Finish")
        break

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness

def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                thickness=text_thickness)
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def get_frame(source, return_list):
  for video in source:
    loadvideo = LoadVideo(video)
    video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
    video_frame = []
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            video_capture.release()
            break
        video_frame.append(frame)
    return_list.put(video_frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                        help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', default='0', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    #reid = REID()
    video1 = ['calliberation/test_video_Trim.mp4']
    video2 = ['calliberation/test_video_Trim.mp4']

    with torch.no_grad():
        #mp.set_start_method('spawn')

        frame_get1 = Manager().Queue()
        frame_get2 = Manager().Queue()
        p4 = Process(target = get_frame, args = (video1, frame_get1))
        p5 = Process(target = get_frame, args = (video2, frame_get2))
        p4.start()
        p5.start()
        p4.join()
        p5.join()
        ids_per_frame1 = Manager().Queue()
        ids_per_frame2 = Manager().Queue()
        return_dict1 = Manager().Queue()
        return_dict2 = Manager().Queue()
        video_get1 = Manager().Queue()
        video_get2 = Manager().Queue()
        coor_get1 = Manager().Queue()
        coor_get2 = Manager().Queue()
        p5 = mp.Process(target=detect, args=(args, frame_get1, return_dict1, ids_per_frame1, 'Video1', video_get1, coor_get1))
        p6 = mp.Process(target=detect, args=(args, frame_get2, return_dict2, ids_per_frame2, 'Video2', video_get2, coor_get2))
        p7 = mp.Process(target=re_identification, args=(return_dict1, return_dict2, ids_per_frame1, ids_per_frame2,
                                                        video_get1, video_get2, coor_get1, coor_get2))
        p5.start()
        p6.start()
        p7.start()
        p5.join()
        p6.join()
        p7.join()
        

