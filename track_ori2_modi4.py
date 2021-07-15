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
import people_counting_v2
import collections
import copy
import numpy as np
import operator
import cv2
import multiprocessing as mp
import queue as Queue

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


def detect(opt, dataset_list, return_dict, ids_per_frame_list, string):
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
    """"
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)
    """
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    """
    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    """

    while True:
        print(string + 'start')
        print(dataset_list.qsize())
        while (dataset_list.empty()):
            print(string + 'is Empty. Not ready for tracking')
            time.sleep(1)
        dataset = dataset_list.get()
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        print('Tracking Start')
        track_cnt = dict()
        #print('time (init) : {}'.format(time.time() - time_init))
        t0 = time.time()
        frame_cnt = 1
        images_by_id = dict()
        ids_per_frame = []
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
                #if webcam:  # batch_size >= 1
                #    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                #else:
                s, im0 = '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                #save_path = str(Path(out) / Path(p).name)

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
                    outputs, images_by_id = deepsort.update(xywhs, confss, im0, images_by_id, ids_per_frame, track_cnt, frame_cnt)
                    """
                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_boxes(im0, bbox_xyxy, identities)
                        # to MOT format
                        tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)
    
                        # Write MOT compliant results to file
                        if save_txt:
                            for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                                bbox_top = tlwh_bbox[0]
                                bbox_left = tlwh_bbox[1]
                                bbox_w = tlwh_bbox[2]
                                bbox_h = tlwh_bbox[3]
                                identity = output[-1]
                                with open(txt_path, 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                                bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                      """
                else:
                    deepsort.increment_ages()

                # Print time (inference + NMS)
                print('{}, {}/{} {}Done. ({}s)'.format(string, frame_cnt, len(dataset), s, t2 - t1))
            """
                # Stream results
                if show_vid:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration
    
                # Save results (image with detections)
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
    
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
    
            if index != 0 and index % 50 == 0:
                print('people counting : {}'.format(people_counting.return_people(reid, images_by_id, ids_per_frame)))
                track_cnt = dict()
                images_by_id = dict()
                ids_per_frame = []
            """
            frame_cnt += 1
        """
        if save_txt or save_vid:
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)
        """
        #print("Result")
        #print(len(images_by_id))
        #for key, item in images_by_id.items():
        #  print('{} : {}'.format(key, len(images_by_id[key])))
        #print('Copy')
        return_dict.put(images_by_id)
        ids_per_frame_list.put(ids_per_frame)
        print(string + ' Tracking Done')
        #for j in return_dict:
        # print('{} : {}'.format(j, len(return_dict[j])))
        #return_list.append(images_by_id)
        #print('\nDone. (%.3fs)' % (time.time() - t0))
        #print(len(images_by_id))
        #return_dict = copy.deepcopy(images_by_id)
       # print(images_by_id)
        # print('people counting : {}'.format(people_counting.return_people(reid, images_by_id, ids_per_frame)))
        # print('ReID : (%.3fs)' % (time.time() - t0))

def re_identification(return_dict1, return_dict2, ids_per_frame1_list, ids_per_frame2_list):
    #print('ReID')
    reid = REID()
    print('Create ReID Model')
    while True:
        while (return_dict1.empty()) or (return_dict2.empty()) or (ids_per_frame1_list.empty()) or (ids_per_frame2_list.empty()):
            print('Not Ready : {} {}, {}, {}'.format(return_dict1.qsize(), return_dict2.qsize(), ids_per_frame1_list.qsize(), ids_per_frame2_list.qsize()))
            time.sleep(1)
        print('Not Ready : {} {}, {}, {}'.format(return_dict1.qsize(), return_dict2.qsize(), ids_per_frame1_list.qsize(), ids_per_frame2_list.qsize()))
        return_list1 = return_dict1.get()
        return_list2 = return_dict2.get()
        #print(len(return_list))
        #print(len(return_list2))
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
        # print(return_list2)
        #  merge_dict(return_list, return_list2)
        images_by_id = copy.deepcopy(return_list)
        #for i in images_by_id:
        #  print('{} : {}'.format(i, len(images_by_id[i])))
        # print(images_by_id)
        print(len(images_by_id))
        #print(return_list)


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
            #print('{} : {} frames'.format(i, len(images_by_id[i])))
            feats[i] = reid._features(images_by_id[i])  # reid._features(images_by_id[i][:min(len(images_by_id[i]),100)])

        #print(len(ids_per_frame))
        #print(len(feats))
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
                            images_by_id[combined_id] += images_by_id[nid]  # images_by_id[combined_id] += images_by_id[nid]
                            final_fuse_id[combined_id].append(nid)
                        else:
                            final_fuse_id[nid] = [nid]

        print('Final ids and their sub-ids:', final_fuse_id)
        print(len(final_fuse_id))

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
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/crowdhuman_yolov5m.pt', help='model.pt path')
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
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    #reid = REID()
    video1 = ['/content/drive/MyDrive/video/112.mp4','/content/drive/MyDrive/video/131.mp4']
    video2 = ['/content/drive/MyDrive/video/131.mp4','/content/drive/MyDrive/video/112.mp4']

    with torch.no_grad():
        #mp.set_start_method('spawn')

        videos1 = Manager().Queue()
        videos2 = Manager().Queue()
        p4 = Process(target = get_frame, args = (video1, videos1))
        p5 = Process(target = get_frame, args = (video2, videos2))
        p4.start()
        p5.start()
        p4.join()
        p5.join()
        ids_per_frame1 = Manager().Queue()
        ids_per_frame2 = Manager().Queue()
        return_dict1 = Manager().Queue()
        return_dict2 = Manager().Queue()
        #print(len(ids_per_frame1))
        #print(len(ids_per_frame2))
        p1 = Process(target=detect, args=(args, videos1, return_dict1, ids_per_frame1, 'Video1'))
        p2 = Process(target=detect, args=(args, videos2, return_dict2, ids_per_frame2, 'Video2'))
        p3 = Process(target = re_identification, args =(return_dict,return_dict2, ids_per_frame1, ids_per_frame2))
        p1.start()
        p2.start()
        p3.start()
        #p1.join()
        #p2.join()
        # print(len(ids_per_frame1))
        #print(len(return_dict))
        #for i in return_dict[0]:
        #  print('{} : {}'.format(i, len(return_dict[0][i])))
        #ti3 = time.time()
        # p3.join()
        p1.join()
        p2.join()
        p3.join()

