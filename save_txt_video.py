# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path
import logging

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import get_yolo_inferer

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git',))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.
    在预测期间初始化用于对象跟踪的跟踪器
    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):
    yolo = YOLO(
        args.yolo_model
    )  # 读取模型

    # yolo.predictor.custom_args = args  #
    # yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))  # 添加回调函数

    results = yolo.track(
        persist=True,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=False,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )  # 获取追踪结果

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))  # 添加回调函数
    yolo.predictor.custom_args = args  #

    # 保存到txt
    source = Path(args.source)
    output_file = source.parent / (source.stem + save_name)
    save_txt_1(results, output_file)

    # for r in results:
    #
    #     img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)
    #
    #     if args.show is True:
    #         cv2.imshow('BoxMOT', img)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord(' ') or key == ord('q'):
    #             break


def save_txt_1(track_results, txt_file):
    global track_id, total_count, class_counts, track_id_set
    texts = []

    if track_results:
        for frame_id, result in enumerate(track_results):
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()  # 从张量转换为列表
                cls = int(box.cls.item())  # 类别
                class_name = result.names[cls] if cls < len(result.names) else "unknown"  # 获取类别名

                # if box.id is None:
                #     track_id = 0
                #     # continue
                # else:
                #     track_id = int(box.id.item())

                if box.id is None:
                    continue
                track_id = int(box.id.item())

                if track_id not in track_id_set:
                    track_id_set.add(track_id)  # 将track_id加入集合
                    total_count += 1  # 更新总数量

                    # 更新每个类别的数量
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                class_name = class_name + '_'
                line = (frame_id, class_name, track_id, int(bbox[0]), int(bbox[1]),
                        int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]), -1, -1, -1, 0)
                texts.append(("%g,%s,%g,%g,%g,%g,%g,%g,%g,%g,%g" % line))

    if texts and save_txt_opt:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # 创建目录
        with open(txt_file, "w") as f:
            f.writelines(text + "\n" for text in texts)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path,
                        default='tracking/weights/yolov8l_bestmodel_dataset3131_cls7_416_416_renamecls.pt',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default='tracking/weights/resnet50_berry_add_1.pth',
                        help='reid model path')
    # parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
    #                     help='reid model path')
    # parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'clip_market1501.pt',
    #                     help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='strongsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    parser.add_argument('--source', type=str,
                        default=r'/home/xplv/huanghanyang/Track_Datasets/test/test_v40.mp4',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=True, action='store_true',
                        help='class-agnostic NMS')

    opt = parser.parse_args()
    return opt


def save_statistics_to_txt(txt_file):
    """保存统计信息到txt文件"""
    with open(txt_file, "w") as f:
        f.write(f"总果实数量: {total_count}\n")
        for class_name, count in class_counts.items():
            f.write(f"{class_name}:{count}\n")


def print_fruit_statistics():
    global total_count, class_counts
    print(f"总果实数量: {total_count}")
    for class_name, count in class_counts.items():
        print(f"类别 '{class_name}' 的数量: {count}")


if __name__ == "__main__":
    total_count = 0  # 总果实数量
    class_counts = {
        "Unripe": 0,
        "Ripe": 0,
        "Ripe7": 0,
        "Ripe4": 0,
        "Ripe2": 0,
        "Flower": 0,
        "Disease": 0
    }
    track_id_set = set()  # 用于记录已统计的track_id

    opt = parse_opt()
    opt.save = True  # 是否保存视频（推理结果）
    save_txt_opt = False  # 是否保存txt
    opt.agnostic_nms = True
    opt.tracking_method = 'botsort'  # help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc'
    opt.reid_model = WEIGHTS / 'resnet50_berry_add_6.pt'  # reid model path
    # opt.reid_model = WEIGHTS / 'osnet_x0_25_msmt17.pt'
    # opt.reid_model = WEIGHTS / 'resnet50_market1501.pt'
    save_name = '_track_results_bot_berry_conf070_1.txt'
    # opt.source = r'/home/xplv/huanghanyang/Track_Datasets/1_艾维/20240113-103852_rack-1_left_RGB.mp4'
    opt.source = r'/home/xplv/huanghanyang/Track_Datasets/1_艾维/20240113-104949_rack-5_right_RGB.mp4'
    # opt.source = r'/home/xplv/huanghanyang/Track_Datasets/2_工厂_phone/0726_redBerry_7_QR.mp4'
    # opt.source = r'/home/xplv/huanghanyang/Track_Datasets/2_工厂_phone/0804_redBerry_6.mp4'
    # opt.source = r'/home/xplv/huanghanyang/Track_Datasets/3_工厂_相机/0725_2.mp4'
    # opt.source = r'/home/xplv/huanghanyang/Track_Datasets/4_工厂_变速/2L_v20_A15.mp4'
    # opt.source = r'/home/xplv/huanghanyang/Track_Datasets/4_工厂_v04/strawberryVideo_20222023testDS_v040_L4_1.mp4'
    # opt.source = r'/home/xplv/huanghanyang/Track_Datasets/6_工厂_v04/part2_1.mp4'
    # opt.source = r'/home/xplv/huanghanyang/Track_Datasets/train/strawberryVideo_20222023testDS_v040_L2_2.mp4'
    # opt.source = r'/home/xplv/huanghanyang/Track_Datasets/bot_test/20240113-103852_rack-1_left_RGB.mp4'
    # opt.source = r'/home/xplv/huanghanyang/Track_Datasets/bot_test/aiwei_2.mp4'
    run(opt)  # 进行跟踪
    print_fruit_statistics()
    source_path = Path(opt.source)
    source_dir = source_path.parent
    source_name = source_path.stem
    result_file = source_dir / f"{source_name}_result_bot_berry_conf070_1.txt"
    if save_txt_opt:
        save_statistics_to_txt(result_file)
