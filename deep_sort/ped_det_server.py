"""
This module gets video in input and outputs the
json file with coordination of bboxes in the video.

"""
from os.path import basename, splitext, join, isfile, isdir, dirname
from os import makedirs

from tqdm import tqdm
import cv2
import argparse
import torch

from detector import build_detector
from deep_sort import build_tracker
from utils.tools import tik_tok, is_video
from utils.draw import compute_color_for_labels
from utils.parser import get_config
from utils.json_logger import BboxToJsonLogger
import warnings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default="./demo/ped.avi")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--write-fps", type=int, default=20)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    args = parser.parse_args()

    assert isfile(args.VIDEO_PATH), "Error: Video not found"
    assert is_video(args.VIDEO_PATH), "Error: Not Supported format"
    if args.frame_interval < 1: args.frame_interval = 1

    return args


class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode!")

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

        # Configure output video and json
        self.logger = BboxToJsonLogger()
        filename, extension = splitext(basename(self.args.VIDEO_PATH))
        self.output_file = join(self.args.save_path, f'{filename}.avi')
        self.json_output = join(self.args.save_path, f'{filename}.json')
        if not isdir(dirname(self.json_output)):
            makedirs(dirname(self.json_output))

    def __enter__(self):
        self.vdo.open(self.args.VIDEO_PATH)
        self.total_frames = int(cv2.VideoCapture.get(self.vdo, cv2.CAP_PROP_FRAME_COUNT))
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_details = {'frame_width': self.im_width,
                         'frame_height': self.im_height,
                         'frame_rate': self.args.write_fps,
                         'video_name': self.args.VIDEO_PATH}
        codec = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(self.output_file, codec, self.args.write_fps,
                                      (self.im_width, self.im_height))
        self.logger.add_video_details(**video_details)

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        idx_frame = 0
        pbar = tqdm(total=self.total_frames + 1)
        while self.vdo.grab():
            if idx_frame % args.frame_interval == 0:
                _, ori_im = self.vdo.retrieve()
                timestamp = self.vdo.get(cv2.CAP_PROP_POS_MSEC)
                frame_id = int(self.vdo.get(cv2.CAP_PROP_POS_FRAMES))
                self.logger.add_frame(frame_id=frame_id, timestamp=timestamp)
                self.detection(frame=ori_im, frame_id=frame_id)
                self.save_frame(ori_im)
                idx_frame += 1
            pbar.update()
        self.logger.json_output(self.json_output)

    @tik_tok
    def detection(self, frame, frame_id):
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)
        if bbox_xywh is not None:
            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            bbox_xywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                frame = self.draw_boxes(img=frame, frame_id=frame_id, output=outputs)

    def draw_boxes(self, img, frame_id, output, offset=(0, 0)):
        for i, box in enumerate(output):
            x1, y1, x2, y2, identity = [int(ii) for ii in box]
            self.logger.add_bbox_to_frame(frame_id=frame_id,
                                          bbox_id=identity,
                                          top=y1,
                                          left=x1,
                                          width=x2 - x1,
                                          height=y2 - y1)
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            # box text and bar
            self.logger.add_label_to_bbox(frame_id=frame_id, bbox_id=identity, category='pedestrian', confidence=0.9)
            color = compute_color_for_labels(identity)
            label = '{}{:d}'.format("", identity)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img

    def save_frame(self, frame) -> None:
        if frame is not None: self.writer.write(frame)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()

