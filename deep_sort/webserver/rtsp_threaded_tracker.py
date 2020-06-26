import warnings
from os import getenv
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))

import torch
from deep_sort import build_tracker
from detector import build_detector
import cv2
from utils.draw import compute_color_for_labels
from concurrent.futures import ThreadPoolExecutor
from redis import Redis

redis_cache = Redis('127.0.0.1')


class RealTimeTracking(object):
    """
    This class is built to get frame from rtsp link and continuously
    assign each frame to an attribute namely as frame in order to
    compensate the network packet loss. then we use flask to give it
    as service to client.
    Args:
        args: parse_args inputs
        cfg: deepsort dict and yolo-model cfg from server_cfg file

    """

    def __init__(self, cfg, args):
        # Create a VideoCapture object
        self.cfg = cfg
        self.args = args
        use_cuda = self.args.use_cuda and torch.cuda.is_available()

        if not use_cuda:
            warnings.warn(UserWarning("Running in cpu mode!"))

        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

        self.vdo = cv2.VideoCapture(self.args.input)
        self.status, self.frame = None, None
        self.total_frames = int(cv2.VideoCapture.get(self.vdo, cv2.CAP_PROP_FRAME_COUNT))
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.output_frame = None

        self.thread = ThreadPoolExecutor(max_workers=1)
        self.thread.submit(self.update)

    def update(self):
        while True:
            if self.vdo.isOpened():
                (self.status, self.frame) = self.vdo.read()

    def run(self):
        print('streaming started ...')
        while getenv('in_progress') != 'off':
            try:
                frame = self.frame.copy()
                self.detection(frame=frame)
                frame_to_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
                redis_cache.set('frame', frame_to_bytes)
            except AttributeError:
                pass
        print('streaming stopped ...')


    def detection(self, frame):
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
                self.draw_boxes(img=frame, output=outputs)

    @staticmethod
    def draw_boxes(img, output, offset=(0, 0)):
        for i, box in enumerate(output):
            x1, y1, x2, y2, identity = [int(ii) for ii in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            # box text and bar
            color = compute_color_for_labels(identity)
            label = '{}{:d}'.format("", identity)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img
