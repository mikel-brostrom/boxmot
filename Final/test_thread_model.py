import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import copy

from load_model import load_model

color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5

class inferThread(threading.Thread):
    def __init__(self, model_yolo, model_tracker, path_video, name):
        # threading.Thread.__init__(self)
        super().__init__()
        self.path_video = path_video
        self.name = name
        self.model_yolo = copy.deepcopy(model_yolo)
        self.model_tracker = copy.deepcopy(model_tracker)


    def run(self):
        cap = cv2.VideoCapture(self.path_video)
        fps = 0.0
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(self.path_video)
                continue
            t1 = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # bboxes = self.model_yolo(frame)
            # bboxes = self.model_yolo.predict(source=frame, save=False, save_txt=False, conf=0.5, classes=0, half=True)[0].boxes
            bboxes = self.model_yolo.predict(source=frame, save=False, save_txt=False, conf=0.5, half=True)[0].boxes
            if len(bboxes) > 0:
                dets = []
                bboxes = bboxes.cpu().numpy()
                for i, bboxe in enumerate(bboxes):
                    x1, y1, x2, y2 = bboxe.xyxy[0]
                    dets.append([int(x1), int(y1), int(x2), int(y2), bboxe.conf[0], bboxe.cls[0]])
                ts = self.model_tracker.update(np.array(dets), frame) # --> (x, y, x, y, id, conf, cls)
                if ts.shape[0] != 0:
                    xyxys = ts[:, 0:4].astype('int') # float64 to int
                    ids = ts[:, 4].astype('int') # float64 to int
                    confs = ts[:, 5]
                    clss = ts[:, 6]
                    # print bboxes with their associated id, cls and conf
                    for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                        frame = cv2.rectangle(
                            frame,
                            (xyxy[0], xyxy[1]),
                            (xyxy[2], xyxy[3]),
                            color,
                            thickness
                        )
                        cv2.putText(
                            frame,
                            f'id: {id}, conf: {conf}, c: {cls}',
                            (xyxy[0], xyxy[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontscale,
                            color,
                            thickness
                        )
            fps = (fps + (1. / (time.time() - t1))) / 2
            cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (50, 30), 0, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Time: {:.3f}'.format(time.time() - t1), (50, 60), 0, 1, (0, 255, 0), 2)
            # if ret == True:
            #     cv2.imshow('frame', frame)
            #     if cv2.waitKey(10) & 0xFF == ord('q'):
            #         break
            # else:
            #     break

            frame_count = frame_count + 1
            # if str(self.name) == "1":
            # cv2.imwrite("img/" + str(self.name) + "/img_" + str(frame_count) + "_" + str(self.name) + ".jpg", frame)
            print('Frame->{}, fps->{:.2f}, saving into output/ ----- ThreadName {}'.format(frame_count, fps, self.name))


model_yolov8 = load_model.model_yolov8
model_tracker = load_model.tracker

# try:
path_video = "/home/aiteam/workspace/datnh14/DangerZones/tracking_demo/video/1.avi"
thread1 = inferThread(model_yolov8, model_tracker, path_video, 1)
thread2 = inferThread(model_yolov8, model_tracker, path_video, 2)
thread3 = inferThread(model_yolov8, model_tracker, path_video, 3)
thread4 = inferThread(model_yolov8, model_tracker, path_video, 4)
# thread5 = inferThread(model_yolov8, path_video, 5)
thread1.start()
thread2.start()
thread3.start()
thread4.start()
# thread5.start()
# finally:
    # destroy the instance
    # yolov5_wrapper.destroy()