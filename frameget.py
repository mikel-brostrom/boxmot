from itertools import chain
from google.cloud import bigquery, storage
import cv2
import os
import queue as Queue
import time

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


def get_frame(i, frame):
    project_id = 'atsm-202107'
    bucket_id = 'sanhak_2021'
    dataset_id = 'sanhak_2021'
    table_id = 'video_sec-10_frame-4'

    storage_client = storage.Client()
    db_client = bigquery.Client()
    bucket = storage_client.bucket(bucket_id)
    select_query = (
        "SELECT camID, date_time, path FROM `{}.{}.{}` WHERE camID = {} ORDER BY date_time LIMIT 1".format(project_id,
                                                                                                        dataset_id,
                                                                                                        table_id, i))
    query_job = db_client.query(select_query)
    results = query_job.result()
    for row in results:
        path = row.path
        dt = row.date_time

    delete_query = (
        "DELETE FROM `{}.{}.{}` WHERE date_time = '{}' AND camID = {}".format(project_id, dataset_id, table_id, dt, i))

    query_job = db_client.query(delete_query)
    results = query_job.result()
    save = []
    cam = cv2.VideoCapture(path)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    start_time = time.time()
    if cam.isOpened():
        while True:
            ret, img = cam.read()
            if ret:
                cv2.waitKey(33)  # what is this??
                save.append(img)
            else:
                break
        frame.put(save)
        print(len(save))
    else:
        print('cannot open the vid #' + str(i))
        exit()
    # while True:
    #     ret, realframe = cam.read()
    #     if (time.time() - start_time) >= 3:
    #         cam.release()
    #         break
    #     frame.append(realframe)
    print("vid {} get_frame finished".format(str(i)))

def get_frame_video(source, return_list, skip_frame, second, fps, resol, end_limit):
    if resol == '1280':
        width, height = 1280, 720
    elif resol == '640':
        width, height = 640, 480

    limit = (second * fps) / skip_frame
    for video in source:
        loadvideo = LoadVideo(video)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        #video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        release = False
        while True:
            video_frame = []
            index = 0
            limit_check = 0
            while limit_check < limit:
                ret, frame = video_capture.read()
                if ret != True:
                    video_capture.release()
                    release = True
                    break
                #print(index)
                if index % skip_frame == 0:
                    limit_check += 1
                    #video_frame.append(frame)
                    video_frame.append(cv2.resize(frame, (width, height)))
                index += 1
            #print(video_frame[0].shape[:2])
            return_list.put(video_frame)
            if release or return_list.qsize() == end_limit:
                break