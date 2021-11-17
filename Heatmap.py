from heatmappy import Heatmapper
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import queue as Queue
import os
import cv2
from threading import Thread
import GPUtil


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def store(video_get1, video_get2, size, coor_get1, coor_get2,
          M1, M2, coor1, coor2, count, num_video, final_fuse_id,
          reid_dict, background, save_vid, save_txt, heatmapcount, example_points, heat_name, date_time):
    #monitor = Monitor(5)
    #print('heatmap start')
    out = './output/video/'
    heatout ='./output/heatmap'
    cortxt = './output/cortxt/'
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder
        os.makedirs(heatout)
        os.makedirs(cortxt)

    example_img_path = background
    example_img = Image.open(example_img_path)
    #example_points = []  # 히트맵 중심 좌표 설정
    if count == 0:
        corfile = open(cortxt + date_time + '.txt', 'w')
    else:
        corfile = open(cortxt + date_time + '.txt', 'a')
    corfile.write('{0}번째 인원 수 : {1}\n'.format(count,len(final_fuse_id)))
    # 히트맵 그리기
    heatmapper = Heatmapper(
        point_diameter=15,  # the size of each point to be drawn
        point_strength=0.5,  # the strength, between 0 and 1, of each point to be drawn
        opacity=0.5,  # the opacity of the heatmap layer
        colours='default',  # 'default' or 'reveal'
        # OR a matplotlib LinearSegmentedColorMap object
        # OR the path to a horizontal scale image
        grey_heatmapper='PIL'  # The object responsible for drawing the points
        # Pillow used by default, 'PySide' option available if installed
    )
    heatmaplist1 = list()
    heatmaplist2 = list()

    if num_video == 2:
        coor1_x, coor1_y = coor1[0], coor1[1]
        coor2_x, coor2_y = coor2[0], coor2[1]
        if save_vid:
            drawimage = video_get1.get()  # list
            size2 = len(drawimage)
            track_cnt1 = video_get1.get()  # dict plus id 해야됨
            imag2 = video_get2.get()
            track_cnt2 = video_get2.get()
            for a in imag2:
                drawimage.append(a)
            for key, value in track_cnt2.items():
                for a in range(len(track_cnt2[key])):
                    track_cnt2[key][a][0] += size2
                track_cnt1[key + size] = track_cnt2[key]

            output = out + date_time + '_' + str(count) + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            shape = drawimage[0].shape[:2]
            size_output = (shape[1], shape[0])
            out2 = cv2.VideoWriter(output, fourcc, 7.5, size_output)
            #print(size_output)
            for frame in range(len(drawimage)):
                img = drawimage[frame]
                for idx in final_fuse_id:
                    for i in final_fuse_id[idx]:  # i = id
                        for f in track_cnt1[i]:
                            if frame == f[0]:
                                text_scale, text_thickness, line_thickness = get_FrameLabels(img)
                                cv2_addBox(idx, img, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                out2.write(img)

            out2.release()
        video1_coor = coor_get1.get()
        video2_coor = coor_get2.get()
        size_video1 = len(video1_coor)
        size_video2 = len(video2_coor)
        if size_video1 > size_video2:
            size_video1 = size_video2

        #print('Video1')
        reid_set_list = list()

        index = 0
        for heatframelist in video1_coor:
            temp_list = list()
            temp_set = set()
            # print(len(heatframelist))
            if len(heatframelist) > 0:
                for key, value in heatframelist.items():
                    temp_list.append(value)
                    temp_set.add(key)
            # print(len(temp_list))
            heatmaplist1.append(temp_list)
            reid_set_list.append(temp_set)
            index += 1
            if index == size_video1:
                break

        index = 0
        for heatframelist in video2_coor:
            temp_list = list()
            if len(heatframelist) > 0:
                for key, value in heatframelist.items():
                    key_size = key + size
                    if key_size not in reid_dict:
                        temp_list.append(value)
                        reid_set_list[index].add(key_size)
                    elif reid_dict[key_size] \
                            not in reid_set_list[index]:
                        temp_list.append(value)
                        reid_set_list[index].add(key_size)

            heatmaplist2.append(temp_list)
            index += 1
            if index == size_video1:
                break
    else:
        coor1_x, coor1_y = coor1[0], coor1[1]
        video1_coor = coor_get1.get()
        size_video1 = len(video1_coor)
        for heatframelist in video1_coor:
            temp_list = list()
            if len(heatframelist) > 0:
                for key, value in heatframelist.items():
                    temp_list.append(value)
            heatmaplist1.append(temp_list)
    pointcheck = list()
    #save_path = 'output/coor/' + str(count)+'/'
    for i in range(size_video1):
        background_image = cv2.imread(background)
        pts1 = heatmaplist1.pop(0)
        if num_video == 2:
            pts2 = heatmaplist2.pop(0)
        if len(pts1) > 0:
            pts1_p = cv2.perspectiveTransform(
                np.array(pts1, dtype=np.float32, ).reshape(1, -1, 2), M1,
            )
            for point in pts1_p[0]:
                a, b = tuple(point)
                x = (int(a) + int(coor1_x), int(b) + int(coor1_y))
                x2 = [int(a) + int(coor1_x), int(b) + int(coor1_y)]
                example_points.append(x)
                cv2.circle(background_image, x, 10, (0, 255, 0), -1)
                pointcheck.append(x2)
        if num_video == 2 and len(pts2) > 0:
            pts2_p = cv2.perspectiveTransform(
                np.array(pts2, dtype=np.float32, ).reshape(1, -1, 2), M2,
            )
            for point in pts2_p[0]:
                a, b = tuple(point)
                x = (int(a) + int(coor2_x), int(b) + int(coor2_y))
                x2 = [int(a) + int(coor2_x), int(b) + int(coor2_y)]
                example_points.append(x)
                cv2.circle(background_image, x, 10, (0, 255, 0), -1)
                pointcheck.append(x2)


        name = str(i) + "_heat.jpg"
        #if i % 20 == 0:
            #cv2.imwrite(os.path.join(save_path, name), background_image)
    if save_txt:
        for a in range(len(pointcheck)):
            corfile.write('coordinate : {0}\n'.format(pointcheck[a]))
    if heatmapcount == 0:
        heatmap = heatmapper.heatmap_on_img(example_points, example_img)
        saveheat = './output/heatmap/' + date_time + '_' + str(heat_name) + '.png'
        heatmap.save(saveheat)
    corfile.close()
    print("Finish")
    #monitor.stop()

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