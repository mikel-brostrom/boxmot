from track import detect
import re_id as re
import frameget as fg
import queue as Queue
from multiprocessing import Process, Manager
import argparse
import os

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
        p5 = Process(target=detect,
                        args=(args, frame_get1, return_dict1, ids_per_frame1, 'Video1', video_get1, coor_get1),
                        daemon=True)
        if args.realtime == 1 or args.num_video == 2:
            p6 = Process(target=detect,
                            args=(args, frame_get2, return_dict2, ids_per_frame2, 'Video2', video_get2, coor_get2),
                            daemon=True)
        p7 = Process(target=re.re_identification,
                        args=(args, return_dict1, return_dict2, ids_per_frame1, ids_per_frame2,
                              video_get1, video_get2, coor_get1, coor_get2), daemon=True)
        p5.start()
        if args.num_video == 2:
            p6.start()
        p7.start()

        p7.join()

