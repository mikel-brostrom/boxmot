from pathlib import Path
import platform
import torch
import cv2
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from trackers.multi_tracker_zoo import create_tracker
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors
from yolov8.ultralytics.yolo.utils import LOGGER

# input
# source = '../data/test_video_mrt.mkv'
source = '0'
yolo_weights = 'yolov8s.pt' # yolov8 model with size n, s, m, l
imgsz = [640, 640] # image size
device = ''
webcam = source.isnumeric() 

# yolo config
agnostic_nms = False
# classes = [67]
classes = [0]
conf_thres = 0.5 
iou_thres = 0.5
max_det = 1000
line_thickness = 2

# deepsort config
tracking_method = 'deepocsort' 
tracking_config = f"./trackers/{tracking_method}/configs/{tracking_method}.yaml"
reid_weights = Path('osnet_x0_25_msmt17.pt')

# output config
save_video = False
output_file_name = 'test_output_1.avi'
out_writter = cv2.VideoWriter(
    output_file_name, 
    cv2.VideoWriter_fourcc('M','J','P','G'), 30, imgsz
)

# Load model
device = select_device(device)
is_seg = '-seg' in str(yolo_weights)
half = device.type != 'cpu'
model = AutoBackend(yolo_weights, device=device, dnn=False, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_imgsz(imgsz, stride=stride)

# Process Dataset
bs = 1

if webcam:
    show_vid = check_imshow(warn=True)
    dataset = LoadStreams(
        source,
        imgsz=imgsz,
        stride=stride,
        auto=pt,
        transforms=getattr(model.model, 'transforms', None),
        vid_stride=1
    )
    bs = len(dataset)
else:
    dataset = LoadImages(
        source,
        imgsz=imgsz,
        stride=stride,
        auto=pt,
        transforms=getattr(model.model, 'transforms', None),
        vid_stride=1)

vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz)) 

# Load Tracker
tracker_list = []
for i in range(bs):
    tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
    tracker_list.append(tracker, )
    # warm up tracker
    if hasattr(tracker_list[i], 'model'):
        if hasattr(tracker_list[i].model, 'warmup'):
            tracker_list[i].model.warmup()
outputs = [None] * bs

# helper function
def find_center(x, y, x2, y2):
    '''Find the center of the rectangle for detection'''
    cx = (x + x2) // 2
    cy = (y + y2) // 2
    return cx, cy

def count_object(
    center, id, left, right, l_count, r_count, mid_list
    ):
    
    ix, iy = center
    if (ix > left) and (ix < right):
        if id not in mid_list:
            mid_list.append(id)

    elif ix < left:
        if id in mid_list:
            mid_list.remove(id)
            l_count += 1

    elif ix > right:
        if id in mid_list:
            mid_list.remove(id)
            r_count += 1
    return l_count, r_count, mid_list

def check_in_gate(center, left, right):
    ix, iy = center
    if (ix > left) and (ix < right):
        return True
    return False

# inference
@torch.no_grad()
def main():
    # counting variables
    middle_list = []
    left_count, right_count = 0, 0
    font_color = (0, 255, 0)
    font_size = 1.
    font_thickness = 2
    font_type = cv2.FONT_HERSHEY_SIMPLEX
    center = (0, 0)
    include_count = 0

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs

    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch

        # step 1: process data format
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  
            im /= 255.0  
            if len(im.shape) == 3:
                im = im[None]  

        # step 2: Inference
        with dt[1]:
            preds = model(im, augment=False, visualize=False)

        # step 3: NMS
        with dt[2]:
            p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process detections after Non Max Suppression
        for i, det in enumerate(p): 
            seen += 1
            if webcam: 
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)
            curr_frames[i] = im0
            ih, iw, _ = im0.shape

            # draw lines
            middle_line_position = iw // 2
            left_line_position = middle_line_position - 100
            right_line_position = middle_line_position + 100
            # cv2.line(im0, (middle_line_position, 0), (middle_line_position, ih), (255, 0, 255), 2)
            cv2.line(im0, (left_line_position, 0), (left_line_position, ih), (0, 255, 0), 2)
            cv2.line(im0, (right_line_position, 0), (right_line_position, ih), (0, 255, 0), 2)
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # camera motion compensation
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # rescale detection boxes to image size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  

                # step 4: pass detections to deepsort tracker and get unique id
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

                include_count = 0
                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]):
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        c, id = int(cls), int(id) 
                        label = f'ID: {id} || Conf: {conf:.2f}'
                        color = colors(c, True)
                        center = find_center(*bbox)

                        left_count, right_count, middle_list = count_object(
                            center, id, left_line_position, right_line_position, 
                            left_count, right_count, middle_list)
                        middle_list = middle_list[-100:]
                        # print(len(middle_list))
                        
                        if check_in_gate(center, left_line_position, right_line_position):
                            include_count += 1
                        annotator.box_label(bbox, label, color=color)
                        cv2.circle(im0, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

            prev_frames[i] = curr_frames[i]

        im0 = annotator.result()
        if include_count > 1:
            cv2.putText(im0, f"ALARM", (ih // 2 - 50, iw // 2 - 150), font_type, 5, (0, 0, 255), 3)

        # display result
        im0 = cv2.flip(im0, 1)
        cv2.putText(im0, f"In --->: {left_count}", (20, 40), font_type, font_size, font_color, font_thickness)
        cv2.putText(im0, f"Out <---: {right_count}", (20, 80), font_type, font_size, font_color, font_thickness)

        if webcam:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                # cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) 
                cv2.namedWindow(str(p), cv2.WINDOW_FREERATIO)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()

        if save_video:
            # TODO: remove this for live video
            frame = im0
            frame = cv2.resize(frame, imgsz, interpolation=cv2.INTER_AREA)
            out_writter.write(frame)

if __name__ == "__main__":
	main()