import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov8 ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device, strip_optimizer
from yolov8.ultralytics.yolo.utils.ops import Profile
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from yolov8.ultralytics.nn.autobackend import AutoBackend
from trackers.multi_tracker_zoo import create_tracker


def load_data(source, imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1):
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=auto,
            transforms=transforms,
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=auto,
            transforms=transforms,
            vid_stride=vid_stride
        )
        bs = 1
    return dataset, bs

@torch.no_grad()
def run(
        source='0',
        yolo_weights='weights/yolov8/yolov5m.pt',  # model.pt path(s),
        reid_weights='weights/reid/osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes        
        show_foot_trajectories= False, # show human foot trajectories
        show_bounding_box= False,      # show bounding box result
        show_segmentation= False,     # show segmentation result for seg model
        show_heatmap= False,        # show human heatmap
        show_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        save_dir='runs/track/exp', # output save directory
        save_log=False,
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False, 
        **kwargs
):
    if save_log:
        LOGGER.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(save_dir, 'logging', 'info.log'))
        fh.setLevel(logging.INFO)
        LOGGER.addHandler(fh)

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    if 'v8' in str(yolo_weights):
        from yolov8.ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes, process_mask, process_mask_native
        model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)

    elif 'v5' in str(yolo_weights):
        from yolov5.models.common import DetectMultiBackend
        from yolov5.utils.general import non_max_suppression, scale_boxes
        from yolov5.utils.segment.general import process_mask, process_mask_native
        model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, fp16=half)

    stride, names, pt = model.stride, model.names, model.pt
    transforms = getattr(model.model, 'transforms', None)

    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    dataset, bs = load_data(source, imgsz=imgsz, stride=stride, auto=pt, 
                            transforms=transforms, vid_stride=vid_stride)

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs

    # Create as many tracker instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, Path(reid_weights), device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    
    outputs = [None] * bs
    
    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        
        if frame_idx==0:
            fme_h, fme_w = np.shape(im0s)[-3:-1]
            if show_foot_trajectories: 
                tj = Trajectory(fme_h, fme_w)
            if show_heatmap: 
                hm = Heatmap()
                emp_h = np.zeros((fme_h, fme_w), np.uint8)
            
        # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # preds = model(im, augment=augment, visualize=visualize)
            preds = model(im, augment=augment)
  
        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1
            if type(path)==list:  # webcam
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                if is_seg:
                    if len(proto.shape) == 3:
                        proto = torch.unsqueeze(proto,0)
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to tracker
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    
                    if is_seg and show_segmentation:
                        # Mask plotting
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if retina_masks else im[i]
                        )
                        im0 = annotator.result()
                    
                    for j, (output) in enumerate(outputs[i]):
                        
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
        
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            
                            if show_bounding_box:
                                annotator.box_label(bbox, label, color=color)
                                im0 = annotator.result()
                            if show_foot_trajectories :
                                im0 = tj.draw_line(im0, output, mode="foot")
                            if show_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(np.array(bbox, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                            
            else:
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()
                
            # Stream results
            if show_vid or save_vid:
                if show_heatmap:
                    emp_h, im0 = hm.get_heatmap(emp_h, im0)
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)

def initialize_path(opt):
    # Directories
    project = Path(opt.output).parent
    name =  Path(opt.output).stem
    if not isinstance(opt.yolo_weights, list):  # single yolo model
        exp_name = Path(opt.yolo_weights).stem
    elif type(opt.yolo_weights) is list and len(opt.yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(opt.yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + opt.reid_weights.stem
    save_dir = increment_path(project / exp_name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'tracks' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'config' if opt.save_config else save_dir).mkdir(parents=True, exist_ok=True)
    (save_dir / 'logging' if opt.save_log else save_dir).mkdir(parents=True, exist_ok=True)
    if opt.save_config:
        with open(os.path.join(save_dir, 'config', 'config.yml'), 'w') as outfile:
            yaml.dump(vars(opt), outfile)

    opt.project = project
    opt.name = name
    opt.save_dir = save_dir
    

def set_default_value(opt, opt_list, key, value):
    if (key.upper() not in opt_list and key not in opt_list) or not getattr(opt, key):
        setattr(opt, key, value)

def initialize_config(opt):
    opt_list = vars(opt).keys()
    set_default_value(opt, opt_list, 'yolo_weights', str(ROOT / 'weights/yolov8/yolov5m.pt'))
    set_default_value(opt, opt_list, 'reid_weights', str(ROOT / 'weights/reid/osnet_x0_25_msmt17.pt'))
    set_default_value(opt, opt_list, 'tracking_method', 'strongsort')
    set_default_value(opt, opt_list, 'tracking_config', str(ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')))
    set_default_value(opt, opt_list, 'imgsz', [640,])
    set_default_value(opt, opt_list, 'conf_thres', 0.5)
    set_default_value(opt, opt_list, 'iou_thres', 0.5)
    set_default_value(opt, opt_list, 'max_det', 1000)
    set_default_value(opt, opt_list, 'device', 'cpu')
    set_default_value(opt, opt_list, 'show_vid', False)
    set_default_value(opt, opt_list, 'save_txt', False)
    set_default_value(opt, opt_list, 'save_config', False)
    set_default_value(opt, opt_list, 'save_log', False)
    set_default_value(opt, opt_list, 'save_conf', False)
    set_default_value(opt, opt_list, 'save_crop', False)
    set_default_value(opt, opt_list, 'show_foot_trajectories', False)
    set_default_value(opt, opt_list, 'show_bounding_box', False)
    set_default_value(opt, opt_list, 'show_segmentation', False)
    set_default_value(opt, opt_list, 'show_heatmap', False)
    set_default_value(opt, opt_list, 'show_trajectories', False)
    set_default_value(opt, opt_list, 'save_vid', False)
    set_default_value(opt, opt_list, 'classes', None)
    set_default_value(opt, opt_list, 'agnostic_nmsp', False)
    set_default_value(opt, opt_list, 'augment', False)
    set_default_value(opt, opt_list, 'update', False)
    set_default_value(opt, opt_list, 'exist_ok', False)
    set_default_value(opt, opt_list, 'line_thickness', 2)
    set_default_value(opt, opt_list, 'hide_labels', False)
    set_default_value(opt, opt_list, 'hide_conf', False)
    set_default_value(opt, opt_list, 'hide_class', False)
    set_default_value(opt, opt_list, 'half', False)
    set_default_value(opt, opt_list, 'dnn', False)
    set_default_value(opt, opt_list, 'vid_stride', 1)
    set_default_value(opt, opt_list, 'retina_masks', False)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--output', type=str, default= str(ROOT / 'runs' / 'track' / 'exp'), help='output folder for the result')  
    parser.add_argument('--track_config', type=str, default='track_configs/default.yml', help='core func config file path')
    parser.add_argument('--out_config', type=str, default='out_configs/default.yml', help='vis func config file path')
    
    opt = parser.parse_args()

    with open(opt.track_config, 'r') as outfile :
        track_config = yaml.safe_load(outfile)
    with open(opt.out_config, 'r') as outfile :
        out_config = yaml.safe_load(outfile)
    for key in track_config:
        setattr(opt, key, track_config[key])
    for key in out_config:
        setattr(opt, key, out_config[key])
    
    return opt

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    initialize_config(opt)
    initialize_path(opt)
    run(**vars(opt))


class Heatmap:
    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2()
    
    def get_heatmap(self, emp_fme, fme):
        """ 
        generate heatmap using bg subtract
        """
        # -- remove background --
        bg_rm = self.bg_sub.apply(fme)
        # -- apply threshold --
        _, th1 = cv2.threshold(bg_rm, 2, 2, cv2.THRESH_BINARY)
        # -- add to empty frame --
        emp_fme = cv2.add(emp_fme, th1)
        # -- apply colormap --
        hm = cv2.applyColorMap(emp_fme, cv2.COLORMAP_JET)
        # -- align to original frame --
        out = cv2.addWeighted(fme, 0.8, hm, 0.3, 0)
        
        return emp_fme, out

class Trajectory:
    def __init__(self, fme_h, fme_w):
        self.empt_fme = np.zeros((fme_h, fme_w, 3), np.uint8)
        self.empt_fme[:] = 255
        self.prev_out = {}
        
    def draw_line(self, fme, c_out, mode="foot", draw_box=False):
    
        id = c_out[4]
        cx1, cy1, cx2, cy2 = c_out[:4]
        if mode == "foot":
            current_p = (int((cx1+cx2)/2) , int(cy2))
        elif mode == "middle":
            current_p = (int((cx1+cx2)/2) , int((cy1+cy2)/2))
        if id not in self.prev_out:
            prev_p = current_p
        else:
            prev_p = self.prev_out[id]
        # -- draw box --
        if draw_box:
            fme = self.draw_box(fme, cx1, cy1, cx2, cy2)
        # -- draw line on empty frame --
        self.empt_fme = cv2.line(self.empt_fme, 
                                prev_p, 
                                current_p, 
                                (0,255,255), 
                                3)

        # -- save current ids to previous id --
        self.prev_out[id] = current_p
        # -- overlay mask on frame --
        print(fme.shape, self.empt_fme.shape)
        return cv2.bitwise_and(fme,self.empt_fme)
        
    @staticmethod
    def draw_box(fme, x1, y1, x2, y2):
        fme = cv2.rectangle(
                            fme,
                            [int(x1),int(y1)],
                            [int(x2),int(y2)],
                            [255,110,0],
                            2,
                            cv2.LINE_AA
        )
        return fme


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
