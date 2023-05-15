# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

from pathlib import Path
import torch
import argparse
import numpy as np
import cv2

from trackers.multi_tracker_zoo import create_tracker
from ultralytics.yolo.engine.model import YOLO, TASK_MAP
from ultralytics.yolo.engine.predictor import BasePredictor, STREAM_WARNING

from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow, print_args, check_requirements
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.data.utils import VID_FORMATS, ROOT


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    for i in range(predictor.dataset.bs):
        tracker = create_tracker('deepocsort', 'trackers/deepocsort/configs/deepocsort.yaml', Path('lmbn_n_duke.pt'), predictor.device, False)
        predictor.trackers.append(tracker)
        if hasattr(predictor.trackers[i], 'model'):
            if hasattr(predictor.trackers[i].model, 'warmup'):
                predictor.trackers[i].model.warmup()
                
                
def write_MOT_results(txt_path, results, frame_idx, i):
    nr_dets = len(results.boxes)
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)
    dont_care = torch.full((nr_dets, 3), -1)
    i = torch.full((nr_dets, 1), i)
    mot = torch.cat([
        frame_idx,
        results.boxes.xywh,
        results.boxes.id.unsqueeze(1),
        dont_care,
        i
    ], dim=1)
    print('mot', mot.shape)

    with open(str(txt_path) + '.txt', 'ab') as f:  # append binary mode
        np.savetxt(f, mot.numpy())


@torch.no_grad()
def run(
    source = '0',
    imgsz = [640, 640],
    save_dir=False,
    vid_stride = 1,
    verbose = True,
    project = None,
    name = None,
    save = True,
    save_txt = True,
    show = False,
    visualize=False,
    plotted_img = False,
    augment = False,
):
    print(project, name)
    if source is None:
        source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
        LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
    
    from ultralytics.yolo.engine.model import YOLO
    model = YOLO('yolov8s.pt')
    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)
    
    predictor = model.predictor
    #predictor.device = 'cpu'
    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=False)

    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/model.py
    #model.predictor.setup_model(model=model.model, verbose=False)
    
    predictor.setup_source(source if source is not None else predictor.args.source)
    predictor.args.conf = 0.5
    predictor.args.project = project
    predictor.args.name = name
    predictor.args.save_txt = True
    predictor.args.save = True
    predictor.write_MOT_results = write_MOT_results
    
    dataset = predictor.dataset
    model = predictor.model
    imgsz = check_imgsz(imgsz, stride=model.model.stride, min_dim=2)  # check image size
    source_type = dataset.source_type
    preprocess = predictor.preprocess
    postprocess = predictor.postprocess
    run_callbacks = predictor.run_callbacks
    save_preds = predictor.save_preds
    predictor.save_dir = increment_path(Path(predictor.args.project) / name, exist_ok=True)
    print(predictor.save_dir)
    
    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (predictor.save_dir / 'labels' if predictor.args.save_txt else predictor.save_dir).mkdir(parents=True, exist_ok=True)
    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz))
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
    predictor.add_callback('on_predict_start', on_predict_start)
    
    run_callbacks('on_predict_start')
    for frame_idx, batch in enumerate(dataset):
        run_callbacks('on_predict_batch_start')
        predictor.batch = batch
        path, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, exist_ok=True, mkdir=True) if visualize and (not source_type.tensor) else False

        # Preprocess
        with predictor.profilers[0]:
            im = preprocess(im0s)

        # Inference
        with predictor.profilers[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Postprocess
        with predictor.profilers[2]:
            predictor.results = postprocess(preds, im, im0s)
        run_callbacks('on_predict_postprocess_end')
        
        # Visualize, save, write results
        n = len(im0s)
        for i in range(n):
            predictor.results[i].speed = {
                'preprocess': predictor.profilers[0].dt * 1E3 / n,
                'inference': predictor.profilers[1].dt * 1E3 / n,
                'postprocess': predictor.profilers[2].dt * 1E3 / n}
            if source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                continue
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)
            
            # get bboxes matrix
            dets = predictor.results[i].boxes.data
            
            # get predictions
            predictor.tracker_outputs[i] = predictor.trackers[i].update(dets.cpu().detach(), im0)
            
            # overwrite bbox results with tracker predictions
            predictor.results[i].boxes = Boxes(
                torch.from_numpy(predictor.tracker_outputs[i]),
                im0.shape,
            )
            
            # write inference results to a file or directory   
            if verbose or save or save_txt or show:
                s += predictor.write_results(i, predictor.results, (p, im, im0))
                predictor.txt_path = Path(predictor.txt_path)
                
                # write MOT specific results
                if source.endswith(VID_FORMATS):
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.stem
                else:
                    # append folder name containing current img
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.name
                    print(predictor.MOT_txt_path)
                write_MOT_results(
                    predictor.MOT_txt_path,
                    predictor.results[i],
                    frame_idx,
                    i
                )

            # display an image in a window using OpenCV imshow()
            if show and plotted_img is not None:
                predictor.show(p)

            # save video predictions
            if save and plotted_img is not None:
                predictor.save_preds(vid_cap, i, str(predictor.save_dir / p.name))

        run_callbacks('on_predict_batch_end')

        # print time (inference-only)
        if verbose:
            LOGGER.info(f'{s}{predictor.profilers[1].dt * 1E3:.1f}ms')

    # Release assets
    if isinstance(predictor.vid_writer[-1], cv2.VideoWriter):
        predictor.vid_writer[-1].release()  # release final video writer

    # Print results
    if verbose and predictor.seen:
        t = tuple(x.t / predictor.seen * 1E3 for x in predictor.profilers)  # speeds per image
        LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                    f'{(1, 3, *imgsz)}' % t)
    if save or args.save_txt or args.save_crop:
        nl = len(list(predictor.save_dir.glob('labels/*.txt')))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}" if predictor.args.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

    run_callbacks('on_predict_end')
    

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8s-seg.pt', help='model.pt path(s)')
    # parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'lmbn_n_cuhk03_d.pt')
    # parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    # parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    # parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    # parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--visualize', action='store_true', help='visualize features')
    # parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    # parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    # opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    #check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)