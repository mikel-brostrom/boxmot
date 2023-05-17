# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

from pathlib import Path
import torch
import argparse
import numpy as np
import cv2

from trackers.multi_tracker_zoo import create_tracker
from ultralytics.yolo.engine.model import YOLO, TASK_MAP

from ultralytics.yolo.utils import LOGGER, SETTINGS, colorstr, ops, is_git_dir
from ultralytics.yolo.utils.checks import check_imgsz, print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.data.utils import VID_FORMATS

WEIGHTS = Path(SETTINGS['weights_dir'])
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root dir
print(ROOT)
WEIGHTS = ROOT / 'weights'


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    predictor.args.tracking_config = \
        Path('trackers') /\
        opt.tracking_method /\
        'configs' /\
        (opt.tracking_method + '.yaml')
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.args.tracking_method,
            predictor.args.tracking_config,
            predictor.args.reid_model,
            predictor.args.device,
            predictor.args.half
        )
        predictor.trackers.append(tracker)
        # if hasattr(predictor.trackers[i], 'model'):
        #     if hasattr(predictor.trackers[i].model, 'warmup'):
        #         predictor.trackers[i].model.warmup()
                
                
def write_MOT_results(txt_path, results, frame_idx, i):
    nr_dets = len(results.boxes)
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)
    dont_care = torch.full((nr_dets, 3), -1)
    i = torch.full((nr_dets, 1), i)
    mot = torch.cat([
        frame_idx,
        results.boxes.id.unsqueeze(1).to('cpu'),
        ops.xyxy2ltwh(results.boxes.xyxy).to('cpu'),
        dont_care,
        i
    ], dim=1)

    with open(str(txt_path) + '.txt', 'ab') as f:  # append binary mode
        np.savetxt(f, mot.numpy(), fmt='%d')  # save as ints instead of scientific notation


@torch.no_grad()
def run(
    yolo_model=WEIGHTS / 'yolov8n.pt',  # model.pt path(s),
    reid_model=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
    tracking_method='strongsort',
    source = '0',
    imgsz = [640, 640],
    save_dir=False,
    vid_stride = 1,
    verbose = True,
    project = None,
    exists_ok = False,
    name = None,
    save = True,
    save_txt = True,
    visualize=False,
    plotted_img = False,
    augment = False,
    conf = 0.5,
    device = '',
    show = False,
    half = True,
    classes = None
):
    if source is None:
        source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
        LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
    
    print(yolo_model)
    model = YOLO(yolo_model)
    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)
    
    predictor = model.predictor

    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/model.py
    #model.predictor.setup_model(model=model.model, verbose=False)
    
    predictor.args.reid_model = reid_model
    predictor.args.tracking_method = tracking_method
    predictor.args.conf = 0.5
    predictor.args.project = project
    predictor.args.name = name
    predictor.args.conf = conf
    predictor.args.half = half
    predictor.args.classes = classes
    predictor.args.imgsz = imgsz
    predictor.args.vid_stride = vid_stride
    predictor.args.save_txt = True
    predictor.args.save = True
    predictor.write_MOT_results = write_MOT_results
    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=False)
    
    predictor.setup_source(source if source is not None else predictor.args.source)
    
    dataset = predictor.dataset
    model = predictor.model
    imgsz = check_imgsz(imgsz, stride=model.model.stride, min_dim=2)  # check image size
    source_type = dataset.source_type
    preprocess = predictor.preprocess
    postprocess = predictor.postprocess
    run_callbacks = predictor.run_callbacks
    save_preds = predictor.save_preds
    predictor.save_dir = increment_path(Path(predictor.args.project) / name, exist_ok=exists_ok)
    
    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (predictor.save_dir / 'labels' if predictor.args.save_txt else predictor.save_dir).mkdir(parents=True, exist_ok=True)
    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz))
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
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
            
            if source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                continue
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)
            
            with predictor.profilers[3]:
                # get raw bboxes tensor
                dets = predictor.results[i].boxes.data
                
                # get predictions
                predictor.tracker_outputs[i] = predictor.trackers[i].update(dets.cpu().detach(), im0)
            
            predictor.results[i].speed = {
                'preprocess': predictor.profilers[0].dt * 1E3 / n,
                'inference': predictor.profilers[1].dt * 1E3 / n,
                'postprocess': predictor.profilers[2].dt * 1E3 / n,
                'tracking': predictor.profilers[3].dt * 1E3 / n
            
            }

            # overwrite bbox results with tracker predictions
            if predictor.tracker_outputs[i].size != 0:
                predictor.results[i].boxes = Boxes(
                    # xyxy, (track_id), conf, cls
                    boxes=torch.from_numpy(predictor.tracker_outputs[i]).to(dets.device),
                    orig_shape=im0.shape[:2],  # (height, width)
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
                    
                if predictor.tracker_outputs[i].size != 0:
                    write_MOT_results(
                        predictor.MOT_txt_path,
                        predictor.results[i],
                        frame_idx,
                        i,
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
        LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess, %.1fms tracking per image at shape '
                    f'{(1, 3, *imgsz)}' % t)
    if save or predictor.args.save_txt or predictor.args.save_crop:
        nl = len(list(predictor.save_dir.glob('labels/*.txt')))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}" if predictor.args.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

    run_callbacks('on_predict_end')
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=str, default=WEIGHTS / 'yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', help='display tracking video results')
    parser.add_argument('--save', action='store_true', help='save video tracking results')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exists-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)