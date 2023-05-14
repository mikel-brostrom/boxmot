# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

from pathlib import Path
import torch

from trackers.multi_tracker_zoo import create_tracker
from ultralytics.yolo.engine.model import YOLO, TASK_MAP
from ultralytics.yolo.engine.predictor import BasePredictor, STREAM_WARNING

from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.engine.results import Boxes


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    for i in range(predictor.dataset.bs):
        tracker = create_tracker('deepocsort', 'trackers/deepocsort/configs/deepocsort.yaml', Path('lmbn_n_duke.pt'), predictor.device, False)
        predictor.trackers.append(tracker)
        if hasattr(predictor.trackers[i], 'model'):
            if hasattr(predictor.trackers[i].model, 'warmup'):
                predictor.trackers[i].model.warmup()


if __name__ == '__main__':
    
    source = '0'
    imgsz = [640, 640]
    save_dir=False
    vid_stride = 1
    verbose = True
    save = True
    save_txt = True
    show = True
    visualize=False
    plotted_img = False
    augment = False
    
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
    predictor.args.save_txt = True
    
    dataset = predictor.dataset
    model = predictor.model
    imgsz = check_imgsz(imgsz, stride=model.model.stride, min_dim=2)  # check image size
    source_type = dataset.source_type
    preprocess = predictor.preprocess
    postprocess = predictor.postprocess
    run_callbacks = predictor.run_callbacks
    write_results = predictor.write_results
    save_preds = predictor.save_preds
    
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
    for batch in dataset:
        run_callbacks('on_predict_batch_start')
        predictor.batch = batch
        path, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem,
                                    mkdir=True) if visualize and (not source_type.tensor) else False

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
    if isinstance(vid_writer[-1], cv2.VideoWriter):
        vid_writer[-1].release()  # release final video writer

    # Print results
    if verbose and seen:
        t = tuple(x.t / seen * 1E3 for x in profilers)  # speeds per image
        LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                    f'{(1, 3, *imgsz)}' % t)
    if save or args.save_txt or args.save_crop:
        nl = len(list(save_dir.glob('labels/*.txt')))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {save_dir / 'labels'}" if args.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    run_callbacks('on_predict_end')