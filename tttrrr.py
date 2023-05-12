# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

from pathlib import Path

from trackers.multi_tracker_zoo import create_tracker
from ultralytics.yolo.engine.model import YOLO, TASK_MAP
from ultralytics.yolo.engine.predictor import BasePredictor

from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.yolo.data import load_inference_source


STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""

def on_predict_postprocess_end(trackers,results,frame):
    # '''
    bs = 1
    im0s = frame
    im0s = im0s if isinstance(im0s, list) else [im0s]
    for i in range(bs):
        det = results[i].boxes.cpu().numpy()
        if len(det) == 0:
            continue
        tracks = trackers[i].update(det, im0s[i])
        if len(tracks) == 0:
            continue
        results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))
        if results[i].masks is not None:
            idx = tracks[:, -1].tolist()
            results[i].masks = results[i].masks[idx]
    # '''
    return results

if __name__ == '__main__':
    
    source = '0'
    imgsz = [640, 640]
    save_dir=False
    vid_stride = 1
    verbose = False
    save = False
    save_txt = False
    show = True
    visualize=False
    plotted_img = False
    augment = False
    
    from ultralytics.yolo.engine.model import YOLO
    model = YOLO('yolov8s.pt')
    model.overrides = model.overrides.copy()
    model.overrides['conf'] = 0.25

    print(model.task)
    print(model.predictor)
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/model.py
    #model.predictor.setup_model(model=model.model, verbose=False)
    predictor = model.predictor
    predictor.setup_model('yolov8s.pt')
    predictor.setup_source(source if source is not None else source)
    dataset = predictor.dataset
    model = predictor.model
    imgsz = check_imgsz(imgsz, stride=model.model.stride, min_dim=2)  # check image size
    #dataset = load_inference_source(source=source, imgsz=imgsz, vid_stride=vid_stride)
    source_type = dataset.source_type
    preprocess = predictor.preprocess
    postprocess = predictor.postprocess
    run_callbacks = predictor.run_callbacks
    write_results = predictor.write_results
    predictor.seen = 0
    
    if not (dataset.mode == 'stream' or  # streams
            len(dataset) > 1000 or  # images
            any(getattr(dataset, 'video_flag', [False]))):  # videos
        LOGGER.warning(STREAM_WARNING)
    vid_path, vid_writer = [None] * dataset.bs, [None] * dataset.bs
    
    
    seen, windows, batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
    run_callbacks('on_predict_start')
    for batch in dataset:
        run_callbacks('on_predict_batch_start')
        path, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem,
                                    mkdir=True) if visualize and (not source_type.tensor) else False

        # Preprocess
        with profilers[0]:
            im = preprocess(im0s)

        # Inference
        with profilers[1]:
            preds = model(im, augment=augment, visualize=visualize)
            print('preds', len(preds))

        # Postprocess
        with profilers[2]:
            results = postprocess(preds, im, im0s)
        run_callbacks('on_predict_postprocess_end')
        print(len(results))
        
        # Visualize, save, write results
        n = len(im0s)
        for i in range(n):
            results[i].speed = {
                'preprocess': profilers[0].dt * 1E3 / n,
                'inference': profilers[1].dt * 1E3 / n,
                'postprocess': profilers[2].dt * 1E3 / n}
            if source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                continue
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)

            if verbose or save or save_txt or show:
                s += write_results(i, results, (p, im, im0))

            if show and plotted_img is not None:
                show(p)

            if save and plotted_img is not None:
                save_preds(vid_cap, i, str(save_dir / p.name))
        run_callbacks('on_predict_batch_end')
        [r for r in results]

        # Print time (inference-only)
        if verbose:
            LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

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