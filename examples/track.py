import argparse
from functools import partial
from pathlib import Path

import torch

from boxmot import DeepOCSORT
from boxmot.utils import EXAMPLES, ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer

__tr = TestRequirements()
__tr.check_packages(('git+https://github.com/mikel-brostrom/ultralytics.git',))  # install

from ultralytics import YOLO
from ultralytics.yolo.data.utils import VID_FORMATS
from ultralytics.yolo.utils.plotting import save_one_box

from examples.utils import write_mot_results


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """
    predictor.args.tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        'deepocsort.yaml'
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = DeepOCSORT(
            model_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
            device=predictor.device,
            fp16=False,
            per_class=False
        )
        trackers.append(tracker)

    predictor.trackers = trackers
    predictor.save_dir = predictor.get_save_dir()


@torch.no_grad()
def run(args):

    yolo = YOLO(
        'yolov8n.pt',
    )
    print(yolo.__dict__.keys())

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    yolo.predictor.args.project = args.project
    yolo.predictor.args.name = args.name
    yolo.predictor.args.exist_ok = args.exist_ok
    yolo.predictor.args.classes = args.classes

    for frame_idx, r in enumerate(results):
        if len(r.boxes.data) != 0:

            if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                yolo.predictor.mot_txt_path = p

            if args.save_mot:
                write_mot_results(
                    yolo.predictor.mot_txt_path,
                    r,
                    frame_idx,
                    frame_idx,
                )

            if args.save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
                    save_one_box(
                        d.xyxy,
                        r.orig_img.copy(),
                        file=(
                            yolo.predictor.save_dir / 'crops' /
                            str(int(d.cls.cpu().numpy().item())) /
                            str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                        ),
                        BGR=True
                    )

    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default='yolov8n', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='/home/mikel.brostrom/Downloads/test.avi',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=EXAMPLES / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
