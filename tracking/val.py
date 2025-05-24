import sys
import argparse
import subprocess
import shutil
import json
import re
import os
from pathlib import Path
import threading
import concurrent.futures

import numpy as np
import torch
from tqdm import tqdm

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, EXAMPLES
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.torch_utils import select_device
from boxmot.utils.misc import increment_path
from boxmot.postprocessing.gsi import gsi

from ultralytics import YOLO
from ultralytics.data.loaders import LoadImagesAndVideos

from tracking.detectors import (
    get_yolo_inferer, default_imgsz, is_ultralytics_model, is_yolox_model
)
from tracking.utils import (
    convert_to_mot_format, write_mot_results,
    download_mot_eval_tools, download_mot_dataset,
    unzip_mot_dataset, eval_setup, split_dataset
)
from boxmot.appearance.reid.auto_backend import ReidAutoBackend

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git',))


def cleanup_mot17(path: Path, keep='FRCNN'):
    dirs = list(path.iterdir())
    seqs = {d.name.split('-')[:2] for d in dirs if d.is_dir()}
    for seq_parts in seqs:
        seq = '-'.join(seq_parts)
        target = path / seq
        if target.exists():
            LOGGER.debug(f"{seq} exists, skipping")
            continue
        keep_dir = path / f"{seq}-{keep}"
        if keep_dir.exists():
            keep_dir.rename(target)
            for other in path.glob(f"{seq}-*"):
                if other != target:
                    shutil.rmtree(other)
        else:
            LOGGER.warning(f"No {keep} for {seq}")


def confirm_overwrite(label: str, file: Path, ci: bool):
    if ci or not sys.stdin.isatty():
        LOGGER.debug(f"Skip overwrite for {file}")
        return False
    print(f"Overwrite {label}? (y/N): ", end='', flush=True)
    resp = []
    def reader():
        resp.append(sys.stdin.readline().strip().lower())
    t = threading.Thread(target=reader, daemon=True)
    t.start(); t.join(3)
    return bool(resp and resp[0] in ('y', 'yes'))


def generate_dets_embs(opt, model_path: Path, seq_dir: Path):
    imgsz = opt.imgsz or default_imgsz(model_path)
    yolo = YOLO(model_path if is_ultralytics_model(model_path) else 'yolov8n.pt')
    results = yolo(
        source=seq_dir / 'img1', conf=opt.conf, iou=opt.iou,
        agnostic_nms=opt.agnostic_nms, device=opt.device,
        stream=True, project=opt.project, name=opt.name,
        exist_ok=opt.exist_ok, classes=opt.classes,
        imgsz=imgsz, vid_stride=opt.vid_stride
    )
    if not is_ultralytics_model(model_path):
        inferer = get_yolo_inferer(model_path)(model=model_path,
                                               device=yolo.predictor.device,
                                               args=yolo.predictor.args)
        yolo.predictor.model = inferer
        if is_yolox_model(model_path):
            yolo.add_callback('on_predict_batch_start', inferer.update_im_paths)
            yolo.predictor.preprocess, yolo.predictor.postprocess = (
                inferer.preprocess, inferer.postprocess)

    dets_file = opt.project / 'dets_n_embs' / model_path.stem / 'dets' / f"{seq_dir.name}.txt"
    embs_folder = opt.project / 'dets_n_embs' / model_path.stem / 'embs'
    dets_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare Reid backends
    reid_backends = []
    for rm_path in opt.reid_model:
        backend = ReidAutoBackend(weights=rm_path,
                                  device=yolo.predictor.device,
                                  half=opt.half).model
        reid_backends.append((rm_path.stem, backend))
        (embs_folder / rm_path.stem / seq_dir.name).parent.mkdir(parents=True, exist_ok=True)

    # Write header
    with open(dets_file, 'w') as df:
        df.write(f"# {seq_dir}\n")

    # Process frames
    for frame_i, r in enumerate(tqdm(results, desc=seq_dir.name), 1):
        boxes = r.boxes.xyxy.cpu().numpy().round().astype(int)
        valid = (boxes[:,0]<boxes[:,2]) & (boxes[:,1]<boxes[:,3])
        dets = np.hstack([
            np.full((valid.sum(),1), frame_i),
            boxes[valid],
            r.boxes.conf.cpu().numpy()[valid, None],
            r.boxes.cls.cpu().numpy()[valid, None]
        ])
        np.savetxt(dets_file, dets, fmt='%f')

        # Save embeddings per Reid model
        for name, backend in reid_backends:
            feats = backend.get_features(dets[:,1:5], r.orig_img)
            embs_file = embs_folder / name / f"{seq_dir.name}.txt"
            np.savetxt(embs_file, feats, fmt='%f')


def _gen_worker(opt_dict, model_path, seq_dir, device):
    opt = argparse.Namespace(**opt_dict)
    opt.device = device
    generate_dets_embs(opt, Path(model_path), Path(seq_dir))


def run_generate(opt):
    seq_dirs = sorted([p for p in Path(opt.source).iterdir() if p.is_dir()])
    cuda_count = torch.cuda.device_count()
    gpu_ids = list(range(cuda_count)) if cuda_count > 0 else []
    devices = gpu_ids or [None]
    max_workers = cuda_count or os.cpu_count()
    opt_dict = vars(opt)

    for model in opt.yolo_model:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, seq_dir in enumerate(seq_dirs):
                dets = opt.project / 'dets_n_embs' / model.stem / 'dets' / f"{seq_dir.name}.txt"
                embs = opt.project / 'dets_n_embs' / model.stem / 'embs' / opt.reid_model[0].stem / f"{seq_dir.name}.txt"
                if dets.exists() and embs.exists() and not confirm_overwrite('Detections', dets, opt.ci):
                    continue
                gpu = devices[idx % len(devices)]
                device = str(gpu) if gpu is not None else 'cpu'
                futures.append(
                    executor.submit(_gen_worker, opt_dict, str(model), str(seq_dir), device)
                )
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    LOGGER.error(f"Error in generation: {e}")


def run_track(opt, evol_cfg=None):
    opt.device = select_device(opt.device)
    tracker = create_tracker(
        opt.tracking_method,
        TRACKER_CONFIGS / f"{opt.tracking_method}.yaml",
        opt.reid_model[0].with_suffix('.pt'),
        opt.device, False, False, evol_cfg
    )
    src = Path(opt.dets_file_path).read_text().splitlines()[0][2:]
    dets = np.loadtxt(opt.dets_file_path, skiprows=1)
    embs = np.loadtxt(opt.embs_file_path)
    data = LoadImagesAndVideos(src)
    results = []
    for fr, (_, imgs, _) in enumerate(tqdm(data, desc='track'), 1):
        mask = dets[:,0]==fr
        tracks = tracker.update(dets[mask,1:7], imgs[0], embs[mask])
        if tracks.size:
            results.append(convert_to_mot_format(tracks, fr))
    out = Path(opt.exp_folder_path) / f"{Path(src).parent.name}.txt"
    write_mot_results(out, np.vstack(results) if results else np.empty((0,0)))
    return {Path(src).parent.name: list(range(1, len(data)+1))}


def run_eval(opt):
    seqs, outDir, motDir, gtDir = eval_setup(opt, opt.val_tools_path)
    cmd = [
        sys.executable, EXAMPLES/'val_utils'/'scripts'/'run_mot_challenge.py',
        "--GT_FOLDER", str(gtDir), "--TRACKERS_FOLDER", str(opt.exp_folder_path),
        "--METRICS", *opt.objectives, "--SEQ_INFO", *(p.name for p in seqs)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr:
        LOGGER.error(proc.stderr)
    scores = re.findall(r"COMBINED.*?([-+]?\d*\.?\d+)", proc.stdout)
    return dict(zip(["HOTA","MOTA","IDF1"], map(float, scores)))


def main():
    parser = argparse.ArgumentParser()
    # Global arguments
    parser.add_argument('--yolo-model', nargs='+', type=Path, default=[WEIGHTS/'yolov8n.pt'], help='Yolo model path')
    parser.add_argument('--reid-model', nargs='+', type=Path, default=[WEIGHTS/'osnet_x0_25_msmt17.pt'], help='ReID model paths')
    parser.add_argument('--source', type=str, required=True, help='Dataset source')
    parser.add_argument('--project', type=Path, default=ROOT/'runs', help='Project directory')
    parser.add_argument('--name', default='', help='Run name')
    parser.add_argument('--exist-ok', action='store_true', help='Existing run OK')
    parser.add_argument('--conf', type=float, default=0.01, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--device', default='', help='CUDA device')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='Filter classes')
    parser.add_argument('--imgsz', nargs='+', type=int, default=None, help='Image size')
    parser.add_argument('--vid-stride', type=int, default=1, help='Video stride')
    parser.add_argument('--half', action='store_true', help='FP16 inference')
    parser.add_argument('--fps', type=int, default=None, help='Target FPS')
    parser.add_argument('--ci', action='store_true', help='CI mode')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='Tracking method')
    parser.add_argument('--dets-file-path', type=Path, help='Detections file path')
    parser.add_argument('--embs-file-path', type=Path, help='Embeddings file path')
    parser.add_argument('--exp-folder-path', type=Path, help='Experiment folder')
    parser.add_argument('--val-tools-path', type=Path, default=EXAMPLES/'val_utils', help='Val tools')
    parser.add_argument('--objectives', nargs='+', default=['HOTA','MOTA','IDF1'], help='Eval metrics')
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    parser.add_argument('--gsi', action='store_true', help='Apply GSI')
    parser.add_argument('--split-dataset', action='store_true', help='Split data')

    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('generate', help='Generate dets & embs')
    subparsers.add_parser('track', help='Generate MOT results')
    subparsers.add_parser('eval', help='Evaluate results')

    opt = parser.parse_args()
    download_mot_eval_tools(opt.val_tools_path)
    if not Path(opt.source).exists():
        zip_path = download_mot_dataset(opt.val_tools_path, Path(opt.source).parent.name)
        unzip_mot_dataset(zip_path, opt.val_tools_path, Path(opt.source).parent.name)
    if Path(opt.source).name == 'MOT17':
        cleanup_mot17(Path(opt.source))
    if opt.split_dataset:
        opt.source = split_dataset(opt.source)

    if opt.command == 'generate':
        run_generate(opt)
    elif opt.command == 'track':
        run_track(opt)
    elif opt.command == 'eval':
        run_eval(opt)
    else:
        run_generate(opt)
        run_track(opt)
        run_eval(opt)


if __name__ == '__main__':
    main()