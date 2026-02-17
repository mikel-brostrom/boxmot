# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import argparse
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import yaml
import cv2
import os
import torch
import sys
import concurrent.futures
from contextlib import nullcontext

from boxmot.trackers.tracker_zoo import create_tracker
from boxmot.utils import NUM_THREADS, ROOT, WEIGHTS, TRACKER_CONFIGS, DATASET_CONFIGS, logger as LOGGER, TRACKEVAL
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.torch_utils import select_device
from boxmot.utils.plots import MetricsPlotter
from boxmot.utils.misc import increment_path, prompt_overwrite
from boxmot.utils.timing import TimingStats, wrap_tracker_reid
from typing import Optional, List, Dict, Generator, Union

from boxmot.utils.dataloaders.dataset import MOTDataset
from boxmot.postprocessing.gsi import gsi

from boxmot.engine.inference import DetectorReIDPipeline, extract_detections, filter_detections
from boxmot.detectors import default_imgsz
from boxmot.utils.mot_utils import convert_to_mot_format, write_mot_results
from boxmot.utils.download import download_eval_data, download_trackeval

checker = RequirementsChecker()
checker.check_packages(('ultralytics', ))  # install


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def load_dataset_cfg(name: str) -> dict:
    """Load the dict from boxmot/configs/datasets/{name}.yaml."""
    path = DATASET_CONFIGS / f"{name}.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def eval_init(args,
              trackeval_dest: Path = TRACKEVAL,
              branch: str = "master",
              overwrite: bool = False
    ) -> None:
    """
    Common initialization: download TrackEval and (if needed) the MOT-challenge
    data for ablation runs, then canonicalize args.source.
    Modifies args in place.
    """
    # 1) download the TrackEval code
    download_trackeval(dest=trackeval_dest, branch=branch, overwrite=overwrite)

    # 2) if doing MOT17/20-ablation, pull down the dataset and rewire args.source/split
    if (DATASET_CONFIGS / f"{args.source}.yaml").exists():
        cfg = load_dataset_cfg(str(args.source))
        
        # Determine dataset destination (under trackeval/data so benchmarks don't mix with TrackEval code)
        bench_name = Path(cfg["benchmark"]["source"]).name
        dataset_url = cfg["download"]["dataset_url"]
        if dataset_url and dataset_url.startswith("hf://"):
            dataset_dest = TRACKEVAL / "data" / bench_name
        elif dataset_url:
            dataset_dest = TRACKEVAL / "data" / f"{bench_name}.zip"
        else:
            # For custom datasets without URL, use the path from config if available, or default to assets
            dataset_dest = Path(cfg["download"].get("dataset_dest", f"assets/{bench_name}"))

        download_eval_data(
            runs_url=cfg["download"]["runs_url"],
            dataset_url=cfg["download"]["dataset_url"],
            dataset_dest=dataset_dest,
            overwrite=overwrite
        )
        args.benchmark = bench_name
        args.split = cfg["benchmark"]["split"]
        if cfg["download"]["dataset_url"]:
            args.source = TRACKEVAL / "data" / f"{args.benchmark}/{args.split}"
        elif "source" in cfg["benchmark"]:
            args.source = Path(cfg["benchmark"]["source"]) / args.split
        else:
            args.source = dataset_dest / args.split

    # 3) finally, make source an absolute Path everywhere
    args.source = Path(args.source).resolve()
    args.project = Path(args.project).resolve()
    args.project.mkdir(parents=True, exist_ok=True)


def parse_mot_results(results: str) -> dict:
    """
    Extracts COMBINED HOTA, MOTA, IDF1, AssA, AssRe, IDSW, and IDs from MOTChallenge evaluation output.
    Returns a dictionary keyed by class name.
    """
    metric_specs = {
        'HOTA':   ('HOTA:',      {'HOTA': 0, 'AssA': 2, 'AssRe': 5}),
        'MOTA':   ('CLEAR:',     {'MOTA': 0, 'IDSW': 12}),
        'IDF1':   ('Identity:',  {'IDF1': 0}),
        'IDs':    ('Count:',     {'IDs': 2}),
    }

    int_fields = {'IDSW', 'IDs'}
    parsed_results = {}
    
    lines = results.splitlines()
    current_class = None
    current_metric_type = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for header lines
        is_header = False
        for metric_name, (prefix, _) in metric_specs.items():
            if line.startswith(prefix):
                is_header = True
                current_metric_type = metric_name
                
                # Format: "HOTA: tracker-classHOTA ..."
                content = line[len(prefix):].strip()
                first_word = content.split()[0]
                if first_word.endswith(metric_name):
                    tracker_class = first_word[:-len(metric_name)]
                    if '-' in tracker_class:
                        current_class = tracker_class.split('-')[-1]
                    else:
                        current_class = 'default'
                    
                    if current_class not in parsed_results:
                        parsed_results[current_class] = {'per_sequence': {}}
                break
        
        if is_header:
            continue
        
        # Check for data rows (COMBINED or sequence names)
        if current_class and current_metric_type:
            fields = line.split()
            if len(fields) > 1:
                row_name = fields[0]  # Either 'COMBINED' or sequence name like 'MOT17-02-FRCNN'
                values = fields[1:]
                _, field_map = metric_specs[current_metric_type]
                
                if row_name == 'COMBINED':
                    # Store COMBINED metrics at class level (backward compatible)
                    for key, idx in field_map.items():
                        if idx < len(values):
                            val = values[idx]
                            parsed_results[current_class][key] = max(0, int(val) if key in int_fields else float(val))
                else:
                    # Store per-sequence metrics
                    if row_name not in parsed_results[current_class]['per_sequence']:
                        parsed_results[current_class]['per_sequence'][row_name] = {}
                    for key, idx in field_map.items():
                        if idx < len(values):
                            val = values[idx]
                            parsed_results[current_class]['per_sequence'][row_name][key] = max(0, int(val) if key in int_fields else float(val))

    return parsed_results


# ---------------------------
# Batched det+emb generation
# ---------------------------
def _sequence_img_dir(seq_dir: Path) -> Path:
    img1 = seq_dir / "img1"
    return img1 if img1.exists() else seq_dir


def _list_sequence_frames(img_dir: Path) -> list[Path]:
    return sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))


def _sequence_name_from_img_dir(img_dir: Path) -> str:
    return img_dir.parent.name if img_dir.name == "img1" else img_dir.name


def _read_image_cv2(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return im


def _collect_seq_info(source: Path) -> tuple[list[Path], dict[str, int]]:
    seq_paths = []
    seq_info: dict[str, int] = {}
    for seq_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        img_dir = _sequence_img_dir(seq_dir)
        frame_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if not frame_files:
            continue
        seq_paths.append(img_dir)
        seq_info[seq_dir.name] = len(frame_files)
    return seq_paths, seq_info


def _autotune_batch_size(yolo, device: str, imgsz, requested: int) -> int:
    dev_lower = str(device).lower()
    use_accel = dev_lower.startswith(("cuda", "0", "1", "2", "3", "4", "5", "6", "7", "mps", "metal"))
    if not use_accel:
        return max(1, requested)

    def _empty_cache():
        if dev_lower.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif dev_lower.startswith(("mps", "metal")) and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    if isinstance(imgsz, (list, tuple)):
        h, w = int(imgsz[0]), int(imgsz[1])
    else:
        h = w = int(imgsz)

    dummy = np.zeros((h, w, 3), dtype=np.uint8)

    bs = max(1, int(requested))
    while bs >= 1:
        try:
            yolo.predict(source=[dummy] * bs, device=device, verbose=False, imgsz=imgsz)
            if bs < requested:
                LOGGER.warning(f"Auto-tuned batch size from {requested} -> {bs} to fit device memory.")
            return bs
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            _empty_cache()
            next_bs = max(1, bs // 2)
            LOGGER.warning(f"Batch size {bs} OOM; retrying with {next_bs}.")
            if next_bs == bs:
                break
            bs = next_bs

    raise RuntimeError("Unable to run even batch size 1; reduce image size or move to CPU.")


def _clear_device_cache(device: str) -> None:
    dev_lower = str(device).lower()
    if dev_lower.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev_lower.startswith(("mps", "metal")) and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _count_data_lines(path: Path, skip_header: bool = False) -> int:
    """Count non-header lines in a txt file, tolerating missing files."""
    try:
        with open(path, "r") as fh:
            if skip_header:
                return sum(1 for line in fh if not line.startswith("#"))
            return sum(1 for _ in fh)
    except FileNotFoundError:
        return 0


def _max_frame_id(path: Path) -> int:
    """Return the maximum frame id (first column) in a dets txt, skipping headers."""
    max_f = 0
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    f_val = int(float(parts[0]))
                except Exception:
                    continue
                if f_val > max_f:
                    max_f = f_val
    except FileNotFoundError:
        return 0
    return max_f


@torch.inference_mode()
def generate_dets_embs_batched(args: argparse.Namespace, y: Path, source_root: Path, timing_stats: Optional[TimingStats] = None) -> None:
    """
    Generate detections and embeddings in batches for evaluation.
    
    Args:
        args: CLI arguments.
        y: Path to YOLO model weights.
        source_root: Root path containing sequence folders.
        timing_stats: Optional TimingStats for timing instrumentation.
    """
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    batch_size = int(getattr(args, "batch_size", 16))
    read_threads_val = getattr(args, "read_threads", None)
    if read_threads_val is None:
        read_threads_val = min(8, (os.cpu_count() or 8))
    read_threads = int(read_threads_val)
    auto_batch = bool(getattr(args, "auto_batch", True))
    resume = bool(getattr(args, "resume", True))

    if args.imgsz is None:
        args.imgsz = default_imgsz(y)

    # Use unified DetectorReIDPipeline with timing for both detection and ReID
    pipeline = DetectorReIDPipeline(
        yolo_model_path=y,
        reid_model_paths=args.reid_model,
        device=args.device,
        imgsz=args.imgsz,
        half=args.half,
        timing_stats=timing_stats,
    )

    # Warmup the model
    pipeline.warmup()

    if auto_batch:
        batch_size = pipeline.autotune_batch_size(batch_size)
        args.batch_size = batch_size

    mot_folder_paths = sorted([p for p in Path(source_root).iterdir() if p.is_dir()])

    seq_states = {}
    det_fhs = {}
    emb_fhs = {r.stem: {} for r in args.reid_model}

    # runs/dets_n_embs/<dataset_name>/y.stem/... when benchmark is set
    benchmark = getattr(args, "benchmark", None)
    dets_base = Path(args.project) / "dets_n_embs"
    if benchmark:
        dets_base = dets_base / benchmark
    dets_folder = dets_base / y.stem / "dets"
    embs_root = dets_base / y.stem / "embs"
    total_frames = 0
    initial_done = 0

    for seq_dir in mot_folder_paths:
        img_dir = _sequence_img_dir(seq_dir)
        frames = _list_sequence_frames(img_dir)
        if not frames:
            continue

        seq_name = _sequence_name_from_img_dir(img_dir)

        dets_path = dets_folder / f"{seq_name}.txt"
        processed = 0

        emb_paths = {}
        any_emb_cached = False
        for r in args.reid_model:
            ep = embs_root / r.stem / f"{seq_name}.txt"
            emb_paths[r.stem] = ep
            if ep.exists():
                any_emb_cached = True

        expected_files = False
        rows_match = False
        det_rows = 0
        det_max_frame = 0
        emb_rows: dict[str, int] = {}

        if resume:
            det_rows = _count_data_lines(dets_path, skip_header=True)
            det_max_frame = _max_frame_id(dets_path)
            emb_rows = {stem: _count_data_lines(ep) for stem, ep in emb_paths.items()}
            expected_files = dets_path.exists() and all(ep.exists() for ep in emb_paths.values())
            rows_match = len(set([det_rows, *emb_rows.values()])) == 1 if expected_files else False
            if expected_files and rows_match and det_rows > 0:
                processed = min(det_max_frame, len(frames))
            elif expected_files and not rows_match:
                LOGGER.warning(
                    f"Cached det/emb rows mismatch for {seq_name}; resetting cached data."
                )
                for p in [dets_path, *emb_paths.values()]:
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
                processed = 0

        if resume and processed >= len(frames) and dets_path.exists() and all(ep.exists() for ep in emb_paths.values()):
            if expected_files and rows_match and det_rows:
                LOGGER.info(
                    f"Skipping {seq_name} (cached complete; {processed}/{len(frames)} frames)."
                )
            else:
                LOGGER.info(f"Skipping {seq_name} (resume: already complete).")
            initial_done += len(frames)
            continue

        if resume and 0 < processed < len(frames):
            LOGGER.info(f"Resuming {seq_name}: cached {processed}/{len(frames)} frames.")

        if (not resume) and dets_path.exists() and any_emb_cached:
            if not prompt_overwrite('Detections and Embeddings', dets_path, args.ci):
                LOGGER.debug(f"Skipping {seq_name} (cached).")
                continue

        dets_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'ab' if (resume and dets_path.exists()) else 'wb'
        det_fhs[seq_name] = open(dets_path, mode, buffering=1024 * 1024)
        if mode == 'wb' or dets_path.stat().st_size == 0:
            np.savetxt(det_fhs[seq_name], [], fmt='%f', header=str(img_dir))

        for r in args.reid_model:
            ep = emb_paths[r.stem]
            ep.parent.mkdir(parents=True, exist_ok=True)
            emb_mode = 'ab' if (resume and ep.exists()) else 'wb'
            emb_fhs[r.stem][seq_name] = open(ep, emb_mode, buffering=1024 * 1024)

        seq_states[seq_name] = {"frames": frames, "i": processed, "img_dir": img_dir}
        total_frames += len(frames)
        initial_done += processed

    if not seq_states:
        LOGGER.info("No sequences to process (all cached or no images).")
        return

    seq_names = list(seq_states.keys())
    rr = 0

    use_cuda = str(args.device).startswith("cuda")
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_cuda and getattr(args, "half", False))
        else nullcontext()
    )

    pbar = tqdm(total=total_frames, desc=f"Batched YOLO+ReID ({y.name}, bs={batch_size})", unit="frame")
    reid_pbar = tqdm(total=0, desc="ReID embeddings", unit="det", dynamic_ncols=True)
    if initial_done:
        pbar.update(initial_done)

    from concurrent.futures import ThreadPoolExecutor

    try:
        with ThreadPoolExecutor(max_workers=read_threads) as pool, amp_ctx:
            alive = True
            while alive:
                batch_items = []
                tried = 0
                while len(batch_items) < batch_size and tried < len(seq_names):
                    seq_name = seq_names[rr % len(seq_names)]
                    rr += 1
                    tried += 1

                    st = seq_states[seq_name]
                    if st["i"] >= len(st["frames"]):
                        continue
                    frame_id = st["i"] + 1
                    img_path = st["frames"][st["i"]]
                    st["i"] += 1
                    batch_items.append((seq_name, frame_id, img_path))

                if not batch_items:
                    alive = False
                    break

                futures = [pool.submit(_read_image_cv2, p) for _, _, p in batch_items]
                imgs = [f.result() for f in futures]

                yolo_results = None
                while True:
                    try:
                        # Use unified batch inference from pipeline
                        yolo_results = pipeline.predict_batch(
                            images=imgs[:batch_size],
                            conf=args.conf,
                            iou=args.iou,
                            agnostic_nms=args.agnostic_nms,
                            classes=args.classes,
                            verbose=False,
                        )
                        break
                    except RuntimeError as e:
                        if "out of memory" not in str(e).lower():
                            raise
                        if batch_size == 1:
                            raise

                        _clear_device_cache(args.device)

                        for seq_name, _, _ in batch_items:
                            seq_states[seq_name]["i"] -= 1

                        new_bs = max(1, batch_size // 2)
                        LOGGER.warning(f"YOLO predict OOM at batch size {batch_size}; retrying with {new_bs}.")
                        batch_size = new_bs
                        args.batch_size = batch_size
                        yolo_results = None
                        break

                if yolo_results is None:
                    continue

                det_counts = [int(r.boxes.shape[0]) if r.boxes is not None else 0 for r in yolo_results]
                emb_dims: dict[str, int] = {}
                LOGGER.info(
                    f"YOLO batch frames={len(batch_items)} | dets/frame={det_counts} | total_dets={sum(det_counts)}"
                )
                touched: set[str] = set()

                for (seq_name, frame_id, _), r, img in zip(batch_items, yolo_results, imgs):
                    # Use unified detection extraction and filtering
                    dets = extract_detections(r)
                    dets = filter_detections(dets, min_area=10.0, remove_degenerate=True)
                    
                    if len(dets) == 0:
                        if timing_stats:
                            timing_stats.frames += 1
                        pbar.update(1)
                        continue

                    boxes = dets[:, :4]
                    confs = dets[:, 4:5]  # Keep as 2D for concatenation
                    clss = dets[:, 5:6]   # Keep as 2D for concatenation

                    # Build detection array with frame_id column: [frame_id, x1, y1, x2, y2, conf, cls]
                    frame_col = np.full((boxes.shape[0], 1), float(frame_id), dtype=np.float32)
                    dets_np = np.concatenate([frame_col, boxes, confs, clss], axis=1)
                    dets_np[:, 1:5] = np.rint(dets_np[:, 1:5])  # round xyxy only

                    np.savetxt(det_fhs[seq_name], dets_np, fmt="%f")

                    det_boxes_np = dets_np[:, 1:5]
                    
                    # Use pipeline's ReID models (with timing instrumentation)
                    all_embs = pipeline.get_all_reid_features(det_boxes_np, img)
                    for reid_name, embs in all_embs.items():
                        if embs.shape[0] != det_boxes_np.shape[0]:
                            raise RuntimeError(
                                f"Embedding count mismatch: dets={det_boxes_np.shape[0]} embs={embs.shape[0]}"
                            )
                        if embs.ndim >= 2 and reid_name not in emb_dims:
                            emb_dims[reid_name] = embs.shape[1]
                        np.savetxt(emb_fhs[reid_name][seq_name], embs, fmt="%f")

                    reid_pbar.update(det_boxes_np.shape[0])

                    # Update frame count for timing
                    if timing_stats:
                        timing_stats.frames += 1
                    
                    pbar.update(1)
                    touched.add(seq_name)

                if emb_dims:
                    LOGGER.info(
                        "ReID embedding dims per model: "
                        + ", ".join([f"{k}={v}" for k, v in emb_dims.items()])
                    )
                else:
                    LOGGER.info("ReID embedding dims per model: n/a (no detections)")

                for seq_name in touched:
                    try:
                        det_fhs[seq_name].flush()
                        for per_reid in emb_fhs.values():
                            if seq_name in per_reid:
                                per_reid[seq_name].flush()
                    except Exception:
                        pass

                del yolo_results, imgs
                _clear_device_cache(args.device)

    finally:
        pbar.close()
        reid_pbar.close()
        for fh in det_fhs.values():
            try:
                fh.close()
            except Exception:
                pass
        for per_reid in emb_fhs.values():
            for fh in per_reid.values():
                try:
                    fh.close()
                except Exception:
                    pass


def run_generate_dets_embs(args: argparse.Namespace, timing_stats: Optional[TimingStats] = None) -> None:
    """
    Generate detections and embeddings for all sequences.
    
    Args:
        args: CLI arguments.
        timing_stats: Optional TimingStats for timing instrumentation.
    """
    source_root = Path(args.source)

    args.batch_size = int(getattr(args, "batch_size", 16))
    if getattr(args, "read_threads", None) is None:
        args.read_threads = min(8, (os.cpu_count() or 8))
    if not hasattr(args, "auto_batch"):
        args.auto_batch = True
    if not hasattr(args, "resume"):
        args.resume = True

    for y in args.yolo_model:
        LOGGER.info(f"Generating dets+embs (batched single-process): {y.name}")
        generate_dets_embs_batched(args, y, source_root, timing_stats=timing_stats)

def build_dataset_eval_settings(
    args: argparse.Namespace,
    gt_folder: Path,
    seq_info: dict[str, int],
) -> dict:
    """Derive dataset-specific evaluation settings (classes, ids, distractors, gt path format).

    This centralizes logic for MOT-style datasets and non-MOT layouts such as VisDrone.
    """

    cfg = {}
    try:
        if hasattr(args, "benchmark"):
            cfg = load_dataset_cfg(args.benchmark)
    except FileNotFoundError:
        cfg = {}
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"Error loading dataset config: {e}")
        cfg = {}

    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    eval_classes_cfg = bench_cfg.get("eval_classes") if isinstance(bench_cfg, dict) else None
    distractor_cfg = bench_cfg.get("distractor_classes") if isinstance(bench_cfg, dict) else None

    classes_to_eval = []
    class_ids = []

    # Filter classes by user provided classes
    if hasattr(args, "classes") and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        classes_to_eval = [COCO_CLASSES[int(i)] for i in class_indices]
        class_ids = [int(i) + 1 for i in class_indices]

    # Match classes by benchmark config
    if isinstance(eval_classes_cfg, dict) and len(eval_classes_cfg) > 0:
        ordered = sorted(((int(k), v) for k, v in eval_classes_cfg.items()), key=lambda kv: kv[0])
        if class_ids:
            class_ids = [k for k, _ in ordered if class_ids and k in class_ids]
            classes_to_eval = [v for k, v in ordered if class_ids and k in class_ids]
        else:
            class_ids = [k for k, _ in ordered]
            classes_to_eval = [v for k, v in ordered]

    # Default classes
    if not classes_to_eval:
        classes_to_eval = ["person"]
    if not class_ids:
        class_ids = [1]

    # Distractors
    distractor_ids: list[int] = []
    if isinstance(distractor_cfg, dict) and len(distractor_cfg) > 0:
        distractor_ids = [int(k) for k in distractor_cfg.keys()]

    # Remove duplicates while preserving order
    seen = set()
    pairs = []
    for name, cid in zip(classes_to_eval, class_ids):
        if name in seen:
            continue
        seen.add(name)
        pairs.append((name, cid))
    classes_to_eval = [name for name, _ in pairs]
    class_ids = [cid for _, cid in pairs]

    # GT path format (VisDrone uses flat txt under annotations)
    gt_loc_format = "{gt_folder}/{seq}/gt/gt_temp.txt"
    is_visdrone = "visdrone" in getattr(args, "benchmark", "").lower() or "visdrone" in str(gt_folder).lower()
    if is_visdrone:
        gt_loc_format = "{gt_folder}/{seq}.txt"

    benchmark_name = getattr(args, "benchmark", "")

    return {
        "classes_to_eval": classes_to_eval,
        "class_ids": class_ids,
        "distractor_ids": distractor_ids,
        "gt_loc_format": gt_loc_format,
        "benchmark_name": benchmark_name,
        "seq_info": seq_info,
    }


def trackeval(
    args: argparse.Namespace,
    seq_paths: list,
    save_dir: Path,
    gt_folder: Path,
    metrics: list = ["HOTA", "CLEAR", "Identity"],
    seq_info: Optional[dict] = None,
) -> str:
    """
    Executes a Python script to evaluate MOT challenge tracking results using specified metrics.

    Args:
        seq_paths (list): List of sequence paths.
        save_dir (Path): Directory to save evaluation results.
        gt_folder (Path): Folder containing ground truth data.
        metrics (list, optional): List of metrics to use for evaluation. Defaults to ["HOTA", "CLEAR", "Identity"].

    Returns:
        str: Standard output from the evaluation script.
    """

    if not seq_info:
        seq_names = [seq_path.parent.name if seq_path.name == "img1" else seq_path.name for seq_path in seq_paths]
        seq_info = {name: None for name in seq_names}

    seq_info_args = []
    for name in sorted(seq_info.keys()):
        length = seq_info[name]
        seq_info_args.append(f"{name}:{length}" if length else name)

    dataset_settings = build_dataset_eval_settings(args, gt_folder, seq_info)
    classes_to_eval = dataset_settings["classes_to_eval"]
    class_ids = dataset_settings["class_ids"]
    distractor_ids = dataset_settings["distractor_ids"]
    gt_loc_format = dataset_settings["gt_loc_format"]
    benchmark_name = dataset_settings["benchmark_name"]

    cmd_args = [
        sys.executable, ROOT / 'boxmot' / 'utils' / 'run_mot_challenge.py',
        "--GT_FOLDER", str(gt_folder),
        "--BENCHMARK", benchmark_name,
        "--TRACKERS_FOLDER", str(args.exp_dir.parent),
        "--TRACKERS_TO_EVAL", args.exp_dir.name,
        "--SPLIT_TO_EVAL", args.split,
        "--METRICS", *metrics,
        "--USE_PARALLEL", "True",
        "--TRACKER_SUB_FOLDER", "",
        "--NUM_PARALLEL_CORES", str(4),
        "--SKIP_SPLIT_FOL", "True",
        "--GT_LOC_FORMAT", gt_loc_format,
        "--CLASSES_TO_EVAL", *classes_to_eval,
        "--CLASS_IDS", *[str(i) for i in class_ids],
        "--DISTRACTOR_CLASS_IDS", *[str(i) for i in distractor_ids],
        "--SEQ_INFO", *seq_info_args
    ]

    p = subprocess.Popen(
        args=cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = p.communicate()

    if stderr:
        print("Standard Error:\n", stderr)
    return stdout



def process_sequence(seq_name: str,
                     mot_root: str,
                     project_root: str,
                     model_name: str,
                     reid_name: str,
                     tracking_method: str,
                     exp_folder: str,
                     target_fps: Optional[int],
                     device: str,
                     cfg_dict: Optional[Dict] = None,
                     dataset_name: Optional[str] = None,
                     ):
    """
    Process a single sequence: run tracker on pre-computed detections/embeddings.
    
    Returns:
        Tuple of (seq_name, kept_frame_ids, timing_dict) where timing_dict contains
        'track_time_ms' and 'num_frames' for aggregating timing stats.
    """
    import time

    device = select_device(device)
    tracker = create_tracker(
        tracker_type=tracking_method,
        tracker_config=TRACKER_CONFIGS / (tracking_method + ".yaml"),
        reid_weights=Path(reid_name + '.pt'),
        device=device,
        half=False,
        per_class=False,
        evolve_param_dict=cfg_dict,
    )

    # load with the user’s FPS
    # runs/dets_n_embs/<dataset_name>/ when dataset_name is set
    det_emb_root = Path(project_root) / "dets_n_embs"
    if dataset_name:
        det_emb_root = det_emb_root / dataset_name
    dataset = MOTDataset(
        mot_root=mot_root,
        det_emb_root=str(det_emb_root),
        model_name=model_name,
        reid_name=reid_name,
        target_fps=target_fps
    )
    sequence = dataset.get_sequence(seq_name)

    all_tracks = []
    kept_frame_ids = []
    total_track_time_ms = 0.0
    num_frames = 0
    
    for frame in sequence:
        fid  = int(frame['frame_id'])
        dets = frame['dets']
        embs = frame['embs']
        img  = frame['img']

        kept_frame_ids.append(fid)
        num_frames += 1

        if dets.size and embs.size:
            if dets.shape[0] != embs.shape[0]:
                msg = (
                    f"Detection/embedding count mismatch for {seq_name} frame {fid}: "
                    f"dets={dets.shape[0]} embs={embs.shape[0]}"
                )
                LOGGER.error(msg)
                raise ValueError(msg)

            # Time the tracker update (association only, embeddings pre-computed)
            t0 = time.perf_counter()
            tracks = tracker.update(dets, img, embs)
            total_track_time_ms += (time.perf_counter() - t0) * 1000
            
            if tracks.size:
                all_tracks.append(convert_to_mot_format(tracks, fid))

    out_arr = np.vstack(all_tracks) if all_tracks else np.empty((0, 0))
    write_mot_results(Path(exp_folder) / f"{seq_name}.txt", out_arr)
    
    timing_dict = {
        'track_time_ms': total_track_time_ms,
        'num_frames': num_frames,
    }
    return seq_name, kept_frame_ids, timing_dict


from boxmot.utils import configure_logging as _configure_logging

def _worker_init():
    # each spawned process needs its own sinks
    _configure_logging()

def run_generate_mot_results(args: argparse.Namespace, evolve_config: dict = None, timing_stats: Optional[TimingStats] = None) -> None:
    """
    Run tracker on pre-computed detections/embeddings and generate MOT result files.
    
    Args:
        args: CLI arguments.
        evolve_config: Optional config dict for hyperparameter tuning.
        timing_stats: Optional TimingStats to record tracking/association time.
    """
    # Prepare experiment folder: runs/mot/<dataset_name>/model_reid_tracker when benchmark is set
    base = args.project / "mot"
    if getattr(args, "benchmark", None):
        base = base / args.benchmark
    base = base / f"{args.yolo_model[0].stem}_{args.reid_model[0].stem}_{args.tracking_method}"
    exp_dir = increment_path(base, sep="_", exist_ok=False)
    exp_dir.mkdir(parents=True, exist_ok=True)
    args.exp_dir = exp_dir

    # Just collect sequence names by scanning directory names
    sequence_names = []
    for d in Path(args.source).iterdir():
        if not d.is_dir():
            continue
        img_dir = d / "img1" if (d / "img1").exists() else d
        if any(img_dir.glob("*.jpg")) or any(img_dir.glob("*.png")):
            sequence_names.append(d.name)
    sequence_names.sort()

    # Build task arguments (include dataset_name for det_emb_root path)
    dataset_name = getattr(args, "benchmark", None)
    task_args = [
        (
            seq,
            str(args.source),
            str(args.project),
            args.yolo_model[0].stem,
            args.reid_model[0].stem,
            args.tracking_method,
            str(exp_dir),
            getattr(args, "fps", None),
            args.device,
            evolve_config,
            dataset_name,
        )
        for seq in sequence_names
    ]

    seq_frame_nums = {}
    total_track_time_ms = 0.0
    total_track_frames = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS, initializer=_worker_init) as executor:
        futures = {
            executor.submit(process_sequence, *args): args[0] for args in task_args
        }

        for fut in concurrent.futures.as_completed(futures):
            seq = futures[fut]
            try:
                seq_name, kept_ids, timing_dict = fut.result()
                seq_frame_nums[seq_name] = kept_ids
                # Aggregate timing from worker process
                total_track_time_ms += timing_dict.get('track_time_ms', 0)
                total_track_frames += timing_dict.get('num_frames', 0)
            except Exception:
                LOGGER.exception(f"Error processing {seq}")
    
    # Record aggregated tracking time in timing_stats
    if timing_stats is not None:
        timing_stats.totals['track'] += total_track_time_ms
        # Also update frame count if not already set (from batch mode)
        if timing_stats.frames == 0 and total_track_frames > 0:
            timing_stats.frames = total_track_frames
        # Log summary
        if total_track_frames > 0:
            avg_track = total_track_time_ms / total_track_frames
            LOGGER.opt(colors=True).info(
                f"<bold>Tracking:</bold> {total_track_frames} frames, "
                f"total: <cyan>{total_track_time_ms:.1f}ms</cyan>, "
                f"avg: <cyan>{avg_track:.2f}ms/frame</cyan>"
            )

    # Optional GSI postprocessing
    if getattr(args, "postprocessing", "none") == "gsi":
        LOGGER.opt(colors=True).info("<cyan>[3b/4]</cyan> Applying GSI postprocessing...")
        from boxmot.postprocessing.gsi import gsi
        gsi(mot_results_folder=exp_dir)

    elif getattr(args, "postprocessing", "none") == "gbrc":
        LOGGER.opt(colors=True).info("<cyan>[3b/4]</cyan> Applying GBRC postprocessing...")
        from boxmot.postprocessing.gbrc import gbrc
        gbrc(mot_results_folder=exp_dir)


def run_trackeval(args: argparse.Namespace, verbose: bool = True) -> dict:
    """
    Runs the trackeval function to evaluate tracking results.

    Args:
        args (Namespace): Parsed command line arguments.
        verbose (bool): Whether to print results summary. Default True.
    """
    seq_paths, seq_info = _collect_seq_info(args.source)
    annotations_dir = args.source.parent / "annotations"
    gt_folder = annotations_dir if annotations_dir.exists() else args.source

    if not seq_paths:
        raise ValueError(f"No sequences with images found under {args.source}")

    if annotations_dir.exists():
        for seq_name in list(seq_info.keys()):
            ann_file = annotations_dir / f"{seq_name}.txt"
            if not ann_file.exists():
                continue
            try:
                with open(ann_file, "r") as f:
                    max_frame = 0
                    for line in f:
                        if not line.strip():
                            continue
                        frame_id = int(float(line.split(",", 1)[0]))
                        if frame_id > max_frame:
                            max_frame = frame_id
                    if max_frame:
                        seq_info[seq_name] = max(seq_info.get(seq_name, 0) or 0, max_frame)
            except Exception:
                LOGGER.warning(f"Failed to read annotation file {ann_file} for sequence length inference")
    # runs/<dataset_name>/<name> when benchmark is set
    if getattr(args, "benchmark", None):
        save_dir = Path(args.project) / args.benchmark / args.name
    else:
        save_dir = Path(args.project) / args.name

    trackeval_results = trackeval(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)
    parsed_results = parse_mot_results(trackeval_results)

    # Load config to filter classes
    # Try to load config from benchmark name first, then fallback to source parent name
    cfg_name = getattr(args, 'benchmark', str(args.source.parent.name))
    try:
        cfg = load_dataset_cfg(cfg_name)
    except FileNotFoundError:
        # If config not found, try to find it by checking if source path ends with a known config name
        # This handles cases where source is a custom path
        found = False
        for config_file in DATASET_CONFIGS.glob("*.yaml"):
            if config_file.stem in str(args.source):
                cfg = load_dataset_cfg(config_file.stem)
                found = True
                break
        
        if not found:
            LOGGER.warning(f"Could not find dataset config for {cfg_name}. Class filtering might be incorrect.")
            cfg = {}

    # Filter parsed_results based on user provided classes (args.classes)
    single_class_mode = False

    # Priority 1: Benchmark config classes (overrides user classes)
    if "benchmark" in cfg:
        bench_cfg = cfg["benchmark"]
        bench_classes = None

        if isinstance(bench_cfg, dict):
            if "eval_classes" in bench_cfg:
                bench_classes = [v for _, v in sorted(bench_cfg["eval_classes"].items(), key=lambda kv: int(kv[0]))]
            elif "classes" in bench_cfg:
                bench_classes = bench_cfg["classes"].split()

        if bench_classes:
            parsed_results = {k: v for k, v in parsed_results.items() if k in bench_classes}
            if len(bench_classes) == 1:
                single_class_mode = True
    # Priority 2: User provided classes
    elif hasattr(args, 'classes') and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        user_classes = [COCO_CLASSES[int(i)] for i in class_indices]
        parsed_results = {k: v for k, v in parsed_results.items() if k in user_classes}
        if len(user_classes) == 1:
            single_class_mode = True

    if single_class_mode and len(parsed_results) > 0:
        final_results = list(parsed_results.values())[0]
    else:
        final_results = parsed_results
    
    # Print results summary
    if verbose:
        LOGGER.info("")
        LOGGER.opt(colors=True).info("<blue>" + "="*105 + "</blue>")
        LOGGER.opt(colors=True).info(f"<bold><cyan>{'📊 RESULTS SUMMARY':^105}</cyan></bold>")
        LOGGER.opt(colors=True).info("<blue>" + "="*105 + "</blue>")
        
        headers = ["Sequence", "HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs"]
        header_str = "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(*headers)
        LOGGER.opt(colors=True).info(f"<bold>{header_str}</bold>")
        LOGGER.opt(colors=True).info("<blue>" + "-"*105 + "</blue>")
        
        for cls, class_metrics in parsed_results.items():
            # Print per-sequence metrics first
            per_sequence = class_metrics.get('per_sequence', {})
            for seq_name in sorted(per_sequence.keys()):
                seq_metrics = per_sequence[seq_name]
                name_col = f"{seq_name:<25}"
                vals = [
                    f"{seq_metrics.get('HOTA', 0):>10.2f}",
                    f"{seq_metrics.get('MOTA', 0):>10.2f}",
                    f"{seq_metrics.get('IDF1', 0):>10.2f}",
                    f"{seq_metrics.get('AssA', 0):>10.2f}",
                    f"{seq_metrics.get('AssRe', 0):>10.2f}",
                    f"{seq_metrics.get('IDSW', 0):>10}",
                    f"{seq_metrics.get('IDs', 0):>10}"
                ]
                vals_str = " ".join([f"<blue>{v}</blue>" for v in vals])
                LOGGER.opt(colors=True).info(f"{name_col} {vals_str}")
            
            # Print COMBINED row (bold, highlighted)
            LOGGER.opt(colors=True).info("<blue>" + "-"*105 + "</blue>")
            name_col = f"{'COMBINED (' + cls + ')':<25}"
            vals = [
                f"{class_metrics.get('HOTA', 0):>10.2f}",
                f"{class_metrics.get('MOTA', 0):>10.2f}",
                f"{class_metrics.get('IDF1', 0):>10.2f}",
                f"{class_metrics.get('AssA', 0):>10.2f}",
                f"{class_metrics.get('AssRe', 0):>10.2f}",
                f"{class_metrics.get('IDSW', 0):>10}",
                f"{class_metrics.get('IDs', 0):>10}"
            ]
            vals_str = " ".join([f"<cyan>{v}</cyan>" for v in vals])
            LOGGER.opt(colors=True).info(f"<bold>{name_col} {vals_str}</bold>")
            
        LOGGER.opt(colors=True).info("<blue>" + "="*105 + "</blue>")

    if args.ci:
        with open(args.tracking_method + "_output.json", "w") as outfile:
            outfile.write(json.dumps(final_results))
    
    return final_results


def main(args):
    # Print evaluation pipeline header (blue palette)
    LOGGER.info("")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info("<bold><cyan>🚀 BoxMOT Evaluation Pipeline</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold>Detector:</bold>  <cyan>{args.yolo_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>ReID:</bold>      <cyan>{args.reid_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Tracker:</bold>   <cyan>{args.tracking_method}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Benchmark:</bold> <cyan>{args.source}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Image size:</bold> <cyan>{getattr(args, 'imgsz', None)}</cyan>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    
    # Initialize timing stats for the evaluation pipeline
    timing_stats = TimingStats()
    
    # Step 1: Download TrackEval
    LOGGER.opt(colors=True).info("<cyan>[1/4]</cyan> Setting up TrackEval...")
    eval_init(args)

    # Step 2: Generate detections and embeddings (with timing)
    LOGGER.opt(colors=True).info("<cyan>[2/4]</cyan> Generating detections and embeddings...")
    run_generate_dets_embs(args, timing_stats=timing_stats)
    
    # Step 3: Generate MOT results (with tracking timing)
    LOGGER.opt(colors=True).info("<cyan>[3/4]</cyan> Running tracker...")
    run_generate_mot_results(args, timing_stats=timing_stats)
    
    # Step 4: Evaluate with TrackEval
    LOGGER.opt(colors=True).info("<cyan>[4/4]</cyan> Evaluating results...")
    results = run_trackeval(args)
    
    # Print timing summary if we collected timing data
    if timing_stats.frames > 0:
        timing_stats.print_summary()
    
    # Only plot if we have results for a single class or handle multi-class plotting differently
    # For now, let's just skip the radar chart if we have multiple classes or complex structure
    # Or pick the first class found?
    # The original code expected a flat dict of metrics. Now we have {class: {metric: val}}
    
    # Let's try to plot for each class or just skip for now to avoid breaking
    # If 'pedestrian' is in results, use that, otherwise use the first key
    
    # Check if results is flat (single class backward compatibility) or nested
    is_flat = False
    if results and isinstance(list(results.values())[0], (int, float)):
        is_flat = True
        metrics_data = results
        plot_class = 'single_class'
    else:
        plot_class = 'pedestrian'
        if plot_class not in results and len(results) > 0:
            plot_class = list(results.keys())[0]
        metrics_data = results.get(plot_class, {})

    if metrics_data:
        plotter = MetricsPlotter(args.exp_dir)
        
        # Filter only the metrics we want to plot
        plot_metrics = ['HOTA', 'MOTA', 'IDF1']
        plot_values = [metrics_data.get(m, 0) for m in plot_metrics]

        plotter.plot_radar_chart(
            {args.tracking_method: plot_values},
            plot_metrics,
            title=f"MOT metrics radar Chart ({plot_class})",
            ylim=(0, 100),
            yticks=[20, 40, 60, 80, 100],
            ytick_labels=['20', '40', '60', '80', '100']
        )
        

if __name__ == "__main__":
    main()