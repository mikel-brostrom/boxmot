# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from boxmot.detectors import default_conf, default_imgsz, get_runtime_detector_cfg
from boxmot.engine.inference import DetectorReIDPipeline, prepare_detections
from boxmot.trackers.tracker_zoo import create_tracker
from boxmot.utils import (
    BENCHMARK_CONFIGS,
    TRACKER_CONFIGS,
    WEIGHTS,
    configure_logging as _configure_logging,
    logger as LOGGER,
)
from boxmot.utils.benchmark_config import (
    ensure_benchmark_detector_model,
    ensure_benchmark_reid_model,
    load_benchmark_cfg,
    should_use_benchmark_detector,
    should_use_benchmark_reid,
)
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.dataloaders.dataset import MOTDataset
from boxmot.utils.evaluation import (
    AppendableNpyWriter,
    COCO_CLASSES,
    _clear_device_cache,
    _collect_seq_info,
    _count_embedding_rows,
    _display_summary_name,
    _existing_cache_path,
    _existing_embedding_cache_path,
    _filter_obb_trackeval_results,
    _known_trackeval_class_names,
    _load_embedding_cache_array,
    _load_numeric_cache_array,
    _load_obb_gt_matrix,
    _max_frame_id,
    _migrate_legacy_embedding_cache,
    _migrate_legacy_numeric_cache,
    _ordered_benchmark_eval_class_names,
    _print_summary_table,
    _read_image_cv2,
    _saved_detection_column_count,
    _select_plot_metrics_data,
    _serialize_eval_detections,
    _summary_sort_keys,
    build_gt_class_remap,
    configure_benchmark_runtime,
    eval_init,
    load_benchmark_cfg_from_args,
    parse_mot_results,
    prepare_aabb_eval_gt,
    resolve_eval_box_type,
    trackeval_aabb,
    trackeval_obb,
)
from boxmot.utils.misc import increment_path, prompt_overwrite, resolve_model_path
from boxmot.utils.mot_utils import convert_to_mmot_obb_format, convert_to_mot_format, write_mot_results
from boxmot.utils.plots import MetricsPlotter
from boxmot.utils.timing import TimingStats
from boxmot.utils.torch_utils import select_device

mp.set_start_method("spawn", force=True)

checker = RequirementsChecker()
checker.check_packages(("ultralytics",))

__all__ = [
    "AppendableNpyWriter",
    "_configure_benchmark_runtime",
    "_existing_cache_path",
    "_existing_embedding_cache_path",
    "_load_benchmark_cfg",
    "_load_embedding_cache_array",
    "_load_numeric_cache_array",
    "_load_obb_gt_matrix",
    "_max_frame_id",
    "_migrate_legacy_embedding_cache",
    "_ordered_benchmark_eval_class_names",
    "_saved_detection_column_count",
    "_select_plot_metrics_data",
    "apply_class_remap",
    "eval_setup",
    "generate_dets_embs_batched",
    "main",
    "parse_mot_results",
    "run_generate_dets_embs",
    "run_generate_mot_results",
    "run_trackeval",
]


def _load_benchmark_cfg(args: argparse.Namespace) -> dict:
    return load_benchmark_cfg_from_args(args)


def _resolve_eval_box_type(args: argparse.Namespace, bench_cfg: Optional[dict] = None) -> str:
    return resolve_eval_box_type(args, bench_cfg)


def _configure_benchmark_runtime(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    return configure_benchmark_runtime(
        args,
        load_benchmark_cfg_fn=_load_benchmark_cfg,
        should_use_benchmark_detector_fn=should_use_benchmark_detector,
        should_use_benchmark_reid_fn=should_use_benchmark_reid,
        ensure_benchmark_detector_model_fn=ensure_benchmark_detector_model,
        ensure_benchmark_reid_model_fn=ensure_benchmark_reid_model,
    )


@torch.inference_mode()
def generate_dets_embs_batched(
    args: argparse.Namespace,
    y: Path,
    source_root: Path,
    timing_stats: Optional[TimingStats] = None,
) -> None:
    """
    Generate detections and embeddings in batches for evaluation.
    """
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    batch_size = int(getattr(args, "batch_size", 16))
    n_threads = int(args.n_threads)
    auto_batch = bool(getattr(args, "auto_batch", True))
    resume = bool(getattr(args, "resume", True))

    if args.imgsz is None:
        args.imgsz = default_imgsz(y)

    expected_det_cols = 8 if str(getattr(args, "eval_box_type", "")).lower() == "obb" else 7

    benchmark = getattr(args, "benchmark", None)
    dets_base = Path(args.project) / "dets_n_embs"
    if benchmark:
        dets_base = dets_base / benchmark
    dets_folder = dets_base / y.stem / "dets"
    embs_root = dets_base / y.stem / "embs"

    mot_folder_paths = sorted([path for path in Path(source_root).iterdir() if path.is_dir()])

    seq_states = {}
    det_writers: dict[str, AppendableNpyWriter] = {}
    emb_writers: dict[str, dict[str, AppendableNpyWriter]] = {reid.stem: {} for reid in args.reid_model}
    total_frames = 0
    initial_done = 0

    for seq_dir in mot_folder_paths:
        img_dir = seq_dir / "img1" if (seq_dir / "img1").exists() else seq_dir
        frames = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if not frames:
            continue

        seq_name = img_dir.parent.name if img_dir.name == "img1" else img_dir.name

        dets_path = dets_folder / f"{seq_name}.npy"
        cached_dets_path = _existing_cache_path(dets_path)
        processed = 0

        emb_paths = {}
        cached_emb_paths = {}
        any_emb_cached = False
        for reid_model in args.reid_model:
            emb_path = embs_root / reid_model.stem / f"{seq_name}.npy"
            emb_paths[reid_model.stem] = emb_path
            cached_emb_path = _existing_embedding_cache_path(emb_path)
            cached_emb_paths[reid_model.stem] = cached_emb_path
            if cached_emb_path is not None:
                any_emb_cached = True

        expected_files = False
        rows_match = False
        det_rows = 0
        det_max_frame = 0
        emb_rows: dict[str, int] = {}

        if resume:
            det_rows = _count_embedding_rows(cached_dets_path) if cached_dets_path is not None else 0
            det_max_frame = _max_frame_id(cached_dets_path) if cached_dets_path is not None else 0
            det_col_count = _saved_detection_column_count(cached_dets_path) if cached_dets_path is not None else 0
            emb_rows = {
                stem: _count_embedding_rows(cached_emb_path if cached_emb_path is not None else emb_paths[stem])
                for stem, cached_emb_path in cached_emb_paths.items()
                if (cached_emb_path is not None or emb_paths[stem].exists())
            }
            expected_files = cached_dets_path is not None and all(
                cached_emb_path is not None for cached_emb_path in cached_emb_paths.values()
            )
            rows_match = len(set([det_rows, *emb_rows.values()])) == 1 if expected_files else False
            schema_match = det_col_count in (0, expected_det_cols)

            if expected_files and not schema_match:
                LOGGER.warning(
                    f"Cached detection schema mismatch for {seq_name}: "
                    f"found {det_col_count} columns, expected {expected_det_cols}. Resetting cached data."
                )
                reset_paths = {
                    dets_path,
                    *(path for path in [cached_dets_path, *emb_paths.values(), *cached_emb_paths.values()] if path is not None),
                }
                for path in reset_paths:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                processed = 0
                expected_files = False
                rows_match = False
            elif expected_files and rows_match and det_rows > 0:
                processed = min(det_max_frame, len(frames))
            elif expected_files and not rows_match:
                LOGGER.warning(f"Cached det/emb rows mismatch for {seq_name}; resetting cached data.")
                reset_paths = {
                    dets_path,
                    *(path for path in [cached_dets_path, *emb_paths.values(), *cached_emb_paths.values()] if path is not None),
                }
                for path in reset_paths:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                processed = 0

        if (
            resume
            and processed >= len(frames)
            and cached_dets_path is not None
            and all(cached_emb_path is not None for cached_emb_path in cached_emb_paths.values())
        ):
            try:
                if _migrate_legacy_numeric_cache(cached_dets_path, dets_path, comments="#"):
                    LOGGER.info(f"Migrated legacy detection cache to {dets_path.name}")
                    cached_dets_path = dets_path
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(f"Failed to migrate detections for {seq_name}: {exc}")
            for stem, cached_emb_path in cached_emb_paths.items():
                if cached_emb_path is None:
                    continue
                try:
                    if _migrate_legacy_embedding_cache(cached_emb_path, emb_paths[stem]):
                        LOGGER.info(f"Migrated legacy embedding cache to {emb_paths[stem].name}")
                        cached_emb_paths[stem] = emb_paths[stem]
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(f"Failed to migrate embeddings for {seq_name}/{stem}: {exc}")
            if expected_files and rows_match and det_rows:
                LOGGER.info(f"Skipping {seq_name} (cached complete; {processed}/{len(frames)} frames).")
            else:
                LOGGER.info(f"Skipping {seq_name} (resume: already complete).")
            initial_done += len(frames)
            continue

        if resume and 0 < processed < len(frames):
            LOGGER.info(f"Resuming {seq_name}: cached {processed}/{len(frames)} frames.")

        if (not resume) and cached_dets_path is not None and any_emb_cached:
            if not prompt_overwrite("Detections and Embeddings", cached_dets_path, args.ci):
                LOGGER.debug(f"Skipping {seq_name} (cached).")
                continue

        dets_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and cached_dets_path is not None and cached_dets_path.suffix == ".txt":
            try:
                if _migrate_legacy_numeric_cache(cached_dets_path, dets_path, comments="#"):
                    LOGGER.info(f"Migrated legacy detection cache to {dets_path.name}")
                    cached_dets_path = dets_path
            except Exception:
                pass
        det_writers[seq_name] = AppendableNpyWriter(
            dets_path,
            dtype=np.float32,
            trailing_shape=(expected_det_cols,),
            empty_trailing_shape=(expected_det_cols,),
        )

        for reid_model in args.reid_model:
            emb_path = emb_paths[reid_model.stem]
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            cached_emb_path = cached_emb_paths[reid_model.stem]
            if resume and cached_emb_path is not None and cached_emb_path.suffix == ".txt":
                try:
                    if _migrate_legacy_embedding_cache(cached_emb_path, emb_path):
                        LOGGER.info(f"Migrated legacy embedding cache to {emb_path.name}")
                        cached_emb_path = emb_path
                except Exception:
                    pass
            emb_writers[reid_model.stem][seq_name] = AppendableNpyWriter(
                emb_path,
                dtype=np.float32,
                trailing_shape=None,
                empty_trailing_shape=(0,),
            )

        seq_states[seq_name] = {"frames": frames, "i": processed, "img_dir": img_dir}
        total_frames += len(frames)
        initial_done += processed

    if not seq_states:
        LOGGER.info("No sequences to process (all cached or no images).")
        return

    pipeline = DetectorReIDPipeline(
        detector_path=y,
        reid_paths=args.reid_model,
        device=args.device,
        reid_device=getattr(args, "reid_device", args.device),
        imgsz=args.imgsz,
        half=args.half,
        reid_half=getattr(args, "reid_half", args.half),
        timing_stats=timing_stats,
    )
    pipeline.warmup()

    if auto_batch:
        batch_size = pipeline.autotune_batch_size(batch_size)
        args.batch_size = batch_size

    use_cuda = str(args.device).startswith("cuda")
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_cuda and getattr(args, "half", False))
        else nullcontext()
    )

    seq_names = list(seq_states.keys())
    rr = 0

    pbar = tqdm(total=total_frames, desc=f"Batched YOLO+ReID ({y.name}, bs={batch_size})", unit="frame")
    reid_pbar = tqdm(total=0, desc="ReID embeddings", unit="det", dynamic_ncols=True)
    if initial_done:
        pbar.update(initial_done)

    from concurrent.futures import ThreadPoolExecutor

    try:
        with ThreadPoolExecutor(max_workers=n_threads) as pool, amp_ctx:
            alive = True
            while alive:
                batch_items = []
                tried = 0
                while len(batch_items) < batch_size and tried < len(seq_names):
                    seq_name = seq_names[rr % len(seq_names)]
                    rr += 1
                    tried += 1

                    state = seq_states[seq_name]
                    if state["i"] >= len(state["frames"]):
                        continue
                    frame_id = state["i"] + 1
                    img_path = state["frames"][state["i"]]
                    state["i"] += 1
                    batch_items.append((seq_name, frame_id, img_path))

                if not batch_items:
                    alive = False
                    break

                futures = [pool.submit(_read_image_cv2, path) for _, _, path in batch_items]
                imgs = [future.result() for future in futures]

                yolo_results = None
                while True:
                    try:
                        yolo_results = pipeline.predict_batch(
                            images=imgs[:batch_size],
                            conf=0.01,
                            iou=args.iou,
                            agnostic_nms=args.agnostic_nms,
                            classes=args.classes,
                        )
                        break
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
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

                det_counts = [len(result.dets) for result in yolo_results]
                emb_dims: dict[str, int] = {}
                LOGGER.info(
                    f"YOLO batch frames={len(batch_items)} | dets/frame={det_counts} | total_dets={sum(det_counts)}"
                )

                for (seq_name, frame_id, _), result, img in zip(batch_items, yolo_results, imgs):
                    dets = prepare_detections(result, img)

                    if len(dets) == 0:
                        if timing_stats:
                            timing_stats.frames += 1
                        pbar.update(1)
                        continue

                    dets_np, det_boxes_np = _serialize_eval_detections(dets, frame_id)
                    det_writers[seq_name].append(dets_np.astype(np.float32, copy=False))

                    all_embs = pipeline.get_all_reid_features(det_boxes_np, img)
                    for reid_name, embs in all_embs.items():
                        if embs.shape[0] != det_boxes_np.shape[0]:
                            raise RuntimeError(
                                f"Embedding count mismatch: dets={det_boxes_np.shape[0]} embs={embs.shape[0]}"
                            )
                        if embs.ndim >= 2 and reid_name not in emb_dims:
                            emb_dims[reid_name] = embs.shape[1]
                        emb_writers[reid_name][seq_name].append(embs.astype(np.float32, copy=False))

                    reid_pbar.update(det_boxes_np.shape[0])

                    if timing_stats:
                        timing_stats.frames += 1

                    pbar.update(1)

                if emb_dims:
                    LOGGER.info(
                        "ReID embedding dims per model: "
                        + ", ".join([f"{name}={dim}" for name, dim in emb_dims.items()])
                    )
                else:
                    LOGGER.info("ReID embedding dims per model: n/a (no detections)")

                del yolo_results, imgs
                _clear_device_cache(args.device)

    finally:
        pbar.close()
        reid_pbar.close()
        for seq_name, writer in det_writers.items():
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(f"Failed to save detections for {seq_name}: {exc}")
        for reid_name, per_seq in emb_writers.items():
            for seq_name, writer in per_seq.items():
                try:
                    writer.close()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(f"Failed to save embeddings for {seq_name}/{reid_name}: {exc}")


def run_generate_dets_embs(args: argparse.Namespace, timing_stats: Optional[TimingStats] = None) -> None:
    """
    Generate detections and embeddings for all sequences.
    """
    if getattr(args, "data", None) and getattr(args, "source", None) is None:
        from boxmot.utils.benchmark_config import apply_benchmark_config

        apply_benchmark_config(args, overwrite=False)

    _configure_benchmark_runtime(args)
    source_root = Path(args.source)

    args.batch_size = int(getattr(args, "batch_size", 16))
    if getattr(args, "n_threads", None) is None:
        args.n_threads = min(8, (os.cpu_count() or 8))
    if not hasattr(args, "auto_batch"):
        args.auto_batch = True
    if not hasattr(args, "resume"):
        args.resume = True

    for yolo_model in args.yolo_model:
        LOGGER.info(f"Generating dets+embs (batched single-process): {yolo_model.name}")
        generate_dets_embs_batched(args, yolo_model, source_root, timing_stats=timing_stats)


def process_sequence(
    seq_name: str,
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
    conf_threshold: float = 0.0,
    progress_queue=None,
):
    """
    Process a single sequence: run tracker on pre-computed detections/embeddings.
    """
    import time

    tracker_device = select_device("cpu")
    tracker = create_tracker(
        tracker_type=tracking_method,
        tracker_config=TRACKER_CONFIGS / f"{tracking_method}.yaml",
        reid_weights=Path(reid_name + ".pt"),
        device=tracker_device,
        half=False,
        per_class=False,
        evolve_param_dict=cfg_dict,
    )

    det_emb_root = Path(project_root) / "dets_n_embs"
    if dataset_name:
        det_emb_root = det_emb_root / dataset_name
    dataset = MOTDataset(
        mot_root=mot_root,
        det_emb_root=str(det_emb_root),
        model_name=model_name,
        reid_name=reid_name,
        target_fps=target_fps,
    )
    sequence = dataset.get_sequence(seq_name, show_progress=False,
                                     progress_queue=progress_queue)

    all_tracks = []
    kept_frame_ids = []
    total_track_time_ms = 0.0
    num_frames = 0

    for frame in sequence:
        frame_id = int(frame["frame_id"])
        dets = frame["dets"]
        embs = frame["embs"]
        img = frame["img"]

        kept_frame_ids.append(frame_id)
        num_frames += 1

        if dets.size and embs.size and conf_threshold > 0:
            conf_col = 5 if dets.shape[1] == 7 else 4
            mask = dets[:, conf_col] >= conf_threshold
            dets = dets[mask]
            embs = embs[mask]

        if dets.size and embs.size:
            if dets.shape[0] != embs.shape[0]:
                message = (
                    f"Detection/embedding count mismatch for {seq_name} frame {frame_id}: "
                    f"dets={dets.shape[0]} embs={embs.shape[0]}"
                )
                LOGGER.error(message)
                raise ValueError(message)

            t0 = time.perf_counter()
            tracks = tracker.update(dets, img, embs)
            total_track_time_ms += (time.perf_counter() - t0) * 1000

            if tracks.size:
                if tracks.ndim == 1:
                    tracks = tracks.reshape(1, -1)
                if tracks.shape[1] >= 9:
                    all_tracks.append(convert_to_mmot_obb_format(tracks, frame_id))
                else:
                    all_tracks.append(convert_to_mot_format(tracks, frame_id))

    out_arr = np.vstack(all_tracks) if all_tracks else np.empty((0, 0))
    write_mot_results(Path(exp_folder) / f"{seq_name}.txt", out_arr)

    timing_dict = {"track_time_ms": total_track_time_ms, "num_frames": num_frames}
    return seq_name, kept_frame_ids, timing_dict


def _worker_init():
    _configure_logging()


def _format_seq_progress(seq_progress: dict, n_display: int = 5) -> str:
    """Format top-N in-progress sequences as mini progress bars."""
    active = {k: v for k, v in seq_progress.items() if v[0] < v[1]}
    if not active:
        return ""
    sorted_seqs = sorted(active.items(), key=lambda x: x[1][0] / max(x[1][1], 1), reverse=True)
    display = sorted_seqs[:n_display]
    name_width = max(len(name) for name, _ in display)
    lines = []
    bar_width = 20
    for name, (current, total) in display:
        pct = current / max(total, 1)
        filled = int(bar_width * pct)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        lines.append(f"  {name:<{name_width}s} {bar} {pct:>5.0%}  ({current}/{total})")
    return "\n".join(lines)


def _drain_progress_queue(q, seq_progress: dict):
    """Read all available messages from the progress queue."""
    import queue
    while True:
        try:
            name, current, total = q.get_nowait()
            seq_progress[name] = (current, total)
        except (queue.Empty, OSError):
            break


def run_generate_mot_results(
    args: argparse.Namespace,
    evolve_config: dict = None,
    timing_stats: Optional[TimingStats] = None,
    quiet: bool = False,
) -> None:
    """
    Run tracker on pre-computed detections/embeddings and generate MOT result files.
    """
    base = args.project / "mot"
    if getattr(args, "benchmark", None):
        base = base / args.benchmark
    base = base / f"{args.yolo_model[0].stem}_{args.reid_model[0].stem}_{args.tracking_method}"
    exp_dir = increment_path(base, sep="_", exist_ok=False)
    exp_dir.mkdir(parents=True, exist_ok=True)
    args.exp_dir = exp_dir

    sequence_names = []
    for path in Path(args.source).iterdir():
        if not path.is_dir():
            continue
        img_dir = path / "img1" if (path / "img1").exists() else path
        if any(img_dir.glob("*.jpg")) or any(img_dir.glob("*.png")):
            sequence_names.append(path.name)
    sequence_names.sort()

    dataset_name = getattr(args, "benchmark", None)
    conf_threshold = getattr(args, "conf", None)
    if conf_threshold is None:
        conf_threshold = default_conf(args.yolo_model[0])

    manager = mp.Manager()
    progress_queue = manager.Queue()

    task_args = [
        (
            seq_name,
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
            conf_threshold,
            progress_queue,
        )
        for seq_name in sequence_names
    ]

    seq_frame_nums = {}
    total_track_time_ms = 0.0
    total_track_frames = 0
    n_seqs = len(sequence_names)
    done_count = 0
    seq_progress: dict = {}
    prev_display_lines = 0

    def _log_progress():
        nonlocal prev_display_lines
        _drain_progress_queue(progress_queue, seq_progress)
        header = f"Tracking: {done_count}/{n_seqs} sequences done"
        seq_display = _format_seq_progress(seq_progress)
        lines = [header] + ([seq_display] if seq_display else [])
        msg = "\n".join(lines)
        # Erase previously printed progress block, then log fresh
        import sys
        if prev_display_lines > 0:
            sys.stderr.write(f"\033[{prev_display_lines}A\033[J")
            sys.stderr.flush()
        LOGGER.opt(colors=True).info(f"<cyan>{msg}</cyan>")
        prev_display_lines = msg.count("\n") + 1

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.n_threads,
        initializer=_worker_init,
    ) as executor:
        futures = {executor.submit(process_sequence, *task_arg): task_arg[0] for task_arg in task_args}
        pending = set(futures.keys())

        while pending:
            done, pending = concurrent.futures.wait(
                pending, timeout=0.3,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                seq_name = futures[future]
                try:
                    sequence_name, kept_ids, timing_dict = future.result()
                    seq_frame_nums[sequence_name] = kept_ids
                    total_track_time_ms += timing_dict.get("track_time_ms", 0)
                    total_track_frames += timing_dict.get("num_frames", 0)
                    done_count += 1
                except Exception:
                    done_count += 1
                    LOGGER.exception(f"Error processing {seq_name}")

            if not quiet:
                _log_progress()

    if not quiet and prev_display_lines > 0:
        import sys
        sys.stderr.write(f"\033[{prev_display_lines}A\033[J")
        sys.stderr.flush()
        LOGGER.opt(colors=True).info(
            f"<cyan>Tracking: {n_seqs}/{n_seqs} sequences done</cyan>"
        )

    args.seq_frame_nums = seq_frame_nums

    if timing_stats is not None:
        timing_stats.totals["track"] += total_track_time_ms
        if timing_stats.frames == 0 and total_track_frames > 0:
            timing_stats.frames = total_track_frames
        if total_track_frames > 0:
            avg_track = total_track_time_ms / total_track_frames
            LOGGER.opt(colors=True).info(
                f"<bold>Tracking:</bold> {total_track_frames} frames, "
                f"total: <cyan>{total_track_time_ms:.1f}ms</cyan>, "
                f"avg: <cyan>{avg_track:.2f}ms/frame</cyan>"
            )

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
    Evaluate tracking results via TrackEval and print a summary.
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
                with open(ann_file, "r") as handle:
                    max_frame = 0
                    for line in handle:
                        if not line.strip():
                            continue
                        frame_id = int(float(line.split(",", 1)[0]))
                        if frame_id > max_frame:
                            max_frame = frame_id
                    if max_frame:
                        seq_info[seq_name] = max(seq_info.get(seq_name, 0) or 0, max_frame)
            except Exception:
                LOGGER.warning(f"Failed to read annotation file {ann_file} for sequence length inference")

    if getattr(args, "benchmark", None):
        save_dir = Path(args.project) / args.benchmark / args.name
    else:
        save_dir = Path(args.project) / args.name

    cfg = _load_benchmark_cfg(args)
    if not cfg:
        cfg_name = (
            getattr(args, "benchmark_id", None)
            or getattr(args, "dataset_id", None)
            or getattr(args, "benchmark", str(args.source.parent.name))
        )
        try:
            cfg = load_benchmark_cfg(cfg_name)
        except FileNotFoundError:
            found = False
            for config_file in BENCHMARK_CONFIGS.glob("*.yaml"):
                if config_file.stem in str(args.source):
                    cfg = load_benchmark_cfg(config_file.stem)
                    found = True
                    break
            if not found:
                LOGGER.warning(f"Could not find benchmark config for {cfg_name}. Class filtering might be incorrect.")
                cfg = {}

    if _resolve_eval_box_type(args, cfg) == "obb":
        trackeval_results = trackeval_obb(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)
    else:
        gt_folder = prepare_aabb_eval_gt(args, gt_folder, seq_info)
        trackeval_results = trackeval_aabb(args, seq_paths, save_dir, gt_folder, seq_info=seq_info)

    parsed_results = parse_mot_results(
        trackeval_results,
        seq_names=set(seq_info.keys()),
        known_classes=_known_trackeval_class_names(args, cfg),
    )
    eval_box_type = _resolve_eval_box_type(args, cfg)

    single_class_mode = False
    if eval_box_type == "obb":
        parsed_results, single_class_mode = _filter_obb_trackeval_results(parsed_results, args, cfg.get("benchmark", {}))
    elif getattr(args, "remapped_class_names", None):
        remapped_lower = {name.lower() for name in args.remapped_class_names}
        parsed_results = {key: value for key, value in parsed_results.items() if key.lower() in remapped_lower}
        if len(args.remapped_class_names) == 1:
            single_class_mode = True
    elif "benchmark" in cfg:
        bench_cfg = cfg["benchmark"]
        bench_classes = _ordered_benchmark_eval_class_names(bench_cfg)
        if bench_classes:
            parsed_results = {key: value for key, value in parsed_results.items() if key in bench_classes}
            if len(bench_classes) == 1:
                single_class_mode = True
    elif hasattr(args, "classes") and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        user_classes = [COCO_CLASSES[int(index)] for index in class_indices]
        parsed_results = {key: value for key, value in parsed_results.items() if key in user_classes}
        if len(user_classes) == 1:
            single_class_mode = True

    final_results = list(parsed_results.values())[0] if single_class_mode and parsed_results else parsed_results

    if verbose:
        LOGGER.info("")
        primary_keys, aggregate_keys = _summary_sort_keys(parsed_results, args, cfg)
        single_sequence = len(seq_info) == 1

        all_names = [_display_summary_name(name) for name in [*primary_keys, *aggregate_keys]]
        for class_metrics in parsed_results.values():
            all_names.extend(class_metrics.get("per_sequence", {}).keys())
        all_names.extend([f"COMBINED ({_display_summary_name(name)})" for name in primary_keys])

        name_width = max(18, max((len(name) for name in all_names), default=18) + 2)
        total_width = name_width + 1 + 76

        LOGGER.opt(colors=True).info("<blue>" + "=" * total_width + "</blue>")
        LOGGER.opt(colors=True).info(f"<bold><cyan>{'📊 RESULTS SUMMARY':^{total_width}}</cyan></bold>")
        LOGGER.opt(colors=True).info("<blue>" + "=" * total_width + "</blue>")

        if len(primary_keys) > 1:
            class_rows = [(_display_summary_name(name), parsed_results[name], False) for name in primary_keys]
            _print_summary_table("Per-Class Combined Metrics", "Class", class_rows, total_width, name_width)

            if aggregate_keys:
                aggregate_rows = [(_display_summary_name(name), parsed_results[name], False) for name in aggregate_keys]
                _print_summary_table("Aggregate Groups", "Group", aggregate_rows, total_width, name_width)

            if not single_sequence:
                for class_name in primary_keys:
                    per_sequence_rows = [
                        (seq_name, seq_metrics, False)
                        for seq_name, seq_metrics in sorted(parsed_results[class_name].get("per_sequence", {}).items())
                    ]
                    per_sequence_rows.append(
                        (f"COMBINED ({_display_summary_name(class_name)})", parsed_results[class_name], True)
                    )
                    _print_summary_table(
                        f"Per-Sequence Details: {_display_summary_name(class_name)}",
                        "Sequence",
                        per_sequence_rows,
                        total_width,
                        name_width,
                    )
        else:
            detail_keys = primary_keys or aggregate_keys or list(parsed_results.keys())
            for class_name in detail_keys:
                per_sequence_rows = [
                    (seq_name, seq_metrics, False)
                    for seq_name, seq_metrics in sorted(parsed_results[class_name].get("per_sequence", {}).items())
                ]
                if not single_sequence or not per_sequence_rows:
                    per_sequence_rows.append(
                        (f"COMBINED ({_display_summary_name(class_name)})", parsed_results[class_name], True)
                    )
                _print_summary_table(
                    _display_summary_name(class_name),
                    "Sequence",
                    per_sequence_rows,
                    total_width,
                    name_width,
                )

    if getattr(args, "ci", False):
        with open(args.tracking_method + "_output.json", "w") as outfile:
            outfile.write(json.dumps(final_results))

    return final_results


def eval_setup(args) -> None:
    """
    Common setup for eval and tune pipelines.
    """
    eval_init(args)
    _, _, dataset_detector_cfg = _configure_benchmark_runtime(args)
    det_cfg = get_runtime_detector_cfg(args.yolo_model[0], dataset_detector_cfg)
    apply_class_remap(args, det_cfg)


def apply_class_remap(args, det_cfg: dict) -> None:
    """
    Remap GT class IDs to match detector output.
    """
    bench_cfg: dict = {}
    benchmark_id = (
        getattr(args, "benchmark_id", None)
        or getattr(args, "dataset_id", None)
        or getattr(args, "benchmark", None)
    )
    if benchmark_id:
        try:
            bench_cfg = (load_benchmark_cfg(benchmark_id) or {}).get("benchmark", {})
        except Exception:
            pass

    if str(bench_cfg.get("box_type", "")).lower() == "obb":
        return

    remap_result = build_gt_class_remap(
        bench_cfg,
        det_cfg,
        benchmark_name=getattr(args, "benchmark", ""),
        model_stem=args.yolo_model[0].stem,
    )
    if remap_result is not None:
        remap_dict, new_class_ids, new_class_names = remap_result
        distractor_ids = [int(key) for key in bench_cfg.get("distractor_classes", {}).keys()]
        args.gt_class_remap = remap_dict
        args.gt_class_distractor_ids = distractor_ids
        args.remapped_class_ids = new_class_ids
        args.remapped_class_names = [name.lower() for name in new_class_names]


def main(args):
    args.yolo_model = [resolve_model_path(model) for model in args.yolo_model]
    args.reid_model = [resolve_model_path(model) for model in args.reid_model]

    LOGGER.opt(colors=True).info("<cyan>[1/4]</cyan> Setting up TrackEval...")
    eval_setup(args)

    LOGGER.info("")
    LOGGER.opt(colors=True).info("<blue>" + "=" * 60 + "</blue>")
    LOGGER.opt(colors=True).info("<bold><cyan>🚀 BoxMOT Evaluation Pipeline</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "=" * 60 + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold>Detector:</bold>  <cyan>{args.yolo_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>ReID:</bold>      <cyan>{args.reid_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Tracker:</bold>   <cyan>{args.tracking_method}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Benchmark:</bold> <cyan>{args.source}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Image size:</bold> <cyan>{getattr(args, 'imgsz', None)}</cyan>")
    LOGGER.opt(colors=True).info("<blue>" + "=" * 60 + "</blue>")

    timing_stats = TimingStats()

    LOGGER.opt(colors=True).info("<cyan>[2/4]</cyan> Generating detections and embeddings...")
    run_generate_dets_embs(args, timing_stats=timing_stats)

    LOGGER.opt(colors=True).info("<cyan>[3/4]</cyan> Running tracker...")
    run_generate_mot_results(args, timing_stats=timing_stats)

    LOGGER.opt(colors=True).info("<cyan>[4/4]</cyan> Evaluating results...")
    results = run_trackeval(args)

    if timing_stats.frames > 0:
        timing_stats.print_summary()

    plot_class, metrics_data = _select_plot_metrics_data(results)
    if metrics_data:
        plotter = MetricsPlotter(args.exp_dir)
        plot_metrics = ["HOTA", "MOTA", "IDF1"]
        plot_values = [metrics_data.get(metric, 0) for metric in plot_metrics]

        plotter.plot_radar_chart(
            {args.tracking_method: plot_values},
            plot_metrics,
            title=f"MOT metrics radar Chart ({plot_class})",
            ylim=(0, 100),
            yticks=[20, 40, 60, 80, 100],
            ytick_labels=["20", "40", "60", "80", "100"],
        )


if __name__ == "__main__":
    main()
