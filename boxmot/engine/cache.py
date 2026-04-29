# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

from boxmot.data.benchmark import configure_benchmark_runtime, load_benchmark_cfg_from_args
from boxmot.data.cache import (
    AppendableNpyWriter,
    _clear_device_cache,
    _count_embedding_rows,
    _existing_cache_path,
    _existing_embedding_cache_path,
    _max_frame_id,
    _migrate_legacy_embedding_cache,
    _migrate_legacy_numeric_cache,
    _read_image_cv2,
    _saved_detection_column_count,
    _serialize_eval_detections,
)
from boxmot.data.dataset import _list_sequence_frames, _sequence_img_dir, _sequence_name_from_img_dir
from boxmot.detectors import default_imgsz
from boxmot.engine.inference import DetectorReIDPipeline, prepare_detections
from boxmot.utils import WEIGHTS, logger as LOGGER
from boxmot.utils.benchmark_config import (
    ensure_benchmark_detector_model,
    ensure_benchmark_reid_model,
    should_use_benchmark_detector,
    should_use_benchmark_reid,
)
from boxmot.utils.misc import prompt_overwrite, resolve_model_path
from boxmot.utils.timing import TimingStats

__all__ = (
    "generate_dets_embs_batched",
    "main",
    "run_generate",
    "run_generate_dets_embs",
)


def _load_benchmark_cfg(args: argparse.Namespace) -> dict:
    return load_benchmark_cfg_from_args(args)


def _configure_benchmark_runtime(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    return configure_benchmark_runtime(
        args,
        load_benchmark_cfg_fn=_load_benchmark_cfg,
        should_use_benchmark_detector_fn=should_use_benchmark_detector,
        should_use_benchmark_reid_fn=should_use_benchmark_reid,
        ensure_benchmark_detector_model_fn=ensure_benchmark_detector_model,
        ensure_benchmark_reid_model_fn=ensure_benchmark_reid_model,
    )


def _ensure_model_list(models) -> list[Path]:
    if isinstance(models, (str, Path)):
        models = [models]
    return [resolve_model_path(model) for model in models]


def _normalize_generate_args(args: argparse.Namespace) -> None:
    args.project = Path(args.project)
    args.detector = _ensure_model_list(args.detector)
    args.reid = _ensure_model_list(args.reid)


def _format_generate_seq_progress(sequence_names: list[str], seq_progress: dict[str, tuple[int, int]]) -> str:
    """Format per-sequence cache generation progress in submission order."""
    if not sequence_names:
        return ""

    name_width = max(len(name) for name in sequence_names)
    lines = []
    bar_width = 20
    for name in sequence_names:
        counts = seq_progress.get(name)
        if counts is None:
            bar = "\u2591" * bar_width
            lines.append(f"  {name:<{name_width}s} {bar}    --  (pending)")
            continue

        current, total = counts
        if total <= 0:
            pct = 1.0 if current >= total else 0.0
        else:
            pct = min(max(current / total, 0.0), 1.0)
        filled = int(bar_width * pct)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        suffix = "(done)" if total > 0 and current >= total else f"({current}/{total})"
        lines.append(f"  {name:<{name_width}s} {bar} {pct:>5.0%}  {suffix}")
    return "\n".join(lines)


def _build_generate_progress_message(
    sequence_names: list[str],
    seq_progress: dict[str, tuple[int, int]],
    processed_frames: int,
    total_frames: int,
) -> str:
    header = f"Generating detections and embeddings: {processed_frames}/{total_frames} frames"
    seq_display = _format_generate_seq_progress(sequence_names, seq_progress)
    return "\n".join([header] + ([seq_display] if seq_display else []))


@torch.inference_mode()
def generate_dets_embs_batched(
    args: argparse.Namespace,
    y: Path,
    source_root: Path,
    timing_stats: Optional[TimingStats] = None,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Generate detections and embeddings in batches for evaluation caches."""
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    verbose = bool(getattr(args, "verbose", False))
    show_progress = bool(getattr(args, "show_progress", True))
    own_terminal_progress = progress_callback is None

    batch_size = int(getattr(args, "batch_size", 16))
    n_threads = int(args.n_threads)
    auto_batch = bool(getattr(args, "auto_batch", True))
    resume = bool(getattr(args, "resume", True))

    if args.imgsz is None:
        args.imgsz = default_imgsz(y)

    expected_det_cols = 8 if str(getattr(args, "eval_box_type", "")).lower() == "obb" else 7

    benchmark = getattr(args, "benchmark", None)
    cache_project = Path(getattr(args, "cache_project", args.project))
    dets_base = cache_project / "dets_n_embs"
    if benchmark:
        dets_base = dets_base / benchmark
    dets_folder = dets_base / y.stem / "dets"
    embs_root = dets_base / y.stem / "embs"
    from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
    preprocess_name = getattr(args, "reid_preprocess", None) or DEFAULT_PREPROCESS

    mot_folder_paths = sorted([path for path in Path(source_root).iterdir() if path.is_dir()])

    seq_states = {}
    det_writers: dict[str, AppendableNpyWriter] = {}
    emb_writers: dict[str, dict[str, AppendableNpyWriter]] = {reid.stem: {} for reid in args.reid}
    total_frames = 0
    initial_done = 0

    for seq_dir in mot_folder_paths:
        img_dir = _sequence_img_dir(seq_dir)
        frames = _list_sequence_frames(img_dir)
        if not frames:
            continue

        seq_name = _sequence_name_from_img_dir(img_dir)

        dets_path = dets_folder / f"{seq_name}.npy"
        cached_dets_path = _existing_cache_path(dets_path)
        processed = 0

        emb_paths = {}
        cached_emb_paths = {}
        any_emb_cached = False
        for reid_model in args.reid:
            emb_path = embs_root / reid_model.stem / preprocess_name / f"{seq_name}.npy"
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
                stem: (
                    _count_embedding_rows(cached_emb_path if cached_emb_path is not None else emb_paths[stem])
                    if (cached_emb_path is not None or emb_paths[stem].exists())
                    else 0
                )
                for stem, cached_emb_path in cached_emb_paths.items()
            }
            expected_files = cached_dets_path is not None and all(
                cached_emb_path is not None for cached_emb_path in cached_emb_paths.values()
            )
            # Treat any divergence between detection rows and *every* ReID embedding
            # row count as a corrupt-cache signal -- including the case where the
            # detection .npy was partially written by a previous interrupted run
            # and the corresponding embedding file is missing or shorter. This
            # prevents downstream native replay from raising
            # "Detection and embedding row counts do not match".
            partial_emb_cache = (
                cached_dets_path is not None
                and det_rows > 0
                and any(
                    emb_rows.get(stem, 0) != det_rows
                    for stem in cached_emb_paths.keys()
                )
            )
            rows_match = (
                len(set([det_rows, *emb_rows.values()])) == 1
                if expected_files
                else False
            )
            schema_match = det_col_count in (0, expected_det_cols)

            if cached_dets_path is not None and not schema_match:
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
                partial_emb_cache = False
            elif partial_emb_cache:
                LOGGER.warning(
                    f"Partial det/emb cache for {seq_name} "
                    f"(det_rows={det_rows}, emb_rows={ {stem: emb_rows.get(stem, 0) for stem in cached_emb_paths.keys()} }); "
                    "resetting cached data."
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
                    if verbose:
                        LOGGER.info(f"Migrated legacy detection cache to {dets_path.name}")
                    cached_dets_path = dets_path
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(f"Failed to migrate detections for {seq_name}: {exc}")
            for stem, cached_emb_path in cached_emb_paths.items():
                if cached_emb_path is None:
                    continue
                try:
                    if _migrate_legacy_embedding_cache(cached_emb_path, emb_paths[stem]):
                        if verbose:
                            LOGGER.info(f"Migrated legacy embedding cache to {emb_paths[stem].name}")
                        cached_emb_paths[stem] = emb_paths[stem]
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(f"Failed to migrate embeddings for {seq_name}/{stem}: {exc}")
            if expected_files and rows_match and det_rows:
                if verbose:
                    LOGGER.info(f"Skipping {seq_name} (cached complete; {processed}/{len(frames)} frames).")
            else:
                if verbose:
                    LOGGER.info(f"Skipping {seq_name} (resume: already complete).")
            initial_done += len(frames)
            continue

        if resume and 0 < processed < len(frames):
            if verbose:
                LOGGER.info(f"Resuming {seq_name}: cached {processed}/{len(frames)} frames.")

        if (not resume) and cached_dets_path is not None and any_emb_cached:
            if not prompt_overwrite("Detections and Embeddings", cached_dets_path, args.ci):
                LOGGER.debug(f"Skipping {seq_name} (cached).")
                continue

        dets_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and cached_dets_path is not None and cached_dets_path.suffix == ".txt":
            try:
                if _migrate_legacy_numeric_cache(cached_dets_path, dets_path, comments="#"):
                    if verbose:
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

        for reid_model in args.reid:
            emb_path = emb_paths[reid_model.stem]
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            cached_emb_path = cached_emb_paths[reid_model.stem]
            if resume and cached_emb_path is not None and cached_emb_path.suffix == ".txt":
                try:
                    if _migrate_legacy_embedding_cache(cached_emb_path, emb_path):
                        if verbose:
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
        if progress_callback is not None:
            progress_callback("No sequences to process (all cached or no images).")
        if verbose:
            LOGGER.info("No sequences to process (all cached or no images).")
        return

    sequence_names = list(seq_states.keys())
    seq_progress = {
        seq_name: (state["i"], len(state["frames"]))
        for seq_name, state in seq_states.items()
    }
    processed_frames = sum(current for current, _ in seq_progress.values())
    last_progress_message = None

    def _report_progress() -> None:
        nonlocal last_progress_message
        if progress_callback is None:
            return
        message = _build_generate_progress_message(
            sequence_names,
            seq_progress,
            processed_frames,
            total_frames,
        )
        if message == last_progress_message:
            return
        progress_callback(message)
        last_progress_message = message

    _report_progress()

    pipeline = DetectorReIDPipeline(
        detector_path=y,
        reid_paths=args.reid,
        device=args.device,
        reid_device=getattr(args, "reid_device", args.device),
        imgsz=args.imgsz,
        half=args.half,
        reid_half=getattr(args, "reid_half", args.half),
        reid_preprocess=getattr(args, "reid_preprocess", None),
        timing_stats=timing_stats,
        tracker_backend=getattr(args, "tracker_backend", None),
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

    pbar = tqdm(
        total=total_frames,
        desc=f"Batched YOLO+ReID ({y.name}, bs={batch_size})",
        unit="frame",
        disable=not (show_progress and own_terminal_progress),
    )
    reid_pbar = tqdm(
        total=0,
        desc="ReID embeddings",
        unit="det",
        dynamic_ncols=True,
        disable=not (show_progress and own_terminal_progress),
    )
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
                            agnostic_nms=getattr(args, "agnostic_nms", False),
                            classes=getattr(args, "classes", None),
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
                if verbose:
                    LOGGER.info(
                        f"YOLO batch frames={len(batch_items)} | dets/frame={det_counts} | total_dets={sum(det_counts)}"
                    )

                for (seq_name, frame_id, _), result, img in zip(batch_items, yolo_results, imgs):
                    dets = prepare_detections(result)

                    if len(dets) == 0:
                        if timing_stats:
                            timing_stats.frames += 1
                        processed_frames += 1
                        seq_progress[seq_name] = (seq_states[seq_name]["i"], len(seq_states[seq_name]["frames"]))
                        pbar.update(1)
                        continue

                    dets_np, det_boxes_np = _serialize_eval_detections(dets, frame_id)

                    # IMPORTANT: compute and append embeddings BEFORE the
                    # detections, so that any failure (ReID error, OOM, killed
                    # process) can never leave the on-disk caches with more
                    # detection rows than embedding rows for this sequence.
                    # The native C++ replay enforces ``det_rows == emb_rows``
                    # and will refuse to load a cache that violates it.
                    all_embs = pipeline.get_all_reid_features(det_boxes_np, img)
                    for reid_name, embs in all_embs.items():
                        if embs.shape[0] != det_boxes_np.shape[0]:
                            raise RuntimeError(
                                f"Embedding count mismatch: dets={det_boxes_np.shape[0]} embs={embs.shape[0]}"
                            )
                        if embs.ndim >= 2 and reid_name not in emb_dims:
                            emb_dims[reid_name] = embs.shape[1]
                        emb_writers[reid_name][seq_name].append(embs.astype(np.float32, copy=False))

                    det_writers[seq_name].append(dets_np.astype(np.float32, copy=False))

                    reid_pbar.update(det_boxes_np.shape[0])

                    if timing_stats:
                        timing_stats.frames += 1

                    processed_frames += 1
                    seq_progress[seq_name] = (seq_states[seq_name]["i"], len(seq_states[seq_name]["frames"]))
                    pbar.update(1)

                _report_progress()

                if verbose:
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


def run_generate_dets_embs(
    args: argparse.Namespace,
    timing_stats: Optional[TimingStats] = None,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Generate detections and embeddings for all sequences."""
    _normalize_generate_args(args)
    verbose = bool(getattr(args, "verbose", False))

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

    for detector in args.detector:
        if verbose:
            LOGGER.info(f"Generating dets+embs (batched single-process): {detector.name}")
        if progress_callback is None:
            generate_dets_embs_batched(
                args,
                detector,
                source_root,
                timing_stats=timing_stats,
            )
        else:
            generate_dets_embs_batched(
                args,
                detector,
                source_root,
                timing_stats=timing_stats,
                progress_callback=progress_callback,
            )


def run_generate(args: argparse.Namespace) -> TimingStats:
    timing_stats = TimingStats()
    run_generate_dets_embs(args, timing_stats=timing_stats)
    return timing_stats


def main(args: argparse.Namespace) -> TimingStats:
    timing_stats = run_generate(args)
    if timing_stats.frames > 0:
        timing_stats.print_summary()
    return timing_stats
