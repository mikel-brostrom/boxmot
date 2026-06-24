# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch

from boxmot.data.benchmark import configure_benchmark_runtime, load_benchmark_cfg_from_args
from boxmot.data.cache import (
    AppendableNpyWriter,
    _clear_device_cache,
    _count_embedding_rows,
    _existing_cache_path,
    _max_frame_id,
    _read_image_cv2,
    _saved_detection_column_count,
    _serialize_eval_detections,
    reid_cache_key,
)
from boxmot.data.dataset import _list_sequence_frames, _sequence_img_dir, _sequence_name_from_img_dir
from boxmot.detectors import default_imgsz
from boxmot.engine.tracking.inference import DetectorReIDPipeline, prepare_detections
from boxmot.utils import WEIGHTS
from boxmot.utils import logger as LOGGER
from boxmot.configs.benchmark import (
    ensure_benchmark_detector_model,
    ensure_benchmark_reid_model,
    should_use_benchmark_detector,
    should_use_benchmark_reid,
)
from boxmot.utils.callbacks import safe_progress_callback
from boxmot.utils.misc import prompt_overwrite, resolve_model_path
from boxmot.utils.rich.generate_reporting import GenerateWorkflowReporter
from boxmot.utils.rich.progress import RichTqdm as tqdm
from boxmot.utils.timing import TimingStats

__all__ = (
    "generate_dets_embs_batched",
    "generate_masks_for_cache",
    "main",
    "run_generate",
    "run_generate_dets_embs",
    "GENERATE_RUN_STEP",
    "GENERATE_SETUP_STEP",
    "GenerateWorkflowReporter",
    "log_generate_pipeline_intro",
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


def _build_reid_only_models(
    reid_paths: list[Path],
    *,
    device: str,
    half: bool,
    preprocess_name: str | None,
    tracker_backend: str | None,
):
    """Instantiate ReID models for the embeddings-only fill loop.

    Mirrors the selection logic used by ``DetectorReIDPipeline._init_reid_models``
    so the cpp ReID C ABI is honoured when ``--tracker-backend cpp`` is set.
    Returns a dict ``{<reid_filename>: <object exposing get_features(boxes, img)>}``.
    """
    models: dict[str, Any] = {}
    use_cpp_reid = (tracker_backend or "").lower() == "cpp"
    cpp_factory = None
    if use_cpp_reid:
        try:
            from boxmot.native.reid import CppOnnxReID
            cpp_factory = CppOnnxReID
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                f"--tracker-backend cpp requested but native ReID C ABI is unavailable: "
                f"{exc}. Falling back to the Python ReID backend for the embeddings-only fill."
            )
            cpp_factory = None
    for reid_path in reid_paths:
        reid_path = Path(reid_path)
        if cpp_factory is not None:
            backend = cpp_factory(weights=reid_path, preprocess_name=preprocess_name)
            models[reid_path.name] = backend.model
        else:
            from boxmot.reid.core import ReID
            backend = ReID(
                weights=reid_path,
                device=device,
                half=half,
                preprocess_name=preprocess_name,
            )
            models[reid_path.name] = backend.model
    return models


def _run_embeddings_only_fill(
    args: argparse.Namespace,
    embed_only_states: dict[str, dict],
    *,
    expected_det_cols: int,
    preprocess_name: str,
    tracker_backend: str | None,
    progress_callback: Callable[[str], None] | None,
    show_progress: bool,
    own_terminal_progress: bool,
    verbose: bool,
) -> None:
    """Compute and append only the missing ReID embedding buckets per sequence.

    The cached detections file is treated as immutable; this loop reads dets
    rows back from disk in stored order, loads each frame's image once, and
    runs only the ReID models that the resume planner flagged as missing.
    Sequences with all-cached buckets never reach this function.
    """
    if not embed_only_states:
        return

    # Resolve the union of (key -> reid_path) across all embed-only sequences,
    # so we instantiate each ReID model at most once.
    key_to_path: dict[str, Path] = {}
    for state in embed_only_states.values():
        for key in state["missing_keys"]:
            if key in key_to_path:
                continue
            for reid_path in args.reid:
                if reid_cache_key(reid_path, tracker_backend=tracker_backend) == key:
                    key_to_path[key] = Path(reid_path)
                    break
    if not key_to_path:
        return

    if verbose:
        LOGGER.info(
            f"Embeddings-only fill: {len(embed_only_states)} sequence(s), "
            f"missing buckets={list(key_to_path.keys())}"
        )

    reid_models = _build_reid_only_models(
        list(key_to_path.values()),
        device=getattr(args, "reid_device", args.device),
        half=getattr(args, "reid_half", getattr(args, "half", False)),
        preprocess_name=preprocess_name,
        tracker_backend=tracker_backend,
    )
    # Map each missing cache key back to its loaded ReID model.
    key_to_model = {
        key: reid_models[Path(reid_path).name] for key, reid_path in key_to_path.items()
    }

    is_obb = expected_det_cols == 8
    box_end = 6 if is_obb else 5  # cols [1:box_end] are the ReID box coords

    total_dets = sum(
        int(np.load(state["dets_path"], mmap_mode="r").shape[0])
        for state in embed_only_states.values()
    )
    pbar = tqdm(
        total=total_dets,
        desc="ReID-only fill",
        unit="det",
        dynamic_ncols=True,
        disable=not (show_progress and own_terminal_progress),
    )
    last_progress_message = None

    def _report(seq_name: str, current: int, total: int) -> None:
        nonlocal last_progress_message
        if progress_callback is None:
            return
        message = f"ReID-only fill: {seq_name} {current}/{total} dets"
        if message == last_progress_message:
            return
        progress_callback(message)
        last_progress_message = message

    try:
        for seq_name, state in embed_only_states.items():
            frames = state["frames"]
            dets_path: Path = state["dets_path"]
            missing_keys: list[str] = state["missing_keys"]
            emb_paths: dict[str, Path] = state["emb_paths"]

            dets_arr = np.load(dets_path).astype(np.float32, copy=False)
            if dets_arr.ndim != 2 or dets_arr.shape[0] == 0:
                continue

            writers: dict[str, AppendableNpyWriter] = {}
            for key in missing_keys:
                emb_path = emb_paths[key]
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                writers[key] = AppendableNpyWriter(
                    emb_path,
                    dtype=np.float32,
                    trailing_shape=None,
                    empty_trailing_shape=(0,),
                )

            try:
                n_rows = dets_arr.shape[0]
                i = 0
                processed = 0
                while i < n_rows:
                    fid = int(dets_arr[i, 0])
                    j = i
                    while j < n_rows and int(dets_arr[j, 0]) == fid:
                        j += 1
                    boxes = dets_arr[i:j, 1:box_end].copy()
                    # frames is 0-indexed; cached frame ids are 1-based.
                    if 1 <= fid <= len(frames):
                        img = _read_image_cv2(frames[fid - 1])
                    else:
                        img = None
                    if img is None:
                        # Should not happen for healthy caches, but guard anyway.
                        i = j
                        continue
                    for key, model in key_to_model.items():
                        feats = model.get_features(boxes, img)
                        feats = np.asarray(feats, dtype=np.float32)
                        if feats.ndim == 1:
                            feats = feats.reshape(1, -1) if feats.size else feats
                        if feats.shape[0] != boxes.shape[0]:
                            raise RuntimeError(
                                f"Embedding count mismatch during fill for {seq_name}/{key}: "
                                f"dets={boxes.shape[0]} embs={feats.shape[0]}"
                            )
                        writers[key].append(feats)
                    processed += boxes.shape[0]
                    pbar.update(boxes.shape[0])
                    _report(seq_name, processed, n_rows)
                    i = j
            finally:
                for key, writer in writers.items():
                    try:
                        writer.close()
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            f"Failed to save filled embeddings for {seq_name}/{key}: {exc}"
                        )
    finally:
        pbar.close()
        # Best-effort release of any cpp/onnxruntime sessions held by reid_models.
        for model in list(reid_models.values()):
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass


def _ensure_public_detector_setup(args: argparse.Namespace, detector: str) -> None:
    """Ensure MOT17 parquet setup has been run for the requested public detector.

    If the source directory already has det/det.txt files, this is a no-op.
    Otherwise triggers the parquet setup to create the MOTChallenge layout
    with the requested detector's detections.
    """
    source_root = Path(args.source)
    # Quick check: if sequences already have det/det.txt, nothing to do
    if source_root.is_dir():
        for seq_dir in source_root.iterdir():
            if seq_dir.is_dir() and (seq_dir / "det" / "det.txt").exists():
                return

    # Run parquet setup
    try:
        from boxmot.utils.mot17_parquet import setup_mot17_from_parquet

        benchmark = getattr(args, "benchmark", None) or "mot17"
        split = getattr(args, "split", None) or "ablation"
        # Resolve the dataset dest from the source path
        # source is usually <root>/<split>, so dest is <root>
        dest = source_root.parent if source_root.name == split else source_root

        setup_mot17_from_parquet(
            dest=dest,
            split=split,
            detector=detector,
        )
    except Exception as e:
        LOGGER.warning(f"Could not set up public detector '{detector}': {e}")


@torch.inference_mode()
def _generate_public_dets_cache(
    args: argparse.Namespace,
    source_root: Path,
    det_key: str = "public",
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Generate detection cache from public det/det.txt files (MOT format).

    Public detections are provided with MOT17/MOT20 sequences in
    ``<seq>/det/det.txt`` with format: frame,id,x,y,w,h,conf,-1,-1,-1.
    This converts them to the standard cache format (frame_id, x1, y1, x2, y2, conf, cls).
    """
    from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
    verbose = bool(getattr(args, "verbose", False))
    resume = bool(getattr(args, "resume", True))
    preprocess_name = getattr(args, "reid_preprocess", None) or DEFAULT_PREPROCESS
    tracker_backend = getattr(args, "tracker_backend", None)

    expected_det_cols = 7  # public dets are always AABB
    benchmark = getattr(args, "benchmark", None)
    split = getattr(args, "split", None)
    cache_project = Path(getattr(args, "cache_project", args.project))
    dets_base = cache_project / "dets_n_embs"
    if benchmark:
        dets_base = dets_base / benchmark
    if split:
        dets_base = dets_base / split
    dets_folder = dets_base / det_key / "dets"
    dets_folder.mkdir(parents=True, exist_ok=True)

    mot_folder_paths = sorted([p for p in source_root.iterdir() if p.is_dir()])
    _seq_pattern = getattr(args, "seq_pattern", None)
    if _seq_pattern:
        from fnmatch import fnmatch
        mot_folder_paths = [p for p in mot_folder_paths if fnmatch(p.name, _seq_pattern)]

    for seq_dir in mot_folder_paths:
        img_dir = _sequence_img_dir(seq_dir)
        frames = _list_sequence_frames(img_dir)
        if not frames:
            continue
        seq_name = _sequence_name_from_img_dir(img_dir)
        dets_path = dets_folder / f"{seq_name}.npy"

        if resume and dets_path.exists():
            if verbose:
                LOGGER.info(f"Skipping public dets for {seq_name} (cached).")
            continue

        det_txt = seq_dir / "det" / "det.txt"
        if not det_txt.exists():
            LOGGER.warning(f"No public detections at {det_txt}, skipping {seq_name}.")
            continue

        # Load MOT format: frame,id,x,y,w,h,conf,-1,-1,-1
        raw = np.loadtxt(det_txt, delimiter=",")
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        # Convert (x, y, w, h) → (x1, y1, x2, y2)
        frame_ids = raw[:, 0].astype(np.int32)
        x, y, w, h = raw[:, 2], raw[:, 3], raw[:, 4], raw[:, 5]
        conf = raw[:, 6] if raw.shape[1] > 6 else np.ones(len(raw))
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        cls = np.zeros(len(raw), dtype=np.float32)

        # Output format: (frame_id, x1, y1, x2, y2, conf, cls)
        dets = np.column_stack([frame_ids, x1, y1, x2, y2, conf, cls]).astype(np.float32)
        np.save(dets_path, dets)
        if verbose:
            LOGGER.info(f"Cached {len(dets)} public detections for {seq_name}.")

    if progress_callback:
        progress_callback("Public detections cached.")


@torch.inference_mode()
def generate_dets_embs_batched(
    args: argparse.Namespace,
    y: Path,
    source_root: Path,
    timing_stats: Optional[TimingStats] = None,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Generate detections and embeddings in batches for evaluation caches."""
    progress_callback = safe_progress_callback(progress_callback)
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
    split = getattr(args, "split", None)
    cache_project = Path(getattr(args, "cache_project", args.project))
    dets_base = cache_project / "dets_n_embs"
    if benchmark:
        dets_base = dets_base / benchmark
    if split:
        dets_base = dets_base / split
    dets_folder = dets_base / y.stem / "dets"
    embs_root = dets_base / y.stem / "embs"
    masks_folder = dets_base / y.stem / "masks" / "seg"
    from boxmot.detectors.registry import is_seg_model
    from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
    preprocess_name = getattr(args, "reid_preprocess", None) or DEFAULT_PREPROCESS

    # Determine if the detector produces masks (seg model)
    _is_seg = is_seg_model(y)

    mot_folder_paths = sorted([path for path in Path(source_root).iterdir() if path.is_dir()])
    # Apply sequence pattern filter if configured (e.g. "*-FRCNN" for public FRCNN detections)
    _seq_pattern = getattr(args, "seq_pattern", None)
    if _seq_pattern:
        from fnmatch import fnmatch
        mot_folder_paths = [p for p in mot_folder_paths if fnmatch(p.name, _seq_pattern)]

    seq_states = {}
    embed_only_states: dict[str, dict] = {}
    cached_seq_names: list[str] = []
    det_writers: dict[str, AppendableNpyWriter] = {}
    mask_writers: dict[str, AppendableNpyWriter] = {}
    tracker_backend = getattr(args, "tracker_backend", None)
    emb_writers: dict[str, dict[str, AppendableNpyWriter]] = {
        reid_cache_key(reid, tracker_backend=tracker_backend): {} for reid in args.reid
    }
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
            key = reid_cache_key(reid_model, tracker_backend=tracker_backend)
            emb_path = embs_root / key / preprocess_name / f"{seq_name}.npy"
            emb_paths[key] = emb_path
            cached_emb_path = _existing_cache_path(emb_path)
            cached_emb_paths[key] = cached_emb_path
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
                # Special case: detections are fully cached and pass the
                # schema check, but one or more ReID buckets are missing
                # or incomplete. Re-running YOLO just to regenerate the
                # detections is wasteful (and would discard a valid cache),
                # so route this sequence through an embeddings-only fill
                # pass that reads the cached dets back from disk and only
                # runs the ReID models for the missing buckets.
                dets_complete = (
                    cached_dets_path is not None
                    and det_rows > 0
                    and schema_match
                    and det_max_frame >= len(frames)
                )
                if dets_complete:
                    missing_keys = [
                        stem
                        for stem in cached_emb_paths.keys()
                        if emb_rows.get(stem, 0) != det_rows
                    ]
                    if missing_keys:
                        if verbose:
                            LOGGER.info(
                                f"Reusing cached detections for {seq_name} "
                                f"(det_rows={det_rows}); regenerating embeddings only for "
                                f"{missing_keys}."
                            )
                        # Drop only the broken/missing emb files so the writers start clean.
                        for stem in missing_keys:
                            for path in [cached_emb_paths.get(stem), emb_paths[stem]]:
                                if path is not None:
                                    try:
                                        path.unlink()
                                    except FileNotFoundError:
                                        pass
                        embed_only_states[seq_name] = {
                            "frames": frames,
                            "img_dir": img_dir,
                            "dets_path": cached_dets_path,
                            "missing_keys": missing_keys,
                            "emb_paths": {k: emb_paths[k] for k in missing_keys},
                        }
                        # Embed-only fill happens after the main loop; book the
                        # frame budget against ``initial_done`` so the main pbar
                        # stays accurate when there are also full-regen seqs.
                        initial_done += len(frames)
                        continue
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
            if verbose:
                LOGGER.info(f"Skipping {seq_name} (cached complete; {processed}/{len(frames)} frames).")
            initial_done += len(frames)
            cached_seq_names.append(seq_name)
            continue

        if resume and 0 < processed < len(frames):
            if verbose:
                LOGGER.info(f"Resuming {seq_name}: cached {processed}/{len(frames)} frames.")

        if (not resume) and cached_dets_path is not None and any_emb_cached:
            if not prompt_overwrite("Detections and Embeddings", cached_dets_path, args.ci):
                LOGGER.debug(f"Skipping {seq_name} (cached).")
                continue

        dets_path.parent.mkdir(parents=True, exist_ok=True)
        det_writers[seq_name] = AppendableNpyWriter(
            dets_path,
            dtype=np.float32,
            trailing_shape=(expected_det_cols,),
            empty_trailing_shape=(expected_det_cols,),
        )

        for reid_model in args.reid:
            key = reid_cache_key(reid_model, tracker_backend=tracker_backend)
            emb_path = emb_paths[key]
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            emb_writers[key][seq_name] = AppendableNpyWriter(
                emb_path,
                dtype=np.float32,
                trailing_shape=None,
                empty_trailing_shape=(0,),
            )

        seq_states[seq_name] = {"frames": frames, "i": processed, "img_dir": img_dir}
        if _is_seg:
            masks_folder.mkdir(parents=True, exist_ok=True)
            mask_writers[seq_name] = AppendableNpyWriter(
                masks_folder / f"{seq_name}.npy",
                dtype=np.uint8,
                trailing_shape=None,
                empty_trailing_shape=(0, 0),
            )
        total_frames += len(frames)
        initial_done += processed

    if not seq_states:
        if embed_only_states:
            _run_embeddings_only_fill(
                args,
                embed_only_states,
                expected_det_cols=expected_det_cols,
                preprocess_name=preprocess_name,
                tracker_backend=tracker_backend,
                progress_callback=progress_callback,
                show_progress=show_progress,
                own_terminal_progress=own_terminal_progress,
                verbose=verbose,
            )
            return
        if progress_callback is not None:
            if cached_seq_names:
                seq_progress = {name: (1, 1) for name in cached_seq_names}
                bars = _format_generate_seq_progress(cached_seq_names, seq_progress)
                progress_callback(
                    f"All {len(cached_seq_names)} sequences loaded from cache\n{bars}"
                )
            else:
                progress_callback("No sequences found.")
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
                        # Pipeline keys embeddings by the raw ReID name; the
                        # writers are bucketed by the backend-aware cache key
                        # (e.g. ``__cpp`` suffix for the C++ backend).
                        writer_key = reid_cache_key(reid_name, tracker_backend=tracker_backend)
                        if embs.ndim >= 2 and writer_key not in emb_dims:
                            emb_dims[writer_key] = embs.shape[1]
                        emb_writers[writer_key][seq_name].append(embs.astype(np.float32, copy=False))

                    det_writers[seq_name].append(dets_np.astype(np.float32, copy=False))

                    # Append masks downsampled + bit-packed (same row order as dets/embs)
                    # Binary masks (N, H, W) → resize to 160×160 → packbits → (N, 160, 20)
                    # 128× smaller than raw storage; IoU ratios are resolution-invariant
                    if _is_seg and result.masks is not None and seq_name in mask_writers:
                        masks_raw = result.masks.astype(np.uint8, copy=False)
                        n = masks_raw.shape[0]
                        masks_small = np.empty((n, 160, 160), dtype=np.uint8)
                        for _mi in range(n):
                            masks_small[_mi] = cv2.resize(
                                masks_raw[_mi], (160, 160),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        packed = np.packbits(masks_small, axis=-1)
                        mask_writers[seq_name].append(packed)

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
        for seq_name, writer in mask_writers.items():
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(f"Failed to save masks for {seq_name}: {exc}")

    # Sequences that already had a complete detections cache but were missing
    # one or more ReID buckets are filled now (post main loop) so we don't
    # rerun the YOLO detector unnecessarily.
    _run_embeddings_only_fill(
        args,
        embed_only_states,
        expected_det_cols=expected_det_cols,
        preprocess_name=preprocess_name,
        tracker_backend=tracker_backend,
        progress_callback=progress_callback,
        show_progress=show_progress,
        own_terminal_progress=own_terminal_progress,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

PERSON_CLASS_ID = 1  # COCO class ID for "person"


def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes. Returns (M, K) matrix."""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def generate_masks_for_cache(
    source_root: Path,
    dets_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    mask_threshold: float = 0.5,
    conf_threshold: float = 0.5,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Generate Mask R-CNN segmentation masks aligned with cached detections.

    For each sequence with a .npy detection cache, runs Mask R-CNN on every
    frame and matches predictions to cached detections via IoU.  Results are
    stored as compressed .npz files: key ``frame_{id}`` -> (N, H, W) uint8.
    """
    import cv2
    import torchvision
    from torchvision.transforms.functional import to_tensor

    seq_names = sorted(p.stem for p in dets_dir.glob("*.npy"))
    if not seq_names:
        LOGGER.warning(f"No detection caches found in {dets_dir}")
        return

    LOGGER.info(f"Generating masks for {len(seq_names)} sequences -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval().to(dev)

    for seq_name in seq_names:
        output_path = output_dir / f"{seq_name}.npz"
        if output_path.exists():
            LOGGER.info(f"  [SKIP] {seq_name} masks already cached")
            continue

        dets_path = dets_dir / f"{seq_name}.npy"
        dets = np.load(str(dets_path))
        frame_ids = np.unique(dets[:, 0].astype(int))

        img_dir = _sequence_img_dir(source_root / seq_name)
        if img_dir is None:
            LOGGER.warning(f"  [SKIP] No image dir for {seq_name}")
            continue

        frames_list = sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if not frames_list:
            frames_list = sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem))
        frame_path_map = {int(p.stem): p for p in frames_list}

        masks_dict: dict[str, np.ndarray] = {}
        total_dets = 0
        total_masks = 0

        for fid in frame_ids:
            img_path = frame_path_map.get(fid)
            if img_path is None:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            frame_mask_sel = dets[:, 0].astype(int) == fid
            frame_dets = dets[frame_mask_sel, 1:]  # (M, 6) x1,y1,x2,y2,conf,cls
            n_dets = frame_dets.shape[0]
            total_dets += n_dets

            if n_dets == 0:
                masks_dict[f"frame_{fid}"] = np.zeros((0, img.shape[0], img.shape[1]), dtype=np.uint8)
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = to_tensor(img_rgb).unsqueeze(0).to(dev)

            with torch.no_grad():
                results = model(img_tensor)[0]

            rcnn_boxes = results["boxes"].cpu().numpy()
            rcnn_scores = results["scores"].cpu().numpy()
            rcnn_labels = results["labels"].cpu().numpy()
            rcnn_masks = results["masks"].cpu().numpy()[:, 0]

            person_sel = (rcnn_labels == PERSON_CLASS_ID) & (rcnn_scores >= conf_threshold)
            rcnn_boxes = rcnn_boxes[person_sel]
            rcnn_masks = rcnn_masks[person_sel]

            frame_masks = np.zeros((n_dets, img.shape[0], img.shape[1]), dtype=np.uint8)

            if len(rcnn_boxes) > 0:
                iou_matrix = _compute_iou_matrix(frame_dets[:, :4], rcnn_boxes)
                for i in range(n_dets):
                    best_j = np.argmax(iou_matrix[i])
                    if iou_matrix[i, best_j] > 0.3:
                        frame_masks[i] = (rcnn_masks[best_j] > mask_threshold).astype(np.uint8)
                        total_masks += 1

            masks_dict[f"frame_{fid}"] = frame_masks

        np.savez_compressed(str(output_path), **masks_dict)
        msg = f"  {seq_name}: {total_masks}/{total_dets} dets matched"
        LOGGER.info(msg)
        if progress_callback:
            progress_callback(msg)


def run_generate_dets_embs(
    args: argparse.Namespace,
    timing_stats: Optional[TimingStats] = None,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Generate detections and embeddings for all sequences."""
    progress_callback = safe_progress_callback(progress_callback)
    _normalize_generate_args(args)
    verbose = bool(getattr(args, "verbose", False))

    if getattr(args, "data", None) and getattr(args, "source", None) is None:
        from boxmot.configs.benchmark import apply_benchmark_config

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

    # Public detections: read from det/det.txt instead of running a detector model
    # Supports "public" (generic) or specific detector names (frcnn, sdp, dpm)
    detection_source = getattr(args, "detection_source", None)
    _public_detector_names = {"public", "frcnn", "sdp", "dpm"}
    if detection_source and detection_source.lower() in _public_detector_names:
        # If a specific public detector is requested, ensure parquet setup ran
        # and sequences have the correct det/det.txt files
        if detection_source.lower() in ("frcnn", "sdp", "dpm"):
            _ensure_public_detector_setup(args, detection_source.upper())
        # Compute det_key early so caches go to the right folder
        det_key = f"mot17_public_{detection_source.lower()}" if detection_source.lower() != "public" else "public"
        _generate_public_dets_cache(args, source_root, det_key=det_key, progress_callback=progress_callback)
        args.detector = [Path(det_key)]

        # Run ReID embeddings fill for the cached public detections
        from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
        _preprocess_name = getattr(args, "reid_preprocess", None) or DEFAULT_PREPROCESS
        _tracker_backend = getattr(args, "tracker_backend", None)
        _benchmark = getattr(args, "benchmark", None)
        _split = getattr(args, "split", None)
        _cache_project = Path(getattr(args, "cache_project", args.project))
        _dets_base = _cache_project / "dets_n_embs"
        if _benchmark:
            _dets_base = _dets_base / _benchmark
        if _split:
            _dets_base = _dets_base / _split
        _dets_folder = _dets_base / det_key / "dets"
        _embs_root = _dets_base / det_key / "embs"

        # Build embed_only_states for sequences that need ReID embeddings
        _mot_folder_paths = sorted([p for p in source_root.iterdir() if p.is_dir()])
        _seq_pattern = getattr(args, "seq_pattern", None)
        if _seq_pattern:
            from fnmatch import fnmatch
            _mot_folder_paths = [p for p in _mot_folder_paths if fnmatch(p.name, _seq_pattern)]

        _embed_only_states: dict[str, dict] = {}
        for _seq_dir in _mot_folder_paths:
            _img_dir = _sequence_img_dir(_seq_dir)
            _frames = _list_sequence_frames(_img_dir)
            if not _frames:
                continue
            _seq_name = _sequence_name_from_img_dir(_img_dir)
            _dets_path = _dets_folder / f"{_seq_name}.npy"
            if not _dets_path.exists():
                continue
            # Check which ReID keys are missing
            _missing_keys: list[str] = []
            _emb_paths: dict[str, Path] = {}
            for _reid_model in args.reid:
                _key = reid_cache_key(_reid_model, tracker_backend=_tracker_backend)
                _emb_path = _embs_root / _key / _preprocess_name / f"{_seq_name}.npy"
                _emb_paths[_key] = _emb_path
                if not _emb_path.exists():
                    _missing_keys.append(_key)
            if _missing_keys:
                _embed_only_states[_seq_name] = {
                    "frames": _frames,
                    "dets_path": _dets_path,
                    "missing_keys": _missing_keys,
                    "emb_paths": {k: _emb_paths[k] for k in _missing_keys},
                }

        if _embed_only_states:
            _run_embeddings_only_fill(
                args,
                _embed_only_states,
                expected_det_cols=7,
                preprocess_name=_preprocess_name,
                tracker_backend=_tracker_backend,
                progress_callback=progress_callback,
                show_progress=bool(getattr(args, "show_progress", True)),
                own_terminal_progress=(progress_callback is None),
                verbose=verbose,
            )
        return

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


def run_generate(
    args: argparse.Namespace,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> TimingStats:
    progress_callback = safe_progress_callback(progress_callback)
    timing_stats = TimingStats()
    run_generate_dets_embs(
        args,
        timing_stats=timing_stats,
        progress_callback=progress_callback,
    )

    # Generate masks when --masks-model is specified (e.g. "maskrcnn")
    # Store at the canonical location: <det_emb_root>/<detector>/masks/<masks_model>/
    masks_model = getattr(args, "masks_model", None)
    masks_dir_override = getattr(args, "masks_dir", None)
    if masks_model or masks_dir_override:
        source_root = Path(args.source)
        project = Path(getattr(args, "cache_project", getattr(args, "project", "runs")))
        det_emb_root = project / "dets_n_embs"
        benchmark = getattr(args, "benchmark", None)
        if benchmark:
            det_emb_root = det_emb_root / benchmark
        split = getattr(args, "split", None)
        if split:
            det_emb_root = det_emb_root / split
        detector_key = args.detector[0].stem if hasattr(args.detector[0], "stem") else str(args.detector[0])
        dets_dir = det_emb_root / detector_key / "dets"

        # Output directory: explicit --masks-dir override, or canonical location
        if masks_dir_override:
            output_dir = Path(masks_dir_override)
        else:
            output_dir = det_emb_root / detector_key / "masks" / masks_model

        device = getattr(args, "device", "cpu")
        generate_masks_for_cache(
            source_root=source_root,
            dets_dir=dets_dir,
            output_dir=output_dir,
            device=str(device),
            progress_callback=progress_callback,
        )

    return timing_stats


def main(args: argparse.Namespace) -> TimingStats:
    pipeline = GenerateWorkflowReporter(args).pipeline()
    toggled = {"done": False}

    def progress_callback(msg: str) -> None:
        if not toggled["done"]:
            pipeline.advance("Generating detections & embeddings...")
            toggled["done"] = True
        pipeline.update(msg)

    verbose = bool(getattr(args, "verbose", False))
    from boxmot.utils.misc import suppress_boxmot_logs

    with pipeline, suppress_boxmot_logs(not verbose, level="WARNING"):
        timing_stats = run_generate(args, progress_callback=progress_callback)

    if timing_stats.frames > 0:
        try:
            summary_text = timing_stats.format_summary()
        except AttributeError:
            summary_text = None
        if summary_text:
            pipeline.update(summary_text)
    pipeline.complete_step()
    return timing_stats
