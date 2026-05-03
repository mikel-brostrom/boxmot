# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

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
    reid_cache_key,
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
from boxmot.utils.download import set_download_status_fn
from boxmot.utils.rich.generate_reporting import (
    GENERATE_RUN_STEP,
    GENERATE_SETUP_STEP,
    GenerateWorkflowReporter,
    log_generate_pipeline_intro,
)
from boxmot.utils.rich.reporting import WorkflowDetailCallback

__all__ = (
    "generate_dets_embs_batched",
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
            from boxmot.native.reid_capi import CppOnnxReID
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
    embed_only_states: dict[str, dict] = {}
    det_writers: dict[str, AppendableNpyWriter] = {}
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
            cached_emb_path = _existing_embedding_cache_path(emb_path)
            # Fall back to the legacy stem-only cache directory so that
            # existing on-disk caches generated before the suffix was added
            # to the key can still be resumed instead of regenerated.
            #
            # The legacy cache predates the per-format suffix split, when
            # ``.pt`` was the only supported runtime. Only trust it when the
            # currently requested weights are also ``.pt`` AND the active
            # tracker backend is the Python one — otherwise a ``.onnx`` (or
            # other-format) request, or any C++ ReID request, would silently
            # consume ``.pt``-generated PyTorch embeddings.
            if (
                cached_emb_path is None
                and reid_model.stem != key
                and reid_model.suffix.lower() == ".pt"
                and (tracker_backend or "python").lower() != "cpp"
            ):
                legacy_path = embs_root / reid_model.stem / preprocess_name / f"{seq_name}.npy"
                cached_emb_path = _existing_embedding_cache_path(legacy_path)
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
                        # Migrate legacy txt dets to the canonical npy path.
                        try:
                            if cached_dets_path.suffix == ".txt" and _migrate_legacy_numeric_cache(
                                cached_dets_path, dets_path, comments="#"
                            ):
                                cached_dets_path = dets_path
                        except Exception as exc:  # noqa: BLE001
                            LOGGER.warning(f"Failed to migrate detections for {seq_name}: {exc}")
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
            key = reid_cache_key(reid_model, tracker_backend=tracker_backend)
            emb_path = emb_paths[key]
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            cached_emb_path = cached_emb_paths[key]
            if resume and cached_emb_path is not None and cached_emb_path.suffix == ".txt":
                try:
                    if _migrate_legacy_embedding_cache(cached_emb_path, emb_path):
                        if verbose:
                            LOGGER.info(f"Migrated legacy embedding cache to {emb_path.name}")
                        cached_emb_path = emb_path
                except Exception:
                    pass
            emb_writers[key][seq_name] = AppendableNpyWriter(
                emb_path,
                dtype=np.float32,
                trailing_shape=None,
                empty_trailing_shape=(0,),
            )

        seq_states[seq_name] = {"frames": frames, "i": processed, "img_dir": img_dir}
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
                        # Pipeline keys embeddings by the raw ReID name; the
                        # writers are bucketed by the backend-aware cache key
                        # (e.g. ``__cpp`` suffix for the C++ backend).
                        writer_key = reid_cache_key(reid_name, tracker_backend=tracker_backend)
                        if embs.ndim >= 2 and writer_key not in emb_dims:
                            emb_dims[writer_key] = embs.shape[1]
                        emb_writers[writer_key][seq_name].append(embs.astype(np.float32, copy=False))

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


def run_generate(
    args: argparse.Namespace,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> TimingStats:
    timing_stats = TimingStats()
    run_generate_dets_embs(
        args,
        timing_stats=timing_stats,
        progress_callback=progress_callback,
    )
    return timing_stats


def main(args: argparse.Namespace) -> TimingStats:
    workflow = log_generate_pipeline_intro(args)
    run_callback = WorkflowDetailCallback(workflow, GENERATE_RUN_STEP)
    setup_callback = WorkflowDetailCallback(workflow, GENERATE_SETUP_STEP)
    set_download_status_fn(setup_callback)
    has_setup = GENERATE_SETUP_STEP in getattr(workflow, "steps", ())
    toggled = {"done": not has_setup}

    def progress_callback(msg: str) -> None:
        if not toggled["done"]:
            workflow.complete(GENERATE_SETUP_STEP, render=False)
            workflow.activate(GENERATE_RUN_STEP, render=False)
            toggled["done"] = True
        run_callback(msg)

    verbose = bool(getattr(args, "verbose", False))
    try:
        from boxmot.engine.workflow_support import suppress_boxmot_logs

        with suppress_boxmot_logs(not verbose, level="WARNING"):
            timing_stats = run_generate(args, progress_callback=progress_callback)
    except BaseException as exc:
        workflow.fail(error=exc)
        raise
    else:
        if timing_stats.frames > 0:
            try:
                summary_text = timing_stats.format_summary()
            except AttributeError:
                summary_text = None
            if summary_text:
                workflow.set_detail(GENERATE_RUN_STEP, summary_text, render=False)
        workflow.complete(GENERATE_RUN_STEP, render=False)
        return timing_stats
    finally:
        set_download_status_fn(None)
        workflow.stop()
