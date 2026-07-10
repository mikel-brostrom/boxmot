# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import queue
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from boxmot.data import MOTDataset
from boxmot.detectors import default_conf
from boxmot.engine.tracking.mot import write_mot_results
from boxmot.engine.tracking.runtime import TrackerRuntime
from boxmot.native import get_native_replay_backend
from boxmot.trackers.specs import normalize_tracker_backend
from boxmot.utils import configure_logging as _base_configure_logging
from boxmot.utils import logger as LOGGER
from boxmot.utils.callbacks import safe_progress_callback
from boxmot.utils.misc import increment_path
from boxmot.utils.rich.core.ui import print_text
from boxmot.utils.timing import TimingStats
from boxmot.utils.torch_utils import select_device

__all__ = (
    "process_sequence",
    "run_generate_mot_results",
)


def _configure_logging(*, main_thread_only: bool = False):
    return _base_configure_logging(main_only=True, main_thread_only=main_thread_only)


def _worker_init() -> None:
    _configure_logging(main_thread_only=True)


def _format_seq_progress(sequence_names: list[str], seq_progress: dict) -> str:
    """Format progress bars for all sequences in submission order.

    When there are more than 10 sequences, the output is arranged in two
    columns so the panel doesn't grow excessively tall.
    """
    if not sequence_names:
        return ""

    name_width = max(len(name) for name in sequence_names)
    bar_width = 20

    def _seq_line(name: str) -> str:
        counts = seq_progress.get(name)
        if counts is None:
            bar = "\u2591" * bar_width
            return f"  {name:<{name_width}s} {bar}    --  (pending)"
        current, total = counts
        # Sentinel (-1, 0) means "in progress" (no percentage known)
        if current < 0:
            filled = bar_width // 4
            bar = "\u2593" * filled + "\u2591" * (bar_width - filled)
            return f"  {name:<{name_width}s} {bar}    --  (processing)"
        if total <= 0:
            pct = 1.0 if current >= total else 0.0
        else:
            pct = min(max(current / total, 0.0), 1.0)
        filled = int(bar_width * pct)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        suffix = "(done)" if total > 0 and current >= total else f"({current}/{total})"
        return f"  {name:<{name_width}s} {bar} {pct:>5.0%}  {suffix}"

    lines = [_seq_line(name) for name in sequence_names]

    # Use two columns when there are many sequences
    if len(lines) > 10:
        mid = (len(lines) + 1) // 2
        col1 = lines[:mid]
        col2 = lines[mid:]
        col_width = max(len(line) for line in lines)
        merged = []
        for i in range(mid):
            left = col1[i].ljust(col_width)
            right = col2[i] if i < len(col2) else ""
            merged.append(f"{left}  {right}")
        return "\n".join(merged)

    return "\n".join(lines)


def _drain_progress_queue(progress_queue, seq_progress: dict) -> None:
    """Read all available messages from a progress queue."""
    while True:
        try:
            name, current, total = progress_queue.get_nowait()
            seq_progress[name] = (current, total)
        except (queue.Empty, OSError):
            break


def _build_task_args(
    args: argparse.Namespace,
    exp_dir: Path,
    sequence_names: list[str],
    evolve_config: dict | None,
    conf_threshold: float,
    cache_project_root: str,
    progress_queue,
) -> list[tuple]:
    from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS
    preprocess_name = getattr(args, "reid_preprocess", None) or DEFAULT_PREPROCESS
    masks_dir = getattr(args, "masks_dir", None)
    kf_tuning = getattr(args, "kf_tuning", None)
    adaptive_kf = getattr(args, "adaptive_kf", False)
    return [
        (
            seq_name,
            str(args.source),
            cache_project_root,
            args.detector[0].stem,
            str(args.reid[0]) if args.reid else "",
            args.tracker,
            str(exp_dir),
            getattr(args, "fps", None),
            evolve_config,
            getattr(args, "benchmark", None),
            conf_threshold,
            preprocess_name,
            getattr(args, "split", None),
            masks_dir,
            kf_tuning,
            progress_queue,
            adaptive_kf,
        )
        for seq_name in sequence_names
    ]


def _apply_kf_tuning_to_runtime(kf_tuning: dict) -> None:
    """Apply tuned KF noise weights to the appropriate Kalman filter class.

    This patches class-level defaults so all new KF instances in this process
    use the tuned values.
    """
    kf_type = kf_tuning.get("kf_type")
    std_pos = kf_tuning.get("std_weight_position")
    std_vel = kf_tuning.get("std_weight_velocity")

    if kf_type in ("xywh", "xyah", "xysr"):
        from boxmot.motion.kalman_filters.base import BaseKalmanFilter
        if std_pos is not None:
            BaseKalmanFilter._tuned_std_weight_position = std_pos
        if std_vel is not None:
            BaseKalmanFilter._tuned_std_weight_velocity = std_vel
        LOGGER.info(
            f"KF tuning applied to BaseKalmanFilter: "
            f"_std_weight_position={std_pos}, _std_weight_velocity={std_vel}"
        )
    elif kf_type == "xyhr":
        from boxmot.motion.kalman_filters.xyhr import ConstantNoiseXYHR
        Q = kf_tuning.get("Q")
        R = kf_tuning.get("R")
        Q_vel_diag = kf_tuning.get("Q_vel_diag")

        # Build the per-class noise registry.
        # Key -1 = global (fallback for classes not explicitly tuned).
        registry: dict[int, dict] = {}

        # Global tuned noise → key -1
        if Q is not None:
            _raw_q = np.asarray(Q, dtype=float)
            _tuned_q_pos_diag = np.abs(np.diag(_raw_q)[:_raw_q.shape[0] // 2])
            if Q_vel_diag is not None:
                _tuned_q_vel_diag = np.abs(np.asarray(Q_vel_diag, dtype=float))
            else:
                _tuned_q_vel_diag = _tuned_q_pos_diag * 0.01
            global_entry: dict = {"q_pos_diag": _tuned_q_pos_diag, "q_vel_diag": _tuned_q_vel_diag}
            if R is not None:
                global_entry["r_diag"] = np.abs(np.diag(np.asarray(R, dtype=float)))
            registry[-1] = global_entry

        # Per-class noise → specific class keys
        per_class_data = kf_tuning.get("per_class")
        if per_class_data:
            for cls_id_key, cls_noise in per_class_data.items():
                cls_id = int(cls_id_key)
                cls_Q = cls_noise.get("Q")
                cls_R = cls_noise.get("R")
                cls_Q_vel = cls_noise.get("Q_vel_diag")
                entry: dict = {}
                if cls_Q is not None:
                    raw_q = np.asarray(cls_Q, dtype=float)
                    dim_z_cls = raw_q.shape[0] // 2
                    entry["q_pos_diag"] = np.abs(np.diag(raw_q)[:dim_z_cls])
                    if cls_Q_vel is not None:
                        entry["q_vel_diag"] = np.abs(np.asarray(cls_Q_vel, dtype=float))
                    else:
                        entry["q_vel_diag"] = entry["q_pos_diag"] * 0.01
                if cls_R is not None:
                    raw_r = np.asarray(cls_R, dtype=float)
                    entry["r_diag"] = np.abs(np.diag(raw_r))
                if entry:
                    registry[cls_id] = entry

        ConstantNoiseXYHR._per_class_noise = registry
        n_classes = len([k for k in registry if k >= 0])
        LOGGER.info(
            f"KF tuning applied to ConstantNoiseXYHR: "
            f"global Q/R + {n_classes} per-class entries"
        )


def process_sequence(
    seq_name: str,
    mot_root: str,
    project_root: str,
    detector_name: str,
    reid_name: str,
    tracker_name: str,
    exp_folder: str,
    target_fps: Optional[int],
    cfg_dict: dict | None = None,
    dataset_name: Optional[str] = None,
    conf_threshold: float = 0.0,
    preprocess_name: Optional[str] = None,
    split: Optional[str] = None,
    masks_dir: Optional[str] = None,
    kf_tuning: dict | None = None,
    progress_queue=None,
    adaptive_kf: bool = False,
):
    """Run a tracker over cached detections and embeddings for one sequence."""
    from boxmot.data.cache import reid_cache_key

    detector_key = Path(detector_name).stem if Path(detector_name).suffix else str(detector_name)
    if reid_name:
        reid_weights = Path(reid_name)
        reid_key = reid_cache_key(reid_name, tracker_backend=None)
        if not reid_weights.suffix:
            reid_weights = reid_weights.with_suffix(".pt")
    else:
        reid_weights = None
        reid_key = None
    precomputed_reid = reid_key is not None

    timing_stats = TimingStats()

    tracker_runtime = TrackerRuntime.create(
        tracker_name=tracker_name,
        reid_weights=reid_weights,
        device=select_device("cpu"),
        half=False,
        per_class=False,
        evolve_param_dict=cfg_dict,
        timing_stats=timing_stats,
        precomputed_reid=precomputed_reid,
    )

    # Apply CLI --adaptive-kf override
    if adaptive_kf and hasattr(tracker_runtime.tracker, "adaptive_kf"):
        tracker_runtime.tracker.adaptive_kf = True

    # Apply KF tuning if provided
    if kf_tuning:
        _apply_kf_tuning_to_runtime(kf_tuning)

    # Trackers with camera motion compensation need real image data
    tracker_obj = tracker_runtime.tracker
    needs_images = hasattr(tracker_obj, "cmc") and tracker_obj.cmc is not None
    needs_precomputed_reid = (
        precomputed_reid
        and bool(getattr(tracker_obj, "with_reid", True))
        and not bool(getattr(tracker_obj, "embedding_off", False))
    )

    # Detect whether real frame files are available for this sequence
    _seq_img_dir = Path(mot_root) / seq_name / "img1"
    _has_real_frames = _seq_img_dir.is_dir() and any(_seq_img_dir.iterdir())

    det_emb_root = Path(project_root) / "dets_n_embs"
    if dataset_name:
        det_emb_root = det_emb_root / dataset_name
    if split:
        det_emb_root = det_emb_root / split
    dataset = MOTDataset(
        mot_root=mot_root,
        det_emb_root=str(det_emb_root),
        model_name=detector_key,
        reid_name=reid_key,
        target_fps=target_fps,
        reid_preprocess=preprocess_name,
        masks_dir=masks_dir,
    )
    sequence = dataset.get_sequence(
        seq_name,
        show_progress=False,
        progress_queue=progress_queue,
        skip_image_load=not needs_images,
    )

    all_tracks = []
    kept_frame_ids = []
    total_track_time_ms = 0.0
    total_reid_time_ms = 0.0
    num_frames = 0

    for frame in sequence:
        frame_id = int(frame["frame_id"])
        dets = frame["dets"]
        embs = frame["embs"]
        img = frame["img"]
        masks = frame.get("masks")

        # Masks are passed at their stored resolution (e.g. 640x640).
        # The tracker handles coordinate scaling internally.

        kept_frame_ids.append(frame_id)
        num_frames += 1

        if dets.size and conf_threshold > 0:
            conf_col = 5 if dets.shape[1] == 7 else 4
            mask = dets[:, conf_col] >= conf_threshold
            dets = dets[mask]
            embs = embs[mask] if embs.size else embs
            if masks is not None:
                masks = masks[mask]

        if dets.size:
            if embs.size and dets.shape[0] != embs.shape[0]:
                message = (
                    f"Detection/embedding count mismatch for {seq_name} frame {frame_id}: "
                    f"dets={dets.shape[0]} embs={embs.shape[0]}"
                )
                LOGGER.error(message)
                raise ValueError(message)

            embs_arg = embs if embs.size else None
            if embs_arg is None and needs_precomputed_reid:
                raise ValueError(
                    f"Cached ReID embeddings are missing for {seq_name} frame {frame_id}; "
                    "regenerate the detection/embedding cache before replay."
                )
            # When running on metadata-only sequences (no real frame files),
            # never pass None embeddings — that would trigger live ReID on
            # blank stub images, which is extremely slow and meaningless.
            if embs_arg is None and not _has_real_frames:
                embs_arg = np.zeros((dets.shape[0], 0), dtype=np.float32)
            masks_arg = masks if (masks is not None and masks.size) else None
            try:
                tracks, elapsed_ms = tracker_runtime.update(dets, img, embs_arg, masks=masks_arg)
            except Exception as exc:
                LOGGER.warning(
                    f"Tracker update failed on {seq_name} frame {frame_id}: {exc}"
                )
                continue
            frame_reid_time_ms = min(timing_stats.get_last_reid_time(), elapsed_ms)
            total_reid_time_ms += frame_reid_time_ms
            total_track_time_ms += max(elapsed_ms - frame_reid_time_ms, 0.0)

            if tracks.size:
                all_tracks.append(TrackerRuntime.format_for_mot(tracks, frame_id))

    # Flush Online GTA: append gap-fill entries (if tracker supports it)
    tracker_obj = tracker_runtime.tracker
    if hasattr(tracker_obj, "flush_gta"):
        gta_entries = tracker_obj.flush_gta()
        if gta_entries.size:
            all_tracks.append(gta_entries)

    out_arr = np.vstack(all_tracks) if all_tracks else np.empty((0, 0))
    write_mot_results(Path(exp_folder) / f"{seq_name}.txt", out_arr)

    timing_dict = {
        "track_time_ms": total_track_time_ms,
        "reid_time_ms": total_reid_time_ms,
        "num_frames": num_frames,
    }
    return seq_name, kept_frame_ids, timing_dict


def _resolve_backend_selection(args: argparse.Namespace) -> tuple[str, str]:
    tracking_backend = str(getattr(args, "tracking_backend", "process")).strip().lower() or "process"
    explicit_tracker_backend = getattr(args, "tracker_backend", None)

    if tracking_backend == "cpp":
        if explicit_tracker_backend not in {None, ""}:
            normalized_tracker_backend = normalize_tracker_backend(explicit_tracker_backend)
            if normalized_tracker_backend != "cpp":
                raise ValueError(
                    "tracking_backend='cpp' conflicts with tracker_backend='python'. "
                    "Use tracking_backend='thread' or 'process' when tracker_backend='python'."
                )
        return "cpp", "thread"

    if tracking_backend not in {"process", "thread"}:
        raise ValueError(
            f"Unsupported tracking backend '{tracking_backend}'. Expected 'process', 'thread', or 'cpp'."
        )

    return normalize_tracker_backend(explicit_tracker_backend, default="python"), tracking_backend


def _run_tracking_tasks(
    args: argparse.Namespace,
    task_args: list[tuple],
    *,
    quiet: bool,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, list[int]], float, float, int]:
    n_seqs = len(task_args)
    seq_frame_nums: dict[str, list[int]] = {}
    total_track_time_ms = 0.0
    total_reid_time_ms = 0.0
    total_track_frames = 0
    done_count = 0
    seq_progress: dict = {}
    prev_display_lines = 0
    last_progress_message = None
    sequence_names = [task[0] for task in task_args]
    tracker_backend, tracking_backend = _resolve_backend_selection(args)

    if tracker_backend == "cpp":
        native_backend = get_native_replay_backend(getattr(args, "tracker", ""))
        return _run_cpp_tracking_tasks(
            args,
            task_args,
            quiet=quiet,
            sequence_names=sequence_names,
            native_backend=native_backend,
            progress_callback=progress_callback,
        )

    def _log_progress(progress_queue) -> None:
        nonlocal prev_display_lines, last_progress_message
        _drain_progress_queue(progress_queue, seq_progress)
        header = f"Tracking: {done_count}/{n_seqs} sequences done"
        seq_display = _format_seq_progress(sequence_names, seq_progress)
        message = "\n".join([header] + ([seq_display] if seq_display else []))
        if message == last_progress_message:
            return
        if progress_callback is not None:
            progress_callback(message)
            last_progress_message = message
            return
        if prev_display_lines > 0:
            sys.stderr.write(f"\033[{prev_display_lines}A\033[J")
            sys.stderr.flush()
        print_text(message, stderr=True)
        prev_display_lines = message.count("\n") + 1
        last_progress_message = message

    _configure_logging(main_thread_only=True)

    def _run_executor(executor, progress_queue) -> None:
        nonlocal total_track_time_ms, total_reid_time_ms, total_track_frames, done_count

        futures = {executor.submit(process_sequence, *task_arg): task_arg[0] for task_arg in bound_task_args}
        pending = set(futures)
        first_error: BaseException | None = None
        failed_seqs: list[str] = []

        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=0.3,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                seq_name = futures[future]
                try:
                    sequence_name, kept_ids, timing_dict = future.result()
                    seq_frame_nums[sequence_name] = kept_ids
                    total_track_time_ms += timing_dict.get("track_time_ms", 0.0)
                    total_reid_time_ms += timing_dict.get("reid_time_ms", 0.0)
                    total_track_frames += timing_dict.get("num_frames", 0)
                    num_frames = int(timing_dict.get("num_frames", 0))
                    seq_progress[sequence_name] = (num_frames, num_frames)
                    done_count += 1
                except Exception as exc:
                    done_count += 1
                    failed_seqs.append(seq_name)
                    if progress_callback is None:
                        LOGGER.exception(f"Error processing {seq_name}")
                    if first_error is None:
                        first_error = exc

            if progress_queue is not None:
                _log_progress(progress_queue)

        if first_error is not None:
            raise RuntimeError(
                f"Tracking failed for {len(failed_seqs)} sequence(s): "
                f"{', '.join(failed_seqs)}"
            ) from first_error

    if tracking_backend == "process":
        spawn_context = mp.get_context("spawn")
        manager_context = spawn_context.Manager() if not quiet else nullcontext()

        with manager_context as manager:
            progress_queue = None if quiet else manager.Queue()
            bound_task_args = (
                task_args
                if progress_queue is None
                else [task[:-2] + (progress_queue, task[-1]) for task in task_args]
            )

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.n_threads,
                initializer=_worker_init,
                mp_context=spawn_context,
            ) as executor:
                _run_executor(executor, progress_queue)
    else:
        progress_queue = None if quiet else queue.Queue()
        bound_task_args = (
            task_args
            if progress_queue is None
            else [task[:-2] + (progress_queue, task[-1]) for task in task_args]
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_threads) as executor:
            _run_executor(executor, progress_queue)

    if progress_queue is not None and prev_display_lines > 0:
        _drain_progress_queue(progress_queue, seq_progress)
        final_display = _format_seq_progress(sequence_names, seq_progress)
        final_message = "\n".join(
            [f"Tracking: {n_seqs}/{n_seqs} sequences done"] + ([final_display] if final_display else [])
        )
        if progress_callback is not None:
            progress_callback(final_message)
        else:
            sys.stderr.write(f"\033[{prev_display_lines}A\033[J")
            sys.stderr.flush()
            print_text(final_message, stderr=True)

    return seq_frame_nums, total_track_time_ms, total_reid_time_ms, total_track_frames


def _run_cpp_tracking_tasks(
    args: argparse.Namespace,
    task_args: list[tuple],
    *,
    quiet: bool,
    sequence_names: list[str],
    native_backend,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, list[int]], float, float, int]:
    n_seqs = len(task_args)
    seq_frame_nums: dict[str, list[int]] = {}
    total_track_time_ms = 0.0
    total_reid_time_ms = 0.0
    total_track_frames = 0
    done_count = 0
    seq_progress: dict[str, tuple[int, int]] = {}
    prev_display_lines = 0
    last_progress_message = None

    progress_queue = None if quiet else queue.Queue()
    bound_task_args = (
        task_args
        if progress_queue is None
        else [task[:-1] + (progress_queue,) for task in task_args]
    )

    def _log_progress() -> None:
        nonlocal prev_display_lines, last_progress_message
        if progress_queue is not None:
            _drain_progress_queue(progress_queue, seq_progress)
        header = f"Tracking: {done_count}/{n_seqs} sequences done"
        seq_display = _format_seq_progress(sequence_names, seq_progress)
        message = "\n".join([header] + ([seq_display] if seq_display else []))
        if message == last_progress_message:
            return
        if progress_callback is not None:
            progress_callback(message)
            last_progress_message = message
            return
        if prev_display_lines > 0:
            sys.stderr.write(f"\033[{prev_display_lines}A\033[J")
            sys.stderr.flush()
        print_text(message, stderr=True)
        prev_display_lines = message.count("\n") + 1
        last_progress_message = message

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        futures = {
            executor.submit(native_backend.process_sequence, *task_arg): task_arg[0]
            for task_arg in bound_task_args
        }
        pending = set(futures)
        first_error: BaseException | None = None
        failed_seqs: list[str] = []

        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=0.3,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                seq_name = futures[future]
                try:
                    sequence_name, kept_ids, timing_dict = future.result()
                    seq_frame_nums[sequence_name] = kept_ids
                    total_track_time_ms += timing_dict.get("track_time_ms", 0.0)
                    total_reid_time_ms += timing_dict.get("reid_time_ms", 0.0)
                    total_track_frames += timing_dict.get("num_frames", 0)
                    num_frames = int(timing_dict.get("num_frames", len(kept_ids)))
                    seq_progress[sequence_name] = (num_frames, num_frames)
                    done_count += 1
                except Exception as exc:
                    done_count += 1
                    failed_seqs.append(seq_name)
                    if progress_callback is None:
                        LOGGER.exception(f"Error processing {seq_name}")
                    if first_error is None:
                        first_error = exc

            if not quiet:
                _log_progress()

        if first_error is not None:
            raise RuntimeError(
                f"Tracking failed for {len(failed_seqs)} sequence(s): "
                f"{', '.join(failed_seqs)}"
            ) from first_error

    if not quiet and prev_display_lines > 0:
        if progress_queue is not None:
            _drain_progress_queue(progress_queue, seq_progress)
        final_display = _format_seq_progress(sequence_names, seq_progress)
        final_message = "\n".join(
            [f"Tracking: {n_seqs}/{n_seqs} sequences done"] + ([final_display] if final_display else [])
        )
        if progress_callback is not None:
            progress_callback(final_message)
        else:
            sys.stderr.write(f"\033[{prev_display_lines}A\033[J")
            sys.stderr.flush()
            print_text(final_message, stderr=True)

    return seq_frame_nums, total_track_time_ms, total_reid_time_ms, total_track_frames


def run_generate_mot_results(
    args: argparse.Namespace,
    evolve_config: dict | None = None,
    timing_stats: Optional[TimingStats] = None,
    quiet: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    postprocess_callback: Callable[[str], None] | None = None,
) -> None:
    """Run trackers over cached detections/embeddings and write MOT result files."""
    progress_callback = safe_progress_callback(progress_callback)
    postprocess_callback = safe_progress_callback(postprocess_callback)
    args.project = Path(args.project)
    cache_project = Path(getattr(args, "cache_project", args.project))
    verbose = bool(getattr(args, "verbose", False))
    base = args.project / "mot"
    if getattr(args, "benchmark", None):
        base = base / args.benchmark
    base = base / f"{args.detector[0].stem}_{args.reid[0].stem if args.reid else 'noreid'}_{args.tracker}"
    exp_dir = increment_path(base, sep="_", exist_ok=False)
    exp_dir.mkdir(parents=True, exist_ok=True)
    args.exp_dir = exp_dir

    sequence_names = MOTDataset(
        mot_root=str(args.source),
        seq_pattern=getattr(args, "seq_pattern", None),
    ).sequence_names()
    conf_threshold = getattr(args, "conf", None)
    if conf_threshold is None:
        conf_threshold = default_conf(args.detector[0])

    task_args = _build_task_args(
        args,
        exp_dir,
        sequence_names,
        evolve_config,
        conf_threshold,
        str(cache_project),
        None,
    )
    seq_frame_nums, total_track_time_ms, total_reid_time_ms, total_track_frames = _run_tracking_tasks(
        args,
        task_args,
        quiet=quiet,
        progress_callback=progress_callback,
    )
    args.seq_frame_nums = seq_frame_nums

    # The metrics evaluator expects a per-sequence tracker txt file. When a sequence has
    # no emitted rows (e.g. no usable cached inputs), keep an empty placeholder
    # so evaluation can proceed instead of failing hard on missing files.
    for seq_name in sequence_names:
        seq_result = exp_dir / f"{seq_name}.txt"
        if not seq_result.exists():
            seq_result.touch()

    if timing_stats is not None:
        timing_stats.metadata["detector_from_cache"] = True
        timing_stats.metadata["reid_from_cache"] = True
        timing_stats.totals["track"] += total_track_time_ms
        timing_stats.totals["reid"] += total_reid_time_ms
        if timing_stats.frames == 0 and total_track_frames > 0:
            timing_stats.frames = total_track_frames
        if verbose and total_track_frames > 0:
            avg_track = total_track_time_ms / total_track_frames
            LOGGER.info(
                f"[bold]Tracking:[/bold] {total_track_frames} frames, "
                f"total: [cyan]{total_track_time_ms:.1f}ms[/cyan], "
                f"avg: [cyan]{avg_track:.2f}ms/frame[/cyan]"
            )

    # Parse postprocessing pipeline (comma-separated, applied in order)
    pp_raw = getattr(args, "postprocessing", "none")
    pp_steps = [s.strip().lower() for s in pp_raw.split(",") if s.strip().lower() not in ("none", "")]
    from boxmot.postprocessing import create_postprocessor, supported_postprocessors

    valid_steps = set(supported_postprocessors())
    for s in pp_steps:
        if s not in valid_steps:
            raise ValueError(
                f"Unknown postprocessing step '{s}'. Valid options: {sorted(valid_steps)}"
            )

    # Collect sequence names from result files for postprocessing progress
    pp_seq_names = sorted(f.stem for f in exp_dir.glob("*.txt"))

    for step_idx, pp_step in enumerate(pp_steps, 1):
        step_label = pp_step.upper()
        if len(pp_steps) > 1:
            step_label = f"{step_label} ({step_idx}/{len(pp_steps)})"

        # Build a per-sequence progress callback that formats progress bars.
        # The callback receives per-track progress: (seq_name, current_track, total_tracks).
        def _make_seq_cb(label: str):
            seq_progress: dict[str, tuple[int, int]] = {}
            total_seqs = len(pp_seq_names)

            def _emit() -> None:
                done = sum(
                    1 for c, t in seq_progress.values() if t > 0 and c >= t
                )
                header = f"{label}: {done}/{total_seqs} sequences done"
                seq_display = _format_seq_progress(pp_seq_names, seq_progress)
                message = "\n".join([header] + ([seq_display] if seq_display else []))
                if postprocess_callback is not None:
                    postprocess_callback(message)

            # Send initial state with all sequences as pending
            _emit()

            def _cb(seq_name: str, current: int, total: int) -> None:
                seq_progress[seq_name] = (current, total)
                _emit()
            return _cb

        if pp_step != "gta":
            postprocessor = create_postprocessor(pp_step)
            if verbose:
                LOGGER.info(
                    f"[cyan]\\[3b/4][/cyan] Applying {postprocessor.display_name} postprocessing..."
                )

            postprocessor.run(
                mot_results_folder=exp_dir,
                progress_callback=_make_seq_cb(step_label) if postprocess_callback else None,
            )
            continue

        if pp_step == "gta":
            postprocessor = create_postprocessor(pp_step)
            if verbose:
                LOGGER.info(
                    f"[cyan]\\[3b/4][/cyan] Applying {postprocessor.display_name} postprocessing..."
                )
            from boxmot.data.cache import reid_cache_key
            from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS

            # Resolve cached embeddings/detections directory
            det_emb_root = cache_project / "dets_n_embs"
            if getattr(args, "benchmark", None):
                det_emb_root = det_emb_root / args.benchmark
            if getattr(args, "split", None):
                det_emb_root = det_emb_root / args.split
            detector_key = args.detector[0].stem
            dets_dir = det_emb_root / detector_key / "dets"
            embs_dir = None
            if args.reid:
                preprocess_name = getattr(args, "reid_preprocess", None) or DEFAULT_PREPROCESS
                embs_root = det_emb_root / detector_key / "embs"
                tracker_backend = getattr(args, "tracker_backend", None)

                key = reid_cache_key(args.reid[0], tracker_backend=tracker_backend)
                candidate_dir = embs_root / key / preprocess_name
                if candidate_dir.exists():
                    embs_dir = candidate_dir

                if embs_dir is None:
                    LOGGER.warning(
                        f"GTA: Could not find embedding cache under {embs_root}. "
                        f"Tried key: {key}"
                    )
            else:
                tracker_name = getattr(args, "tracker", "this tracker")
                skip_msg = (
                    f"GTA skipped: '{tracker_name}' has no ReID embeddings.\n"
                    f"Use a ReID tracker (botsort, deepocsort) or remove --postprocessing gta."
                )
                if postprocess_callback is not None:
                    postprocess_callback(skip_msg)
                else:
                    LOGGER.warning(skip_msg)
                continue

            postprocessor = create_postprocessor(
                "gta",
                embs_dir=embs_dir,
                dets_dir=dets_dir if dets_dir.exists() else None,
            )
            postprocessor.run(
                mot_results_folder=exp_dir,
                progress_callback=_make_seq_cb(step_label) if postprocess_callback else None,
            )
