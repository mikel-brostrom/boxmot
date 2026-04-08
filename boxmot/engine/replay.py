# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import queue
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from boxmot.data import MOTDataset
from boxmot.detectors import default_conf
from boxmot.engine.tracker import TrackerRuntime
from boxmot.utils import configure_logging as _base_configure_logging, logger as LOGGER
from boxmot.utils.misc import increment_path
from boxmot.utils.timing import TimingStats
from boxmot.utils.mot_utils import write_mot_results
from boxmot.utils.torch_utils import select_device

__all__ = (
    "process_sequence",
    "run_generate_mot_results",
)


def _configure_logging(*, main_thread_only: bool = False):
    return _base_configure_logging(main_only=main_thread_only)


def _worker_init() -> None:
    _configure_logging(main_thread_only=True)


def _format_seq_progress(seq_progress: dict, n_display: int = 5) -> str:
    """Format top-N in-progress sequences as mini progress bars."""
    active = {name: counts for name, counts in seq_progress.items() if counts[0] < counts[1]}
    if not active:
        return ""

    sorted_seqs = sorted(active.items(), key=lambda item: item[1][0] / max(item[1][1], 1), reverse=True)
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
    progress_queue,
) -> list[tuple]:
    return [
        (
            seq_name,
            str(args.source),
            str(args.project),
            args.yolo_model[0].stem,
            str(args.reid_model[0]),
            args.tracking_method,
            str(exp_dir),
            getattr(args, "fps", None),
            getattr(args, "device", None),
            evolve_config,
            getattr(args, "benchmark", None),
            conf_threshold,
            progress_queue,
        )
        for seq_name in sequence_names
    ]


def process_sequence(
    seq_name: str,
    mot_root: str,
    project_root: str,
    model_name: str,
    reid_name: str,
    tracking_method: str,
    exp_folder: str,
    target_fps: Optional[int],
    device: Optional[str] = None,
    cfg_dict: dict | None = None,
    dataset_name: Optional[str] = None,
    conf_threshold: float = 0.0,
    progress_queue=None,
):
    """Run a tracker over cached detections and embeddings for one sequence."""
    model_key = Path(model_name).stem if Path(model_name).suffix else str(model_name)
    reid_weights = Path(reid_name)
    reid_key = reid_weights.stem if reid_weights.suffix else str(reid_weights)
    if not reid_weights.suffix:
        reid_weights = reid_weights.with_suffix(".pt")

    tracker_runtime = TrackerRuntime.create(
        tracking_method=tracking_method,
        reid_weights=reid_weights,
        device=select_device("cpu"),
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
        model_name=model_key,
        reid_name=reid_key,
        target_fps=target_fps,
    )
    sequence = dataset.get_sequence(
        seq_name,
        show_progress=False,
        progress_queue=progress_queue,
    )

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

            tracks, elapsed_ms = tracker_runtime.update(dets, img, embs)
            total_track_time_ms += elapsed_ms

            if tracks.size:
                all_tracks.append(TrackerRuntime.format_for_mot(tracks, frame_id))

    out_arr = np.vstack(all_tracks) if all_tracks else np.empty((0, 0))
    write_mot_results(Path(exp_folder) / f"{seq_name}.txt", out_arr)

    timing_dict = {"track_time_ms": total_track_time_ms, "num_frames": num_frames}
    return seq_name, kept_frame_ids, timing_dict


def _run_tracking_tasks(
    args: argparse.Namespace,
    task_args: list[tuple],
    *,
    quiet: bool,
) -> tuple[dict[str, list[int]], float, int]:
    n_seqs = len(task_args)
    seq_frame_nums: dict[str, list[int]] = {}
    total_track_time_ms = 0.0
    total_track_frames = 0
    done_count = 0
    seq_progress: dict = {}
    prev_display_lines = 0
    last_progress_message = None

    def _log_progress(progress_queue) -> None:
        nonlocal prev_display_lines, last_progress_message
        _drain_progress_queue(progress_queue, seq_progress)
        header = f"Tracking: {done_count}/{n_seqs} sequences done"
        seq_display = _format_seq_progress(seq_progress)
        message = "\n".join([header] + ([seq_display] if seq_display else []))
        if message == last_progress_message:
            return
        if prev_display_lines > 0:
            sys.stderr.write(f"\033[{prev_display_lines}A\033[J")
            sys.stderr.flush()
        LOGGER.opt(colors=True).info(f"<cyan>{message}</cyan>")
        prev_display_lines = message.count("\n") + 1
        last_progress_message = message

    _configure_logging(main_thread_only=True)
    spawn_context = mp.get_context("spawn")

    with spawn_context.Manager() as manager:
        progress_queue = manager.Queue()

        bound_task_args = [task[:-1] + (progress_queue,) for task in task_args]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.n_threads,
            initializer=_worker_init,
            mp_context=spawn_context,
        ) as executor:
            futures = {executor.submit(process_sequence, *task_arg): task_arg[0] for task_arg in bound_task_args}
            pending = set(futures)

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
                        total_track_frames += timing_dict.get("num_frames", 0)
                        done_count += 1
                    except Exception:
                        done_count += 1
                        LOGGER.exception(f"Error processing {seq_name}")

                if not quiet:
                    _log_progress(progress_queue)

        if not quiet and prev_display_lines > 0:
            sys.stderr.write(f"\033[{prev_display_lines}A\033[J")
            sys.stderr.flush()
            LOGGER.opt(colors=True).info(f"<cyan>Tracking: {n_seqs}/{n_seqs} sequences done</cyan>")

    return seq_frame_nums, total_track_time_ms, total_track_frames


def run_generate_mot_results(
    args: argparse.Namespace,
    evolve_config: dict | None = None,
    timing_stats: Optional[TimingStats] = None,
    quiet: bool = False,
) -> None:
    """Run trackers over cached detections/embeddings and write MOT result files."""
    args.project = Path(args.project)
    base = args.project / "mot"
    if getattr(args, "benchmark", None):
        base = base / args.benchmark
    base = base / f"{args.yolo_model[0].stem}_{args.reid_model[0].stem}_{args.tracking_method}"
    exp_dir = increment_path(base, sep="_", exist_ok=False)
    exp_dir.mkdir(parents=True, exist_ok=True)
    args.exp_dir = exp_dir

    sequence_names = MOTDataset(mot_root=str(args.source)).sequence_names()
    conf_threshold = getattr(args, "conf", None)
    if conf_threshold is None:
        conf_threshold = default_conf(args.yolo_model[0])

    task_args = _build_task_args(
        args,
        exp_dir,
        sequence_names,
        evolve_config,
        conf_threshold,
        None,
    )
    seq_frame_nums, total_track_time_ms, total_track_frames = _run_tracking_tasks(args, task_args, quiet=quiet)
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