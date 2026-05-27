import argparse
import concurrent.futures
import multiprocessing as mp
import queue
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from boxmot.utils import logger as LOGGER
from boxmot.utils.callbacks import safe_seq_progress_callback
from boxmot.utils.rich.progress import RichTqdm as tqdm


def linear_interpolation(data: np.ndarray, interval: int) -> np.ndarray:
    """
    Apply linear interpolation between rows in the tracking results.

    Assumes col0=frame, col1=id. Interpolates gaps:
        previous_frame + 1 < current_frame < previous_frame + interval
    """
    if data.size == 0:
        return data

    # Sort by id then frame (same as your other script style)
    sorted_data = data[np.lexsort((data[:, 0], data[:, 1]))]

    result_rows = []
    previous_id = None
    previous_frame = None
    previous_row = None

    for row in sorted_data:
        current_frame, current_id = int(row[0]), int(row[1])

        if (
            previous_id is not None
            and current_id == previous_id
            and previous_frame + 1 < current_frame < previous_frame + interval
        ):
            gap = current_frame - previous_frame - 1
            for i in range(1, gap + 1):
                new_row = previous_row + (row - previous_row) * (
                    i / (current_frame - previous_frame)
                )
                result_rows.append(new_row)

        result_rows.append(row)
        previous_id, previous_frame, previous_row = current_id, current_frame, row

    out = np.array(result_rows)
    return out[np.lexsort((out[:, 0], out[:, 1]))]


def _fit_predict_1d(regr: GradientBoostingRegressor, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit regressor on (t -> y) and return predictions shaped (n,)."""
    regr.fit(t, y.ravel())
    return regr.predict(t)


def gradient_boosting_smooth(
    data: np.ndarray,
    n_estimators: int = 115,
    learning_rate: float = 0.065,
    min_samples_split: int = 6,
    progress_fn: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """
    Smooth columns 2..5 (x,y,w,h) per track id using GradientBoostingRegressor.
    """
    if data.size == 0:
        return data

    smoothed_rows = []
    unique_ids = np.unique(data[:, 1])
    total_ids = len(unique_ids)

    for idx, obj_id in enumerate(unique_ids):
        tracks = data[data[:, 1] == obj_id]
        if len(tracks) == 0:
            continue

        t = tracks[:, 0].reshape(-1, 1)

        # If a track is extremely short, boosting can still fit,
        # but smoothing isn't really meaningful; we still run it for consistency.
        regr = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
        )

        tracks[:, 2] = _fit_predict_1d(regr, t, tracks[:, 2])
        tracks[:, 3] = _fit_predict_1d(regr, t, tracks[:, 3])
        tracks[:, 4] = _fit_predict_1d(regr, t, tracks[:, 4])
        tracks[:, 5] = _fit_predict_1d(regr, t, tracks[:, 5])

        smoothed_rows.append(tracks)
        if progress_fn is not None:
            progress_fn(idx + 1, total_ids)

    out = np.concatenate(smoothed_rows)
    return out[np.lexsort((out[:, 0], out[:, 1]))]


def process_file(
    file_path: Path,
    interval: int,
    n_estimators: int,
    learning_rate: float,
    min_samples_split: int,
    progress_queue=None,
):
    """
    Process a single MOT results file by applying linear interpolation and gradient boosting smoothing.
    """
    LOGGER.debug(f"Applying GBRC/GBI to: {file_path}")
    seq_name = file_path.stem if progress_queue is not None else None
    if progress_queue is not None:
        progress_queue.put((seq_name, -1, 0))  # mark as processing
    tracking_results = np.loadtxt(file_path, delimiter=",")

    if tracking_results.size == 0:
        LOGGER.warning(f"No tracking results in {file_path}. Skipping...")
        if progress_queue is not None:
            progress_queue.put((seq_name, 1, 1))  # mark done
        return

    interpolated = linear_interpolation(tracking_results, interval)

    pq_fn = None
    if progress_queue is not None:
        # Report 0% after interpolation, before smoothing starts
        n_tracks = len(np.unique(interpolated[:, 1]))
        progress_queue.put((seq_name, 0, n_tracks))
        def pq_fn(current: int, total: int) -> None:
            progress_queue.put((seq_name, current, total))

    smoothed = gradient_boosting_smooth(
        interpolated,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
        progress_fn=pq_fn,
    )

    # Save like your GSI script: overwrite the same file
    # Use integer frame/id, floats for bbox, then ints for trailing fields.
    # Note: if you want commas instead, switch fmt + delimiter.
    fmt = ["%d", "%d"] + ["%.2f"] * (smoothed.shape[1] - 2)
    np.savetxt(
        file_path,
        smoothed,
        fmt=fmt,
        delimiter=",",
    )


def _drain_queue(progress_queue, seq_progress: dict) -> None:
    """Read all available messages from a progress queue."""
    while True:
        try:
            name, current, total = progress_queue.get_nowait()
            seq_progress[name] = (current, total)
        except (queue.Empty, OSError):
            break


def gbrc(
    mot_results_folder: Path,
    interval: int = 20,
    n_estimators: int = 115,
    learning_rate: float = 0.065,
    min_samples_split: int = 6,
    progress_callback: Callable[[str, int, int], None] | None = None,
):
    """
    Apply GBRC/GBI-style postprocessing to all MOT*.txt files in a folder.
    """
    tracking_files = sorted(mot_results_folder.glob("*.txt"))
    total_files = len(tracking_files)
    LOGGER.debug(f"GBRC: Found {total_files} file(s) to process.")

    progress_callback = safe_seq_progress_callback(progress_callback)
    use_queue = progress_callback is not None
    spawn_ctx = mp.get_context("spawn")
    manager_ctx = spawn_ctx.Manager() if use_queue else nullcontext()

    with manager_ctx as manager:
        progress_queue = manager.Queue() if use_queue else None

        with concurrent.futures.ProcessPoolExecutor(mp_context=spawn_ctx) as executor:
            futures = {
                executor.submit(
                    process_file,
                    file_path,
                    interval,
                    n_estimators,
                    learning_rate,
                    min_samples_split,
                    progress_queue,
                ): file_path
                for file_path in tracking_files
            }

            if progress_callback is not None:
                # Poll-based loop (like tracking) for live progress
                seq_progress: dict[str, tuple[int, int]] = {}
                pending = set(futures)

                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=0.3,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        file_path = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            LOGGER.error(f"Error processing file {file_path}: {e}")
                        seq_progress[file_path.stem] = (1, 1)  # mark complete

                    # Drain queue and emit latest state
                    _drain_queue(progress_queue, seq_progress)
                    # Send the latest update for each seq to the callback
                    for name, (cur, tot) in seq_progress.items():
                        progress_callback(name, cur, tot)
            else:
                # Simple tqdm loop (no callback)
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=total_files,
                    desc="Processing files",
                ):
                    file_path = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        LOGGER.error(f"Error processing file {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply Gradient Boosting Reconnection Context (GBRC/GBI) postprocessing to tracking results."
    )
    parser.add_argument("--path", type=str, required=True, help="Path to MOT results folder")
    parser.add_argument("--interval", type=int, default=20, help="Maximum gap to interpolate (default: 20)")
    parser.add_argument("--n_estimators", type=int, default=115, help="GBR n_estimators (default: 115)")
    parser.add_argument("--learning_rate", type=float, default=0.065, help="GBR learning_rate (default: 0.065)")
    parser.add_argument("--min_samples_split", type=int, default=6, help="GBR min_samples_split (default: 6)")
    args = parser.parse_args()

    mot_results_folder = Path(args.path)
    gbrc(
        mot_results_folder,
        interval=args.interval,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        min_samples_split=args.min_samples_split,
    )


if __name__ == "__main__":
    main()
