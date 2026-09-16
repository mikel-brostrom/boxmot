from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF

from boxmot.postprocessing.base import FileWorker, MotFilePostprocessor, ProgressCallback
from boxmot.utils import logger as LOGGER


def linear_interpolation(data: np.ndarray, interval: int) -> np.ndarray:
    """Fill short gaps within each track/class pair in canonical MOT rows.

    MOT rows contain frame, track ID, x, y, width, height, confidence, class ID,
    and detection index. Only geometry and confidence are interpolated; new
    rows retain their class and use detection index ``-1``. A gap is filled
    when its endpoint-frame difference is strictly smaller than ``interval``.
    """
    if data.size == 0:
        return data.copy()
    rows = np.asarray(data, dtype=np.float64)
    ordered = rows[np.lexsort((rows[:, 0], rows[:, 7], rows[:, 1]))]
    result_rows = []
    previous = None
    for row in ordered:
        if previous is not None and np.array_equal(row[[1, 7]], previous[[1, 7]]):
            gap = int(row[0] - previous[0])
            if 1 < gap < interval:
                for offset in range(1, gap):
                    interpolated = previous.copy()
                    interpolated[0] = previous[0] + offset
                    interpolated[2:7] += (row[2:7] - previous[2:7]) * (offset / gap)
                    interpolated[8] = -1
                    result_rows.append(interpolated)
        result_rows.append(row)
        previous = row
    result = np.asarray(result_rows)
    return result[np.lexsort((result[:, 0], result[:, 1]))]


def gaussian_smooth(
    data: np.ndarray,
    tau: float,
    progress_fn: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Smooth MOT box coordinates independently for each track/class pair.

    Frame numbers, IDs, confidence, classes, and detection indices are retained
    exactly. Empty inputs and single-observation trajectories are unchanged.
    ``progress_fn`` receives the number of completed track/class pairs.
    """
    if data.size == 0:
        return data.copy()
    rows = np.asarray(data, dtype=np.float64)
    track_classes = np.unique(rows[:, [1, 7]], axis=0)
    smoothed_output = []
    for index, (track_id, class_id) in enumerate(track_classes):
        tracks = rows[(rows[:, 1] == track_id) & (rows[:, 7] == class_id)].copy()
        if len(tracks) > 1:
            length_scale = np.clip(tau * np.log(tau**3 / len(tracks)), tau**-1, tau**2)
            times = tracks[:, 0].reshape(-1, 1)
            kernel = RBF(length_scale, length_scale_bounds="fixed")
            regressor = GPR(kernel)
            tracks[:, 2:6] = regressor.fit(times, tracks[:, 2:6]).predict(times)
        smoothed_output.append(tracks)
        if progress_fn is not None:
            progress_fn(index + 1, len(track_classes))
    result = np.concatenate(smoothed_output)
    return result[np.lexsort((result[:, 0], result[:, 1]))]


def process_file(file_path: Path, interval: int, tau: float, progress_queue: Any | None = None) -> None:
    """
    Process a single MOT results file by applying linear interpolation and Gaussian smoothing.

    Parameters:
        file_path (Path): Path to the tracking results file.
        interval (int): Interval for linear interpolation.
        tau (float): Smoothing parameter for Gaussian process.
        progress_queue: Optional multiprocessing queue for per-track progress.
    """
    LOGGER.debug(f"Applying GSI to: {file_path}")
    seq_name = file_path.stem if progress_queue is not None else None
    if progress_queue is not None:
        progress_queue.put((seq_name, -1, 0))  # mark as processing
    try:
        tracking_results = np.loadtxt(file_path, delimiter=",")
    except (ValueError, OSError) as exc:
        LOGGER.warning(f"GSI: could not load {file_path}: {exc}. Skipping...")
        if progress_queue is not None:
            progress_queue.put((seq_name, 1, 1))
        return
    if tracking_results.ndim == 1 and tracking_results.size > 0:
        tracking_results = tracking_results.reshape(1, -1)
    if tracking_results.size != 0:
        interpolated_results = linear_interpolation(tracking_results, interval)
        pq_fn = None
        if progress_queue is not None:
            n_tracks = len(np.unique(interpolated_results[:, [1, 7]], axis=0))
            progress_queue.put((seq_name, 0, n_tracks))

            def pq_fn(current: int, total: int) -> None:
                progress_queue.put((seq_name, current, total))

        smoothed_results = gaussian_smooth(interpolated_results, tau, progress_fn=pq_fn)
        np.savetxt(
            file_path,
            smoothed_results,
            fmt=["%d", "%d", *(["%.17g"] * 5), "%d", "%d"],
            delimiter=",",
        )
    else:
        LOGGER.warning(f"No tracking results in {file_path}. Skipping...")
        if progress_queue is not None:
            progress_queue.put((seq_name, 1, 1))  # mark done


class GSIPostprocessor(MotFilePostprocessor):
    """Gaussian-smoothed interpolation postprocessor."""

    name = "gsi"
    display_name = "GSI"

    def __init__(self, interval: int = 20, tau: float = 10) -> None:
        self.interval = interval
        self.tau = tau

    def worker(self) -> FileWorker:
        return process_file

    def worker_args(self) -> tuple[int, float]:
        return self.interval, self.tau


def gsi(
    mot_results_folder: Path,
    interval: int = 20,
    tau: float = 10,
    progress_callback: ProgressCallback | None = None,
):
    """
    Apply Gaussian Smoothed Interpolation (GSI) to all tracking result files in a folder.

    Parameters:
        mot_results_folder (Path): Path to the folder containing MOT result files.
        interval (int, optional): Maximum gap to perform interpolation. Defaults to 20.
        tau (float, optional): Smoothing parameter for Gaussian process. Defaults to 10.
        progress_callback: Called with (seq_name, current_track, total_tracks) per track.
    """
    GSIPostprocessor(interval=interval, tau=tau).run(
        mot_results_folder,
        progress_callback=progress_callback,
    )


__all__ = (
    "GSIPostprocessor",
    "gaussian_smooth",
    "gsi",
    "linear_interpolation",
)
