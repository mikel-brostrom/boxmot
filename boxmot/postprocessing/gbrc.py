from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

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
    """Smooth MOT box coordinates independently for each track/class pair.

    All metadata columns are preserved, including observed detection indices.
    Empty inputs and single-observation trajectories are returned unchanged.
    """
    if data.size == 0:
        return data.copy()
    rows = np.asarray(data, dtype=np.float64)
    track_classes = np.unique(rows[:, [1, 7]], axis=0)
    smoothed_rows = []
    for index, (track_id, class_id) in enumerate(track_classes):
        tracks = rows[(rows[:, 1] == track_id) & (rows[:, 7] == class_id)].copy()
        if len(tracks) > 1:
            times = tracks[:, 0].reshape(-1, 1)
            regressor = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                min_samples_split=min_samples_split,
                random_state=0,
            )
            for column in range(2, 6):
                tracks[:, column] = _fit_predict_1d(regressor, times, tracks[:, column])
        smoothed_rows.append(tracks)
        if progress_fn is not None:
            progress_fn(index + 1, len(track_classes))
    result = np.concatenate(smoothed_rows)
    return result[np.lexsort((result[:, 0], result[:, 1]))]


def process_file(
    file_path: Path,
    interval: int,
    n_estimators: int,
    learning_rate: float,
    min_samples_split: int,
    progress_queue: Any | None = None,
) -> None:
    """
    Process a single MOT results file by applying linear interpolation and gradient boosting smoothing.
    """
    LOGGER.debug(f"Applying GBRC/GBI to: {file_path}")
    seq_name = file_path.stem if progress_queue is not None else None
    if progress_queue is not None:
        progress_queue.put((seq_name, -1, 0))  # mark as processing
    try:
        tracking_results = np.loadtxt(file_path, delimiter=",")
    except (ValueError, OSError) as exc:
        LOGGER.warning(f"GBRC: could not load {file_path}: {exc}. Skipping...")
        if progress_queue is not None:
            progress_queue.put((seq_name, 1, 1))
        return
    if tracking_results.ndim == 1 and tracking_results.size > 0:
        tracking_results = tracking_results.reshape(1, -1)

    if tracking_results.size == 0:
        LOGGER.warning(f"No tracking results in {file_path}. Skipping...")
        if progress_queue is not None:
            progress_queue.put((seq_name, 1, 1))  # mark done
        return

    interpolated = linear_interpolation(tracking_results, interval)

    pq_fn = None
    if progress_queue is not None:
        # Report 0% after interpolation, before smoothing starts
        n_tracks = len(np.unique(interpolated[:, [1, 7]], axis=0))
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

    np.savetxt(
        file_path,
        smoothed,
        fmt=["%d", "%d", *(["%.17g"] * 5), "%d", "%d"],
        delimiter=",",
    )


class GBRCPostprocessor(MotFilePostprocessor):
    """Gradient-boosting reconnection context postprocessor."""

    name = "gbrc"
    display_name = "GBRC"

    def __init__(
        self,
        interval: int = 20,
        n_estimators: int = 115,
        learning_rate: float = 0.065,
        min_samples_split: int = 6,
    ) -> None:
        self.interval = interval
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split

    def worker(self) -> FileWorker:
        return process_file

    def worker_args(self) -> tuple[int, int, float, int]:
        return (
            self.interval,
            self.n_estimators,
            self.learning_rate,
            self.min_samples_split,
        )


def gbrc(
    mot_results_folder: Path,
    interval: int = 20,
    n_estimators: int = 115,
    learning_rate: float = 0.065,
    min_samples_split: int = 6,
    progress_callback: ProgressCallback | None = None,
):
    """
    Apply GBRC/GBI-style postprocessing to all MOT*.txt files in a folder.
    """
    GBRCPostprocessor(
        interval=interval,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
    ).run(mot_results_folder, progress_callback=progress_callback)


__all__ = (
    "GBRCPostprocessor",
    "gbrc",
    "gradient_boosting_smooth",
    "linear_interpolation",
)
