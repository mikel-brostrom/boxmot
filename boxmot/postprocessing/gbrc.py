import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

from boxmot.utils import logger as LOGGER


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
) -> np.ndarray:
    """
    Smooth columns 2..5 (x,y,w,h) per track id using GradientBoostingRegressor.

    Output format (per your original):
        [frame, id, x, y, w, h, 1, -1, -1, -1]
    """
    if data.size == 0:
        return np.empty((0, 10), dtype=float)

    smoothed_rows = []
    unique_ids = np.unique(data[:, 1])

    for obj_id in unique_ids:
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

        xx = _fit_predict_1d(regr, t, tracks[:, 2]).reshape(-1, 1)
        yy = _fit_predict_1d(regr, t, tracks[:, 3]).reshape(-1, 1)
        ww = _fit_predict_1d(regr, t, tracks[:, 4]).reshape(-1, 1)
        hh = _fit_predict_1d(regr, t, tracks[:, 5]).reshape(-1, 1)

        # Build output rows: frame, id, x,y,w,h, 1,-1,-1,-1
        # (keeping your intended MOT-like 10 columns)
        for i in range(len(tracks)):
            smoothed_rows.append(
                [
                    float(tracks[i, 0]),
                    float(obj_id),
                    float(xx[i, 0]),
                    float(yy[i, 0]),
                    float(ww[i, 0]),
                    float(hh[i, 0]),
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ]
            )

    out = np.array(smoothed_rows, dtype=float)
    return out[np.lexsort((out[:, 0], out[:, 1]))]


def process_file(
    file_path: Path,
    interval: int,
    n_estimators: int,
    learning_rate: float,
    min_samples_split: int,
):
    """
    Process a single MOT results file by applying linear interpolation and gradient boosting smoothing.
    """
    LOGGER.info(f"Applying GBRC/GBI to: {file_path}")
    tracking_results = np.loadtxt(file_path, delimiter=",")

    if tracking_results.size == 0:
        LOGGER.warning(f"No tracking results in {file_path}. Skipping...")
        return

    interpolated = linear_interpolation(tracking_results, interval)
    smoothed = gradient_boosting_smooth(
        interpolated,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
    )

    # Save like your GSI script: overwrite the same file
    # Use integer frame/id, floats for bbox, then ints for trailing fields.
    # Note: if you want commas instead, switch fmt + delimiter.
    np.savetxt(
        file_path,
        smoothed,
        fmt="%d %d %.2f %.2f %.2f %.2f %d %d %d %d",
    )


def gbrc(
    mot_results_folder: Path,
    interval: int = 20,
    n_estimators: int = 115,
    learning_rate: float = 0.065,
    min_samples_split: int = 6,
):
    """
    Apply GBRC/GBI-style postprocessing to all MOT*.txt files in a folder.
    """
    tracking_files = list(mot_results_folder.glob("MOT*.txt"))
    total_files = len(tracking_files)
    LOGGER.debug(f"GBRC: Found {total_files} file(s) to process.")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_file,
                file_path,
                interval,
                n_estimators,
                learning_rate,
                min_samples_split,
            ): file_path
            for file_path in tracking_files
        }

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
