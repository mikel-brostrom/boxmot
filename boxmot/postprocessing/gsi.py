from pathlib import Path
import numpy as np
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from boxmot.utils import logger as LOGGER
import concurrent.futures
from tqdm import tqdm


def linear_interpolation(data: np.ndarray, interval: int) -> np.ndarray:
    """
    Apply linear interpolation between rows in the tracking results.

    The function assumes the first two columns of `data` represent frame number and object ID.
    Interpolated rows are added when consecutive rows for the same ID have a gap of more than 1
    frame but less than the specified interval.

    Parameters:
        data (np.ndarray): Input tracking results.
        interval (int): Maximum gap to perform interpolation.

    Returns:
        np.ndarray: Tracking results with interpolated rows included.
    """
    # Sort data by frame and then by ID
    sorted_data = data[np.lexsort((data[:, 0], data[:, 1]))]
    output = sorted_data.copy()

    previous_id = -1
    previous_frame = -1
    previous_row = np.zeros(sorted_data.shape[1])

    for row in sorted_data:
        current_frame, current_id = row[:2].astype(int)
        # If the same ID and the frame gap is appropriate, interpolate missing frames
        if current_id == previous_id and previous_frame + 1 < current_frame < previous_frame + interval:
            for i, frame in enumerate(range(previous_frame + 1, current_frame), start=1):
                # Linear interpolation for each missing frame
                step = (row - previous_row) / (current_frame - previous_frame) * i
                new_row = previous_row + step
                output = np.vstack((output, new_row))
        previous_id, previous_row, previous_frame = current_id, row, current_frame

    # Resort the output after adding interpolated rows
    return output[np.lexsort((output[:, 0], output[:, 1]))]


def gaussian_smooth(data: np.ndarray, tau: float) -> np.ndarray:
    """
    Apply Gaussian process smoothing to specified columns in the tracking results.

    For each unique object ID in the data, this function smooths columns 2 through 5 using
    a Gaussian Process with an RBF kernel. Additional columns (columns 6 and 7) and a constant
    value (-1) are appended to each row.

    Parameters:
        data (np.ndarray): Tracking results.
        tau (float): Smoothing parameter.

    Returns:
        np.ndarray: Tracking results with smoothed columns.
    """
    smoothed_output = []

    # Extract unique IDs from the second column
    unique_ids = set(data[:, 1])
    for obj_id in unique_ids:
        tracks = data[data[:, 1] == obj_id]
        num_tracks = len(tracks)
        # Determine length scale using logarithmic scaling with clipping
        length_scale = np.clip(tau * np.log(tau ** 3 / num_tracks), tau ** -1, tau ** 2)
        # Reshape the time/frame column for GP input
        t = tracks[:, 0].reshape(-1, 1)
        kernel = RBF(length_scale, length_scale_bounds="fixed")
        gpr = GPR(kernel)

        # Smooth columns 2 to 5
        smoothed_columns = []
        for col_index in range(2, 6):
            column_data = tracks[:, col_index].reshape(-1, 1)
            gpr.fit(t, column_data)
            smoothed_prediction = gpr.predict(t).reshape(-1, 1)
            smoothed_columns.append(smoothed_prediction)

        # Build new rows with the smoothed data, retaining other columns and appending -1
        for j in range(len(t)):
            smoothed_values = [smoothed_columns[k][j, 0] for k in range(len(smoothed_columns))]
            new_row = [t[j, 0], obj_id, *smoothed_values, tracks[j, 6], tracks[j, 7], -1]
            smoothed_output.append(new_row)

    return np.array(smoothed_output)


def process_file(file_path: Path, interval: int, tau: float):
    """
    Process a single MOT results file by applying linear interpolation and Gaussian smoothing.

    Parameters:
        file_path (Path): Path to the tracking results file.
        interval (int): Interval for linear interpolation.
        tau (float): Smoothing parameter for Gaussian process.
    """
    LOGGER.info(f"Applying GSI to: {file_path}")
    tracking_results = np.loadtxt(file_path, delimiter=',')
    if tracking_results.size != 0:
        interpolated_results = linear_interpolation(tracking_results, interval)
        smoothed_results = gaussian_smooth(interpolated_results, tau)
        np.savetxt(file_path, smoothed_results, fmt='%d %d %d %d %d %d %d %d %d')
    else:
        LOGGER.warning(f'No tracking results in {file_path}. Skipping...')


def gsi(mot_results_folder: Path, interval: int = 20, tau: float = 10):
    """
    Apply Gaussian Smoothed Interpolation (GSI) to all tracking result files in a folder.

    Parameters:
        mot_results_folder (Path): Path to the folder containing MOT result files.
        interval (int, optional): Maximum gap to perform interpolation. Defaults to 20.
        tau (float, optional): Smoothing parameter for Gaussian process. Defaults to 10.
    """
    tracking_files = list(mot_results_folder.glob('MOT*.txt'))
    total_files = len(tracking_files)
    LOGGER.info(f"Found {total_files} file(s) to process.")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file_path, interval, tau): file_path for file_path in tracking_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="Processing files"):
            file_path = futures[future]
            try:
                future.result()
            except Exception as e:
                LOGGER.error(f"Error processing file {file_path}: {e}")


def main():
    """
    Parse command line arguments and run the Gaussian Smoothed Interpolation process.
    """
    parser = argparse.ArgumentParser(
        description='Apply Gaussian Smoothed Interpolation (GSI) to tracking results.'
    )
    parser.add_argument('--path', type=str, required=True, help='Path to MOT results folder')
    args = parser.parse_args()

    mot_results_folder = Path(args.path)
    gsi(mot_results_folder)


if __name__ == "__main__":
    main()
