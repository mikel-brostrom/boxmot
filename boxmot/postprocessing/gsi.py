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
    result_rows = []
    previous_id = None
    previous_frame = None
    previous_row = None

    for row in sorted_data:
        current_frame, current_id = int(row[0]), int(row[1])
        if previous_id is not None and current_id == previous_id and previous_frame + 1 < current_frame < previous_frame + interval:
            gap = current_frame - previous_frame - 1
            for i in range(1, gap + 1):
                # Linear interpolation for each missing frame
                new_row = previous_row + (row - previous_row) * (i / (current_frame - previous_frame))
                result_rows.append(new_row)
        result_rows.append(row)
        previous_id, previous_frame, previous_row = current_id, current_frame, row

    result_array = np.array(result_rows)
    # Resort the array
    return result_array[np.lexsort((result_array[:, 0], result_array[:, 1]))]


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
    unique_ids = np.unique(data[:, 1])
    for obj_id in unique_ids:
        tracks = data[data[:, 1] == obj_id]
        num_tracks = len(tracks)
        # Determine length scale using logarithmic scaling with clipping
        length_scale = np.clip(tau * np.log(tau ** 3 / num_tracks), tau ** -1, tau ** 2)
        t = tracks[:, 0].reshape(-1, 1)
        kernel = RBF(length_scale, length_scale_bounds="fixed")
        gpr = GPR(kernel)
        
        # Smooth columns 2 to 5 simultaneously (if supported by your version of scikit-learn)
        smoothed_columns = gpr.fit(t, tracks[:, 2:6]).predict(t)
        
        # Build new rows with the smoothed data, retaining other columns and appending -1
        for i in range(len(tracks)):
            new_row = np.concatenate(([tracks[i, 0], obj_id], smoothed_columns[i], tracks[i, 6:8], [-1]))
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
