from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from boxmot.utils import logger as LOGGER


def linear_interpolation(input_, interval):
    """
    Perform linear interpolation on the input data to fill in missing frames within the specified interval.

    Args:
        input_ (np.ndarray): Input array with shape (n, m) where n is the number of rows and m is the number of columns.
        interval (int): The maximum frame gap to interpolate.

    Returns:
        np.ndarray: Interpolated array with additional rows for the interpolated frames.
    """
    input_ = input_[np.lexsort((input_[:, 0], input_[:, 1]))]
    output_ = input_.copy()

    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    for row in input_:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre and f_pre + 1 < f_curr < f_pre + interval:
            for i, f in enumerate(range(f_pre + 1, f_curr), start=1):
                step = (row - row_pre) / (f_curr - f_pre) * i
                row_new = row_pre + step
                output_ = np.vstack((output_, row_new))
        id_pre, row_pre, f_pre = id_curr, row, f_curr
    return output_[np.lexsort((output_[:, 0], output_[:, 1]))]


def gaussian_smooth(input_, tau):
    """
    Apply Gaussian smoothing to the input data.

    Args:
        input_ (np.ndarray): Input array with shape (n, m) where n is the number of rows and m is the number of columns.
        tau (float): Time constant for Gaussian smoothing.

    Returns:
        np.ndarray: Smoothed array with the same shape as the input.
    """
    output_ = []
    ids = set(input_[:, 1])
    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        t = tracks[:, 0].reshape(-1, 1)
        gpr = GPR(RBF(len_scale, 'fixed'))
        smoothed_data = []
        # x, y, w, h
        for i in range(2, 6):
            data = tracks[:, i].reshape(-1, 1)
            gpr.fit(t, data)
            smoothed_data.append(gpr.predict(t).reshape(-1, 1))
        for j in range(len(t)):
            output_.append([
                t[j, 0], id_, *[data[j, 0] for data in smoothed_data], tracks[j, 6], tracks[j, 7], -1
            ])
    return np.array(output_)


def gsi(mot_results_folder=Path('examples/runs/val/exp87/labels'), interval=20, tau=10):
    """
    Apply Gaussian Smoothed Interpolation (GSI) to the tracking results files.

    Args:
        mot_results_folder (Path): Path to the folder containing the tracking results files.
        interval (int): The maximum frame gap to interpolate.
        tau (float): Time constant for Gaussian smoothing.

    Returns:
        None
    """
    tracking_results_files = mot_results_folder.glob('MOT*FRCNN.txt')
    for p in tracking_results_files:
        LOGGER.info(f"Applying gaussian smoothed interpolation (GSI) to: {p}")
        tracking_results = np.loadtxt(p, dtype=int, delimiter=' ')
        if tracking_results.size != 0:
            li = linear_interpolation(tracking_results, interval)
            gsi = gaussian_smooth(li, tau)
            np.savetxt(p, gsi, fmt='%d %d %d %d %d %d %d %d %d')
        else:
            print(f'No tracking result in {p}. Skipping...')


def main():
    """
    Main function to run GSI on the specified folder.
    """
    gsi()


if __name__ == "__main__":
    main()