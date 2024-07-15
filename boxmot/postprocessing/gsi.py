# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from boxmot.utils import logger as LOGGER

def linear_interpolation(input_, interval):
    """
    Perform linear interpolation on the input array.

    Args:
        input_ (np.ndarray): Input array with shape (n, m) where n is the number of entries and m is the number of features.
        interval (int): Maximum interval for interpolation.

    Returns:
        np.ndarray: Interpolated array.
    """
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]
    output_ = input_.copy()

    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    for row in input_:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre:
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:
            id_pre = id_curr
        row_pre = row
        f_pre = f_curr
    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]
    return output_

def gaussian_smooth(input_, tau):
    """
    Apply Gaussian smoothing to the input array.

    Args:
        input_ (np.ndarray): Input array with shape (n, m) where n is the number of entries and m is the number of features.
        tau (float): Smoothing parameter.

    Returns:
        list: Smoothed array.
    """
    output_ = []
    ids = set(input_[:, 1])
    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        
        t = tracks[:, 0].reshape(-1, 1)
        features = [tracks[:, i].reshape(-1, 1) for i in range(2, 6)]
        smoothed_features = []

        for feature in features:
            gpr.fit(t, feature)
            smoothed_features.append(gpr.predict(t))

        for i in range(len(t)):
            output_.append([t[i, 0], id_, *[sf[i, 0] for sf in smoothed_features], tracks[i, 6], tracks[i, 7], -1])

    return output_

def gsi(mot_results_folder=Path('examples/runs/val/exp87/labels'), interval=20, tau=10):
    """
    Apply Gaussian Smoothed Interpolation (GSI) to tracking results.

    Args:
        mot_results_folder (Path): Path to the folder containing tracking result files.
        interval (int): Maximum interval for interpolation.
        tau (float): Smoothing parameter.
    """
    tracking_results_files = mot_results_folder.glob('MOT*FRCNN.txt')
    for p in tracking_results_files:
        LOGGER.info(f"Applying gaussian smoothed interpolation (GSI) to: {p}")
        tracking_results = np.loadtxt(p, dtype=int, delimiter=' ')
        if tracking_results.size != 0:
            li = linear_interpolation(tracking_results, interval)
            gsi_results = gaussian_smooth(li, tau)
            np.savetxt(p, gsi_results, fmt='%d %d %d %d %d %d %d %d %d')
        else:
            LOGGER.info(f'No tracking result in {p}. Skipping...')

def main():
    """
    Main function to apply Gaussian Smoothed Interpolation (GSI) to tracking results.
    """
    gsi()

if __name__ == "__main__":
    main()