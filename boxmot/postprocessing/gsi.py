from pathlib import Path
import numpy as np
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from boxmot.utils import logger as LOGGER
import concurrent.futures
from tqdm import tqdm

def linear_interpolation(input_, interval):
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
    output_ = []
    ids = set(input_[:, 1])
    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        t = tracks[:, 0].reshape(-1, 1)
        gpr = GPR(RBF(len_scale, 'fixed'))
        smoothed_data = []
        for i in range(2, 6):
            data = tracks[:, i].reshape(-1, 1)
            gpr.fit(t, data)
            smoothed_data.append(gpr.predict(t).reshape(-1, 1))
        for j in range(len(t)):
            output_.append([
                t[j, 0], id_, *[data[j, 0] for data in smoothed_data],
                tracks[j, 6], tracks[j, 7], -1
            ])
    return np.array(output_)

def process_file(p: Path, interval: int, tau: int):
    LOGGER.info(f"Applying GSI to: {p}")
    tracking_results = np.loadtxt(p, delimiter=',')
    if tracking_results.size != 0:
        li = linear_interpolation(tracking_results, interval)
        gsi_result = gaussian_smooth(li, tau)
        np.savetxt(p, gsi_result, fmt='%d %d %d %d %d %d %d %d %d')
    else:
        LOGGER.warning(f'No tracking result in {p}. Skipping...')

def gsi(mot_results_folder: Path, interval=20, tau=10):
    tracking_results_files = list(mot_results_folder.glob('MOT*.txt'))
    total_files = len(tracking_results_files)
    LOGGER.info(f"Found {total_files} file(s) to process.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, p, interval, tau): p for p in tracking_results_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="Processing files"):
            p = futures[future]
            try:
                future.result()
            except Exception as e:
                LOGGER.error(f"Error processing file {p}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Apply Gaussian Smoothed Interpolation (GSI) to tracking results.'
    )
    parser.add_argument('--path', type=str, required=True, help='Path to MOT results folder')
    args = parser.parse_args()

    mot_results_folder = Path(args.path)
    gsi(mot_results_folder)

if __name__ == "__main__":
    main()