# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF

from boxmot.utils import logger as LOGGER


def linear_interpolation(input_, interval):
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
    output_ = list()
    print('input_', input_)
    ids = set(input_[:, 1])
    for i, id_ in enumerate(ids):
        tracks = input_[input_[:, 1] == id_]
        print('tracks', tracks)
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        t = tracks[:, 0].reshape(-1, 1)
        x = tracks[:, 2].reshape(-1, 1)
        y = tracks[:, 3].reshape(-1, 1)
        w = tracks[:, 4].reshape(-1, 1)
        h = tracks[:, 5].reshape(-1, 1)
        gpr.fit(t, x)
        xx = gpr.predict(t)
        gpr.fit(t, y)
        yy = gpr.predict(t)
        gpr.fit(t, w)
        ww = gpr.predict(t)
        gpr.fit(t, h)
        hh = gpr.predict(t)
        # frame count, id, x, y, w, h, conf, cls, -1 (don't care)
        output_.extend([
            [t[j, 0], id_, xx[j], yy[j], ww[j], hh[j], tracks[j, 6], tracks[j, 7], -1] for j in range(len(t))
        ])
    return output_


def gsi(mot_results_folder=Path('examples/runs/val/exp87/labels'), interval=20, tau=10):
    tracking_results_files = mot_results_folder.glob('MOT*FRCNN.txt')
    for p in tracking_results_files:
        LOGGER.info(f"Applying gaussian smoothed interpolation (GSI) to: {p}")
        tracking_results = np.loadtxt(p, dtype=int, delimiter=' ')
        if tracking_results.size != 0:
            li = linear_interpolation(tracking_results, interval)
            gsi = gaussian_smooth(li, tau)
            np.savetxt(p, gsi, fmt='%d %d %d %d %d %d %d %d %d')
        else:
            print('No tracking result in {p}. Skipping...')


def main():
    gsi()


if __name__ == "__main__":
    main()
