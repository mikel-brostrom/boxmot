"""
@Author: Du Yunhao
@Filename: GSI.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/1 9:18
@Discription: Gaussian-smoothed interpolation
"""
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF


def LinearInterpolation(input_, interval):
    print(input_)
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]  # 按ID和帧排序
    output_ = input_.copy()
    '''线性插值'''
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    for row in input_:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre:  # 同ID
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):  # 逐框插值
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:  # 不同ID
            id_pre = id_curr
        row_pre = row
        f_pre = f_curr
    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]
    return output_


def GaussianSmooth(input_, tau):
    output_ = list()
    ids = set(input_[:, 1])
    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        print(len_scale)
        gpr = GPR(RBF(len_scale, 'fixed'))
        t = tracks[:, 0].reshape(-1, 1)
        x = tracks[:, 2].reshape(-1, 1)
        y = tracks[:, 3].reshape(-1, 1)
        w = tracks[:, 4].reshape(-1, 1)
        h = tracks[:, 5].reshape(-1, 1)
        gpr.fit(t, x)
        xx = gpr.predict(t)[0]
        gpr.fit(t, y)
        yy = gpr.predict(t)[0]
        gpr.fit(t, w)
        ww = gpr.predict(t)[0]
        gpr.fit(t, h)
        hh = gpr.predict(t)[0]
        print('t', t[0])
        print('id_', id_)
        print('xx', xx)
        output_.extend([
            [int(t[0][0]), int(id_), int(xx), int(yy), int(ww), int(hh), 1, -1, -1] for i in range(len(t))
        ])
    return output_


def gsi_interpolation(mot_results_folder=Path('examples/runs/val/exp50/labels'), interval=20, tau=10):
    tracking_results_files = mot_results_folder.glob('MOT*FRCNN.txt')
    for p in tracking_results_files:
        print(f"Processing: {p}")
        tracking_results = np.loadtxt(p, delimiter=' ')
        li = LinearInterpolation(tracking_results, interval)
        gsi = GaussianSmooth(li, tau)

        # print(gsi)
        print(f"Saving: {p.parent / (p.stem + 'proc.txt')}")
        np.savetxt(p.parent / (p.stem + 'proc.txt'), gsi, fmt='%d %d %d %d %d %d %d %d %d')


gsi_interpolation()
