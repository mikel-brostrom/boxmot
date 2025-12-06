import numpy as np

from boxmot.postprocessing.gsi import gaussian_smooth, linear_interpolation
from boxmot.postprocessing.gta import gta


def test_gsi():
    tracking_results = np.array(
        [
            [1, 1, 1475, 419, 75, 169, 0, 0, -1],
            [2, 1, 1475, 419, 75, 169, 0, 0, -1],
            [4, 1, 1475, 419, 75, 169, 0, 0, -1],
            [6, 1, 1475, 419, 75, 169, 0, 0, -1],
        ]
    )
    li = linear_interpolation(tracking_results, interval=20)
    gsi = gaussian_smooth(li, tau=10)
    assert len(gsi) == 6


def test_gta_merging(tmp_path):
    data = np.array(
        [
            [1, 1, 0, 0, 1, 1, 0.9, -1, -1],
            [2, 1, 0, 0, 1, 1, 0.9, -1, -1],
            [5, 2, 0, 0, 1, 1, 0.9, -1, -1],
            [6, 2, 0, 0, 1, 1, 0.9, -1, -1],
        ]
    )
    mot_file = tmp_path / "MOT17-01.txt"
    np.savetxt(mot_file, data, delimiter=",", fmt="%.1f")

    gta(mot_results_folder=tmp_path, use_split=False, use_connect=True, merge_dist_thres=0.5)

    processed = np.loadtxt(mot_file, delimiter=",")
    assert np.unique(processed[:, 1]).size == 1
