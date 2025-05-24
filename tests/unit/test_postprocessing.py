import numpy as np

from boxmot.postprocessing.gsi import gaussian_smooth, linear_interpolation


def test_gsi():
    tracking_results = np.array([
        [1, 1, 1475, 419, 75, 169, 0, 0, -1],
        [2, 1, 1475, 419, 75, 169, 0, 0, -1],
        [4, 1, 1475, 419, 75, 169, 0, 0, -1],
        [6, 1, 1475, 419, 75, 169, 0, 0, -1]
    ])
    li = linear_interpolation(tracking_results, interval=20)
    gsi = gaussian_smooth(li, tau=10)
    assert len(gsi) == 6