import numpy as np

from boxmot.postprocessing.gsi import gaussian_smooth, linear_interpolation
from boxmot.postprocessing.gbrc import gradient_boosting_smooth
from boxmot.postprocessing.gbrc import linear_interpolation as gbrc_linear_interpolation


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


def test_gbrc():
    tracking_results = np.array(
        [
            [1, 1, 1475, 419, 75, 169, 0, 0, -1],
            [2, 1, 1475, 419, 75, 169, 0, 0, -1],
            [4, 1, 1475, 419, 75, 169, 0, 0, -1],
            [6, 1, 1475, 419, 75, 169, 0, 0, -1],
        ]
    )
    li = gbrc_linear_interpolation(tracking_results, interval=20)
    gbrc = gradient_boosting_smooth(li)
    assert len(gbrc) == 6
    assert gbrc.shape[1] == 10
