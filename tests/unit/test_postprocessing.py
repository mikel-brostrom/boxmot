import numpy as np
import pytest

from boxmot.postprocessing import MotFilePostprocessor, Postprocessor, create_postprocessor, supported_postprocessors
from boxmot.postprocessing.gbrc import gradient_boosting_smooth
from boxmot.postprocessing.gbrc import linear_interpolation as gbrc_linear_interpolation
from boxmot.postprocessing.gsi import gaussian_smooth, linear_interpolation


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
    assert gbrc.shape[1] == 9


def test_postprocessor_factory_creates_file_postprocessors():
    assert supported_postprocessors() == ("gsi", "gbrc", "gta")

    gsi_postprocessor = create_postprocessor("gsi", interval=7, tau=3)
    gbrc_postprocessor = create_postprocessor("gbrc", interval=9)

    assert isinstance(gsi_postprocessor, Postprocessor)
    assert isinstance(gsi_postprocessor, MotFilePostprocessor)
    assert gsi_postprocessor.interval == 7
    assert gsi_postprocessor.tau == 3

    assert isinstance(gbrc_postprocessor, Postprocessor)
    assert isinstance(gbrc_postprocessor, MotFilePostprocessor)
    assert gbrc_postprocessor.interval == 9


def test_postprocessor_factory_rejects_unknown_step():
    with pytest.raises(ValueError, match="Unknown postprocessing step"):
        create_postprocessor("missing")
