import pytest
import numpy as np
from pathlib import Path

from boxmot import OcSort, DeepOcSort
from boxmot.utils import WEIGHTS
from boxmot.trackers.ocsort.ocsort import KalmanBoxTracker as OCSortKalmanBoxTracker
from boxmot.trackers.deepocsort.deepocsort import (
    KalmanBoxTracker as DeepOCSortKalmanBoxTracker,
)

TRACKER_CREATORS = {
    OcSort: lambda bbox, cls, idx, tr: OCSortKalmanBoxTracker(
        bbox, cls, idx,
        Q_xy_scaling=tr.Q_xy_scaling,
        Q_s_scaling=tr.Q_s_scaling
    ),
    DeepOcSort: lambda bbox, cls, idx, tr: DeepOCSortKalmanBoxTracker(
        np.concatenate([bbox, [cls, idx]]),
        Q_xy_scaling=tr.Q_xy_scaling,
        Q_s_scaling=tr.Q_s_scaling
    ),
}

@pytest.mark.parametrize("Tracker, init_args", [
    (OcSort, {}),
    (DeepOcSort, {
        "reid_weights": Path(WEIGHTS / "osnet_x0_25_msmt17.pt"),
        "device": "cpu", "half": True
    }),
])
def test_Q_matrix_scaling(Tracker, init_args):
    bbox = np.array([0, 0, 100, 100, 0.9])
    cls, idx = 1, 0
    scale_xy, scale_s = 0.05, 0.0005

    tracker = Tracker(
        Q_xy_scaling=scale_xy,
        Q_s_scaling=scale_s,
        **init_args
    )
    kf = TRACKER_CREATORS[Tracker](bbox, cls, idx, tracker).kf

    assert kf.Q[4, 4] == scale_xy
    assert kf.Q[5, 5] == scale_xy
    assert kf.Q[6, 6] == scale_s
