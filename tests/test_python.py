# pytest tests/test_python.py

import numpy as np
import torch
from pathlib import Path

from boxmot.utils import WEIGHTS
from boxmot.trackers import StrongSORT
from boxmot.trackers import OCSort as OCSORT
from boxmot.trackers import BYTETracker
from boxmot.trackers import BoTSORT
from boxmot.trackers import DeepOCSort as DeepOCSORT
from boxmot.tracker_zoo import create_tracker, get_tracker_config


def test_tracker_output():
    tracker_conf = get_tracker_config('deepocsort')
    tracker = create_tracker(
        tracker_type='deepocsort',
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=False
    )
    rgb = np.random.randint(255, size=(640, 640, 3),dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])
    output = tracker.update(det, rgb)
    assert output.shape == (2, 7)  # two inputs should give two outputs


def test_strongsort_instantiation():
    ss = StrongSORT(
        model_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
        device='cpu',
        fp16=True,
    )


def test_botsort_instantiation():
    bs = BoTSORT(
        model_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
        device='cpu',
        fp16=True,
    )
    

def test_deepocsort_instantiation():
    dos = DeepOCSORT(
        model_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
        device='cpu',
        fp16=True,
        per_class=False
    )


def test_ocsort_instantiation():
    os = OCSORT(
        per_class=False
    )


def test_bytetrack_instantiation():
    bt = BYTETracker()
    

