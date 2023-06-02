# pytest tests/test_python.py

import numpy as np
import torch
from pathlib import Path

from boxmot.utils import WEIGHTS
from boxmot.strongsort.strong_sort import StrongSORT
from boxmot.ocsort.ocsort import OCSort as OCSORT
from boxmot.bytetrack.byte_tracker import BYTETracker
from boxmot.botsort.bot_sort import BoTSORT
from boxmot.deepocsort.ocsort import OCSort as DeepOCSORT
from boxmot.tracker_zoo import create_tracker, get_tracker_config


def test_tracker_output():
    tracker_conf = get_tracker_config('deepocsort')
    tracker = create_tracker(
        'deepocsort',
        tracker_conf,
        WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        'cpu',
        False
    )
    rgb = np.random.randint(255, size=(640, 640, 3),dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])
    det = torch.from_numpy(det)
    output = tracker.update(det, rgb)
    assert output.shape == (2, 7)  # two inputs should give two outputs
    
def test_strongsort():
    ss = StrongSORT()
    
def test_ocsort():
    os = OCSORT()
    
def test_bytetrack():
    bt = BYTETracker()
    
def test_botsort():
    bs = BoTSORT()
    
def test_deepocsort():
    dos = DeepOCSORT()
