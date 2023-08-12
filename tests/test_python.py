# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

# pytest tests/test_python.py

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from boxmot import (OCSORT, BoTSORT, BYTETracker, DeepOCSORT, StrongSORT,
                    create_tracker, get_tracker_config)
from boxmot.postprocessing.gsi import gaussian_smooth, linear_interpolation
from boxmot.utils import WEIGHTS


def test_strongsort_instantiation():
    StrongSORT(
        model_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
        device='cpu',
        fp16=True,
    )


def test_botsort_instantiation():
    BoTSORT(
        model_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
        device='cpu',
        fp16=True,
    )


def test_deepocsort_instantiation():
    DeepOCSORT(
        model_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
        device='cpu',
        fp16=True,
        per_class=False
    )


def test_ocsort_instantiation():
    OCSORT(
        per_class=False
    )


def test_bytetrack_instantiation():
    BYTETracker()


def test_deepocsort_output():
    tracker_conf = get_tracker_config('deepocsort')
    tracker = create_tracker(
        tracker_type='deepocsort',
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=False
    )
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])
    output = tracker.update(det, rgb)
    # Works since frame count is less than min hits (1 <= 2)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = np.flip(np.delete(output, [4, 7], axis=1), axis=0)
    assert_allclose(det, output, atol=1, rtol=7e-3, verbose=True)

    # Instantiate new tracker and ensure minimum number of hits works
    tracker = create_tracker(
        tracker_type='deepocsort',
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=False
    )
    tracker.min_hits = 2
    output = tracker.update(np.empty((0, 6)), rgb)
    assert output.size == 0
    output = tracker.update(np.empty((0, 6)), rgb)
    assert output.size == 0
    output = tracker.update(det, rgb)
    assert output.size == 0
    output = tracker.update(det, rgb)
    assert output.size == 0
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = np.flip(np.delete(output, [4, 7], axis=1), axis=0)
    assert_allclose(det, output, atol=1, rtol=7e-3, verbose=True)


def test_ocsort_output():
    tracker_conf = get_tracker_config('ocsort')
    tracker = create_tracker(
        tracker_type='ocsort',
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=False
    )
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])
    output = tracker.update(det, rgb)
    # Works since frame count is less than min hits (1 <= 2)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = np.flip(np.delete(output, [4, 7], axis=1), axis=0)
    assert_allclose(det, output, atol=1, rtol=7e-3, verbose=True)

    # Instantiate new tracker and ensure minimum number of hits works
    tracker = create_tracker(
        tracker_type='ocsort',
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=False
    )
    tracker.min_hits = 2
    output = tracker.update(np.empty((0, 6)), rgb)
    assert output.size == 0
    output = tracker.update(np.empty((0, 6)), rgb)
    assert output.size == 0
    output = tracker.update(det, rgb)
    assert output.size == 0
    output = tracker.update(det, rgb)
    assert output.size == 0
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = np.flip(np.delete(output, [4, 7], axis=1), axis=0)
    assert_allclose(det, output, atol=1, rtol=7e-3, verbose=True)


def test_botsort_output():
    tracker_conf = get_tracker_config('botsort')
    tracker = create_tracker(
        tracker_type='botsort',
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=False
    )
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = np.delete(output, [4, 7], axis=1)
    assert_allclose(det, output, atol=1, rtol=7e-3, verbose=True)


def test_bytetrack_output():
    tracker_conf = get_tracker_config('bytetrack')
    tracker = create_tracker(
        tracker_type='bytetrack',
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=False
    )
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.86, 65]])
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = np.delete(output, [4, 7], axis=1)
    assert_allclose(det, output, atol=1, rtol=7e-3, verbose=True)


def test_strongsort_output():
    tracker_conf = get_tracker_config('strongsort')
    tracker = create_tracker(
        tracker_type='strongsort',
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=False
    )
    tracker.n_init = 1
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])
    output = tracker.update(det, rgb)
    assert output.size == 0
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    output = np.delete(output, [4, 7], axis=1)
    assert_allclose(det, output, atol=1, rtol=7e-3, verbose=True)


def test_gsi():
    tracking_results = np.array([
        [1, 1, 1475, 419, 75, 169, 0, 0, -1],
        [2, 1, 1475, 419, 75, 169, 0, 0, -1],
        [4, 1, 1475, 419, 75, 169, 0, 0, -1],
    ])
    li = linear_interpolation(tracking_results, interval=20)
    gsi = gaussian_smooth(li, tau=10)
    assert len(gsi) == 4
