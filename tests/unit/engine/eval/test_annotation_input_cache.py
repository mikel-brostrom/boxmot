"""Image mask and box scoring reuse ground truth while every trial stays fresh."""

from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from boxmot.engine.eval.kitti_boxes import run_kitti_box_metrics
from boxmot.engine.eval.mots import run_mots_metrics


@pytest.mark.parametrize("masks", [False, True])
def test_scoring_caches_png_inputs_but_not_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, masks: bool
) -> None:
    from pycocotools import mask as mask_utils

    labels = np.zeros((6, 8), dtype=np.uint16)
    labels[1:4, 2:5] = 1001
    labels[0] = 10000
    annotation = tmp_path / "000000.png"
    assert cv2.imwrite(str(annotation), labels)
    prediction = tmp_path / "0000.txt"
    if masks:
        counts = mask_utils.encode(np.asfortranarray(labels == 1001, dtype=np.uint8))["counts"].decode("ascii")
        rows = f"0 7 1 6 8 {counts}\n"
    else:
        rows = "1,7,2,1,3,3,1,1,-1\n"
    prediction.write_text(rows)
    args = Namespace(
        exp_dir=tmp_path,
        cache_inputs=False,
        evaluation_config={"mots_gt_frames": {"0000": [(0, annotation, 6, 8)]}},
    )
    evaluate = run_mots_metrics if masks else run_kitti_box_metrics

    def score() -> dict:
        return evaluate(args, [Path("0000")], tmp_path, tmp_path, seq_info={"0000": 1})

    expected = score()
    assert expected["car"]["HOTA"] == 100
    args.cache_inputs = True
    assert score() == expected
    read_image = cv2.imread
    reads = []

    def counted(path: str, flags: int) -> np.ndarray:
        reads.append(path)
        return read_image(path, flags)

    monkeypatch.setattr(cv2, "imread", counted)
    assert score() == expected
    assert reads == []
    prediction.write_text("")
    assert score()["car"]["HOTA"] == 0
    prediction.write_text(rows)
    assert score() == expected
    assert reads == []
    labels[4:6, 6:8] = 1002
    assert cv2.imwrite(str(annotation), labels)
    assert score()["car"]["GT_Dets"] == 2
    assert reads == [str(annotation)]
