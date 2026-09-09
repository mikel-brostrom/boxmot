"""Pixel geometry, KITTI ignore rules, and report behavior for MOTS."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from boxmot.engine.eval.mots import _mask_ious, _preprocess_frame, run_mots_metrics

mask_utils = pytest.importorskip("pycocotools.mask")


def _encode(mask: np.ndarray) -> dict:
    """Encode a fixture independently of the production results writer."""
    return mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))


def _row(frame: int, identity: int, class_id: int, mask: np.ndarray) -> str:
    """Write an official MOTS row with a compressed COCO mask."""
    return f"{frame} {identity} {class_id} {mask.shape[0]} {mask.shape[1]} {_encode(mask)['counts'].decode('ascii')}\n"


def _fixture(
    tmp_path: Path,
    labels: list[np.ndarray],
    rows: str = "",
    *,
    seq_name: str = "0000",
    frame_indices: list[int] | None = None,
) -> Namespace:
    """Create selected ground-truth metadata and one tracker results file."""
    annotations = tmp_path / "instances" / seq_name
    annotations.mkdir(parents=True, exist_ok=True)
    exp_dir = tmp_path / "tracker"
    exp_dir.mkdir(exist_ok=True)
    frames = []
    for index, image in zip(range(len(labels)) if frame_indices is None else frame_indices, labels):
        path = annotations / f"{index:06d}.png"
        assert cv2.imwrite(str(path), image)
        frames.append((index, path, *image.shape[:2]))
    (exp_dir / f"{seq_name}.txt").write_text(rows)
    return Namespace(exp_dir=exp_dir, evaluation_config={"mots_gt_frames": {seq_name: tuple(frames)}})


def _evaluate(args: Namespace, *, seq_info: dict[str, int] | None = None) -> dict:
    """Invoke the same public runner used by evaluation and tuning."""
    seq_paths = [Path(name) for name in args.evaluation_config["mots_gt_frames"]]
    return run_mots_metrics(args, seq_paths, Path("unused"), Path("unused"), seq_info=seq_info)


def test_perfect_multiclass_masks_and_sequence_aggregation(tmp_path: Path) -> None:
    labels = np.zeros((8, 10), dtype=np.uint16)
    labels[1:4, 1:4] = 1001
    labels[4:7, 6:9] = 2001
    args = _fixture(
        tmp_path,
        [labels, labels],
        "".join(
            _row(frame, identity, class_id, labels == label)
            for frame in range(2)
            for identity, class_id, label in ((8, 1, 1001), (9, 2, 2001))
        ),
    )
    second = _fixture(tmp_path, [labels], _row(0, 8, 1, labels == 1001), seq_name="0001")
    args.evaluation_config["mots_gt_frames"].update(second.evaluation_config["mots_gt_frames"])

    results = _evaluate(args)

    assert set(results) == {"car", "pedestrian", "cls_comb_cls_av", "cls_comb_det_av"}
    assert results["car"]["HOTA"] == pytest.approx(100)
    assert results["car"]["sMOTA"] == pytest.approx(100)
    assert results["car"]["GT_Dets"] == 3
    assert results["car"]["GT_IDs"] == 2
    assert results["car"]["IDs"] == 2
    assert results["car"]["Frames"] == 3
    assert results["pedestrian"]["HOTA"] == pytest.approx(np.sqrt(2 / 3) * 100)
    assert results["pedestrian"]["MOTA"] == pytest.approx(200 / 3)
    assert results["cls_comb_det_av"]["GT_Dets"] == 6
    assert results["cls_comb_det_av"]["Dets"] == 5
    assert results["cls_comb_det_av"]["MOTA"] == pytest.approx(500 / 6)
    assert set(results["car"]["per_sequence"]) == {"0000", "0001"}


def test_masks_with_identical_bounding_boxes_have_different_iou(tmp_path: Path) -> None:
    mask = np.eye(7, dtype=bool)
    prediction = np.fliplr(mask)
    labels = mask.astype(np.uint16) * 1001
    args = _fixture(tmp_path, [labels], _row(0, 5, 1, prediction))

    result = _evaluate(args)["car"]

    assert _mask_ious([_encode(mask)], [_encode(prediction)])[0, 0] == pytest.approx(1 / 13)
    assert result["HOTA"] == pytest.approx(100 / 19)
    assert result["MOTA"] == -100
    assert result["CLR_FN"] == result["CLR_FP"] == 1


def test_partial_mask_localization_affects_soft_accuracy(tmp_path: Path) -> None:
    labels = np.zeros((4, 6), dtype=np.uint16)
    labels[0:2, 0:4] = 1001
    prediction = np.zeros_like(labels, dtype=bool)
    prediction[0:2, 0:3] = True
    args = _fixture(tmp_path, [labels], _row(0, 3, 1, prediction))

    result = _evaluate(args)["car"]

    assert result["MOTA"] == 100
    assert result["MOTP"] == result["sMOTA"] == 75
    assert result["HOTA"] == pytest.approx(1500 / 19)


@pytest.mark.parametrize(("ignored_columns", "remaining"), [(2, 1), (3, 0)])
def test_only_strict_majority_of_ignore_removes_prediction(
    tmp_path: Path, ignored_columns: int, remaining: int
) -> None:
    labels = np.zeros((3, 5), dtype=np.uint16)
    labels[0, :ignored_columns] = 10000
    prediction = np.zeros_like(labels, dtype=bool)
    prediction[0, :4] = True
    args = _fixture(tmp_path, [labels], _row(0, 9, 1, prediction))

    assert _evaluate(args)["car"]["Dets"] == remaining


def test_ignore_preprocessing_preserves_matched_predictions() -> None:
    mask = _encode(np.ones((3, 4), dtype=bool))
    # An overlapping synthetic ignore region isolates the matched/unmatched rule.
    tracker_ids, similarity = _preprocess_frame(np.array([3]), [mask], np.array([9]), [mask], mask)

    np.testing.assert_array_equal(tracker_ids, [9])
    np.testing.assert_array_equal(similarity, [[1]])


def test_ignore_matching_includes_exact_half_iou() -> None:
    ground_truth = np.ones((2, 4), dtype=bool)
    prediction = ground_truth.copy()
    prediction[:, 2:] = False
    tracker_ids, similarity = _preprocess_frame(
        np.array([3]), [_encode(ground_truth)], np.array([9]), [_encode(prediction)], _encode(prediction)
    )

    np.testing.assert_array_equal(tracker_ids, [9])
    np.testing.assert_array_equal(similarity, [[0.5]])


def test_ignore_removal_happens_separately_for_each_class(tmp_path: Path) -> None:
    labels = np.full((3, 3), 1001, dtype=np.uint16)
    args = _fixture(tmp_path, [labels], _row(0, 9, 2, labels == 1001))

    results = _evaluate(args)

    assert results["car"]["CLR_FN"] == 1
    assert results["car"]["CLR_TP"] == 0
    assert results["pedestrian"]["CLR_FP"] == 1
    assert results["pedestrian"]["GT_Dets"] == 0


def test_identity_switches_reduce_association_metrics(tmp_path: Path) -> None:
    labels = np.full((3, 3), 1001, dtype=np.uint16)
    args = _fixture(tmp_path, [labels, labels], _row(0, 9, 1, labels == 1001) + _row(1, 10, 1, labels == 1001))

    result = _evaluate(args)["car"]

    assert result["IDSW"] == 1
    assert result["MOTA"] == result["sMOTA"] == result["IDF1"] == 50
    assert result["HOTA"] == pytest.approx(np.sqrt(0.5) * 100)


@pytest.mark.parametrize("has_gt", [False, True])
def test_empty_tracker_file_is_valid(tmp_path: Path, has_gt: bool) -> None:
    labels = np.full((3, 3), 1001 if has_gt else 0, dtype=np.uint16)
    args = _fixture(tmp_path, [labels])

    result = _evaluate(args)["car"]

    assert result["HOTA"] == 0
    assert result["Dets"] == 0
    assert result["GT_Dets"] == int(has_gt)
    assert result["CLR_FN"] == int(has_gt)
    assert result["Frames"] == 1


def test_sparse_frame_indices_preserve_timeline_and_exact_annotation_pairing(tmp_path: Path) -> None:
    labels = np.full((3, 3), 1001, dtype=np.uint16)
    args = _fixture(
        tmp_path,
        [labels, labels],
        _row(4, 9, 1, labels == 1001) + _row(7, 9, 1, labels == 1001),
        frame_indices=[4, 7],
    )

    result = _evaluate(args, seq_info={"0000": 8})["car"]

    assert result["HOTA"] == 100
    assert result["GT_Dets"] == 2
    assert result["Frames"] == 8


@pytest.mark.parametrize("frame", [0, 5, 8])
def test_rejects_tracker_rows_outside_selected_frames(tmp_path: Path, frame: int) -> None:
    labels = np.full((3, 3), 1001, dtype=np.uint16)
    args = _fixture(tmp_path, [labels], _row(frame, 9, 1, labels == 1001), frame_indices=[4])

    with pytest.raises(ValueError, match="[Ff]rame|timestep"):
        _evaluate(args, seq_info={"0000": 8})


def test_missing_tracker_results_are_rejected(tmp_path: Path) -> None:
    args = _fixture(tmp_path, [np.zeros((2, 2), dtype=np.uint16)])
    (args.exp_dir / "0000.txt").unlink()

    with pytest.raises(FileNotFoundError):
        _evaluate(args)


@pytest.mark.parametrize("class_id", [0, 3, 10])
def test_unknown_tracker_classes_are_rejected(tmp_path: Path, class_id: int) -> None:
    labels = np.full((2, 2), 1001, dtype=np.uint16)
    args = _fixture(tmp_path, [labels], _row(0, 9, class_id, labels == 1001))

    with pytest.raises(ValueError, match="[Cc]lass"):
        _evaluate(args)


def test_duplicate_ids_across_classes_are_rejected(tmp_path: Path) -> None:
    labels = np.array([[1001, 2001]], dtype=np.uint16)
    args = _fixture(tmp_path, [labels], _row(0, 9, 1, labels == 1001) + _row(0, 9, 2, labels == 2001))

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        _evaluate(args)


def test_overlapping_masks_across_classes_are_rejected(tmp_path: Path) -> None:
    labels = np.full((2, 2), 1001, dtype=np.uint16)
    args = _fixture(tmp_path, [labels], _row(0, 9, 1, labels == 1001) + _row(0, 10, 2, labels == 1001))

    with pytest.raises(ValueError, match="[Oo]verlap"):
        _evaluate(args)


@pytest.mark.parametrize("invalid_label", [1, 999, 3000, 10001, 65535])
def test_invalid_annotation_labels_are_rejected(tmp_path: Path, invalid_label: int) -> None:
    args = _fixture(tmp_path, [np.full((2, 2), invalid_label, dtype=np.uint16)])

    with pytest.raises(ValueError, match="unsupported labels"):
        _evaluate(args)


def test_uint8_annotations_are_rejected(tmp_path: Path) -> None:
    args = _fixture(tmp_path, [np.zeros((2, 2), dtype=np.uint8)])

    with pytest.raises(ValueError, match="uint16"):
        _evaluate(args)


def test_annotation_dimensions_must_match_catalog(tmp_path: Path) -> None:
    args = _fixture(tmp_path, [np.zeros((2, 2), dtype=np.uint16)])
    frame, path, _, _ = args.evaluation_config["mots_gt_frames"]["0000"][0]
    args.evaluation_config["mots_gt_frames"]["0000"] = ((frame, path, 3, 2),)

    with pytest.raises(ValueError, match="dimensions"):
        _evaluate(args)


def test_explicit_class_selection(tmp_path: Path) -> None:
    args = _fixture(tmp_path, [np.zeros((2, 2), dtype=np.uint16)])
    args.remapped_class_names = ["pedestrian"]
    args.remapped_class_ids = [2]

    assert set(_evaluate(args)) == {"pedestrian"}


@pytest.mark.parametrize("class_ids", [[], [0], [3], [1, 2, 10]])
def test_invalid_selected_classes_are_rejected(tmp_path: Path, class_ids: list[int]) -> None:
    args = _fixture(tmp_path, [np.zeros((2, 2), dtype=np.uint16)])
    args.classes = class_ids

    with pytest.raises(ValueError, match="only car"):
        _evaluate(args)
