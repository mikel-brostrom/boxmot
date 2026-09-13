"""KITTI instance-bound evaluation without prediction masks or COCO tooling."""

from __future__ import annotations

import builtins
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from boxmot.engine.eval.kitti_boxes import _box_ignore_ioa, run_kitti_box_metrics


def _row(frame: int, identity: int, class_id: int, box: tuple[float, float, float, float]) -> str:
    """Write BoxMOT's AABB9 format, with one-based frames and xywh bounds."""
    return f"{frame},{identity},{','.join(map(str, box))},0.9,{class_id},-1\n"


def _fixture(
    tmp_path: Path,
    labels: list[np.ndarray],
    rows: str = "",
    *,
    sequence: str = "0000",
    frame_indices: list[int] | None = None,
) -> Namespace:
    """Create annotation files with catalog frame IDs independent of filenames."""
    annotations = tmp_path / "instances" / sequence
    annotations.mkdir(parents=True, exist_ok=True)
    exp_dir = tmp_path / "tracker"
    exp_dir.mkdir(exist_ok=True)
    frames = []
    frame_indices = list(range(len(labels))) if frame_indices is None else frame_indices
    for file_index, (frame_index, labels_image) in enumerate(zip(frame_indices, labels)):
        path = annotations / f"{file_index * 2:06d}.png"
        assert cv2.imwrite(str(path), labels_image)
        frames.append((frame_index, path, *labels_image.shape[:2]))
    (exp_dir / f"{sequence}.txt").write_text(rows)
    return Namespace(exp_dir=exp_dir, evaluation_config={"mots_gt_frames": {sequence: frames}})


def _evaluate(args: Namespace, *, seq_info: dict[str, int] | None = None) -> dict:
    """Use the evaluation/tuning adapter with the exact selected frame metadata."""
    return run_kitti_box_metrics(
        args,
        [Path(name) for name in args.evaluation_config["mots_gt_frames"]],
        Path("unused"),
        Path("unused"),
        seq_info=seq_info,
    )


def test_perfect_holey_instance_bounds_need_no_pycocotools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = np.eye(7, dtype=np.uint16) * 1001
    args = _fixture(tmp_path, [labels], _row(1, 8, 1, (0, 0, 7, 7)))
    original_import = builtins.__import__

    def reject_coco(name: str, *args: object, **kwargs: object) -> object:
        """Fail on any attempted import of optional mask evaluation tooling."""
        if name.startswith("pycocotools"):
            raise AssertionError("Box evaluation must not require pycocotools")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_coco)

    result = _evaluate(args)["car"]

    assert result["HOTA"] == result["MOTA"] == result["MOTP"] == result["sMOTA"] == result["IDF1"] == 100
    assert result["GT_Dets"] == result["Dets"] == 1


def test_multiclass_and_sequence_aggregation(tmp_path: Path) -> None:
    labels = np.zeros((8, 10), dtype=np.uint16)
    labels[1:4, 1:4] = 1000
    labels[4:7, 6:9] = 2000
    rows = "".join(
        _row(frame, identity, class_id, box)
        for frame in (1, 2)
        for identity, class_id, box in ((8, 1, (1, 1, 3, 3)), (9, 2, (6, 4, 3, 3)))
    )
    args = _fixture(tmp_path, [labels, labels], rows)
    second = _fixture(tmp_path, [labels], _row(1, 8, 1, (1, 1, 3, 3)), sequence="0001")
    args.evaluation_config["mots_gt_frames"].update(second.evaluation_config["mots_gt_frames"])

    results = _evaluate(args)

    assert results["car"]["HOTA"] == results["car"]["sMOTA"] == 100
    assert results["car"]["GT_Dets"] == 3
    assert results["car"]["GT_IDs"] == results["car"]["IDs"] == 2
    assert results["car"]["Frames"] == 3
    assert results["pedestrian"]["MOTA"] == pytest.approx(200 / 3)
    assert results["cls_comb_det_av"]["GT_Dets"] == 6
    assert results["cls_comb_det_av"]["Dets"] == 5
    assert results["cls_comb_det_av"]["MOTA"] == pytest.approx(500 / 6)


def test_box_localization_changes_metrics(tmp_path: Path) -> None:
    labels = np.full((2, 4), 1001, dtype=np.uint16)
    args = _fixture(tmp_path, [labels], _row(1, 3, 1, (0, 0, 3, 2)))

    result = _evaluate(args)["car"]

    assert result["MOTA"] == 100
    assert result["MOTP"] == result["sMOTA"] == 75
    assert result["HOTA"] == pytest.approx(1500 / 19)


@pytest.mark.parametrize(("ignored_columns", "remaining"), [(2, 1), (3, 0)])
def test_only_strict_majority_ignore_coverage_removes_boxes(
    tmp_path: Path, ignored_columns: int, remaining: int
) -> None:
    labels = np.zeros((3, 5), dtype=np.uint16)
    labels[0, :ignored_columns] = 10000
    args = _fixture(tmp_path, [labels], _row(1, 9, 1, (0, 0, 4, 1)))

    assert _evaluate(args)["car"]["Dets"] == remaining


def test_ignore_area_accounts_for_fractional_edges_and_unclipped_box_area() -> None:
    ignore_mask = np.array([[True, False], [False, True]])
    boxes = np.array([[0.5, 0.25, 1, 1], [-1, -1, 2, 2], [3, 3, 1, 1], [0, 0, 2, 2]])

    np.testing.assert_allclose(_box_ignore_ioa(boxes, ignore_mask), [0.5, 0.25, 0, 0.5])


def test_matching_preserves_box_over_ignored_holes_but_removes_unmatched_duplicate(tmp_path: Path) -> None:
    labels = np.full((7, 7), 10000, dtype=np.uint16)
    np.fill_diagonal(labels, 1001)
    args = _fixture(tmp_path, [labels], _row(1, 8, 1, (0, 0, 7, 7)) + _row(1, 9, 1, (0, 0, 7, 7)))

    result = _evaluate(args)["car"]

    assert result["Dets"] == result["CLR_TP"] == 1
    assert result["CLR_FP"] == 0
    assert result["HOTA"] == 100


def test_class_filtering_does_not_match_other_class(tmp_path: Path) -> None:
    labels = np.full((3, 3), 1001, dtype=np.uint16)
    args = _fixture(tmp_path, [labels], _row(1, 9, 2, (0, 0, 3, 3)))

    result = _evaluate(args)

    assert result["car"]["CLR_FN"] == result["pedestrian"]["CLR_FP"] == 1
    args.classes = [1]
    assert set(_evaluate(args)) == {"car"}


def test_identity_switch_changes_association_metrics(tmp_path: Path) -> None:
    labels = np.full((3, 3), 1001, dtype=np.uint16)
    args = _fixture(tmp_path, [labels, labels], _row(1, 9, 1, (0, 0, 3, 3)) + _row(2, 10, 1, (0, 0, 3, 3)))

    result = _evaluate(args)["car"]

    assert result["IDSW"] == 1
    assert result["MOTA"] == result["sMOTA"] == result["IDF1"] == 50
    assert result["HOTA"] == pytest.approx(np.sqrt(0.5) * 100)


@pytest.mark.parametrize("frame_indices", [[0, 1], [4, 7]])
def test_catalog_mapping_supports_fps_remapping_and_sparse_native_ids(tmp_path: Path, frame_indices: list[int]) -> None:
    labels = np.full((3, 3), 1001, dtype=np.uint16)
    args = _fixture(
        tmp_path,
        [labels, labels],
        "".join(_row(index + 1, 9, 1, (0, 0, 3, 3)) for index in frame_indices),
        frame_indices=frame_indices,
    )

    result = _evaluate(args, seq_info={"0000": max(frame_indices) + 1})["car"]

    assert result["HOTA"] == 100
    assert result["GT_Dets"] == 2
    assert result["Frames"] == max(frame_indices) + 1


@pytest.mark.parametrize("frame", [0, 1, 4, 6])
def test_rejects_box_rows_outside_selected_frames(tmp_path: Path, frame: int) -> None:
    args = _fixture(tmp_path, [np.zeros((2, 2), dtype=np.uint16)], _row(frame, 9, 1, (0, 0, 1, 1)), frame_indices=[4])

    with pytest.raises(ValueError, match="frame"):
        _evaluate(args)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("broken", "Invalid"),
        ("1,8,0,0,1,1,1,1", "nine finite"),
        ("1,8,0,0,nan,1,1,1,-1", "nine finite"),
        ("1,8.5,0,0,1,1,1,1,-1", "integers"),
        ("1,-1,0,0,1,1,1,1,-1", "non-negative int64"),
        ("1,1e30,0,0,1,1,1,1,-1", "non-negative int64"),
        ("1,8,0,0,0,1,1,1,-1", "positive box area"),
        ("1,8,0,0,1,-1,1,1,-1", "positive box area"),
        ("1,8,0,0,1,1,1,3,-1", "class"),
    ],
)
def test_invalid_box_results_are_rejected(tmp_path: Path, row: str, message: str) -> None:
    args = _fixture(tmp_path, [np.zeros((2, 2), dtype=np.uint16)], row)

    with pytest.raises(ValueError, match=message):
        _evaluate(args)


def test_duplicate_ids_across_classes_are_rejected(tmp_path: Path) -> None:
    args = _fixture(
        tmp_path,
        [np.zeros((2, 2), dtype=np.uint16)],
        _row(1, 9, 1, (0, 0, 1, 1)) + _row(1, 9, 2, (1, 1, 1, 1)),
    )

    with pytest.raises(ValueError, match="Duplicate"):
        _evaluate(args)


@pytest.mark.parametrize("has_gt", [False, True])
def test_empty_results_file_is_valid(tmp_path: Path, has_gt: bool) -> None:
    args = _fixture(tmp_path, [np.full((2, 2), 1001 if has_gt else 0, dtype=np.uint16)])

    result = _evaluate(args)["car"]

    assert result["Dets"] == 0
    assert result["GT_Dets"] == result["CLR_FN"] == int(has_gt)
    assert result["Frames"] == 1


def test_missing_results_file_is_rejected(tmp_path: Path) -> None:
    args = _fixture(tmp_path, [np.zeros((2, 2), dtype=np.uint16)])
    (args.exp_dir / "0000.txt").unlink()

    with pytest.raises(FileNotFoundError):
        _evaluate(args)


@pytest.mark.parametrize(
    ("labels", "message"),
    [(np.zeros((2, 2), dtype=np.uint8), "uint16"), (np.full((2, 2), 3000, dtype=np.uint16), "unsupported labels")],
)
def test_invalid_instance_pngs_are_rejected(tmp_path: Path, labels: np.ndarray, message: str) -> None:
    args = _fixture(tmp_path, [labels])

    with pytest.raises(ValueError, match=message):
        _evaluate(args)
