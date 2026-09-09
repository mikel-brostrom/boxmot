from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from boxmot.datasets.kitti_fusion import KittiFusionSequence, _decode_mask


def _pointgnn_row(label: str = "Car", score: str = "103") -> str:
    """Use distinct geometry values to expose incorrect KITTI field ordering."""
    return f"{label} -1 -1 0 0 0 4 3 1.5 2 4 5 6 20 0.25 {score}\n"


def _trackrcnn_row(frame: int = 0, **replacements: str) -> str:
    """Create a full-frame mask and the released 128-dimensional embedding tail."""
    fields = [str(frame), "0.25", "0.5", "3.75", "2.5", "0.9", "1", "3", "4", "0<"] + ["0"] * 128
    for field, value in replacements.items():
        fields[int(field)] = value
    return " ".join(fields) + "\n"


def _fixture(tmp_path: Path, frames: int = 3) -> tuple[Path, Path]:
    """Create aligned downloaded-style files without external data or models."""
    root = tmp_path / "eagermot-data"
    images = tmp_path / "training/image_02"
    image_sequence = images / "0000"
    image_sequence.mkdir(parents=True)
    for frame in range(frames):
        Image.new("RGB", (4, 3)).save(image_sequence / f"{frame:06d}.png")
    calibration = root / "calib/training/calib/0000.txt"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("P2: 100 0 2 0.4 0 100 1.5 0.2 0 0 1 0.0027\n")
    poses = root / "ego_motion/0000.npy"
    poses.parent.mkdir(parents=True)
    values = np.repeat(np.eye(4)[None], frames, axis=0)
    values[:, 0, 3] = np.arange(frames)
    np.save(poses, values)
    for folder in ("results_tracking_car_auto_t3_trainval", "results_tracking_ped_cyl_auto_trainval"):
        directory = root / "pointgnn/training" / folder / "0000/data"
        directory.mkdir(parents=True)
        for frame in range(frames):
            (directory / f"{frame:06d}.txt").write_text(_pointgnn_row("Car" if "car_auto" in folder else "Pedestrian"))
    trackrcnn = root / "trackrcnn_detections/0000.txt"
    trackrcnn.parent.mkdir(parents=True)
    trackrcnn.write_text(_trackrcnn_row())
    return root, images


def test_fusion_reorders_geometry_bounds_scores_and_preserves_projection(tmp_path: Path) -> None:
    root, images = _fixture(tmp_path)
    sequence = KittiFusionSequence(root, images, "0000")

    sample = sequence[0]

    assert len(sequence) == 3
    assert sequence.image_size == sample.image_size == (3, 4)
    assert sample.frame_index == 0
    assert sample.detections.sample_id == sample.detections_3d.sample_id == "train:0000:0"
    assert sample.detections.geometry.values.tolist() == [[0.25, 0.5, 3.75, 2.5]]
    assert sample.detections.embeddings is None
    assert sample.detections.masks.values.shape == (1, 3, 4)
    assert sample.detections.masks.values.all()
    assert sample.detections_3d.geometry.values.tolist() == [[5, 6, 20, 0.25, 4, 2, 1.5]] * 2
    assert sample.detections_3d.class_ids.tolist() == [1, 2]
    torch.testing.assert_close(sample.detections_3d.scores, torch.tensor([103 / 104, 103 / 104]))
    assert sample.camera.projection[2, 3].item() == pytest.approx(0.0027)
    assert sequence[2].camera.camera_to_world[0, 3].item() == 2


def test_fusion_retains_missing_3d_and_empty_2d_frames(tmp_path: Path) -> None:
    root, images = _fixture(tmp_path)
    for directory in (root / "pointgnn/training").glob("*/0000/data"):
        (directory / "000001.txt").unlink()
    sequence = KittiFusionSequence(root, images, "0000")

    assert sequence.missing_3d_frames == {"car": (1,), "pedestrian": (1,)}
    samples = list(sequence)
    assert [sample.frame_index for sample in samples] == [0, 1, 2]
    assert [len(sample.detections_3d) for sample in samples] == [2, 0, 2]
    assert len(samples[1].detections) == 0
    assert samples[1].detections.masks.values.shape == (0, 3, 4)
    assert samples[1].detections_3d.geometry.values.shape == (0, 7)
    assert sequence[-1].frame_index == 2
    assert [sample.frame_index for sample in sequence[1:]] == [1, 2]


def test_fusion_does_not_decode_rgb_or_decode_masks_before_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, images = _fixture(tmp_path)
    trackrcnn = root / "trackrcnn_detections/0000.txt"
    trackrcnn.write_text(_trackrcnn_row(2, **{"9": "malformed"}))

    def reject_pixels(*args: object, **kwargs: object) -> None:
        raise AssertionError("RGB pixels should never be loaded")

    monkeypatch.setattr(Image.Image, "load", reject_pixels)
    sequence = KittiFusionSequence(root, images, "0000")
    assert len(sequence[0].detections) == 0
    with pytest.raises(ValueError, match=r"0000.txt:1: RLE"):
        sequence[2]


def test_fusion_excludes_cyclists_and_retains_zero_area_masks(tmp_path: Path) -> None:
    root, images = _fixture(tmp_path)
    pedestrian = root / "pointgnn/training/results_tracking_ped_cyl_auto_trainval/0000/data/000000.txt"
    pedestrian.write_text(_pointgnn_row("Cyclist"))
    (root / "trackrcnn_detections/0000.txt").write_text(_trackrcnn_row(**{"9": "<"}))

    sample = KittiFusionSequence(root, images, "0000")[0]

    assert sample.detections_3d.class_ids.tolist() == [1]
    assert len(sample.detections) == 1
    assert not sample.detections.masks.values.any()


@pytest.mark.parametrize("raw_score,expected", [("0", 0.0), ("1", 0.5), ("1e308", 1.0)])
def test_fusion_handles_finite_extreme_pointgnn_scores(tmp_path: Path, raw_score: str, expected: float) -> None:
    root, images = _fixture(tmp_path)
    car = root / "pointgnn/training/results_tracking_car_auto_t3_trainval/0000/data/000000.txt"
    car.write_text(_pointgnn_row(score=raw_score))

    sample = KittiFusionSequence(root, images, "0000")[0]

    assert sample.detections_3d.scores[0].item() == expected


@pytest.mark.parametrize(
    "row,error",
    [
        (_pointgnn_row(score="-1"), "score must be nonnegative"),
        (_pointgnn_row(score="nan"), "must be finite"),
        (_pointgnn_row(score="inf"), "must be finite"),
        (_pointgnn_row("Truck"), "unsupported PointGNN class"),
        ("Car 1 2\n", "16 KITTI detection fields"),
        (_pointgnn_row().replace("1.5 2 4", "0 2 4"), "positive dimensions"),
        (_pointgnn_row().replace("1.5 2 4", "1e-100 2 4"), "positive in float32"),
    ],
)
def test_fusion_reports_pointgnn_file_and_line_for_invalid_input(tmp_path: Path, row: str, error: str) -> None:
    root, images = _fixture(tmp_path)
    car = root / "pointgnn/training/results_tracking_car_auto_t3_trainval/0000/data/000000.txt"
    car.write_text(row)
    sequence = KittiFusionSequence(root, images, "0000")

    with pytest.raises(ValueError, match=f"000000.txt:1: .*{error}"):
        sequence[0]


@pytest.mark.parametrize(
    "fields,error",
    [
        ({"0": "3"}, "no corresponding image"),
        ({"0": "-1"}, "nonnegative integers"),
        ({"6": "3"}, "class must be"),
        ({"7": "4"}, "mask dimensions"),
        ({"1": "nan"}, "finite float32"),
        ({"3": "0"}, "positive area"),
        ({"5": "2"}, "score must be"),
        ({"5": "1.000000001"}, "score must be"),
    ],
)
def test_fusion_rejects_unaligned_trackrcnn_rows(tmp_path: Path, fields: dict[str, str], error: str) -> None:
    root, images = _fixture(tmp_path)
    (root / "trackrcnn_detections/0000.txt").write_text(_trackrcnn_row(**fields))

    with pytest.raises(ValueError, match=f"0000.txt:1: .*{error}"):
        KittiFusionSequence(root, images, "0000")


def test_fusion_requires_detector_format_not_ground_truth(tmp_path: Path) -> None:
    root, images = _fixture(tmp_path)
    (root / "trackrcnn_detections/0000.txt").write_text("0 1001 1 3 4 0<\n")

    with pytest.raises(ValueError, match="138 fields"):
        KittiFusionSequence(root, images, "0000")


@pytest.mark.parametrize(
    "calibration,error",
    [
        ("P0: 1 0 0\n", "missing P2"),
        ("P2: 1 2 3\n", "12 finite float32"),
        ("P2: 0 0 0 0 0 0 0 0 0 0 0 0\n", "nonsingular"),
        ("P2: 100 0 2 0 0 100 1.5 0 0 0 1 0\n" * 2, "duplicate P2"),
    ],
)
def test_fusion_reports_invalid_calibration_path(tmp_path: Path, calibration: str, error: str) -> None:
    root, images = _fixture(tmp_path)
    (root / "calib/training/calib/0000.txt").write_text(calibration)

    with pytest.raises(ValueError, match=error) as caught:
        KittiFusionSequence(root, images, "0000")
    assert "calib/0000.txt" in str(caught.value)


@pytest.mark.parametrize("defect", ["image_gap", "image_size", "pose_count", "reflection", "nonfinite_pose"])
def test_fusion_rejects_image_or_pose_misalignment(tmp_path: Path, defect: str) -> None:
    root, images = _fixture(tmp_path)
    poses_path = root / "ego_motion/0000.npy"
    poses = np.load(poses_path)
    if defect == "image_gap":
        (images / "0000/000001.png").unlink()
    elif defect == "image_size":
        Image.new("RGB", (5, 3)).save(images / "0000/000001.png")
    elif defect == "pose_count":
        np.save(poses_path, poses[:2])
    elif defect == "reflection":
        poses[1, 0, 0] = -1
        np.save(poses_path, poses)
    else:
        poses[1, 0, 3] = np.nan
        np.save(poses_path, poses)

    with pytest.raises(ValueError, match="images|dimensions|ego motion|pose"):
        KittiFusionSequence(root, images, "0000")


def test_fusion_requires_selected_car_variant_and_preserves_native_frame_bounds(tmp_path: Path) -> None:
    root, images = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="results_tracking_car_auto_t2_train"):
        KittiFusionSequence(root, images, "0000", car_variant="t2-train")
    base = root / "pointgnn/training"
    (base / "results_tracking_car_auto_t3_trainval").rename(base / "results_tracking_car_auto_t2_train")
    sequence = KittiFusionSequence(root, images, "0000", car_variant="t2-train")
    assert len(sequence[0].detections_3d) == 2
    (base / "results_tracking_car_auto_t2_train/0000/data/000003.txt").write_text("")
    with pytest.raises(ValueError, match="no corresponding image"):
        KittiFusionSequence(root, images, "0000", car_variant="t2-train")


@pytest.mark.parametrize("counts", [b"", b"P", b"M", b"=", b"11", b"~"])
def test_compressed_mask_decoder_rejects_invalid_counts(counts: bytes) -> None:
    with pytest.raises(ValueError, match="RLE"):
        _decode_mask(counts, (3, 4))


def test_compressed_mask_decoder_matches_official_coco_codec() -> None:
    codec = pytest.importorskip("pycocotools.mask")
    rng = np.random.default_rng(37)
    for probability in (0, 0.05, 0.5, 0.9, 1):
        expected = rng.random((15, 21)) < probability
        encoded = codec.encode(np.asfortranarray(expected, dtype=np.uint8))
        decoded = _decode_mask(encoded["counts"], expected.shape)
        np.testing.assert_array_equal(decoded, expected)
        assert decoded.dtype == np.bool_
        assert decoded.flags.c_contiguous


def test_fusion_preserves_full_rigid_ego_pose(tmp_path: Path) -> None:
    root, images = _fixture(tmp_path)
    poses_path = root / "ego_motion/0000.npy"
    poses = np.load(poses_path)
    angle = 0.1
    poses[1, :3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    np.save(poses_path, poses)

    sample = KittiFusionSequence(root, images, "0000")[1]

    np.testing.assert_array_equal(sample.camera.camera_to_world.numpy(), poses[1].astype(np.float32))
