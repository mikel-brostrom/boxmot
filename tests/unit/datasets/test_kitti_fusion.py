from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from boxmot.datasets.kitti_fusion import KittiFusionSequence
from boxmot.datasets.trackrcnn import _decode_mask


def _pointgnn_row(label: str = "Car", score: str = "103") -> str:
    """Use distinct geometry values to expose incorrect KITTI field ordering."""
    return f"{label} -1 -1 0 0 0 4 3 1.5 2 4 5 6 20 0.25 {score}\n"


def _trackrcnn_row(frame: int = 0, **replacements: str) -> str:
    """Create a full-frame mask and the released 128-dimensional embedding tail."""
    fields = [str(frame), "0.25", "0.5", "3.75", "2.5", "0.9", "1", "3", "4", "0<"] + ["0"] * 128
    for field, value in replacements.items():
        fields[int(field)] = value
    return " ".join(fields) + "\n"


def _fixture(tmp_path: Path, frames: int = 3) -> dict[str, Path]:
    """Create explicit sequence inputs independently of source download folders."""
    root = tmp_path / "sequences/0000"
    paths = {
        "images": root / "images",
        "detections_2d": root / "detections_2d.txt",
        "calibration": root / "calibration.txt",
        "poses": root / "poses.npy",
        "car_detections_3d": root / "detections_3d/car",
        "pedestrian_detections_3d": root / "detections_3d/pedestrian",
    }
    paths["images"].mkdir(parents=True)
    for frame in range(frames):
        Image.new("RGB", (4, 3)).save(paths["images"] / f"{frame:06d}.png")
    paths["calibration"].write_text("P2: 100 0 2 0.4 0 100 1.5 0.2 0 0 1 0.0027\n")
    values = np.repeat(np.eye(4)[None], frames, axis=0)
    values[:, 0, 3] = np.arange(frames)
    np.save(paths["poses"], values)
    for key, label in (("car_detections_3d", "Car"), ("pedestrian_detections_3d", "Pedestrian")):
        directory = paths[key]
        directory.mkdir(parents=True)
        for frame in range(frames):
            (directory / f"{frame:06d}.txt").write_text(_pointgnn_row(label))
    paths["detections_2d"].write_text(_trackrcnn_row())
    return paths


def test_fusion_reorders_geometry_bounds_scores_and_preserves_projection(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    sequence = KittiFusionSequence("0000", **paths)

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
    paths = _fixture(tmp_path)
    for directory in (paths["car_detections_3d"], paths["pedestrian_detections_3d"]):
        (directory / "000001.txt").unlink()
    sequence = KittiFusionSequence("0000", **paths)

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
    paths = _fixture(tmp_path)
    trackrcnn = paths["detections_2d"]
    trackrcnn.write_text(_trackrcnn_row(2, **{"9": "malformed"}))

    def reject_pixels(*args: object, **kwargs: object) -> None:
        raise AssertionError("RGB pixels should never be loaded")

    monkeypatch.setattr(Image.Image, "load", reject_pixels)
    sequence = KittiFusionSequence("0000", **paths)
    assert len(sequence[0].detections) == 0
    with pytest.raises(ValueError, match=r"detections_2d.txt:1: RLE"):
        sequence[2]


def test_fusion_excludes_cyclists_and_retains_zero_area_masks(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    pedestrian = paths["pedestrian_detections_3d"] / "000000.txt"
    pedestrian.write_text(_pointgnn_row("Cyclist"))
    paths["detections_2d"].write_text(_trackrcnn_row(**{"9": "<"}))

    sample = KittiFusionSequence("0000", **paths)[0]

    assert sample.detections_3d.class_ids.tolist() == [1]
    assert len(sample.detections) == 1
    assert not sample.detections.masks.values.any()


@pytest.mark.parametrize("raw_score,expected", [("0", 0.0), ("1", 0.5), ("1e308", 1.0)])
def test_fusion_handles_finite_extreme_pointgnn_scores(tmp_path: Path, raw_score: str, expected: float) -> None:
    paths = _fixture(tmp_path)
    car = paths["car_detections_3d"] / "000000.txt"
    car.write_text(_pointgnn_row(score=raw_score))

    sample = KittiFusionSequence("0000", **paths)[0]

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
    paths = _fixture(tmp_path)
    car = paths["car_detections_3d"] / "000000.txt"
    car.write_text(row)
    sequence = KittiFusionSequence("0000", **paths)

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
    paths = _fixture(tmp_path)
    paths["detections_2d"].write_text(_trackrcnn_row(**fields))

    with pytest.raises(ValueError, match=f"detections_2d.txt:1: .*{error}"):
        KittiFusionSequence("0000", **paths)


def test_fusion_requires_detector_format_not_ground_truth(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["detections_2d"].write_text("0 1001 1 3 4 0<\n")

    with pytest.raises(ValueError, match="138 fields"):
        KittiFusionSequence("0000", **paths)


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
    paths = _fixture(tmp_path)
    paths["calibration"].write_text(calibration)

    with pytest.raises(ValueError, match=error) as caught:
        KittiFusionSequence("0000", **paths)
    assert str(paths["calibration"]) in str(caught.value)


@pytest.mark.parametrize("defect", ["image_gap", "image_size", "pose_count", "reflection", "nonfinite_pose"])
def test_fusion_rejects_image_or_pose_misalignment(tmp_path: Path, defect: str) -> None:
    paths = _fixture(tmp_path)
    poses_path = paths["poses"]
    poses = np.load(poses_path)
    if defect == "image_gap":
        (paths["images"] / "000001.png").unlink()
    elif defect == "image_size":
        Image.new("RGB", (5, 3)).save(paths["images"] / "000001.png")
    elif defect == "pose_count":
        np.save(poses_path, poses[:2])
    elif defect == "reflection":
        poses[1, 0, 0] = -1
        np.save(poses_path, poses)
    else:
        poses[1, 0, 3] = np.nan
        np.save(poses_path, poses)

    with pytest.raises(ValueError, match="images|dimensions|ego motion|pose"):
        KittiFusionSequence("0000", **paths)


def test_fusion_requires_selected_3d_directory_and_preserves_native_frame_bounds(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    selected = paths["car_detections_3d"].with_name("selected-car-predictions")
    selected_paths = {**paths, "car_detections_3d": selected}
    with pytest.raises(FileNotFoundError, match="selected-car-predictions"):
        KittiFusionSequence("0000", **selected_paths)
    paths["car_detections_3d"].rename(selected)
    sequence = KittiFusionSequence("0000", **selected_paths)
    assert len(sequence[0].detections_3d) == 2
    (selected / "000003.txt").write_text("")
    with pytest.raises(ValueError, match="no corresponding image"):
        KittiFusionSequence("0000", **selected_paths)


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
    paths = _fixture(tmp_path)
    poses_path = paths["poses"]
    poses = np.load(poses_path)
    angle = 0.1
    poses[1, :3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    np.save(poses_path, poses)

    sample = KittiFusionSequence("0000", **paths)[1]

    np.testing.assert_array_equal(sample.camera.camera_to_world.numpy(), poses[1].astype(np.float32))
