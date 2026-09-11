from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from boxmot.datasets.inputs import ModalityInput, SequenceInputs
from boxmot.datasets.readers.detections import _decode_mask
from boxmot.datasets.sequence import MultimodalSequence

_CLASSES = {"car": {"id": 1, "evaluation": "target"}, "pedestrian": {"id": 2, "evaluation": "target"}}


def _sequence(sequence_id: str, *, fps: float = 10.0, **paths: Path) -> MultimodalSequence:
    """Declare source encodings explicitly, independently of sequence identity."""
    formats = {
        "images": "image-directory",
        "detections_2d": "trackrcnn",
        "calibration": "kitti-p2",
        "poses": "camera-to-world-npy",
    }
    modalities = {name: ModalityInput(format, (paths[name],), {}) for name, format in formats.items()}
    modalities["detections_3d"] = ModalityInput(
        "kitti-detections",
        (paths["car_detections_3d"], paths["pedestrian_detections_3d"]),
        {"score_transform": "odds", "ignore_classes": ["Cyclist"]},
    )
    return MultimodalSequence(SequenceInputs(sequence_id, modalities), classes=_CLASSES, fps=fps)


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


def test_multimodal_reorders_geometry_bounds_scores_and_preserves_projection(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    sequence = _sequence("0000", **paths)

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
    assert sequence.fps == 10.0
    assert [frame.timestamp_s for frame in sequence] == [0.0, 0.1, 0.2]


def test_multimodal_preserves_custom_sequence_name_and_frame_rate(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    sequence = _sequence("downtown-drive", **paths, fps=25.0)

    assert sequence.sequence_id == "downtown-drive"
    assert sequence.fps == 25.0
    assert [frame.timestamp_s for frame in sequence] == [0.0, 0.04, 0.08]
    assert sequence[2].detections.sample_id == "train:downtown-drive:2"
    assert sequence[2].detections_3d.sample_id == sequence[2].detections.sample_id


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True, "25", None])
def test_multimodal_rejects_invalid_frame_rate_before_loading_inputs(tmp_path: Path, fps: object) -> None:
    paths = {
        name: tmp_path / name
        for name in ("images", "detections_2d", "calibration", "poses", "car_detections_3d", "pedestrian_detections_3d")
    }

    with pytest.raises(ValueError, match="fps must be a positive finite number"):
        _sequence("downtown-drive", **paths, fps=fps)


def test_multimodal_retains_missing_3d_and_empty_2d_frames(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    for directory in (paths["car_detections_3d"], paths["pedestrian_detections_3d"]):
        (directory / "000001.txt").unlink()
    sequence = _sequence("0000", **paths)

    assert sequence.missing_3d_frames == {
        str(paths["car_detections_3d"]): (1,),
        str(paths["pedestrian_detections_3d"]): (1,),
    }
    samples = list(sequence)
    assert [sample.frame_index for sample in samples] == [0, 1, 2]
    assert [len(sample.detections_3d) for sample in samples] == [2, 0, 2]
    assert len(samples[1].detections) == 0
    assert samples[1].detections.masks.values.shape == (0, 3, 4)
    assert samples[1].detections_3d.geometry.values.shape == (0, 7)
    assert sequence[-1].frame_index == 2
    assert [sample.frame_index for sample in sequence[1:]] == [1, 2]


def test_multimodal_does_not_decode_rgb_or_decode_masks_before_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    trackrcnn = paths["detections_2d"]
    trackrcnn.write_text(_trackrcnn_row(2, **{"9": "malformed"}))

    def reject_pixels(*args: object, **kwargs: object) -> None:
        raise AssertionError("RGB pixels should never be loaded")

    monkeypatch.setattr(Image.Image, "load", reject_pixels)
    sequence = _sequence("0000", **paths)
    assert len(sequence[0].detections) == 0
    with pytest.raises(ValueError, match=r"detections_2d.txt:1: RLE"):
        sequence[2]


def test_multimodal_excludes_cyclists_and_retains_zero_area_masks(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    pedestrian = paths["pedestrian_detections_3d"] / "000000.txt"
    pedestrian.write_text(_pointgnn_row("Cyclist"))
    paths["detections_2d"].write_text(_trackrcnn_row(**{"9": "<"}))

    sample = _sequence("0000", **paths)[0]

    assert sample.detections_3d.class_ids.tolist() == [1]
    assert len(sample.detections) == 1
    assert not sample.detections.masks.values.any()


@pytest.mark.parametrize("raw_score,expected", [("0", 0.0), ("1", 0.5), ("1e308", 1.0)])
def test_multimodal_handles_finite_extreme_pointgnn_scores(tmp_path: Path, raw_score: str, expected: float) -> None:
    paths = _fixture(tmp_path)
    car = paths["car_detections_3d"] / "000000.txt"
    car.write_text(_pointgnn_row(score=raw_score))

    sample = _sequence("0000", **paths)[0]

    assert sample.detections_3d.scores[0].item() == expected


@pytest.mark.parametrize(
    "row,error",
    [
        (_pointgnn_row(score="-1"), "score must be nonnegative"),
        (_pointgnn_row(score="nan"), "must be finite"),
        (_pointgnn_row(score="inf"), "must be finite"),
        (_pointgnn_row("Truck"), "unsupported 3D detection class"),
        ("Car 1 2\n", "16 KITTI detection fields"),
        (_pointgnn_row().replace("1.5 2 4", "0 2 4"), "positive dimensions"),
        (_pointgnn_row().replace("1.5 2 4", "1e-100 2 4"), "positive in float32"),
    ],
)
def test_multimodal_reports_pointgnn_file_and_line_for_invalid_input(tmp_path: Path, row: str, error: str) -> None:
    paths = _fixture(tmp_path)
    car = paths["car_detections_3d"] / "000000.txt"
    car.write_text(row)
    sequence = _sequence("0000", **paths)

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
def test_multimodal_rejects_unaligned_trackrcnn_rows(tmp_path: Path, fields: dict[str, str], error: str) -> None:
    paths = _fixture(tmp_path)
    paths["detections_2d"].write_text(_trackrcnn_row(**fields))

    with pytest.raises(ValueError, match=f"detections_2d.txt:1: .*{error}"):
        _sequence("0000", **paths)


def test_multimodal_requires_detector_format_not_ground_truth(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["detections_2d"].write_text("0 1001 1 3 4 0<\n")

    with pytest.raises(ValueError, match="138 fields"):
        _sequence("0000", **paths)


@pytest.mark.parametrize(
    "calibration,error",
    [
        ("P0: 1 0 0\n", "missing P2"),
        ("P2: 1 2 3\n", "12 finite float32"),
        ("P2: 0 0 0 0 0 0 0 0 0 0 0 0\n", "nonsingular"),
        ("P2: 100 0 2 0 0 100 1.5 0 0 0 1 0\n" * 2, "duplicate P2"),
    ],
)
def test_multimodal_reports_invalid_calibration_path(tmp_path: Path, calibration: str, error: str) -> None:
    paths = _fixture(tmp_path)
    paths["calibration"].write_text(calibration)

    with pytest.raises(ValueError, match=error) as caught:
        _sequence("0000", **paths)
    assert str(paths["calibration"]) in str(caught.value)


@pytest.mark.parametrize("defect", ["image_gap", "image_size", "pose_count", "reflection", "nonfinite_pose"])
def test_multimodal_rejects_image_or_pose_misalignment(tmp_path: Path, defect: str) -> None:
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

    with pytest.raises(ValueError, match="images|dimensions|Ego motion|pose"):
        _sequence("0000", **paths)


def test_multimodal_requires_selected_3d_directory_and_preserves_native_frame_bounds(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    selected = paths["car_detections_3d"].with_name("selected-car-predictions")
    selected_paths = {**paths, "car_detections_3d": selected}
    with pytest.raises(FileNotFoundError, match="selected-car-predictions"):
        _sequence("0000", **selected_paths)
    paths["car_detections_3d"].rename(selected)
    sequence = _sequence("0000", **selected_paths)
    assert len(sequence[0].detections_3d) == 2
    (selected / "000003.txt").write_text("")
    with pytest.raises(ValueError, match="no corresponding image"):
        _sequence("0000", **selected_paths)


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


def test_multimodal_preserves_full_rigid_ego_pose(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    poses_path = paths["poses"]
    poses = np.load(poses_path)
    angle = 0.1
    poses[1, :3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    np.save(poses_path, poses)

    sample = _sequence("0000", **paths)[1]

    np.testing.assert_array_equal(sample.camera.camera_to_world.numpy(), poses[1].astype(np.float32))


def test_multimodal_custom_class_mapping_and_split_apply_to_both_detection_modalities(tmp_path: Path) -> None:
    """Source labels and benchmark identities are declared independently."""
    paths = _fixture(tmp_path)
    paths["detections_2d"].write_text(_trackrcnn_row(**{"6": "7"}))
    inputs = SequenceInputs(
        "warehouse",
        {
            "images": ModalityInput("image-directory", (paths["images"],), {}),
            "detections_2d": ModalityInput("trackrcnn", (paths["detections_2d"],), {}),
            "detections_3d": ModalityInput(
                "kitti-detections",
                (paths["car_detections_3d"], paths["pedestrian_detections_3d"]),
                {"score_transform": "odds", "class_map": {"Car": "vehicle", "Pedestrian": 9}},
            ),
        },
    )
    classes = {"vehicle": {"id": 7, "evaluation": "target"}, "person": {"id": 9, "evaluation": "target"}}

    sample = MultimodalSequence(inputs, classes=classes, fps=25, split="validation")[0]

    assert sample.detections.sample_id == sample.detections_3d.sample_id == "validation:warehouse:0"
    assert sample.detections.class_ids.tolist() == [7]
    assert sample.detections_3d.class_ids.tolist() == [7, 9]
    assert sample.camera is None


def test_multimodal_optional_inputs_keep_images_and_empty_canonical_detections(tmp_path: Path) -> None:
    """An image-only selection needs no sensor-specific assets or class names."""
    paths = _fixture(tmp_path)
    inputs = SequenceInputs("front-door", {"images": ModalityInput("image-directory", (paths["images"],), {})})
    sequence = MultimodalSequence(inputs, classes={"parcel": {"id": 4, "evaluation": "target"}}, fps=5)

    sample = sequence[2]

    assert len(sequence) == 3
    assert sample.timestamp_s == 0.4
    assert sample.detections.geometry.values.shape == (0, 4)
    assert sample.detections_3d.geometry.values.shape == (0, 7)
    assert sample.camera is None


@pytest.mark.parametrize(
    "options",
    [
        {"coordinate_frame": "lidar"},
        {"box_origin": "center"},
        {"dimensions": "lwh"},
        {"yaw_axis": "z"},
        {"score_transform": "sigmoid"},
        {"unknown_option": True},
    ],
)
def test_multimodal_rejects_unhandled_3d_coordinate_and_score_conventions(tmp_path: Path, options: dict) -> None:
    """Readers fail before interpreting an authored but unsupported convention."""
    paths = _fixture(tmp_path)
    inputs = SequenceInputs(
        "test",
        {
            "images": ModalityInput("image-directory", (paths["images"],), {}),
            "detections_3d": ModalityInput("kitti-detections", (paths["car_detections_3d"],), options),
        },
    )
    with pytest.raises(ValueError, match="kitti-detections"):
        MultimodalSequence(inputs, classes=_CLASSES, fps=10)


def test_multimodal_requires_explicit_score_transform_for_unbounded_scores(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    inputs = SequenceInputs(
        "test",
        {
            "images": ModalityInput("image-directory", (paths["images"],), {}),
            "detections_3d": ModalityInput("kitti-detections", (paths["car_detections_3d"],), {}),
        },
    )
    sequence = MultimodalSequence(inputs, classes=_CLASSES, fps=10)

    with pytest.raises(ValueError, match="identity score_transform"):
        sequence[0]


def test_multimodal_rejects_unused_reader_options(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    inputs = SequenceInputs(
        "test", {"images": ModalityInput("image-directory", (paths["images"],), {"resize": [100, 100]})}
    )

    with pytest.raises(ValueError, match="does not support reader options"):
        MultimodalSequence(inputs, classes=_CLASSES, fps=10)
