from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from boxmot.data.dataset import compute_fps_mask
from boxmot.native import _common
from boxmot.native.reid.capi import ensure_reid_capi_library


@pytest.fixture(scope="module")
def io_probe():
    ensure_reid_capi_library()
    build_dir = _common.tracker_build_dir("base")
    completed = subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "boxmot_io_probe",
            "--parallel",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    executable_name = "boxmot_io_probe.exe" if os.name == "nt" else "boxmot_io_probe"
    candidates = (build_dir / "Release" / executable_name, build_dir / executable_name)
    executable = next((path for path in candidates if path.is_file()), None)
    assert executable is not None
    return executable


def test_native_npy_image_matches_python_tracking_replay_channels(io_probe, tmp_path):
    image = np.zeros((2, 3, 8), dtype=np.uint8)
    image[0, 0] = np.arange(8, dtype=np.uint8) * 10
    image_path = tmp_path / "multispectral.npy"
    np.save(image_path, image)

    completed = subprocess.run(
        [str(io_probe), str(image_path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    # MOTSequence truncates multispectral frames to the first three channels.
    assert completed.stdout.split() == ["2", "3", "3", "0", "10", "20"]


def test_native_npy_image_matches_python_grayscale_conversion(io_probe, tmp_path):
    image_path = tmp_path / "grayscale.npy"
    np.save(image_path, np.full((2, 3), 17, dtype=np.uint8))

    completed = subprocess.run(
        [str(io_probe), str(image_path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["2", "3", "3", "17", "17", "17"]


def test_native_npy_matrix_rejects_non_2d_cache_like_python(io_probe, tmp_path):
    matrix_path = tmp_path / "invalid_detections.npy"
    np.save(matrix_path, np.zeros((2, 7, 2), dtype=np.float32))

    completed = subprocess.run(
        [str(io_probe), "--matrix-shape", str(matrix_path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 1
    assert "npy matrix must be 2D" in completed.stderr


def test_native_fps_selection_matches_numpy_float64_arange(io_probe):
    frames = np.arange(1, 61)
    expected = frames[compute_fps_mask(frames, orig_fps=24, target_fps=5)]

    completed = subprocess.run(
        [str(io_probe), "--wanted-frames", "24", "5", "60"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    actual = np.fromstring(completed.stdout, sep=" ", dtype=int)
    np.testing.assert_array_equal(actual, expected)


def test_native_sequence_frames_support_each_image_extension(io_probe, tmp_path):
    expected = ["000001.jpg", "000002.jpeg", "000003.npy", "000004.png"]
    for filename in expected:
        (tmp_path / filename).touch()

    completed = subprocess.run(
        [str(io_probe), "--list-frames", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == expected


@pytest.mark.parametrize(
    ("first_extension", "second_extension"),
    [(".jpg", ".npy"), (".jpeg", ".png")],
)
def test_native_sequence_frames_reject_duplicate_stems(
    io_probe,
    tmp_path,
    first_extension,
    second_extension,
):
    first = tmp_path / f"000001{first_extension}"
    second = tmp_path / f"000001{second_extension}"
    first.touch()
    second.touch()

    completed = subprocess.run(
        [str(io_probe), "--list-frames", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 1
    assert "Multiple image files found for frame stem '000001'" in completed.stderr
    assert first.name in completed.stderr
    assert second.name in completed.stderr
    assert "Keep exactly one of .jpg, .jpeg, .png, or .npy per frame." in completed.stderr
