"""BoxMOT DeepStream adapter — Python build helper.

Provides utilities for building the ``libnvds_boxmot_tracker.so`` shared library
and generating config files for use with NVIDIA DeepStream's Gst-nvtracker plugin.

Usage:
    from boxmot.native.deepstream_adapter import build_adapter, generate_config

    # Build the shared library
    lib_path = build_adapter()

    # Generate a tracker config file
    config_path = generate_config(
        algorithm="botsort",
        reid_onnx="/path/to/reid_model.onnx",
        output_path="/path/to/config_tracker_boxmot.yml"
    )
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Root of the deepstream adapter source
_ADAPTER_DIR = Path(__file__).parent / "trackers" / "deepstream"
_DEFAULT_CONFIG = _ADAPTER_DIR / "config_tracker_boxmot.yml"


def _find_deepstream_root() -> Path | None:
    """Locate the DeepStream SDK installation."""
    candidates = [
        Path("/opt/nvidia/deepstream/deepstream"),
        Path("/opt/nvidia/deepstream/deepstream-7.0"),
        Path("/opt/nvidia/deepstream/deepstream-6.4"),
        Path("/opt/nvidia/deepstream/deepstream-6.3"),
    ]
    for path in candidates:
        if (path / "sources" / "includes" / "nvdstracker.h").exists():
            return path
    return None


def _find_tensorrt_root() -> Path | None:
    """Locate TensorRT installation."""
    candidates = [
        Path("/usr/local/tensorrt"),
        Path("/usr"),
    ]
    for path in candidates:
        nvinfer_h = path / "include" / "NvInfer.h"
        if not nvinfer_h.exists():
            nvinfer_h = path / "include" / "x86_64-linux-gnu" / "NvInfer.h"
        if nvinfer_h.exists():
            return path
    return None


def build_adapter(
    *,
    build_dir: Path | None = None,
    deepstream_root: Path | str | None = None,
    tensorrt_root: Path | str | None = None,
    cmake_args: list[str] | None = None,
    verbose: bool = False,
) -> Path:
    """Build the DeepStream adapter shared library.

    Returns the path to the built ``libnvds_boxmot_tracker.so``.

    Raises:
        RuntimeError: If build prerequisites are missing or build fails.
    """
    if build_dir is None:
        build_dir = _ADAPTER_DIR / "build"

    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Resolve DeepStream root
    if deepstream_root is None:
        deepstream_root = _find_deepstream_root()
    if deepstream_root is None:
        raise RuntimeError(
            "DeepStream SDK not found. Install it or pass deepstream_root=."
        )
    deepstream_root = Path(deepstream_root)

    # Resolve TensorRT root
    if tensorrt_root is None:
        tensorrt_root = _find_tensorrt_root()
    if tensorrt_root is None:
        raise RuntimeError(
            "TensorRT not found. Install it or pass tensorrt_root=."
        )
    tensorrt_root = Path(tensorrt_root)

    # CMake configure
    cmake_cmd = [
        "cmake",
        str(_ADAPTER_DIR),
        f"-DDEEPSTREAM_ROOT={deepstream_root}",
        f"-DTENSORRT_ROOT={tensorrt_root}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if cmake_args:
        cmake_cmd.extend(cmake_args)

    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.PIPE

    result = subprocess.run(
        cmake_cmd, cwd=build_dir, stdout=stdout, stderr=stderr
    )
    if result.returncode != 0:
        err_msg = result.stderr.decode() if result.stderr else ""
        raise RuntimeError(f"CMake configure failed:\n{err_msg}")

    # CMake build
    result = subprocess.run(
        ["cmake", "--build", ".", "--parallel"],
        cwd=build_dir, stdout=stdout, stderr=stderr
    )
    if result.returncode != 0:
        err_msg = result.stderr.decode() if result.stderr else ""
        raise RuntimeError(f"CMake build failed:\n{err_msg}")

    # Find the output library
    lib_path = build_dir / "libnvds_boxmot_tracker.so"
    if not lib_path.exists():
        raise RuntimeError(
            f"Build succeeded but library not found at {lib_path}"
        )

    return lib_path


def generate_config(
    *,
    algorithm: str = "botsort",
    reid_onnx: str | Path | None = None,
    reid_engine: str | Path | None = None,
    reid_feature_size: int = 256,
    reid_batch_size: int = 100,
    reid_network_mode: int = 1,
    reid_infer_dims: list[int] | None = None,
    track_high_thresh: float = 0.6,
    track_low_thresh: float = 0.1,
    new_track_thresh: float = 0.7,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    max_targets_per_stream: int = 150,
    frame_rate: int = 30,
    output_path: str | Path | None = None,
    **extra_params: Any,
) -> Path:
    """Generate a DeepStream tracker config YAML file.

    Args:
        algorithm: BoxMOT tracker to use (botsort, bytetrack, ocsort, sfsort, occluboost).
        reid_onnx: Path to ONNX ReID model.
        reid_engine: Path to pre-built TensorRT engine.
        reid_feature_size: ReID embedding dimension.
        reid_batch_size: Max crops per TensorRT inference call.
        reid_network_mode: 0=FP32, 1=FP16, 2=INT8.
        reid_infer_dims: Network input [H, W, C].
        track_high_thresh: High confidence detection threshold.
        track_low_thresh: Low confidence detection threshold.
        new_track_thresh: Minimum confidence for new track creation.
        track_buffer: Frames to retain lost tracks.
        match_thresh: IoU matching threshold.
        max_targets_per_stream: Maximum targets per video stream.
        frame_rate: Video frame rate.
        output_path: Where to write the YAML (default: temp location).
        **extra_params: Additional parameters to include.

    Returns:
        Path to the generated config file.
    """
    if reid_infer_dims is None:
        reid_infer_dims = [128, 64, 3]

    config = {
        "BaseConfig": {
            "algorithm": algorithm,
            "minDetectorConfidence": track_low_thresh,
            "frameRate": frame_rate,
        },
        "BoxMOT": {
            "trackHighThresh": track_high_thresh,
            "trackLowThresh": track_low_thresh,
            "newTrackThresh": new_track_thresh,
            "trackBuffer": track_buffer,
            "matchThresh": match_thresh,
            "withReId": reid_onnx is not None or reid_engine is not None,
        },
        "TargetManagement": {
            "maxTargetsPerStream": max_targets_per_stream,
            "probationAge": 3,
            "maxShadowTrackingAge": 30,
            "earlyTerminationAge": 1,
            "supportPastFrame": 1,
        },
        "DataAssociator": {
            "associationMatcherType": 0,
            "checkClassMatch": 0,
            "matchingScoreWeight4Iou": 1.0,
            "matchingScoreWeight4ReIDSimilarity": 0.5 if reid_onnx else 0.0,
        },
        "ReID": {
            "reidType": 1 if (reid_onnx or reid_engine) else 0,
            "onnxFile": str(reid_onnx) if reid_onnx else "",
            "modelEngineFile": str(reid_engine) if reid_engine else "",
            "batchSize": reid_batch_size,
            "networkMode": reid_network_mode,
            "workspaceSize": 20,
            "inferDims": reid_infer_dims,
            "inputOrder": 1,
            "colorFormat": 0,
            "netScaleFactor": 1.0,
            "offsets": [0.0, 0.0, 0.0],
            "reidFeatureSize": reid_feature_size,
            "reidHistorySize": 100,
            "addFeatureNormalization": 1,
            "keepAspc": 1,
        },
    }

    # Apply extra params
    if extra_params:
        for key, value in extra_params.items():
            # Try to place in the most appropriate section
            placed = False
            for section in config.values():
                if isinstance(section, dict) and key in section:
                    section[key] = value
                    placed = True
                    break
            if not placed:
                config["BoxMOT"][key] = value

    if output_path is None:
        output_path = Path("/tmp/config_tracker_boxmot.yml")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def get_default_config_path() -> Path:
    """Return path to the bundled default config file."""
    return _DEFAULT_CONFIG


def check_prerequisites() -> dict[str, bool]:
    """Check if build prerequisites are available.

    Returns a dict with availability status for each component.
    """
    return {
        "deepstream": _find_deepstream_root() is not None,
        "tensorrt": _find_tensorrt_root() is not None,
        "cmake": shutil.which("cmake") is not None,
        "nvcc": shutil.which("nvcc") is not None,
    }
