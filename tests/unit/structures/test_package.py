"""Public structure identities and independently loaded implementation modules."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("name", "module_name"),
    [
        ("Boxes3D", "geometry"),
        ("CameraModel", "camera"),
        ("Detections3D", "detections"),
        ("Tracks3D", "tracks"),
        ("MultimodalTracks", "tracks"),
    ],
)
def test_public_spatial_classes_share_their_canonical_module_identity(name: str, module_name: str) -> None:
    """Public and direct imports must resolve to the same defining class."""

    structures = importlib.import_module("boxmot.structures")
    implementation = importlib.import_module(f"boxmot.structures.{module_name}")
    canonical = getattr(implementation, name)

    assert getattr(structures, name) is canonical
    assert canonical.__module__ == implementation.__name__
    assert name in structures.__all__
    assert name in implementation.__all__


def test_structure_namespace_and_camera_geometry_imports_remain_independent() -> None:
    """Resolving a camera or box type must not load detection and track modules."""

    script = """
import sys

import boxmot.structures as structures

assert not any(name.startswith("boxmot.structures.") for name in sys.modules)
assert not any(name in sys.modules for name in ("cv2", "numpy", "torch"))

boxes = structures.Boxes3D
assert boxes.__module__ == "boxmot.structures.geometry"
assert "boxmot.structures.camera" not in sys.modules
assert "boxmot.structures.detections" not in sys.modules
assert "boxmot.structures.tracks" not in sys.modules

camera = structures.CameraModel
assert camera.__module__ == "boxmot.structures.camera"
assert "boxmot.structures.detections" not in sys.modules
assert "boxmot.structures.tracks" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_removed_spatial_module_has_no_importable_wrapper() -> None:
    structures_root = Path(__file__).resolve().parents[3] / "boxmot" / "structures"

    assert importlib.util.find_spec("boxmot.structures.spatial") is None
    assert not (structures_root / "spatial.py").exists()
