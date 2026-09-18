"""Supported box-model names come from installed Ultralytics asset metadata."""

from __future__ import annotations

import json
import subprocess
import sys

from boxmot.detectors._ultralytics_models import ultralytics_detector_names
from tests._paths import REPO_ROOT


def test_inventory_filter_includes_box_families_and_excludes_nonbox_tasks() -> None:
    supported = {
        "yolov8n.pt",
        "yolo26n.pt",
        "yolo26n-obb.pt",
        "yolo26n-seg.pt",
        "yolo26n-pose.pt",
        "yolo11n-grayscale.pt",
        "yolov8s-worldv2.pt",
        "yoloe-26n-seg-pf.pt",
        "rtdetr-l.pt",
        "FastSAM-x.pt",
        "yolo_nas_s.pt",
        "yolo100n.pt",
    }
    unsupported = {
        "yolo26n-cls.pt",
        "yolo26n-sem.pt",
        "yolo26n-cls-custom.pt",
        "sam_b.pt",
        "sam2.1_l.pt",
        "mobile_sam.pt",
        "mobileclip_blt.ts",
        "calibration_data.npy.zip",
        "unrelated.pt",
        "yolo26n.yaml",
        "directory/yolo26n.pt",
        "directory\\yolo26n.pt",
    }
    assert ultralytics_detector_names(supported | unsupported) == tuple(sorted(name[:-3] for name in supported))


def test_inventory_filter_is_deterministic_and_does_not_modify_inputs() -> None:
    inventory = ["yolo26s.pt", "FastSAM-x.pt", "yolo26n.pt", "yolo26n.pt"]
    original = inventory.copy()
    assert ultralytics_detector_names(inventory) == ("FastSAM-x", "yolo26n", "yolo26s")
    assert ultralytics_detector_names(reversed(inventory)) == ultralytics_detector_names(inventory)
    assert inventory == original
    assert ultralytics_detector_names(()) == ()


def test_installed_inventory_contains_every_supported_official_box_checkpoint() -> None:
    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

    expected = {
        name[:-3]
        for name in GITHUB_ASSETS_NAMES
        if name.endswith(".pt")
        and name.startswith(("yolo", "rtdetr-", "FastSAM-"))
        and "-cls" not in name
        and "-sem" not in name
    }
    assert set(ultralytics_detector_names()) == expected
    assert {"yolo26n", "rtdetr-l", "FastSAM-s", "yolo_nas_m", "yolo11n-grayscale"} <= expected


def test_metadata_helper_import_and_explicit_filter_do_not_import_ultralytics() -> None:
    probe = """
import json, sys
from boxmot.detectors._ultralytics_models import ultralytics_detector_names
assert ultralytics_detector_names(['yolo26n.pt']) == ('yolo26n',)
print(json.dumps(sorted(name for name in sys.modules if name in {'ultralytics', 'torch', 'yaml', 'numpy', 'cv2'})))
"""
    completed = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []
