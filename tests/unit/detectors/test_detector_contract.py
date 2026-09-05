"""Strict structured boundary tests for concrete detector backends."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from boxmot.detectors import DetectorCapabilities, DetectorSpec
from boxmot.detectors.factory import detector_capabilities
from tests._paths import REPO_ROOT


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    (
        ("boxmot.detectors.backends.ultralytics", "UltralyticsDetector"),
        ("boxmot.detectors.backends.yolox", "YoloXDetector"),
        ("boxmot.detectors.backends.rtdetr", "RTDetrDetector"),
    ),
)
def test_concrete_backends_expose_no_raw_staged_api(module_name: str, class_name: str) -> None:
    probe = (
        "import importlib, json; "
        f"cls = getattr(importlib.import_module({module_name!r}), {class_name!r}); "
        "forbidden = ('__call__', 'preprocess', 'process', 'postprocess'); "
        "print(json.dumps({'predict': callable(getattr(cls, 'predict', None)), "
        "'forbidden': [name for name in forbidden if name in cls.__dict__]}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"predict": True, "forbidden": []}


def test_capability_lookup_is_model_free_and_geometry_specific() -> None:
    probe = (
        "import json, sys; "
        "from boxmot.detectors import DetectorSpec; "
        "from boxmot.detectors.factory import detector_capabilities; "
        "value = detector_capabilities(DetectorSpec('ultralytics', artifact='model-seg.pt')); "
        "print(json.dumps({'masks': value.provides_masks, 'aabb': value.supports_aabb, "
        "'obb': value.supports_obb, 'backends': sorted(name for name in sys.modules "
        "if name.startswith('boxmot.detectors.backends.'))}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "masks": True,
        "aabb": True,
        "obb": True,
        "backends": [],
    }
    assert detector_capabilities(DetectorSpec("yolox", geometry_mode="aabb")) == DetectorCapabilities()
    with pytest.raises(ValueError, match="Unknown detector backend"):
        detector_capabilities(DetectorSpec("missing"))
