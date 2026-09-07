"""YOLOX detector registration tests for the explicit v24 backend selector."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from boxmot.detectors import DetectorSpec
from boxmot.detectors.factory import _DETECTOR_FACTORIES
from tests._paths import REPO_ROOT


def test_yolox_is_selected_by_backend_instead_of_artifact_filename() -> None:
    spec = DetectorSpec(backend="yolox", artifact="custom-weights.pt")

    factory = _DETECTOR_FACTORIES.resolve(spec.backend)

    assert callable(factory)
    assert factory.__name__ == "YoloXDetector"


def test_resolving_yolox_factory_keeps_implementation_module_lazy() -> None:
    probe = (
        "import json, sys; "
        "from boxmot.detectors.factory import _DETECTOR_FACTORIES; "
        "before = 'boxmot.detectors.backends.yolox' in sys.modules; "
        "factory = _DETECTOR_FACTORIES.resolve('yolox'); "
        "print(json.dumps({'factory': factory.__name__, 'before': before, "
        "'after': 'boxmot.detectors.backends.yolox' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "factory": "YoloXDetector",
        "before": False,
        "after": True,
    }


@pytest.mark.parametrize("artifact", ("yolox_s.pt", "yolox_n.pt", "renamed-model.pt"))
def test_yolox_artifact_names_do_not_change_explicit_backend_selection(artifact: str) -> None:
    spec = DetectorSpec(backend="yolox", artifact=artifact)

    assert _DETECTOR_FACTORIES.resolve(spec.backend) is _DETECTOR_FACTORIES.resolve("yolox")


def test_unknown_detector_backend_is_rejected_without_filename_inference() -> None:
    with pytest.raises(ValueError) as error:
        _DETECTOR_FACTORIES.resolve("weights-file")

    assert str(error.value) == (
        "Unknown detector backend 'weights-file'. Available backends: rtdetr, ultralytics, yolox."
    )
