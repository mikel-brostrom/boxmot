"""Check native loader dependencies across Docker stage inheritance."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "Dockerfile"
OPENCV_RUNTIME_PACKAGES = frozenset(
    {
        "libopencv-calib3d406",
        "libopencv-core406",
        "libopencv-dnn406",
        "libopencv-imgproc406",
        "libopencv-video406",
    }
)


def _stages() -> dict[str, tuple[str, list[str]]]:
    """Resolve named FROM stages and join continued Docker instructions."""
    source = DOCKERFILE.read_text(encoding="utf-8").replace("\\\n", " ")
    headers = list(re.finditer(r"^FROM (\S+) AS (\S+)\s*$", source, re.MULTILINE))
    return {
        header[2]: (
            header[1],
            source[header.end() : headers[index + 1].start() if index + 1 < len(headers) else len(source)].splitlines(),
        )
        for index, header in enumerate(headers)
    }


def _installed_packages(instruction: str) -> set[str]:
    """Read explicit apt packages from an instruction, excluding command flags."""
    match = re.search(r"apt-get install\s+(.+?)(?:&&|;|$)", instruction)
    return {word for word in shlex.split(match[1]) if not word.startswith("-")} if match else set()


def _packages_at_end(stages: dict[str, tuple[str, list[str]]], name: str) -> set[str]:
    """Only FROM inherits system packages; COPY imports files, not apt installs."""
    if name not in stages:
        return set()
    parent, instructions = stages[name]
    packages = _packages_at_end(stages, parent)
    for instruction in instructions:
        packages.update(_installed_packages(instruction))
    return packages


def test_native_library_smokes_have_opencv_before_loading() -> None:
    """A later runtime stage cannot satisfy a builder's earlier ctypes smoke."""
    stages = _stages()
    checked = []
    for name, (parent, instructions) in stages.items():
        packages = _packages_at_end(stages, parent)
        for instruction in instructions:
            before_load = instruction.split("ctypes.CDLL", 1)[0]
            packages.update(_installed_packages(before_load))
            if "ctypes.CDLL" in instruction:
                missing = OPENCV_RUNTIME_PACKAGES - packages
                assert not missing, f"{name} loads native libraries without runtime packages: {sorted(missing)}"
                checked.append(name)
    assert checked, "Retain native-library load validation in CLI wheel builders."


@pytest.mark.parametrize("target", ("cli-cpu", "cli-gpu", "default"))
def test_cli_targets_inherit_opencv_runtime_packages(target: str) -> None:
    """The published images and default target can load their native artifacts."""
    missing = OPENCV_RUNTIME_PACKAGES - _packages_at_end(_stages(), target)
    assert not missing, f"{target} lacks native runtime packages: {sorted(missing)}"
