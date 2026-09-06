from __future__ import annotations

from pathlib import Path

import pytest

from tests._paths import REPO_ROOT

TRACK_DOC = REPO_ROOT / "docs" / "modes" / "track.md"
MATERIALIZE_DOC = REPO_ROOT / "docs" / "modes" / "materialize.md"
EVAL_DOC = REPO_ROOT / "docs" / "modes" / "eval.md"
README = REPO_ROOT / "README.md"
PUBLIC_DOC_ROOTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "docs",
    REPO_ROOT / "examples",
    REPO_ROOT / "boxmot" / "configs" / "README.md",
)


def _public_text_files() -> list[Path]:
    paths: list[Path] = []
    for root in PUBLIC_DOC_ROOTS:
        if root.is_file():
            paths.append(root)
            continue
        paths.extend(root.rglob("*.md"))
        paths.extend(root.rglob("*.ipynb"))
    return paths


@pytest.mark.parametrize(
    "heading",
    (
        "## Inference sources",
        "## Geometry",
        "## Masks and appearance",
        "## Sequence state",
        "## Working with results",
        "## Native trackers",
    ),
)
def test_track_guide_documents_v24_pipeline_sections(heading: str) -> None:
    content = TRACK_DOC.read_text(encoding="utf-8")

    assert heading in content


@pytest.mark.parametrize("source", ("images", "directories", "finite videos", "webcams", "URLs"))
def test_track_guide_lists_engine_frame_sources(source: str) -> None:
    content = TRACK_DOC.read_text(encoding="utf-8")

    assert source in content


def test_track_guide_uses_structured_tracker_contract() -> None:
    content = TRACK_DOC.read_text(encoding="utf-8")

    assert "TrackerSpec" in content
    assert "TrackingPipeline" in content
    assert "step_detections(frame, detections)" in content


def test_readme_minimal_usage_accepts_numpy_rows() -> None:
    content = README.read_text(encoding="utf-8")

    assert "import numpy as np" in content
    assert "from boxmot import ByteTrack" in content
    assert "tracker = ByteTrack()" in content
    assert "dets = np.array" in content
    assert "tracks = tracker.update(dets)" in content
    assert "tracks[:, 4].astype(int)" in content


def test_materialize_guide_documents_build_workflow() -> None:
    content = MATERIALIZE_DOC.read_text(encoding="utf-8")

    assert "boxmot materialize" in content
    assert "--build BUILD_ID" in content
    assert "never selects a “latest” build" in content
    assert "run canonical materialization automatically" in content
    assert ":command: materialize" in content


def test_eval_guide_documents_automatic_and_explicit_build_selection() -> None:
    content = EVAL_DOC.read_text(encoding="utf-8")

    assert "when omitted" in content
    assert "Dataset-only evaluation requires `--build`" in content
    assert "There is no latest-build selection" in content
    assert ":command: eval" in content


@pytest.mark.parametrize(
    "removed_surface",
    (
        "from boxmot import BoxMOT",
        "from boxmot import Detector",
        "ReIDModel(",
        "boxmot generate",
        "boxmot.engine.cli generate",
        "boxmot.api.functional",
    ),
)
def test_guides_and_examples_do_not_teach_removed_surface(removed_surface: str) -> None:
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in _public_text_files()
        if path != REPO_ROOT / "docs" / "guides" / "v24-migration.md"
        and removed_surface in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


@pytest.mark.parametrize("option", ("--dataset", "--split", "--save-crop"))
def test_track_guide_does_not_advertise_unsupported_options(option: str) -> None:
    content = TRACK_DOC.read_text(encoding="utf-8")

    assert f"`{option}`" not in content
