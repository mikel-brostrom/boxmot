from __future__ import annotations

import pytest

from boxmot.segmentors.config import resolve_segmentor_spec


def test_segmentor_defaults_are_part_of_canonical_spec(tmp_path) -> None:
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"resolved model bytes")

    spec, _ = resolve_segmentor_spec(
        {"backend": "maskrcnn", "artifact": str(artifact)},
        geometry="obb",
        allow_download=False,
    )

    assert spec.geometry_mode == "obb"
    assert spec.option_values() == {
        "class_mapping": (),
        "mask_threshold": 0.5,
        "match_iou": 0.5,
        "score_threshold": 0.0,
    }


def test_segmentor_resolution_requires_artifact_before_build_planning() -> None:
    with pytest.raises(ValueError, match="Segmentor"):
        resolve_segmentor_spec({"backend": "fixture"}, geometry="aabb", allow_download=False)
