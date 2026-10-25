from __future__ import annotations

from typing import Any

import pytest
import torch

from boxmot.datasets.storage import embedding_records, instance_records, mask_records
from boxmot.structures import Boxes, Detections, MaskBatch


def _detections(sample_id: str | None, *, count: int) -> Detections:
    """Build complete rows so record validation isolates sample identity."""
    return Detections(
        geometry=Boxes(torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32).repeat(count, 1)),
        scores=torch.ones(count, dtype=torch.float32),
        class_ids=torch.zeros(count, dtype=torch.int64),
        sample_id=sample_id,
        instance_ids=tuple(f"build:{sample_id}:{index}" for index in range(count)),
        masks=MaskBatch(torch.ones((count, 2, 2), dtype=torch.bool)),
        embeddings=torch.ones((count, 3), dtype=torch.float32),
    )


def _records(artifact: str, detections: Detections) -> list[dict[str, Any]]:
    """Serialize one of the keyed detection artifacts."""
    if artifact == "instances":
        return instance_records(detections, build_id="build")
    if artifact == "masks":
        return mask_records(detections)
    return embedding_records(detections, encoder_fingerprint="a" * 64)


@pytest.mark.parametrize("artifact", ("instances", "masks", "embeddings"))
@pytest.mark.parametrize("count", (0, 1))
def test_detection_records_require_explicit_sample_identity(artifact: str, count: int) -> None:
    """Optional live identities must never produce null or synthetic cache keys."""
    with pytest.raises(ValueError, match="require an explicit sample_id"):
        _records(artifact, _detections(None, count=count))


@pytest.mark.parametrize("artifact", ("instances", "masks", "embeddings"))
def test_detection_records_preserve_explicit_sample_identity(artifact: str) -> None:
    """Identified rows preserve the shared sample and instance keys."""
    records = _records(artifact, _detections("sample", count=1))

    assert len(records) == 1
    assert records[0]["sample_id"] == "sample"
    assert records[0]["instance_id"] == "build:sample:0"
