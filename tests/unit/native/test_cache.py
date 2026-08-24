from __future__ import annotations

import numpy as np

from boxmot.native._common import embedding_cache_is_complete


def test_native_cache_completeness_rejects_nonempty_zero_width_embeddings(tmp_path):
    detections = tmp_path / "detections.npy"
    embeddings = tmp_path / "embeddings.npy"
    np.save(detections, np.zeros((2, 7), dtype=np.float32))
    np.save(embeddings, np.empty((2, 0), dtype=np.float32))

    assert embedding_cache_is_complete(embeddings, detections) is False
