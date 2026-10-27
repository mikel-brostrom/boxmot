"""Prepared runtime warps preserve each tracker's existing camera mask contract."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot import create_tracker
from boxmot.datasets.cached import DatasetSample
from boxmot.engine.calibration import cmc
from boxmot.structures import Boxes, Detections
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.motion.cmc.sof import SOF
from tests.unit.engine.calibration.test_cmc_candidates import _source_build


def _mask_warp(boxes: np.ndarray, frame_index: int) -> np.ndarray:
    """Make the selected mask rows observable in an otherwise valid transform."""
    warp = np.eye(2, 3, dtype=np.float32)
    if frame_index:
        warp[0, 2] = 1.0 + 0.02 * boxes[:, 0].sum() + 0.25 * len(boxes)
    return warp


def _record_sof(self: SOF, image: np.ndarray, dets: np.ndarray | None = None) -> np.ndarray:
    """Record actual production CMC inputs, including diagnostic subclasses."""
    if not hasattr(self, "_test_mask_inputs"):
        self._test_mask_inputs = []
        self._test_warps = []
    frame_index = len(self._test_mask_inputs)
    self._test_mask_inputs.append(dets.copy())
    self.was_initializing = frame_index == 0
    self.inliers_accepted = self.transform_valid = frame_index > 0
    self.match_count = self.inlier_count = 30
    warp = _mask_warp(dets, frame_index)
    self._test_warps.append(warp.copy())
    return warp


class _Dataset(list[DatasetSample]):
    """Use real canonical detections without invoking artifact reader fixtures."""

    manifest = SimpleNamespace(box_type="aabb")

    def __init__(self, samples: list[DatasetSample], image_root: Path) -> None:
        super().__init__(samples)
        self._image_root = str(image_root)


def _sample(index: int, image_root: Path) -> DatasetSample:
    """Include low scores and exact threshold boundaries across two classes."""
    scores = torch.tensor([0.05, 0.1, 0.3, 0.5, 0.95], dtype=torch.float32)
    class_ids = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int64)
    boxes = torch.tensor([[5 + 15 * row, 10, 15 + 15 * row, 30] for row in range(5)], dtype=torch.float32)
    if index == 0:
        # Share initialization history before native/default masking diverges.
        scores.fill_(0.95)
    if index == 2:
        # Retain only one detected class while previous class state may survive.
        selected = class_ids == 1
        boxes, scores, class_ids = boxes[selected], scores[selected], class_ids[selected]
    sample_id = f"sequence/{index}"
    image_ref = f"frame-{index}.png"
    (image_root / image_ref).write_bytes(b"pixels decoded by the instrumented reader")
    return DatasetSample(
        sample_id=sample_id,
        split="train",
        sequence_id="sequence",
        frame_index=index,
        timestamp_s=None,
        image_size=(100, 100),
        image_ref=image_ref,
        frame=None,
        detections=Detections(
            geometry=Boxes(boxes),
            scores=scores,
            class_ids=class_ids,
            sample_id=sample_id,
            embeddings=torch.ones((len(scores), 4), dtype=torch.float32),
        ),
    )


@pytest.mark.parametrize(
    "tracker_name",
    ["bytetrack", "botsort", "ocsort", "deepocsort", "boosttrack", "hybridsort", "occluboost", "strongsort"],
)
@pytest.mark.parametrize("explicit_threshold", [None, 0.5])
@pytest.mark.parametrize("per_class", [False, True])
def test_runtime_warp_archive_matches_live_masks_without_extra_image_decodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tracker_name: str,
    explicit_threshold: float | None,
    per_class: bool,
) -> None:
    """Analysis decisions remain unchanged while replay matches live camera calls."""
    build, image_root = tmp_path / "build", tmp_path / "images"
    _source_build(build)
    image_root.mkdir()
    samples = [_sample(index, image_root) for index in range(3)]
    dataset = _Dataset(samples, image_root)
    monkeypatch.setattr(cmc.CachedVisionDataset, "_for_sequence", lambda *args, **kwargs: dataset)
    decoded = []

    def decode(sample: DatasetSample, root: Path | str) -> np.ndarray:
        decoded.append(sample.sample_id)
        return np.full((100, 100, 3), sample.frame_index, dtype=np.uint8)

    monkeypatch.setattr(cmc, "_bgr_image", decode)
    monkeypatch.setattr(SOF, "apply", _record_sof)
    options = {"use_cmc": True, "cmc_method": "sof", "cmc_min_conf": explicit_threshold}
    spec = TrackerSpec(tracker_name, per_class=per_class, options=tuple(sorted(options.items())))
    tracker = create_tracker(spec)
    try:
        for sample in samples:
            detections = sample.detections
            if not tracker.requirements.embeddings:
                detections = replace(detections, embeddings=None)
            tracker.update(detections, np.zeros((100, 100, 3), dtype=np.uint8))
        expected = np.asarray(tracker.cmc._test_warps)
    finally:
        tracker.reset()
        close = getattr(tracker, "close", None)
        if callable(close):
            close()

    args = Namespace(
        tracker=tracker_name,
        tracker_backend="python",
        geometry="aabb",
        build_path=build,
        split="train",
        seq_info={"sequence": len(samples)},
        sequence_names=None,
        per_class=per_class,
    )
    result = cmc.prepare_cmc_candidates(args, options, tmp_path / "analysis")
    archive_keys = result.report["sequences"]["sequence"]["archive_keys"]
    with np.load(result.warps_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive[archive_keys["runtime_warps"]], expected)
        threshold = result.mask_min_conf
        original_policy_warps = np.asarray([
            _mask_warp(
                sample.detections.geometry.values.numpy()[sample.detections.scores.numpy() >= threshold],
                index,
            )
            for index, sample in enumerate(samples)
        ])
        np.testing.assert_array_equal(archive[archive_keys["warps"]], original_policy_warps)
    assert result.policy.sequences == (("sequence", True),)
    assert decoded == [sample.sample_id for sample in samples]
