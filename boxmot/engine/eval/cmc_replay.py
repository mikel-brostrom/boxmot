"""Immutable camera transforms and their replay input provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from boxmot.engine.calibration.cmc import (
    cmc_analysis_settings,
    cmc_build_signature,
    cmc_image_signature,
    cmc_runtime_mask_settings,
)
from boxmot.engine.config.cmc import CMC_ANALYSIS_RULE
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.config import load_tracker_config
from boxmot.trackers.common.motion.cmc.integration import resolve_cmc_min_conf


def cmc_tracker_inputs(spec: TrackerSpec) -> dict[str, Any]:
    """Describe all tracker settings that determine automatic camera estimates."""
    options = load_tracker_config(spec.name, None, spec.option_dict)
    return {
        "tracker": spec.name,
        "backend": spec.backend,
        "geometry": spec.geometry,
        "per_class": spec.per_class,
        "class_ids": None if spec.class_ids is None else list(spec.class_ids),
        "class_names": [list(entry) for entry in spec.class_names],
        "rule": options.get("calibration.cmc_rule", CMC_ANALYSIS_RULE),
        "method": "sof",
        "opencv_version": cv2.__version__,
        "settings": cmc_analysis_settings(options),
        "mask_min_conf": float(resolve_cmc_min_conf(options)),
        "runtime_mask": cmc_runtime_mask_settings(options),
    }


@dataclass(frozen=True, slots=True)
class CMCReplaySequence:
    """Validated frame-aligned affine transforms with immutable backing bytes."""

    sequence_id: str
    sample_ids: tuple[str, ...]
    frame_indices: tuple[int, ...]
    warp_bytes: bytes
    build: str
    split: str | None
    source_signature: tuple[tuple[str, tuple[int, ...]], ...]
    image_root: str
    image_refs: tuple[str, ...]
    image_signature: tuple[tuple[str, tuple[int, ...]], ...]
    tracker_inputs: str

    def __post_init__(self) -> None:
        count = len(self.sample_ids)
        if (
            not self.sequence_id
            or not count
            or any(not isinstance(value, str) or not value for value in self.sample_ids)
            or len(set(self.sample_ids)) != count
            or self.frame_indices != tuple(range(count))
            or len(self.image_refs) != count
            or not isinstance(self.warp_bytes, bytes)
            or len(self.warp_bytes) != count * 6 * np.dtype("float32").itemsize
        ):
            raise ValueError("Prepared CMC transforms must match unique samples in delivered frame order.")
        if not np.isfinite(self.warps).all() or not np.array_equal(self.warps[0], np.eye(2, 3, dtype=np.float32)):
            raise ValueError("Prepared CMC transforms must be finite with identity at the first frame.")

    @property
    def warps(self) -> np.ndarray:
        """Expose a read-only array, including after serialization to a worker."""
        return np.frombuffer(self.warp_bytes, dtype=np.float32).reshape(len(self.sample_ids), 2, 3)

    def validate_sources(self) -> None:
        """Reject changed build files or images immediately before worker replay."""
        try:
            build_signature = cmc_build_signature(Path(self.build))
            image_signature = cmc_image_signature(self.image_refs, self.image_root)
        except (OSError, ValueError) as exc:
            raise ValueError("Prepared CMC source files changed; prepare camera analysis again.") from exc
        expected_build = {name: list(value) for name, value in self.source_signature}
        expected_images = {name: list(value) for name, value in self.image_signature}
        if build_signature != expected_build or image_signature != expected_images:
            raise ValueError("Prepared CMC source files changed; prepare camera analysis again.")

    def validate_selection(self, build: Path, split: str | None) -> None:
        """Reject transforms prepared for another materialized build or split."""
        if str(Path(build).resolve()) != self.build or split != self.split:
            raise ValueError("Prepared CMC transforms do not match the replay build and split.")

    def validate_tracker(self, spec: TrackerSpec) -> None:
        """Require the effective replay tracker to use this analysis's camera inputs."""
        options = load_tracker_config(spec.name, None, spec.option_dict)
        if (
            not options.get("use_cmc")
            or options.get("cmc_method") != "sof"
            or json.loads(self.tracker_inputs) != cmc_tracker_inputs(spec)
        ):
            raise ValueError("Prepared CMC transforms do not match the replay tracker camera inputs.")
