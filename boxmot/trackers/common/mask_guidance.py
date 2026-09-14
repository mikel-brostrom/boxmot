"""Configuration and track bookkeeping for McByte++ mask guidance.

The optional temporal model is independent of detection segmentation. Its masks
are keyed by track identity and guide association with current detection boxes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from boxmot.structures import Frame
from boxmot.utils.devices import normalize_device

if TYPE_CHECKING:
    from boxmot.segmentors.propagation.edgetam import EdgeTAMMaskPropagator
    from boxmot.trackers.common.specs import TrackerSpec


MASK_GUIDANCE_OPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "edgetam.min_coverage": "min_coverage",
        "edgetam.min_fill": "min_fill",
        "edgetam.prompt_overlap": "prompt_overlap",
        "edgetam.max_objects": "max_objects",
    }
)


def _validated_mask_guidance_values(values: Mapping[str, object]) -> dict[str, int | float]:
    """Validate and normalize scalar guidance settings without loading a model."""
    unknown = set(values).difference(MASK_GUIDANCE_OPTIONS.values())
    if unknown:
        raise TypeError("Unknown edgetam parameters: " + ", ".join(sorted(map(str, unknown))))
    normalized: dict[str, int | float] = {}
    for name, value in values.items():
        label = f"MaskGuidanceConfig.{name}"
        if name == "max_objects":
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{label} must be a positive integer.")
            if value < 1:
                raise ValueError(f"{label} must be a positive integer.")
            normalized[name] = int(value)
        else:
            interval = "(0, 1]" if name == "prompt_overlap" else "[0, 1]"
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{label} must be a finite number in {interval}.")
            if not np.isfinite(value) or not 0 <= value <= 1 or (name == "prompt_overlap" and value == 0):
                raise ValueError(f"{label} must be a finite number in {interval}.")
            normalized[name] = float(value)
    return normalized


def mask_guidance_config_from_options(
    checkpoint: str | Path, device: str, options: Mapping[str, object]
) -> MaskGuidanceConfig:
    """Construct guidance from resolved tracker options, retaining missing defaults."""
    values = {field: options[key] for key, field in MASK_GUIDANCE_OPTIONS.items() if key in options}
    return MaskGuidanceConfig(checkpoint=checkpoint, device=device, **values)


def validate_mask_guidance_spec(spec: TrackerSpec, *, resolved_options: Mapping[str, object] | None = None) -> None:
    """Check the supported shared box-guidance contract before model loading."""
    from boxmot.trackers.common.config import load_tracker_config
    from boxmot.trackers.common.registry import get_tracker_definition
    from boxmot.trackers.common.specs import TrackerFamily

    if spec.backend != "python" or get_tracker_definition(spec.name).capabilities.family is not TrackerFamily.BOX:
        raise ValueError("Mask guidance requires a Python 2D box tracker.")
    options = load_tracker_config(spec.name, None, spec.option_dict) if resolved_options is None else resolved_options
    if (
        spec.geometry != "aabb"
        or spec.per_class
        or options.get("per_class", False)
        or options.get("asso_func", "iou") != "iou"
    ):
        raise ValueError("Mask guidance requires AABB, asso_func='iou', and per_class=False.")


@dataclass(frozen=True, slots=True)
class MaskGuidanceConfig:
    """Configure EdgeTAM guidance for frames delivered to a tracker.

    Args:
        checkpoint: Local EdgeTAM checkpoint, or ``edgetam.pt`` to download into ``models``.
        device: Torch device on which to run mask propagation.
        max_objects: Maximum identities with temporal mask memory at one time.
        min_coverage: Minimum fraction of a propagated mask inside a candidate box.
        min_fill: Minimum fraction of a candidate box covered by the mask.
        prompt_overlap: Block prompts whose overlap with a foreground box reaches this fraction.
    """

    checkpoint: str | Path
    device: str = "cuda"
    max_objects: int = 32
    min_coverage: float = 0.90
    min_fill: float = 0.05
    prompt_overlap: float = 0.10

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, (str, Path)) or not str(self.checkpoint).strip():
            raise TypeError("MaskGuidanceConfig.checkpoint must be a non-empty path.")
        object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser())
        if not isinstance(self.device, str) or not self.device.strip():
            raise TypeError("MaskGuidanceConfig.device must be a non-empty Torch device string.")
        values = _validated_mask_guidance_values(
            {field: getattr(self, field) for field in MASK_GUIDANCE_OPTIONS.values()}
        )
        for name, value in values.items():
            object.__setattr__(self, name, value)


class MaskGuidance:
    """Keep previous observations and propagated masks aligned to track IDs."""

    def __init__(self, config: MaskGuidanceConfig, *, propagator: EdgeTAMMaskPropagator | None = None) -> None:
        """Own one tracker's temporal state, optionally using a shared-model propagator."""
        if not isinstance(config, MaskGuidanceConfig):
            raise TypeError("config must be a MaskGuidanceConfig.")
        if propagator is not None:
            if propagator.max_objects != config.max_objects:
                raise ValueError("Injected propagator max_objects must match MaskGuidanceConfig.max_objects.")
            if normalize_device(propagator.device) != normalize_device(config.device):
                raise ValueError("Injected propagator device must match MaskGuidanceConfig.device.")
            if propagator.prompt_overlap != config.prompt_overlap:
                raise ValueError("Injected propagator prompt_overlap must match MaskGuidanceConfig.prompt_overlap.")
        self.config = config
        self._propagator = propagator
        self._active_boxes: dict[int, np.ndarray] = {}
        self._new_boxes: dict[int, np.ndarray] = {}
        self._masks: dict[int, np.ndarray] = {}
        self._sequence_id: str | None = None
        self._pending_sequence_id: str | None = None
        self._source_frame_index: int | None = None
        self._pending_frame_index: int | None = None

    @property
    def masks(self) -> Mapping[int, np.ndarray]:
        """Borrow current CPU masks for synchronous rendering without copying pixels.

        The mapping is read-only and the propagator's arrays are immutable.
        Consume it before the next update; retaining snapshots retains their
        mask storage outside the tracker's temporal memory budget.
        """
        return MappingProxyType(self._masks)

    def validate_frame(self, frame: Frame | np.ndarray | None) -> None:
        """Check available source metadata without advancing temporal memory.

        Stage the sequence ID for binding only after propagation succeeds, so
        an otherwise invalid first update does not select the sequence. NumPy
        frames and absent metadata retain the caller's source-order contract.
        """
        self._pending_sequence_id = None
        self._pending_frame_index = None
        if not isinstance(frame, Frame):
            return
        if (
            frame.frame_index is not None
            and self._source_frame_index is not None
            and frame.frame_index <= self._source_frame_index
        ):
            raise ValueError(
                "Mask guidance requires increasing source frame indices; "
                f"received {frame.frame_index} after {self._source_frame_index}."
            )
        if frame.sequence_id is not None and self._sequence_id is not None and frame.sequence_id != self._sequence_id:
            raise ValueError(
                f"Mask guidance is bound to sequence {self._sequence_id!r}; "
                f"reset before processing sequence {frame.sequence_id!r}."
            )
        self._pending_sequence_id = frame.sequence_id
        self._pending_frame_index = frame.frame_index

    def advance(self, frame_index: int, frame: np.ndarray) -> None:
        """Propagate previous observations before any current-frame matching."""
        if self._propagator is None:
            from boxmot.segmentors.propagation.edgetam import EdgeTAMMaskPropagator

            self._propagator = EdgeTAMMaskPropagator(
                self.config.checkpoint,
                device=self.config.device,
                max_objects=self.config.max_objects,
                prompt_overlap=self.config.prompt_overlap,
            )
        # The propagator owns the previous masks needed for reseeding. Drop our
        # borrowed views before it replaces or evicts those masks during inference.
        self._masks = {}
        self._masks = self._propagator.propagate(frame_index, frame, self._active_boxes, self._new_boxes)
        if self._sequence_id is None:
            self._sequence_id = self._pending_sequence_id
        if self._pending_frame_index is not None:
            self._source_frame_index = self._pending_frame_index

    def condition(
        self,
        costs: np.ndarray,
        track_ids: Sequence[int],
        detection_boxes: np.ndarray,
        *,
        threshold: float,
    ) -> np.ndarray:
        """Apply the paper's gated mask cue with masks aligned to track rows."""
        from boxmot.trackers.common.association.masks import apply_mask_guidance

        return apply_mask_guidance(
            costs,
            detection_boxes,
            [self._masks.get(int(track_id)) for track_id in track_ids],
            threshold=threshold,
            min_coverage=self.config.min_coverage,
            min_fill=self.config.min_fill,
        )

    def observe(
        self,
        active_boxes: Mapping[int, np.ndarray],
        newly_confirmed_ids: Sequence[int],
        *,
        retained_track_ids: Sequence[int],
    ) -> None:
        """Save matched detection boxes for next-frame mask prompts.

        The reference prompts masks after confirmation, using detection boxes
        rather than Kalman-filtered geometry. First-frame tracks are seeded
        together by the propagation backend.
        """
        self._active_boxes = {int(key): np.array(box, dtype=np.float32, copy=True) for key, box in active_boxes.items()}
        self._new_boxes = {
            int(key): self._active_boxes[int(key)].copy()
            for key in newly_confirmed_ids
            if int(key) in self._active_boxes
        }
        retained = set(retained_track_ids)
        self._masks = {key: mask for key, mask in self._masks.items() if key in retained}
        if self._propagator is not None:
            self._propagator.retain_tracks(retained)

    def reset(self) -> None:
        """Clear sequence memory before tracker IDs can be reused."""
        if self._propagator is not None:
            self._propagator.reset()
        self._active_boxes.clear()
        self._new_boxes.clear()
        self._masks.clear()
        self._sequence_id = None
        self._pending_sequence_id = None
        self._source_frame_index = None
        self._pending_frame_index = None
