"""Track manually selected objects using EdgeTAM's temporal model memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral
from typing import Protocol

import numpy as np
import torch
from typing_extensions import Unpack

from boxmot.segmentors.propagation.prompts import EdgeTAMPrompt
from boxmot.structures import Detections, Frame, GeometryKind, Tracks
from boxmot.trackers.common.base import BaseTracker
from boxmot.trackers.common.constructor import TrackerMetadataOptions, validate_runtime_options
from boxmot.trackers.common.input import frame_image_size
from boxmot.trackers.common.specs import TrackerCapabilities, TrackerFamily
from boxmot.trackers.edgetam.config import EdgeTAMConfig, EdgeTAMModelConfig


class _PromptPropagator(Protocol):
    """Minimal temporal backend contract supporting injected test implementations."""

    def process(
        self,
        frame_index: int,
        frame: np.ndarray,
        prompts: Sequence[EdgeTAMPrompt] = (),
    ) -> dict[int, torch.Tensor]: ...

    def remove_object(self, object_id: int) -> None: ...

    def reset(self) -> None: ...


def _identifier(value: object, name: str) -> int:
    """Validate a canonical nonnegative int64 identity without lossy coercion."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    if not 0 <= value <= np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be within the nonnegative int64 range.")
    return int(value)


class EdgeTAM(BaseTracker):
    """Propagate prompted objects independently of detector observations.

    Queue boxes or foreground/background points with ``add_box`` or
    ``add_points``, then call ``update`` with empty detections and the selected
    frame. Subsequent frames retain the selected identities through temporal
    model memory. Point corrections refine the specified object's propagated
    mask; a replacement box initializes it again on the incoming frame.
    Canonical output includes masks and their enclosing AABBs; empty masks
    remain in model memory but produce no visible track row.
    """

    capabilities = TrackerCapabilities(
        family=TrackerFamily.MULTIMODAL,
        geometry_kinds=frozenset({GeometryKind.AABB}),
        requires_frame=True,
        accepts_frame=True,
    )
    _requires_frame = True
    uses_frame_dimensions_for_association = False

    def __init__(
        self,
        config: EdgeTAMConfig | None = None,
        *,
        prompt_model: EdgeTAMModelConfig | None = None,
        propagator: _PromptPropagator | None = None,
        **kwargs: Unpack[TrackerMetadataOptions],
    ) -> None:
        """Configure prompting without allocating or downloading model weights.

        Args:
            config: Immutable object capacity and propagation batch settings.
            prompt_model: Checkpoint selection and inference device, separate from algorithm settings.
            propagator: Optional prepared temporal backend for component injection.
            **kwargs: Runtime ``class_ids`` and ``class_names`` metadata; ``per_class`` must be false.
        """
        config = EdgeTAMConfig.resolve(config)
        validate_runtime_options(kwargs, box=False)
        per_class = kwargs.get("per_class", False)
        if not isinstance(per_class, bool):
            raise TypeError("per_class must be bool.")
        if per_class:
            raise ValueError("EdgeTAM prompts share one object namespace; per_class is not supported.")
        if prompt_model is not None and not isinstance(prompt_model, EdgeTAMModelConfig):
            raise TypeError("prompt_model must be an EdgeTAMModelConfig object or None.")
        if propagator is not None and any(
            not callable(getattr(propagator, name, None)) for name in ("process", "remove_object", "reset")
        ):
            raise TypeError("propagator must provide process(), remove_object(), and reset().")
        if kwargs.get("class_ids") is not None:
            kwargs["class_ids"] = tuple(_identifier(value, "class_id") for value in kwargs["class_ids"])
        if kwargs.get("class_names") is not None:
            names = kwargs["class_names"]
            if not isinstance(names, Mapping):
                raise TypeError("class_names must be a mapping of class IDs to names.")
            if any(not isinstance(value, str) or not value for value in names.values()):
                raise ValueError("class_names must contain nonempty string names.")
            kwargs["class_names"] = {_identifier(key, "class_id"): value for key, value in names.items()}
        self.config = config
        self.prompt_model = prompt_model or EdgeTAMModelConfig()
        self._propagator = propagator
        super().__init__(
            max_age=config.max_age,
            max_obs=config.max_obs,
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold,
            asso_func=config.asso_func,
            centroid_max_distance=config.centroid_max_distance,
            **kwargs,
        )
        self._object_classes: dict[int, int] = {}
        self._tracked_object_ids: set[int] = set()
        self._pending_prompts: dict[int, EdgeTAMPrompt] = {}
        self._next_object_id = 0
        self._output_object_ids: tuple[int, ...] = ()
        self._propagation_index = 0
        self._image_size: tuple[int, int] | None = None

    @property
    def object_ids(self) -> tuple[int, ...]:
        """Return selected identities, including queued and temporarily invisible objects."""
        return tuple(self._object_classes)

    def _prompt_identity(self, object_id: int | None, class_id: int) -> tuple[int, int]:
        """Resolve a proposed identity and class without reserving either."""
        object_id = _identifier(self._next_object_id if object_id is None else object_id, "object_id")
        class_id = _identifier(class_id, "class_id")
        if object_id in self._object_classes:
            existing = self._object_classes[object_id]
            if class_id not in (0, existing):
                raise ValueError("Corrections must preserve the selected object's class_id.")
            class_id = existing
        elif len(self._object_classes) >= self.config.max_objects:
            raise ValueError(f"EdgeTAM supports at most {self.config.max_objects} selected objects; remove one first.")
        self.class_catalog.validate_ids((class_id,))
        return object_id, class_id

    def _queue_prompt(self, prompt: EdgeTAMPrompt, class_id: int) -> int:
        """Commit one validated prompt and reserve its stable identity."""
        if self._image_size is not None:
            prompt.validate_bounds(self._image_size)
        self._pending_prompts[prompt.object_id] = prompt
        self._object_classes[prompt.object_id] = class_id
        self._next_object_id = max(self._next_object_id, prompt.object_id + 1)
        return prompt.object_id

    def add_box(self, box: Sequence[float], *, object_id: int | None = None, class_id: int = 0) -> int:
        """Queue an XYXY box, preserving the object's class and any pending points."""
        object_id, class_id = self._prompt_identity(object_id, class_id)
        previous = self._pending_prompts.get(object_id)
        prompt = EdgeTAMPrompt(
            object_id,
            box=tuple(box),
            points=() if previous is None else previous.points,
            labels=() if previous is None else previous.labels,
        )
        return self._queue_prompt(prompt, class_id)

    def add_points(
        self,
        points: Sequence[Sequence[float]],
        labels: Sequence[int] | None = None,
        *,
        object_id: int | None = None,
        class_id: int = 0,
    ) -> int:
        """Queue foreground/background points to initialize or refine an object.

        Labels default to foreground (1). Each call replaces the pending point
        set and preserves a pending box. Include all points intended for this
        correction. Existing objects use their propagated current-frame mask
        as context; new objects need a foreground point or a pending box.
        """
        object_id, class_id = self._prompt_identity(object_id, class_id)
        points = tuple(tuple(point) for point in points)
        labels = (1,) * len(points) if labels is None else tuple(labels)
        previous = self._pending_prompts.get(object_id)
        prompt = EdgeTAMPrompt(
            object_id,
            box=None if previous is None else previous.box,
            points=points,
            labels=labels,
        )
        if not prompt.points:
            raise ValueError("Point prompts must contain a nonempty point set.")
        if prompt.box is None and 1 not in prompt.labels and object_id not in self._tracked_object_ids:
            raise ValueError("New objects require a foreground point or a pending box.")
        return self._queue_prompt(prompt, class_id)

    def remove_object(self, object_id: int) -> None:
        """Remove an object's pending prompts and model memory; absent IDs are harmless."""
        object_id = _identifier(object_id, "object_id")
        if self._propagator is not None:
            self._propagator.remove_object(object_id)
        self._pending_prompts.pop(object_id, None)
        self._object_classes.pop(object_id, None)
        self._tracked_object_ids.discard(object_id)

    def reset(self) -> None:
        """Clear sequence state and identity allocation while retaining the loaded model."""
        if self._propagator is not None:
            self._propagator.reset()
        self._reset_common_state()
        self._object_classes.clear()
        self._tracked_object_ids.clear()
        self._pending_prompts.clear()
        self._next_object_id = 0
        self._output_object_ids = ()
        self._propagation_index = 0
        self._image_size = None

    def _validate_detection_inputs(self, detections: Detections) -> None:
        """Reject detector observations before the shared public wrapper changes state."""
        if len(detections):
            raise ValueError(
                "EdgeTAM uses manual prompts; pass empty detections and select objects with add_box/add_points."
            )
        super()._validate_detection_inputs(detections)

    def _update_arrays(
        self,
        *,
        geometry: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        sample_id: str | None,
        frame: Frame | np.ndarray | None,
        embeddings: np.ndarray | None,
        masks: np.ndarray | None,
        mask_image_size: tuple[int, int] | None,
        numpy_output: bool,
    ) -> Tracks | np.ndarray:
        """Keep user object IDs exact across the shared float32 kernel adapter."""
        if len(scores):
            raise ValueError(
                "EdgeTAM uses manual prompts; pass empty detections and select objects with add_box/add_points."
            )
        assert frame is not None
        image_size = frame_image_size(frame)
        if self._image_size is not None and image_size != self._image_size:
            raise ValueError("EdgeTAM requires constant frame dimensions; reset before a new sequence.")
        for prompt in self._pending_prompts.values():
            prompt.validate_bounds(image_size)
        if numpy_output and any(
            value > 2**53 for values in (self._object_classes, self._object_classes.values()) for value in values
        ):
            raise ValueError(
                "Packed object and class IDs must fit the exact float64 integer range; use Detections input."
            )
        result = super()._update_arrays(
            geometry=geometry,
            scores=scores,
            class_ids=class_ids,
            sample_id=sample_id,
            frame=frame,
            embeddings=embeddings,
            masks=masks,
            mask_image_size=mask_image_size,
            numpy_output=numpy_output,
        )
        if isinstance(result, Tracks):
            return replace(result, track_ids=torch.tensor(self._output_object_ids, dtype=torch.int64))
        result[:, self.detection_layout.schema.track_id_index] = self._output_object_ids
        return result

    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray | None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate memory and derive visible AABBs from current-frame masks."""
        del dets, embs, masks
        assert img is not None
        height, width = img.shape[:2]
        propagated: Mapping[int, torch.Tensor] = {}
        if self._object_classes:
            if self._propagator is None:
                from boxmot.segmentors.propagation.prompted import EdgeTAMPromptPropagator

                self._propagator = EdgeTAMPromptPropagator(
                    self.prompt_model.checkpoint,
                    device=self.prompt_model.device,
                    max_objects=self.config.max_objects,
                    batch_size=self.config.batch_size,
                )
            propagated = self._propagator.process(self._propagation_index, img, tuple(self._pending_prompts.values()))
            if not isinstance(propagated, Mapping):
                raise TypeError("EdgeTAM propagator must return a mapping of object IDs to masks.")
            if propagated.keys() - self._object_classes.keys():
                raise ValueError("EdgeTAM propagator returned an unselected object ID.")
        rows: list[list[float]] = []
        visible_ids: list[int] = []
        visible_masks: list[np.ndarray] = []
        for object_id in sorted(propagated):
            mask = propagated[object_id]
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != (height, width):
                raise ValueError("EdgeTAM propagator masks must be boolean tensors matching frame dimensions.")
            owned_mask = mask.detach().cpu().numpy().copy()
            # Reduce each axis instead of allocating coordinates for every foreground pixel.
            y = np.flatnonzero(owned_mask.any(axis=1))
            if not len(y):
                continue
            x = np.flatnonzero(owned_mask.any(axis=0))
            class_code = self._encode_kernel_class_id(self._object_classes[object_id])
            rows.append([float(x[0]), float(y[0]), float(x[-1] + 1), float(y[-1] + 1), len(rows), 1.0, class_code, -1])
            visible_ids.append(object_id)
            visible_masks.append(owned_mask)
        self._tracked_object_ids.update(self._object_classes)
        self._pending_prompts.clear()
        self._output_object_ids = tuple(visible_ids)
        self._image_size = (height, width)
        self.frame_count += 1
        if self._object_classes:
            self._propagation_index += 1
        raw = np.asarray(rows, dtype=np.float32).reshape(-1, self.detection_layout.output_cols)
        output_masks = np.stack(visible_masks) if visible_masks else np.empty((0, height, width), dtype=np.bool_)
        return raw, output_masks


__all__ = ("EdgeTAM",)
