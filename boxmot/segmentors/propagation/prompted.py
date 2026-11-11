"""Bounded EdgeTAM tracking from explicit, current-frame user prompts."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boxmot.segmentors.propagation.edgetam import _EdgeTAMStream
from boxmot.segmentors.propagation.model import inference_context
from boxmot.segmentors.propagation.prompts import EdgeTAMPrompt


class EdgeTAMPromptPropagator(_EdgeTAMStream):
    """Propagate selected objects while preserving their independent histories.

    Boxes initialize or replace their object's memory on the submitted frame.
    Point corrections refine an existing object's current propagated mask.
    Other objects continue tracking. Returned mappings borrow device boolean
    masks and must not be modified by callers.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        max_objects: int = 32,
        batch_size: int = 8,
        predictor: Any | None = None,
    ) -> None:
        """Load one shared model or use an injected official video predictor."""
        super().__init__(checkpoint, device=device, max_objects=max_objects, batch_size=batch_size, predictor=predictor)

    def process(
        self,
        frame_index: int,
        frame: np.ndarray,
        prompts: Sequence[EdgeTAMPrompt] = (),
    ) -> dict[int, torch.Tensor]:
        """Consume one consecutive BGR frame and emit same-frame prompted masks."""
        self._validate_frame(frame_index, frame)
        if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)):
            raise TypeError("prompts must be a sequence of EdgeTAMPrompt objects.")
        prompts = tuple(prompts)
        if any(not isinstance(prompt, EdgeTAMPrompt) for prompt in prompts):
            raise TypeError("prompts must contain only EdgeTAMPrompt objects.")
        prompted_ids = {prompt.object_id for prompt in prompts}
        if len(prompted_ids) != len(prompts):
            raise ValueError("Only one prompt per object may be submitted for a frame.")
        if len(self._objects.keys() | prompted_ids) > self.max_objects:
            raise ValueError(f"EdgeTAM supports at most {self.max_objects} selected objects; remove an object first.")
        for prompt in prompts:
            prompt.validate_bounds(tuple(frame.shape[:2]))
            if prompt.box is None and 1 not in prompt.labels and prompt.object_id not in self._objects:
                raise ValueError("New objects require a foreground point or a box.")

        frame_index = int(frame_index)
        with inference_context(self.device):
            self._append_frame(frame_index, frame)
            for track_ids in self._object_batches():
                continuing = [object_id for object_id in track_ids if object_id not in prompted_ids]
                if continuing:
                    self._frame_masks(frame_index, continuing)
            for prompt in prompts:
                self._seed_prompt(frame_index, prompt)
            self._prune_history(frame_index)
        self.last_frame_index = frame_index
        return dict(self._masks)

    def _seed_prompt(self, frame_index: int, prompt: EdgeTAMPrompt) -> None:
        """Commit one official prompt/preflight result without resetting other IDs."""
        assert self._state is not None and self.frame_shape is not None
        refining = prompt.box is None and prompt.object_id in self._objects
        state = self._refinement_state(frame_index, prompt.object_id) if refining else self._prompt_state()
        inputs = {}
        if prompt.box is not None:
            inputs["box"] = np.asarray(prompt.box, dtype=np.float32)
        if prompt.points:
            inputs["points"] = np.asarray(prompt.points, dtype=np.float32)
            inputs["labels"] = np.asarray(prompt.labels, dtype=np.int32)
        result_frame, object_ids, logits = self._predictor.add_new_points_or_box(
            state, frame_index, prompt.object_id, **inputs
        )
        if result_frame != frame_index or list(object_ids) != [prompt.object_id]:
            raise RuntimeError("EdgeTAM prompt output must identify the submitted frame and object.")
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 1, *self.frame_shape):
            raise RuntimeError("EdgeTAM must return one original-resolution mask for a prompted object.")
        self._predictor.propagate_in_video_preflight(state)
        storage_key = "non_cond_frame_outputs" if refining else "cond_frame_outputs"
        current = state["output_dict"][storage_key][frame_index]
        self._objects[prompt.object_id] = {
            "cond_frame_outputs": {frame_index: self._object_memory(current, 0, batch_size=1)},
            "non_cond_frame_outputs": {},
        }
        self._masks[prompt.object_id] = (logits[0, 0] > 0).detach()
        self._state["cached_features"] = state["cached_features"]

    def _refinement_state(self, frame_index: int, object_id: int) -> dict[str, Any]:
        """Give upstream clicks current-frame logits and the object's temporal context."""
        assert self._state is not None
        history = self._objects[object_id]
        current, _ = self._predictor._run_single_frame_inference(
            inference_state=self._state,
            output_dict=history,
            frame_idx=frame_index,
            batch_size=1,
            is_init_cond_frame=False,
            point_inputs=None,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=False,
        )
        state = self._prompt_state()
        obj_idx = self._predictor._obj_id_to_idx(state, object_id)
        outputs = {kind: dict(frames) for kind, frames in history.items()}
        outputs["non_cond_frame_outputs"][frame_index] = current
        state["output_dict_per_obj"][obj_idx] = outputs
        state["frames_already_tracked"][frame_index] = {"reverse": False}
        state["tracking_has_started"] = True
        # add_new_points_or_box looks up pred_masks for this exact frame and
        # supplies the logits to its decoder alongside the correction clicks.
        # Preflight then encodes only the corrected result as future memory.
        return state

    def remove_object(self, object_id: int) -> None:
        """Release an object's memory and mask; an already absent ID is harmless."""
        if isinstance(object_id, bool) or not isinstance(object_id, Integral) or object_id < 0:
            raise ValueError("object_id must be a nonnegative integer.")
        self._objects.pop(int(object_id), None)
        self._masks.pop(int(object_id), None)


__all__ = ("EdgeTAMPromptPropagator",)
