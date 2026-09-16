"""Bounded, causal mask propagation using the official EdgeTAM predictor."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection, Iterator, Mapping
from numbers import Integral, Real
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from PIL import Image

from boxmot.segmentors.propagation.model import build_edgetam_predictor, inference_context
from boxmot.utils.devices import resolve_device

_MEMORY_FIELDS = ("maskmem_features", "maskmem_pos_enc", "obj_ptr")
DEFAULT_OBJECT_BATCH_SIZE = 4
_UINT8_TO_FLOAT32 = (np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)


def _boxes_by_id(boxes: Mapping[int, np.ndarray], *, name: str) -> dict[int, np.ndarray]:
    """Validate previous-frame AABB prompts indexed by tracker identity."""
    if not isinstance(boxes, Mapping):
        raise TypeError(f"{name} must map track IDs to xyxy boxes.")
    validated: dict[int, np.ndarray] = {}
    for track_id, box in boxes.items():
        if isinstance(track_id, bool) or not isinstance(track_id, Integral) or track_id < 0:
            raise ValueError(f"{name} track IDs must be nonnegative integers.")
        values = np.asarray(box, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError(f"{name} boxes must contain four finite xyxy coordinates.")
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"{name} boxes must have positive width and height.")
        validated[int(track_id)] = values.copy()
    return validated


def _can_prompt(box: np.ndarray, active_boxes: Mapping[int, np.ndarray], *, prompt_overlap: float = 0.10) -> bool:
    """Apply McByte++'s maximum overlap gate for boxes with a lower bottom edge."""
    area = float((box[2] - box[0]) * (box[3] - box[1]))
    for other in active_boxes.values():
        if other[3] <= box[3]:
            continue
        width = max(0.0, float(min(box[2], other[2]) - max(box[0], other[0])))
        height = max(0.0, float(min(box[3], other[3]) - max(box[1], other[1])))
        if width * height / area >= prompt_overlap:
            return False
    return True


class EdgeTAMMaskPropagator:
    """Propagate previous-frame prompts with bounded objects and temporal state.

    One official model and one image-feature cache serve every object. Compatible
    temporal histories run in bounded object batches, limiting decoder and attention
    workspaces. New prompts trigger McByte++'s collective reset and reseed lifecycle.

    Returned dictionaries borrow device boolean tensors; callers must not modify
    their contents. Call ``retain_tracks``
    after association to release retired identities. At capacity, visible existing
    objects precede promptable visible newcomers, followed by the most recently
    observed lost objects. IDs break ties.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda",
        predictor: Any | None = None,
        max_objects: int = 96,
        batch_size: int = DEFAULT_OBJECT_BATCH_SIZE,
        prompt_overlap: float = 0.10,
    ) -> None:
        """Load a shared model or accept one, retaining at most ``max_objects`` IDs."""
        if isinstance(max_objects, bool) or not isinstance(max_objects, Integral) or max_objects < 1:
            raise ValueError("EdgeTAM max_objects must be a positive integer.")
        self.max_objects = int(max_objects)
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("EdgeTAM batch_size must be a positive integer.")
        self.batch_size = int(batch_size)
        if isinstance(prompt_overlap, bool) or not isinstance(prompt_overlap, Real):
            raise TypeError("EdgeTAM prompt_overlap must be a finite number in (0, 1].")
        if not np.isfinite(prompt_overlap) or not 0 < prompt_overlap <= 1:
            raise ValueError("EdgeTAM prompt_overlap must be a finite number in (0, 1].")
        self.prompt_overlap = float(prompt_overlap)
        self.device = resolve_device(device)
        self._predictor = build_edgetam_predictor(checkpoint, self.device) if predictor is None else predictor
        model = self._predictor
        if (
            model.training
            or model.add_tpos_enc_to_obj_ptrs
            or model.memory_temporal_stride_for_eval != 1
            or model.non_overlap_masks
            or model.non_overlap_masks_for_mem_enc
            or model.clear_non_cond_mem_around_input
            or model.add_all_frames_to_correct_as_cond
            or model.num_maskmem < 2
            or not model.use_obj_ptrs_in_encoder
            or model.max_obj_ptrs_in_encoder < model.num_maskmem
        ):
            raise ValueError("Streaming requires the reference EdgeTAM evaluation memory configuration.")
        if predictor is not None and (perceiver := getattr(predictor, "spatial_perceiver", None)) is not None:
            from boxmot.segmentors.propagation.perceiver import _forward_2d

            # A caller can provide an official predictor without going through
            # our loader. Fix its batch layout while leaving shared weights intact.
            perceiver.forward_2d = MethodType(_forward_2d, perceiver)
        self._state: dict[str, Any] | None = None
        self._objects: dict[int, dict[str, dict[int, dict[str, Any]]]] = {}
        self._masks: dict[int, torch.Tensor] = {}
        self._last_observed: dict[int, int] = {}
        self.frame_shape: tuple[int, int] | None = None
        self.last_frame_index = -1
        self._mean = torch.tensor((0.485, 0.456, 0.406), device=self.device, dtype=torch.float32)[:, None, None]
        self._std = torch.tensor((0.229, 0.224, 0.225), device=self.device, dtype=torch.float32)[:, None, None]

    @property
    def predictor(self) -> Any:
        """Return the loaded official model for explicit reuse by another adapter."""
        return self._predictor

    def propagate(
        self,
        frame_index: int,
        frame: np.ndarray,
        active_tracks: Mapping[int, np.ndarray],
        new_tracks: Mapping[int, np.ndarray],
    ) -> dict[int, torch.Tensor]:
        """Return this frame's masks using only confirmed previous-frame tracks."""
        if isinstance(frame_index, bool) or not isinstance(frame_index, Integral):
            raise TypeError("EdgeTAM frame_index must be an integer.")
        if frame_index != self.last_frame_index + 1:
            raise ValueError(
                "EdgeTAM requires consecutive frame indices starting at zero; reset before a new sequence."
            )
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("EdgeTAM frames must be uint8 HxWx3 BGR arrays.")
        shape = tuple(frame.shape[:2])
        if min(shape) < 1 or (self.frame_shape is not None and shape != self.frame_shape):
            raise ValueError("EdgeTAM requires positive, constant frame dimensions; reset before a new sequence.")
        active = _boxes_by_id(active_tracks, name="active_tracks")
        new = _boxes_by_id(new_tracks, name="new_tracks")
        if not set(new).issubset(active):
            raise ValueError("new_tracks must contain only newly confirmed IDs present in active_tracks.")
        frame_index = int(frame_index)
        with inference_context(self.device):
            self._append_frame(frame_index, frame)
            if frame_index:
                prompts = self._select_objects(frame_index, active)
                if prompts:
                    self._seed_previous_frame(frame_index - 1, prompts)
                for track_ids in self._object_batches():
                    self._frame_masks(frame_index, track_ids)
                self._prune_history(frame_index)
        self.last_frame_index = frame_index
        return dict(self._masks)

    def _select_objects(self, frame_index: int, active: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Choose a deterministic bounded mask pool, retrying unseeded visible IDs."""
        visible = sorted(set(active) & self._objects.keys())
        promptable = sorted(
            track_id
            for track_id, box in active.items()
            if track_id not in self._objects and _can_prompt(box, active, prompt_overlap=self.prompt_overlap)
        )
        lost = sorted(
            self._objects.keys() - active.keys(),
            key=lambda track_id: (-self._last_observed[track_id], track_id),
        )
        selected = (visible + promptable + lost)[: self.max_objects]
        self.retain_tracks(selected)
        for track_id in selected:
            if track_id in active:
                self._last_observed[track_id] = frame_index - 1
        return {track_id: active[track_id] for track_id in selected if track_id not in self._objects}

    def _seed_previous_frame(self, frame_index: int, prompts: Mapping[int, np.ndarray]) -> None:
        """Use official singleton prompt/preflight APIs, discarding interactive state."""
        assert self._state is not None
        self._objects.clear()
        for track_id in sorted(self._masks.keys() | prompts.keys()):
            state = self._prompt_state()
            if track_id in prompts:
                self._predictor.add_new_points_or_box(
                    state, frame_index, track_id, box=prompts[track_id].astype(np.float32)
                )
            else:
                self._predictor.add_new_mask(state, frame_index, track_id, mask=self._masks[track_id])
            self._predictor.propagate_in_video_preflight(state)
            current = state["output_dict"]["cond_frame_outputs"][frame_index]
            self._objects[track_id] = {
                "cond_frame_outputs": {frame_index: {name: current[name] for name in _MEMORY_FIELDS}},
                "non_cond_frame_outputs": {},
            }
            # Upstream replaces this mapping on a cache miss. Reuse its result in
            # subsequent objects while releasing the complete interactive state.
            self._state["cached_features"] = state["cached_features"]
            del current, state

    def _prompt_state(self) -> dict[str, Any]:
        """Create only the temporary upstream state required to initialize one ID."""
        assert self._state is not None
        return {
            **self._state,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
            "tracking_has_started": False,
            "frames_already_tracked": {},
        }

    def _object_batches(self) -> Iterator[list[int]]:
        """Group IDs with identical temporal layouts without padding missing memories."""
        groups: dict[tuple[Any, ...], list[int]] = {}
        for track_id, outputs in self._objects.items():
            layout = tuple(
                (kind, tuple((index, tuple(sorted(memory))) for index, memory in sorted(history.items())))
                for kind, history in outputs.items()
            )
            groups.setdefault(layout, []).append(track_id)
        for track_ids in groups.values():
            for offset in range(0, len(track_ids), self.batch_size):
                yield track_ids[offset : offset + self.batch_size]

    def _batch_history(self, track_ids: list[int]) -> dict[str, Any]:
        """Stack compatible objects along the predictor's leading batch dimension."""
        first = self._objects[track_ids[0]]
        if len(track_ids) == 1:
            return first
        outputs: dict[str, Any] = {}
        for kind, history in first.items():
            outputs[kind] = {}
            for index, memory in history.items():
                members = [self._objects[track_id][kind][index] for track_id in track_ids]
                stacked: dict[str, Any] = {}
                for name, value in memory.items():
                    if value is None:
                        stacked[name] = None
                    elif name == "maskmem_pos_enc":
                        positions = []
                        for i, tensor in enumerate(value):
                            tensors = [item[name][i] for item in members]
                            shared = all(
                                other.device == tensor.device
                                and other.dtype == tensor.dtype
                                and other.shape == tensor.shape
                                and other.stride() == tensor.stride()
                                and other.storage_offset() == tensor.storage_offset()
                                and other.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
                                for other in tensors[1:]
                            )
                            positions.append(
                                tensor.expand(len(track_ids), *tensor.shape[1:])
                                if shared
                                else torch.cat(tensors, dim=0)
                            )
                        stacked[name] = positions
                    else:
                        stacked[name] = torch.cat([item[name] for item in members], dim=0)
                outputs[kind][index] = stacked
        return outputs

    @staticmethod
    def _object_memory(current: dict[str, Any], index: int, *, batch_size: int) -> dict[str, Any]:
        """Own each object's memory while sharing cached singleton position constants."""
        if batch_size == 1:
            return {name: current[name] for name in _MEMORY_FIELDS}
        memory: dict[str, Any] = {}
        for name in _MEMORY_FIELDS:
            value = current[name]
            if value is None:
                memory[name] = None
            elif name == "maskmem_pos_enc":
                positions = []
                for tensor in value:
                    position = tensor[index : index + 1]
                    # Upstream broadcasts a cached, single-object constant. Keep
                    # sharing that storage, but own any genuinely batched output.
                    if (
                        tensor.stride(0) != 0
                        or tensor.untyped_storage().nbytes() != position.numel() * position.element_size()
                    ):
                        position = position.clone()
                    positions.append(position)
                memory[name] = positions
            else:
                memory[name] = value[index : index + 1].clone()
        return memory

    def _frame_masks(self, frame_index: int, track_ids: list[int]) -> None:
        """Infer compatible objects and retain their masks on the predictor device."""
        assert self._state is not None and self.frame_shape is not None
        batch_size = len(track_ids)
        current, logits = self._predictor._run_single_frame_inference(
            inference_state=self._state,
            output_dict=self._batch_history(track_ids),
            frame_idx=frame_index,
            batch_size=batch_size,
            is_init_cond_frame=False,
            point_inputs=None,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=True,
        )
        _, logits = self._predictor._get_orig_video_res_output(self._state, logits)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (batch_size, 1, *self.frame_shape):
            raise RuntimeError("EdgeTAM must return one original-resolution mask per object ID.")
        values = (logits[:, 0] > 0).detach()
        for index, track_id in enumerate(track_ids):
            self._masks[track_id] = values[index].clone() if batch_size > 1 else values[index]
            self._objects[track_id]["non_cond_frame_outputs"][frame_index] = self._object_memory(
                current, index, batch_size=batch_size
            )

    def _append_frame(self, frame_index: int, frame: np.ndarray) -> None:
        """Match official decoded-image preprocessing while retaining two frames."""
        size = self._predictor.image_size
        rgb = np.array(Image.fromarray(frame[..., ::-1]).resize((size, size)))
        # A byte lookup preserves reference float64 division then float32 rounding
        # without allocating a full-image float64 intermediate on the CPU.
        tensor = torch.from_numpy(_UINT8_TO_FLOAT32[rgb]).permute(2, 0, 1).to(self.device)
        tensor = (tensor - self._mean) / self._std
        if self._state is None:
            self.frame_shape = tuple(frame.shape[:2])
            self._state = {
                "images": {},
                "num_frames": 0,
                "offload_video_to_cpu": False,
                "offload_state_to_cpu": False,
                "video_height": self.frame_shape[0],
                "video_width": self.frame_shape[1],
                "device": self.device,
                "storage_device": self.device,
                "cached_features": {},
                "constants": {},
            }
        self._state["images"][frame_index] = tensor
        self._state["num_frames"] = frame_index + 1
        for index in tuple(self._state["images"]):
            if index < frame_index - 1:
                del self._state["images"][index]
        # Empty frames do not run the image encoder. Its cached image/features
        # must still expire with the frame window after all objects retire.
        for index in tuple(self._state["cached_features"]):
            if index not in self._state["images"]:
                del self._state["cached_features"][index]

    def _prune_history(self, frame_index: int) -> None:
        """Keep six recent spatial memories and fifteen pointers with official defaults."""
        spatial_start = frame_index - self._predictor.num_maskmem + 2
        pointer_start = frame_index - self._predictor.max_obj_ptrs_in_encoder + 2
        for outputs in self._objects.values():
            recent = outputs["non_cond_frame_outputs"]
            for index in tuple(recent):
                if index < pointer_start:
                    del recent[index]
                elif index < spatial_start:
                    recent[index] = {"obj_ptr": recent[index]["obj_ptr"]}

    def retain_tracks(self, track_ids: Collection[int]) -> None:
        """Release retired identities and masks without copying survivors."""
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in track_ids):
            raise ValueError("Retained track IDs must be nonnegative integers.")
        retained = set(track_ids)
        for container in (self._objects, self._masks, self._last_observed):
            for track_id in container.keys() - retained:
                del container[track_id]

    def reset(self) -> None:
        """Release stream state and reuse the loaded model for the next sequence."""
        self._state = None
        self._objects.clear()
        self._masks.clear()
        self._last_observed.clear()
        self.frame_shape = None
        self.last_frame_index = -1
