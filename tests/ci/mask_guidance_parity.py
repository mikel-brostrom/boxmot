"""Compare streaming masks with the unpruned official EdgeTAM video predictor.

Run explicitly with the optional dependency and a real checkpoint:
``python -m tests.ci.mask_guidance_parity --checkpoint /path/to/edgetam.pt``.
The offline oracle intentionally preloads JPEGs and keeps unpruned state for
each object; the production backend receives only each newly decoded frame.
Run the same fixture with ``--device cpu`` and the target accelerator to check
exact streaming/offline parity independently on each device.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from boxmot.segmentors.propagation.edgetam import EdgeTAMMaskPropagator
from boxmot.segmentors.propagation.model import inference_context
from tests.ci.mask_guidance_smoke import _assert_bounded_history


def _object_boxes(index: int) -> tuple[np.ndarray, np.ndarray]:
    """Move both objects while preserving their sizes and spatial separation."""
    first_phase, second_phase = index % 42, index % 40
    first_x, first_y = 8 + min(first_phase, 42 - first_phase) // 3, 8 + (index // 4) % 3
    second_x, second_y = 72 - min(second_phase, 40 - second_phase) // 4, 16 + (index // 3) % 2
    return (
        np.array([first_x, first_y, first_x + 32, first_y + 72], dtype=np.float32),
        np.array([second_x, second_y, second_x + 40, second_y + 56], dtype=np.float32),
    )


def _parity_frame(index: int) -> np.ndarray:
    """Generate a moving textured scene that exercises temporal mask memory."""
    y, x = np.indices((96, 128))
    shifted_x, shifted_y = x + index, y + index // 4
    image = np.stack(
        (
            18 + (3 * shifted_x + shifted_y) % 23,
            24 + (shifted_x + 2 * shifted_y) % 19,
            20 + (2 * shifted_x + shifted_y) % 17,
        ),
        axis=-1,
    ).astype(np.uint8)
    for number, box in enumerate(_object_boxes(index)):
        x1, y1, x2, y2 = box.astype(int)
        local_y, local_x = np.indices((y2 - y1, x2 - x1))
        texture = ((local_x // 4 + local_y // 6) % 2) * 24
        base = np.array([40, 180, 225] if number == 0 else [195, 105, 40], dtype=np.uint8)
        image[y1:y2, x1:x2] = base + texture[..., None]
    return image


class _OfflinePropagation:
    """Run official unpruned singleton iterators using one shared loaded model.

    Separate official inference states provide an independent singleton oracle
    for production object batching without changing model operations,
    preprocessing, or output postprocessing.
    """

    def __init__(self, predictor: Any, source: Path) -> None:
        self.predictor = predictor
        self.source = source
        self.state = predictor.init_state(video_path=str(source))
        self.states: dict[int, dict[str, Any]] = {}
        self.iterators: dict[int, Generator[tuple[int, list[int], torch.Tensor], None, None]] = {}
        self.masks: dict[int, np.ndarray] = {}

    def propagate(self, frame_index: int, prompts: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Propagate unpruned history, retaining every prompted identity."""
        if frame_index == 0:
            return {}
        if prompts:
            self._close_iterators()
            for track_id, mask in self.masks.items():
                state = self.states[track_id]
                self.predictor.reset_state(state)
                self.predictor.add_new_mask(
                    state, frame_idx=frame_index - 1, obj_id=track_id, mask=mask.astype(np.float32)
                )
            for track_id, box in prompts.items():
                assert track_id not in self.states, f"Identity {track_id} was already prompted"
                state = (
                    self.state if not self.states else self.predictor.init_state(video_path=str(self.source))
                )
                self.states[track_id] = state
                self.predictor.add_new_points_or_box(
                    state, frame_idx=frame_index - 1, obj_id=track_id, box=box
                )
            for track_id, state in self.states.items():
                iterator = self.predictor.propagate_in_video(state, start_frame_idx=frame_index - 1)
                self.iterators[track_id] = iterator
                conditioning_index, ids, _ = next(iterator)
                assert conditioning_index == frame_index - 1
                assert ids == [track_id], "The official oracle must use singleton states"
        assert self.iterators
        masks = {}
        for track_id, iterator in self.iterators.items():
            actual_index, ids, logits = next(iterator)
            assert actual_index == frame_index
            assert ids == [track_id]
            assert logits.shape == (1, 1, self.state["video_height"], self.state["video_width"])
            masks[track_id] = (logits[0, 0] > 0).detach().cpu().numpy().copy()
        self.masks = masks
        return self.masks

    def _close_iterators(self) -> None:
        """Release suspended per-object generators before reseeding their states."""
        for iterator in self.iterators.values():
            iterator.close()
        self.iterators.clear()

    def close(self) -> None:
        """Release official iterators and full-video states without unloading the model."""
        self._close_iterators()
        for state in self.states.values():
            state.clear()
        self.states.clear()
        self.state.clear()
        self.masks.clear()


def main() -> None:
    """Check pruning, late reseeding, and retired-object survivor parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=22, help="At least 22 frames to cross retention and reseeding.")
    args = parser.parse_args()
    if args.frames < 22:
        parser.error("--frames must be at least 22")
    torch.set_num_threads(2)
    streaming = EdgeTAMMaskPropagator(args.checkpoint, device=args.device)
    with tempfile.TemporaryDirectory(prefix="boxmot-edgetam-parity-") as directory:
        source = Path(directory)
        for index in range(args.frames):
            Image.fromarray(_parity_frame(index)[..., ::-1]).save(source / f"{index:05d}.jpg", quality=95)
        with inference_context(streaming.device):
            offline = _OfflinePropagation(streaming.predictor, source)
        try:
            for index in range(args.frames):
                with Image.open(source / f"{index:05d}.jpg") as image:
                    # Both paths use PIL JPEG decoding, so this checks the memory
                    # adaptation independently of differences between image codecs.
                    frame = np.asarray(image.convert("RGB"))[..., ::-1].copy()
                first_box, second_box = _object_boxes(max(0, index - 1))
                active = {0: first_box}
                prompts = {0: first_box} if index == 1 else {}
                if index >= 18:
                    active[1] = second_box
                if index == 18:
                    prompts[1] = second_box
                if index >= 20:
                    streaming.retain_tracks({1})
                    active.pop(0)
                actual = streaming.propagate(index, frame, active, prompts if index == 18 else {})
                # The oracle uses unchanged public predictor APIs and keeps
                # every object's history, including retired identity zero.
                with inference_context(streaming.device):
                    expected = offline.propagate(index, prompts)
                _assert_bounded_history(streaming)
                torch.testing.assert_close(
                    streaming._state["images"][index], offline.state["images"][index], rtol=0, atol=0
                )
                expected_ids = set(expected) if index < 20 else {1}
                assert set(actual) == expected_ids
                for track_id in actual:
                    assert expected[track_id].any(), f"Offline mask is empty at frame {index}, identity {track_id}"
                    np.testing.assert_array_equal(
                        actual[track_id].detach().cpu().numpy(),
                        expected[track_id],
                        err_msg=f"frame {index}, identity {track_id}",
                    )
                if index == 17:
                    assert len(offline.states[0]["output_dict"]["non_cond_frame_outputs"]) > 15
                    recent = streaming._objects[0]["non_cond_frame_outputs"]
                    assert len(recent) == 15
                    assert sum("maskmem_features" in output for output in recent.values()) == 6
        finally:
            offline.close()
            streaming.reset()
    print(
        f"Streaming/offline EdgeTAM parity passed on {streaming.device} over {args.frames} frames: "
        "identical normalized pixels and masks, "
        "bounded history, late identity reseeding, and unchanged survivor masks after retirement."
    )


if __name__ == "__main__":
    main()
