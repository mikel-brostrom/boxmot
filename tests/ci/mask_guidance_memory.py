"""Measure real mask-guided replay memory in the production spawned workers.

Run with an existing perception build, for example::

    uv run --no-sync python -m tests.ci.mask_guidance_memory --build BUILD \
      --checkpoint models/edgetam.pt --device mps --workers 1 --output runs/mask-memory

Writes small per-sequence JSONL samples; never retains frames or model outputs.
RSS, live tensor allocations, and driver allocations are separate measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import weakref
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import partial
from importlib import import_module
from itertools import islice
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import psutil
import torch

from boxmot.engine.eval import replay
from boxmot.segmentors.propagation.weights import resolve_edgetam_checkpoint
from boxmot.trackers import TrackerSpec
from tests.ci.mask_guidance_smoke import _assert_bounded_history, _synchronize

_REPLAY_TASK = replay._replay_sequence_task


class _LimitedInputs:
    """Select an initial diagnostic frame range without changing the cached build."""

    def __init__(self, dataset: Any, limit: int) -> None:
        self.dataset, self.limit = dataset, min(len(dataset), limit)
        self.manifest = dataset.manifest

    def __len__(self) -> int:
        return self.limit

    def __iter__(self):
        return islice(iter(self.dataset), self.limit)

    def close(self) -> None:
        close = getattr(self.dataset, "close", None)
        if callable(close):
            close()


@contextmanager
def _trace_stages(path: Path, device: torch.device, frame_number: Any, *, detail: str):
    """Measure individual reference operations inside this diagnostic worker only."""
    targets = [
        ("sam2.sam2_video_predictor", "SAM2VideoPredictor", name)
        for name in (
            "add_new_points_or_box", "add_new_mask", "_consolidate_temp_output_across_obj",
            "_get_orig_video_res_output", "_run_single_frame_inference", "_run_memory_encoder",
        )
    ] + [
        ("sam2.modeling.sam2_base", "SAM2Base", "_encode_new_memory"),
        ("sam2.modeling.memory_encoder", "MemoryEncoder", "forward"),
    ]
    if detail == "all":
        targets.extend([
            ("sam2.modeling.memory_attention", "MemoryAttention", "forward"),
            ("sam2.modeling.sam.mask_decoder", "MaskDecoder", "forward"),
            ("torch.nn", "functional", "scaled_dot_product_attention"),
            ("torch.nn", "functional", "interpolate"),
        ])

    def wrap(original, label):
        def measured(*args, **kwargs):
            shapes = [list(value.shape) for value in args if isinstance(value, torch.Tensor)]
            _sample(path, stage=f"enter:{label}:{shapes}", frame=frame_number(), device=device)
            result = original(*args, **kwargs)
            _sample(path, stage=f"exit:{label}:{shapes}", frame=frame_number(), device=device)
            return result
        return measured

    with ExitStack() as stack:
        for module, owner_name, name in targets:
            owner = getattr(import_module(module), owner_name)
            stack.enter_context(patch.object(owner, name, wrap(getattr(owner, name), f"{owner_name}.{name}")))
        yield


def _state_storage_bytes(value: Any) -> int:
    """Count distinct backing storages, including the full storage behind views."""
    seen: set[int] = set()
    storages: dict[tuple[str, int], int] = {}
    arrays: dict[int, int] = {}

    def visit(item: Any) -> None:
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, np.ndarray):
            owner = item
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            arrays[id(owner)] = owner.nbytes
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return sum(storages.values()) + sum(arrays.values())


def _mask_counts(tracker: Any) -> dict[str, int]:
    """Assert every temporal collection is bounded and record current object load."""
    guidance = tracker._mask_guidance
    propagator = guidance._propagator
    state = propagator._state
    _assert_bounded_history(propagator)
    ids = set(propagator._objects)
    retained = {track.id for track in tracker.active_tracks + tracker.lost_stracks}
    assert ids.issubset(retained)
    assert set(guidance._masks).issubset(ids)
    return {
        "objects": len(ids),
        "active_tracks": len(tracker.active_tracks),
        "lost_tracks": len(tracker.lost_stracks),
        "removed_tracks": len(tracker.removed_stracks),
        "images": len(state["images"]),
        "cached_features": len(state["cached_features"]),
        "conditioning_frames": sum(len(o["cond_frame_outputs"]) for o in propagator._objects.values()),
        "recent_frames": sum(len(o["non_cond_frame_outputs"]) for o in propagator._objects.values()),
    }


def _sample(path: Path, *, stage: str, frame: int, device: torch.device, tracker: Any = None) -> None:
    """Record synchronized allocator and process measurements without flushing caches."""
    _synchronize(device)
    record = {
        "pid": os.getpid(), "time_s": time.time(), "stage": stage, "frame": frame,
        "rss_bytes": psutil.Process().memory_info().rss,
    }
    if device.type == "mps":
        record["live_device_bytes"] = torch.mps.current_allocated_memory()
        record["driver_device_bytes"] = torch.mps.driver_allocated_memory()
    elif device.type == "cuda":
        record["live_device_bytes"] = torch.cuda.memory_allocated(device)
        record["driver_device_bytes"] = torch.cuda.memory_reserved(device)
    if tracker is not None:
        record.update(_mask_counts(tracker))
        propagator = tracker._mask_guidance._propagator
        record["mask_state_storage_bytes"] = _state_storage_bytes(
            (propagator._state, propagator._objects, propagator._masks)
        )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _measured_task(
    task: replay._SequenceReplayTask, *, flush_cache: bool = False,
    max_frames: int | None = None, trace_stages: str | None = None,
) -> replay._SequenceReplayResult:
    """Instrument one otherwise unchanged production sequence task inside its worker."""
    device = torch.device(task.mask_guidance_device)
    path = Path(task.output_path).parent.parent / f"{task.sequence_id}.memory.jsonl"
    factory = replay.create_tracker
    tracker_ref = None
    model_refs = []
    completed = 0
    if max_frames is not None:
        task = replace(task, frame_total=min(task.frame_total, max_frames))

    def create(*args: Any, **kwargs: Any) -> Any:
        nonlocal tracker_ref
        tracker = factory(*args, **kwargs)
        tracker_ref = weakref.ref(tracker)
        return tracker

    def observe(_frame: replay.ReplayFrame) -> None:
        nonlocal completed
        completed += 1
        tracker = tracker_ref()
        if completed == 1:
            propagator = tracker._mask_guidance._propagator
            model_refs.extend((weakref.ref(propagator), weakref.ref(propagator._predictor)))
        _mask_counts(tracker)
        if completed in (1, 2, 4, 8, 16, task.frame_total) or completed % 25 == 0:
            _sample(path, stage="frame", frame=completed, device=device, tracker=tracker)
        if flush_cache and device.type == "mps":
            torch.mps.empty_cache()
            if completed in (1, 2, 4, 8, 16, task.frame_total) or completed % 25 == 0:
                _sample(path, stage="after_empty_cache", frame=completed, device=device, tracker=tracker)

    replay.create_tracker = create
    _sample(path, stage="before_sequence", frame=0, device=device)
    try:
        with ExitStack() as contexts:
            if max_frames is not None:
                worker_inputs = replay._worker_inputs
                contexts.enter_context(patch.object(
                    replay, "_worker_inputs",
                    lambda *args, **kwargs: _LimitedInputs(worker_inputs(*args, **kwargs), max_frames),
                ))
            if trace_stages:
                contexts.enter_context(_trace_stages(
                    path.with_suffix(".stages.jsonl"), device, lambda: completed + 1, detail=trace_stages,
                ))
            result = _REPLAY_TASK(task, frame_callback=observe)
    finally:
        replay.create_tracker = factory
    _sample(path, stage="after_sequence", frame=completed, device=device)
    assert tracker_ref() is None, "Completed replay retained its tracker"
    assert all(reference() is None for reference in model_refs), "Completed replay retained its mask model"
    return result


def main() -> None:
    """Replay real sequences with one worker by default and save allocator measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=96)
    parser.add_argument("--split", default="ablation")
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--max-frames", type=int, help="Measure only the first N selected frames of each sequence.")
    parser.add_argument(
        "--trace-stages", nargs="?", const="layers", choices=("layers", "all"),
        help="Sample model boundaries; use 'all' to include individual attention and interpolation operations.",
    )
    parser.add_argument(
        "--flush-cache", action="store_true", help="Diagnostic: release unused MPS buffers after each frame."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty to keep memory measurements from different runs separate")
    checkpoint = resolve_edgetam_checkpoint(args.checkpoint)
    replay._replay_sequence_task = partial(
        _measured_task, flush_cache=args.flush_cache, max_frames=args.max_frames, trace_stages=args.trace_stages,
    )

    def progress(event: replay.ReplayProgressEvent) -> None:
        if event.status == "completed" or (event.completed and event.completed % 50 == 0):
            print(f"{event.sequence_id}: {event.completed}/{event.total}", flush=True)

    result = replay.replay_build(
        args.build, TrackerSpec(name="bytetrack"), split=args.split, output_dir=args.output,
        workers=args.workers, sequence_ids=None if args.sequence is None else tuple(args.sequence),
        mask_guidance_weights=checkpoint, mask_guidance_device=args.device, progress_callback=progress,
        mask_guidance_max_objects=args.max_objects,
    )
    print(
        f"Mask-history and tracker-release checks passed over {result.frames} frames; "
        f"inspect driver-memory samples in {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
