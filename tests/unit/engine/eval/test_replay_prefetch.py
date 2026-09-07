from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from boxmot.datasets import DatasetSample
from boxmot.engine.eval import replay as replay_module
from boxmot.structures import Boxes, Detections, Tracks
from boxmot.trackers import TrackerRequirements, TrackerSpec


def _sample(index: int) -> DatasetSample:
    """Build a tiny canonical sample without image decoding or model inference."""
    sample_id = f"sequence:{index}"
    return DatasetSample(
        sample_id=sample_id,
        split="validation",
        sequence_id="sequence",
        frame_index=index,
        timestamp_s=None,
        image_size=(2, 2),
        image_ref=None,
        frame=None,
        detections=Detections(
            geometry=Boxes(torch.empty((0, 4), dtype=torch.float32)),
            scores=torch.empty(0, dtype=torch.float32),
            class_ids=torch.empty(0, dtype=torch.int64),
            sample_id=sample_id,
        ),
    )


class _ObservedIterator:
    """Observe generator ownership and reject close while a read is active."""

    def __init__(self, values: Iterator[DatasetSample]) -> None:
        self.values = values
        self.produced: list[DatasetSample] = []
        self.producer_threads: set[int] = set()
        self.second_ready = threading.Event()
        self.active = False
        self.close_count = 0

    def __iter__(self) -> _ObservedIterator:
        return self

    def __next__(self) -> DatasetSample:
        self.producer_threads.add(threading.get_ident())
        self.active = True
        try:
            sample = next(self.values)
            self.produced.append(sample)
            if len(self.produced) == 2:
                self.second_ready.set()
            return sample
        finally:
            self.active = False

    def close(self) -> None:
        assert not self.active, "dataset closed while its producer was still reading"
        self.close_count += 1
        close = getattr(self.values, "close", None)
        if close is not None:
            close()


def test_prefetch_preserves_order_with_only_one_sample_ahead() -> None:
    expected = [_sample(index) for index in range(4)]
    source = _ObservedIterator(iter(expected))
    consumer_thread = threading.get_ident()

    with replay_module._prefetch_samples(source) as samples:
        assert next(samples) is expected[0]
        assert source.second_ready.wait(timeout=5)
        # The second sample can load during tracking of the first. A third
        # read requires the consumer to advance, regardless of producer speed.
        assert source.produced == expected[:2]
        assert list(samples) == expected[1:]

    assert len(source.producer_threads) == 1
    assert consumer_thread not in source.producer_threads
    assert source.close_count == 1


def test_prefetch_closes_an_empty_iterator() -> None:
    source = _ObservedIterator(iter(()))

    with replay_module._prefetch_samples(source) as samples:
        assert list(samples) == []

    assert source.close_count == 1


def test_prefetch_propagates_dataset_read_errors_and_closes() -> None:
    first = _sample(0)
    failure = OSError("cached image could not be decoded")

    def failing_samples() -> Iterator[DatasetSample]:
        yield first
        raise failure

    source = _ObservedIterator(failing_samples())
    with pytest.raises(OSError) as caught:
        with replay_module._prefetch_samples(source) as samples:
            assert next(samples) is first
            next(samples)

    assert caught.value is failure
    assert source.close_count == 1


def test_prefetch_waits_for_active_read_before_closing_after_consumer_failure() -> None:
    first, second = _sample(0), _sample(1)
    reading_second = threading.Event()
    release_reader = threading.Event()
    consumer_exiting = threading.Event()
    failure = RuntimeError("tracker failed")
    observed: list[BaseException] = []

    def blocked_samples() -> Iterator[DatasetSample]:
        yield first
        reading_second.set()
        assert release_reader.wait(timeout=5)
        yield second

    source = _ObservedIterator(blocked_samples())

    def consume() -> None:
        try:
            with replay_module._prefetch_samples(source) as samples:
                assert next(samples) is first
                assert reading_second.wait(timeout=5)
                consumer_exiting.set()
                raise failure
        except BaseException as exc:
            observed.append(exc)

    consumer = threading.Thread(target=consume)
    consumer.start()
    try:
        assert consumer_exiting.wait(timeout=5)
        assert source.active
        assert source.close_count == 0
        assert consumer.is_alive()
    finally:
        release_reader.set()
        consumer.join(timeout=5)

    assert not consumer.is_alive()
    assert observed == [failure]
    assert source.close_count == 1


@pytest.mark.parametrize("needs_pixels", [False, True])
def test_sequence_replay_prefetch_keeps_tracker_on_worker_thread(tmp_path, monkeypatch, needs_pixels: bool) -> None:
    source = _ObservedIterator(iter([_sample(0), _sample(1)]))
    consumer_thread = threading.get_ident()
    tracker_threads: list[int] = []

    class _Dataset:
        manifest = SimpleNamespace()

        def __len__(self) -> int:
            return 2

        def __iter__(self) -> _ObservedIterator:
            return source

    class _Tracker:
        supports_obb = False
        requirements = TrackerRequirements(frame=needs_pixels)

        def update(self, detections: Detections, frame=None) -> Tracks:
            tracker_threads.append(threading.get_ident())
            return Tracks(
                geometry=detections.geometry,
                track_ids=torch.empty(0, dtype=torch.int64),
                scores=detections.scores,
                class_ids=detections.class_ids,
                detection_indices=torch.empty(0, dtype=torch.int64),
                sample_id=detections.sample_id,
            )

        def reset(self) -> None:
            return None

    @contextmanager
    def owned_tracker(_spec):
        yield _Tracker()

    monkeypatch.setattr(replay_module, "_owned_tracker", owned_tracker)
    monkeypatch.setattr(
        replay_module.CachedVisionDataset,
        "_stream_sequence",
        classmethod(lambda *args, **kwargs: _Dataset()),
    )
    monkeypatch.setattr(replay_module, "validate_build_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay_module, "_WORKER_PROGRESS_QUEUE", None)

    result = replay_module._replay_sequence_task(
        replay_module._SequenceReplayTask(
            build=str(tmp_path / "build"),
            tracker_spec=TrackerSpec(name="bytetrack"),
            split="validation",
            sequence_id="sequence",
            frame_total=2,
            output_path=str(tmp_path / "sequence.txt"),
            ordinal=0,
        )
    )

    assert result.frames == 2
    assert tracker_threads == [consumer_thread, consumer_thread]
    assert (consumer_thread not in source.producer_threads) == needs_pixels
    assert source.close_count == int(needs_pixels)
