"""Process isolation and fixed-capacity camera capture contracts."""

import multiprocessing as mp
import time

import numpy as np
import pytest

from boxmot.engine.streaming.capture import CapturedFrame, CaptureDisconnected, FrameSlot, NetworkCapture


def produce(_url, _timeout, slot, stop, _timestamp_source):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    index = 0
    while not stop.wait(0.002):
        slot.put(CapturedFrame(image, index, index / 30, time.monotonic(), time.time()))
        index += 1


def blocked(_url, _timeout, slot, stop, _timestamp_source):
    # Simulate a decoder killed while holding the shared slot lock.
    slot.lock.acquire()
    while True:
        time.sleep(1)


def test_slot_replaces_pending_frames_without_growing_and_keeps_capture_time():
    slot = FrameSlot(mp.get_context("spawn"), 16 * 16 * 3)
    for index in range(10000):
        slot.put(CapturedFrame(np.full((16, 16, 3), index % 256, np.uint8), index, index / 30, 123.0, 456.0))
    frame = slot.get(-1)
    assert frame.index == 9999 and frame.timestamp_s == 9999 / 30
    assert frame.captured_at == 123 and frame.captured_wall_time == 456
    assert frame.image[0, 0, 0] == 9999 % 256
    assert slot.get(9999) is None
    assert len(slot.pixels) == 768
    with pytest.raises(ValueError, match="capacity"):
        slot.put(CapturedFrame(np.zeros((32, 32, 3), np.uint8), 10000, None, 0))


def test_slow_consumer_gets_recent_frames_and_drops_are_observable():
    capture = NetworkCapture("fixture", target=produce, max_frame_bytes=768)
    try:
        deadline = time.monotonic() + 10
        first = None
        while first is None and time.monotonic() < deadline:
            first = capture.next()
        assert first is not None
        time.sleep(0.1)
        latest = capture.next()
        assert latest.index > first.index + 1
        assert latest.timestamp_s > first.timestamp_s
        assert time.monotonic() - latest.captured_at < 0.1
    finally:
        capture.close()
        capture.close()


def test_stuck_capture_has_bounded_wait_and_can_be_replaced():
    capture = NetworkCapture("fixture", timeout_s=0.1, target=blocked, max_frame_bytes=768)
    started = time.monotonic()
    try:
        with pytest.raises(CaptureDisconnected):
            while True:
                capture.next()
    finally:
        capture.close()
    assert time.monotonic() - started < 5
    replacement = NetworkCapture("fixture", target=produce, max_frame_bytes=768)
    try:
        deadline = time.monotonic() + 10
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = replacement.next()
        assert frame is not None
    finally:
        replacement.close()
