"""Live prediction time follows receipt or explicit PTS, never advertised FPS."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from boxmot.engine.streaming import capture
from boxmot.engine.tracking.timestamps import LiveTimestamps, SourceTimestamps


@pytest.mark.parametrize("source,expected", [("capture", [0, 0.2, 0.6]), ("media", [0, 0.04, 0.08])])
def test_capture_clock_is_explicit_and_does_not_consult_advertised_fps(monkeypatch, source, expected):
    """An MJPEG backend's fake 25 FPS/PTS cannot compress real 5 FPS capture."""
    frames = []

    class Decoder:
        index = -1
        released = False

        def isOpened(self):
            return True

        def read(self):
            self.index += 1
            return self.index < 3, np.zeros((2, 2, 3), np.uint8)

        def get(self, key):
            assert key == cv2.CAP_PROP_POS_MSEC, "Live timing must never use nominal FPS."
            return self.index * 40

        def release(self):
            self.released = True

    decoder = Decoder()
    clock = iter((100, 100.2, 100.6))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args: decoder)
    monkeypatch.setattr(capture.os, "dup2", lambda *_args: None)
    monkeypatch.setattr(capture, "time", SimpleNamespace(monotonic=lambda: next(clock), time=lambda: 500))
    capture._capture("http://fixture", 10, SimpleNamespace(put=frames.append), threading.Event(), source)
    assert decoder.released
    assert [frame.timestamp_s for frame in frames] == pytest.approx(expected)
    assert [frame.captured_at for frame in frames] == [100, 100.2, 100.6]


def test_skipped_frames_keep_elapsed_gap_and_reconnect_starts_new_timeline():
    timestamps = LiveTimestamps()
    decoded = [timestamps.resolve(captured_at=value) for value in (10, 10.2, 10.4, 10.9)]
    # A latest-frame consumer sees frame zero and frame three, not three nominal steps.
    assert decoded[3] - decoded[0] == pytest.approx(0.9)
    replacement = LiveTimestamps()
    assert replacement.resolve(captured_at=23) == 0
    assert replacement.resolve(captured_at=23.3) == pytest.approx(0.3)


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -1, 1000, 900])
def test_trusted_media_fails_on_missing_or_nonincreasing_pts_without_switching_clocks(invalid):
    timestamps = LiveTimestamps("media")
    assert timestamps.resolve(captured_at=10, position_ms=1000) == 0
    with pytest.raises(ValueError, match="Live media timestamps"):
        timestamps.resolve(captured_at=10.2, position_ms=invalid)


def test_media_time_preserves_irregular_intervals_independent_of_receipt_delay():
    timestamps = LiveTimestamps("media")
    assert timestamps.resolve(captured_at=100, position_ms=9000) == 0
    assert timestamps.resolve(captured_at=100.001, position_ms=9400) == pytest.approx(0.4)


def test_file_timestamp_fallback_remains_media_based():
    timestamps = SourceTimestamps(25)
    assert timestamps.resolve(0, frame_index=0) == 0
    assert timestamps.resolve(0, frame_index=5) == pytest.approx(0.2)
    assert timestamps.resolve(500, frame_index=6) == pytest.approx(0.5)
