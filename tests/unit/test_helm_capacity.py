"""Contracts for independent capacity evidence and the real paced network fixture."""

from __future__ import annotations

import copy
import threading
import urllib.request
from types import SimpleNamespace

import pytest

from tests.e2e.fixtures import camera as camera_fixture
from tests.e2e.fixtures.camera import camera_server
from tests.e2e.helm_capacity import CapacityTest, summarize


def samples() -> list[dict]:
    state = {
        "pod_uid": "worker-1", "restarts": 0, "frames": 10, "dropped_frames": 2,
        "completed_events": 1, "pending_batches": 3, "queued_crop_bytes": 1024,
        "oldest_batch_age_s": 5, "mean_frame_ms": 200, "reconnects": 0,
        "observed_at_s": 1, "attributes_observed_at_s": 0.5,
    }
    final = {
        **state, "frames": 40, "dropped_frames": 12, "completed_events": 3, "oldest_batch_age_s": 7,
        "observed_at_s": 11, "attributes_observed_at_s": 10.5,
    }
    return [{"elapsed_s": 1, "cameras": {"camera-01": state}},
            {"elapsed_s": 11, "cameras": {"camera-01": final}}]


def test_capacity_rates_use_deltas_after_warmup() -> None:
    result = summarize(samples())["camera-01"]
    assert result["processed_fps"] == 3
    assert result["attribute_events_per_second"] == 0.2
    assert result["dropped_fraction"] == 0.25
    assert result["peak_oldest_batch_age_s"] == 7
    assert CapacityTest.promotable is False


@pytest.mark.parametrize("field,value", [("pod_uid", "replacement"), ("restarts", 1), ("frames", 0), ("reconnects", 1)])
def test_capacity_rejects_restarts_and_counter_resets(field: str, value: object) -> None:
    rows = copy.deepcopy(samples())
    rows[-1]["cameras"]["camera-01"][field] = value
    with pytest.raises(AssertionError):
        summarize(rows)


def test_capacity_rates_use_each_observation_time_despite_slow_later_polls() -> None:
    rows = samples()
    rows[-1]["elapsed_s"] = 51  # Later camera probes took another 40 seconds.
    rows[-1]["cameras"]["camera-01"]["attributes_observed_at_s"] = 20.5

    result = summarize(rows)["camera-01"]

    assert result["processed_fps"] == 3
    assert result["attribute_events_per_second"] == 0.1


def test_capacity_rejects_mid_window_counter_reset_even_when_final_count_recovers() -> None:
    rows = samples()
    middle = copy.deepcopy(rows[0])
    middle["cameras"]["camera-01"].update(frames=0, observed_at_s=5, attributes_observed_at_s=4.5)
    rows.insert(1, middle)

    with pytest.raises(AssertionError, match="counters regressed"):
        summarize(rows)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_fixture_requires_finite_positive_fps(value: float) -> None:
    with pytest.raises(ValueError, match="finite positive FPS"):
        camera_server([b"jpeg"], value, ("127.0.0.1", 0))


def test_fixture_does_not_burst_after_socket_backpressure(monkeypatch) -> None:
    clock = 0.0
    pauses = []

    def pause(seconds: float) -> None:
        nonlocal clock
        pauses.append(seconds)
        clock += seconds

    class SlowSocket:
        writes = 0

        def write(self, _payload: bytes) -> None:
            self.writes += 1
            if self.writes > 3:
                raise BrokenPipeError

        def flush(self) -> None:
            nonlocal clock
            clock += 1.0  # The client blocks each write for five frame intervals.

    monkeypatch.setattr(camera_fixture, "time", SimpleNamespace(monotonic=lambda: clock, sleep=pause))
    monkeypatch.setattr(camera_fixture, "ThreadingHTTPServer", lambda address, handler: handler)
    handler_type = camera_server([b"jpeg"], 5, ("127.0.0.1", 0))
    handler = object.__new__(handler_type)
    handler.path = "/camera"
    handler.wfile = SlowSocket()
    handler.send_response = handler.send_header = handler.end_headers = lambda *args: None

    handler.do_GET()

    assert pauses == pytest.approx([0.2, 0.2, 0.2])


def test_network_fixture_loops_distinct_frames() -> None:
    server = camera_server([b"first-jpeg", b"second-jpeg"], 100, ("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/camera-01", timeout=5) as response:
            frames = []
            for _ in range(3):
                assert response.readline() == b"--frame\r\n"
                assert response.readline() == b"Content-Type: image/jpeg\r\n"
                length = int(response.readline().split(b":")[1])
                assert response.readline() == b"\r\n"
                frames.append(response.read(length))
                assert response.readline() == b"\r\n"
            assert set(frames) == {b"first-jpeg", b"second-jpeg"}
            assert frames[0] == frames[2]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
