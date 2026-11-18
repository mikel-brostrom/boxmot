"""Readiness, bounded profiling, authored stream settings and CLI entrypoints."""

import threading
import time
from collections import deque

import pytest
from click.testing import CliRunner

from boxmot.engine.commands.stream import maintain, stream
from boxmot.engine.config.streams import StreamConfig, load_stream_config
from boxmot.engine.streaming.health import StreamHealth
from boxmot.engine.tracking.profiling import RuntimeProfiler
from boxmot.utils.config import ConfigurationError


def test_readiness_separates_outages_from_supervisor_health():
    state = StreamHealth()
    assert state.snapshot()["live"] and not state.snapshot()["ready"]
    state.owned = state.connected = True
    state.last_frame = time.monotonic()
    assert state.snapshot()["ready"]
    state.connected = False
    assert state.snapshot()["live"] and not state.snapshot()["ready"]
    state.connected = True
    state.ownership_lost = threading.Event()
    state.ownership_lost.set()
    assert not state.snapshot()["ready"]
    state.heartbeat -= 61
    assert not state.snapshot()["live"]


def test_long_stream_releases_profiler_samples_with_fixed_timing_window():
    profiler = RuntimeProfiler()
    timings = deque(maxlen=300)
    for index in range(30000):
        sample = str(index)
        profiler.begin_pipeline(sample)
        profiler.end_pipeline(sample)
        profiler.finish_sample(sample, 1.0)
        timings.append(profiler.release_sample(sample))
    assert len(timings) == 300
    assert not profiler._samples
    assert profiler.totals_ms()["overall"] == 30000
    profiler.release_sample("29999")
    assert profiler.totals_ms()["overall"] == 30000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"camera_id": "../camera"},
        {"camera_id": "cam", "max_pending_jobs": 0},
        {"camera_id": "cam", "capture_timeout_s": float("nan")},
        {"camera_id": "cam", "tracking": {"show": True}},
    ],
)
def test_invalid_stream_config(kwargs):
    with pytest.raises(ConfigurationError):
        StreamConfig(**kwargs)


def test_config_uses_engine_defaults_and_cli_help(tmp_path):
    path = tmp_path / "camera.yaml"
    path.write_text("camera_id: front-door\ntracking:\n  tracker: bytetrack\n")
    config = load_stream_config(path)
    assert config.max_pending_jobs == 1000 and config.max_crop_bytes == 256 * 1024 * 1024
    args = config.tracking_args()
    assert args.device == "cpu" and not args.show and args.tracker == "bytetrack"
    assert CliRunner().invoke(stream, ["--help"]).exit_code == 0
    result = CliRunner().invoke(maintain, ["--camera-id", "front-door"])
    assert result.exit_code == 2 and "provided together" in result.output


def test_repeated_termination_signals_do_not_extend_deadline(monkeypatch):
    import signal

    from boxmot.engine.streaming.runtime import termination

    timers = []

    class Timer:
        def __init__(self, interval, callback, args):
            self.interval, self.callback, self.args = interval, callback, args
            self.cancelled = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(threading, "Timer", Timer)
    stop = threading.Event()
    original = signal.getsignal(signal.SIGTERM)
    with termination(stop):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert stop.is_set() and len(timers) == 1 and timers[0].interval == 20
    assert timers[0].cancelled and signal.getsignal(signal.SIGTERM) == original


def test_http_probes_report_startup_readiness_and_camera_outage():
    import json
    from urllib.error import HTTPError
    from urllib.request import urlopen

    from boxmot.engine.streaming.health import HealthServer

    state = StreamHealth()
    server = HealthServer(state, 0)
    endpoint = f"http://127.0.0.1:{server.server.server_port}"
    try:
        with urlopen(endpoint + "/healthz", timeout=2) as response:
            assert response.status == 200 and json.load(response)["live"]
        with pytest.raises(HTTPError) as error:
            urlopen(endpoint + "/startupz", timeout=2)
        assert error.value.code == 503
        state.initialized = state.owned = state.connected = True
        state.last_frame = time.monotonic()
        with urlopen(endpoint + "/readyz", timeout=2) as response:
            assert response.status == 200 and json.load(response)["ready"]
        state.connected = False
        with pytest.raises(HTTPError) as error:
            urlopen(endpoint + "/readyz", timeout=2)
        assert error.value.code == 503
        with urlopen(endpoint + "/healthz", timeout=2) as response:
            assert response.status == 200
    finally:
        server.close()
