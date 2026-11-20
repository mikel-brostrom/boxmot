"""Exercise reconnects and SIGTERM with real embedded PostgreSQL scheduling."""

import asyncio
import json
import os
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.session import AttributeEnricher
from boxmot.engine.config.streams import StreamConfig
from boxmot.engine.streaming import runtime
from boxmot.engine.streaming.capture import CapturedFrame, CaptureDisconnected
from boxmot.structures import Boxes, Tracks


def test_reconnect_resets_session_and_sigterm_cancels_inference(attribute_database, monkeypatch, tmp_path):
    backends, sessions, captures, health_states = [], [], [], []

    class Backend:
        def __init__(self):
            self.entered = threading.Event()
            self.cancelled = False
            self.closed = False

        async def extract(self, *_args):
            self.entered.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def aclose(self):
            self.closed = True

    def enricher(config, **kwargs):
        backend = Backend()
        backends.append(backend)
        session = AttributeEnricher(config, backend=backend, **kwargs)
        sessions.append(session)
        return session

    class Pipeline:
        attributes = None
        resets = 0

        def reset(self):
            self.resets += 1
            self.attributes.reset()

        def step(self, frame):
            tracks = Tracks(
                Boxes(torch.tensor([[1, 1, 14, 14]], dtype=torch.float32)),
                torch.tensor([1]),
                torch.ones(1),
                torch.zeros(1, dtype=torch.int64),
                torch.zeros(1, dtype=torch.int64),
                frame.sample_id,
            )
            latest = self.attributes.update(frame, tracks)
            assert backends[-1].entered.wait(2)
            return SimpleNamespace(tracks=tracks, attributes=latest)

    class Capture:
        def __init__(self, *_args, **_kwargs):
            assert _kwargs["timestamp_source"] == "capture"
            self.index = -1
            self.closed = False
            captures.append(self)

        def next(self):
            self.index += 1
            if self.index == 2:
                if len(captures) == 1:
                    raise CaptureDisconnected()
                os.kill(os.getpid(), signal.SIGTERM)
                return None
            return CapturedFrame(
                np.zeros((16, 16, 3), np.uint8), self.index, self.index / 30, time.monotonic(), time.time()
            )

        def close(self):
            self.closed = True

    pipeline = Pipeline()

    def health_server(state, _port):
        health_states.append(state)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setenv("BOXMOT_CAMERA_URL", "rtsp://private:secret@camera/stream")
    monkeypatch.setattr(runtime, "HealthServer", health_server)
    monkeypatch.setattr(runtime, "NetworkCapture", Capture)
    monkeypatch.setattr(runtime, "AttributeEnricher", enricher)
    monkeypatch.setattr(runtime, "build_tracking_pipeline", lambda *_args, **_kwargs: pipeline)
    config = AttributeConfig(model="fake", prompt="Describe", schema={"type": "object"}, batch_size=1)
    started = time.monotonic()
    runtime.run_stream(
        StreamConfig(camera_id="fixture-runtime", attributes=config, output_dir=tmp_path, reconnect_max_s=0.01)
    )
    assert time.monotonic() - started < 20
    assert len(sessions) == pipeline.resets == len(captures) == 2
    assert sessions[0].run_id != sessions[1].run_id
    assert all(b.cancelled and b.closed for b in backends)
    assert all(c.closed for c in captures)
    records = [json.loads(line) for path in tmp_path.rglob("tracks-*.jsonl") for line in path.read_text().splitlines()]
    assert len(records) == 4
    assert {r["session_id"] for r in records} == {s.run_id for s in sessions}
    assert {r["generation"] for r in records} == {1}
    assert all(r["tracks"][0]["track_id"] == 1 for r in records)
    assert "secret" not in json.dumps(records)
    outcomes = [
        json.loads(line) for path in tmp_path.rglob("attributes-*.jsonl") for line in path.read_text().splitlines()
    ]
    assert len(outcomes) == 2 and all(e["status"] == "cancelled" and e["event_id"] for e in outcomes)
    health = health_states[0]
    assert health.frames == 4 and health.reconnects == 1 and health.processing_seconds > 0
    assert health.attribute_events["cancelled"] == 2
    assert health.pending_batches == health.queued_crop_bytes == health.oldest_batch_age_s == 0
    attribute_database.execute("DELETE FROM boxmot_attributes.camera_leases WHERE camera_id='fixture-runtime'")
