"""Supervise one network camera with fresh sessions and fenced attribute output."""

from __future__ import annotations

import os
import random
import signal
import threading
import time
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import Any, Iterator
from urllib.parse import urlsplit

from boxmot.attributes.config import resolve_attribute_config
from boxmot.attributes.session import AttributeEnricher
from boxmot.attributes.streams import CameraJournal, CameraLease
from boxmot.engine.config.streams import StreamConfig
from boxmot.engine.streaming.capture import CaptureDisconnected, NetworkCapture
from boxmot.engine.streaming.health import HealthServer, StreamHealth
from boxmot.engine.streaming.outputs import AttributeEventWriter, RotatingJsonl, export_events, prune_outputs
from boxmot.engine.tracking.profiling import RuntimeProfiler
from boxmot.engine.tracking.sinks import _json_tracks
from boxmot.engine.tracking.sources import frame_from_bgr
from boxmot.engine.tracking.workflow import build_tracking_pipeline
from boxmot.utils import logger as LOGGER
from boxmot.utils.config import ConfigurationError


@contextmanager
def termination(stop: threading.Event) -> Iterator[None]:
    """Signals request cooperative cleanup with a hard twenty-second ceiling."""
    watchdog: threading.Timer | None = None

    def request_stop(_number: int, _frame: Any) -> None:
        nonlocal watchdog
        stop.set()
        if watchdog is None:
            watchdog = threading.Timer(20, os._exit, args=(1,))
            watchdog.daemon = True
            watchdog.start()

    previous = {number: signal.signal(number, request_stop) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        if watchdog is not None:
            watchdog.cancel()
        for number, handler in previous.items():
            signal.signal(number, handler)


def _wait(stop: threading.Event, lease: CameraLease, health: StreamHealth, seconds: float) -> None:
    until = time.monotonic() + seconds
    while not stop.is_set() and time.monotonic() < until:
        lease.check()
        health.heartbeat = time.monotonic()
        stop.wait(min(0.1, max(0, until - time.monotonic())))


def run_stream(config: StreamConfig) -> None:
    """Own model setup, capture recovery, metadata, and bounded shutdown."""
    url = os.environ.get(config.source_url_env, "").strip()
    try:
        parts = urlsplit(url)
        valid = parts.scheme in {"rtsp", "rtsps", "http", "https", "rtmp", "udp", "tcp"} and bool(parts.netloc)
    except ValueError:
        valid = False
    if not valid:
        raise ConfigurationError(f"{config.source_url_env} must contain a network camera URL.")
    attributes = resolve_attribute_config(config.attributes)
    if config.attributes_host is not None:
        attributes = replace(attributes, host=config.attributes_host)
    stop = threading.Event()
    health = StreamHealth(config.ready_timeout_s)
    timings: deque[dict[str, float]] = deque(maxlen=300)
    with termination(stop), ExitStack() as resources:
        server = HealthServer(health, config.health_port)
        resources.callback(server.close)
        lease = CameraLease(config.camera_id, attributes.database_url_env)
        resources.callback(lease.close)
        journal = CameraJournal(lease)
        resources.callback(journal.close)
        health.owned = True
        health.ownership_lost = lease.lost
        profiler = RuntimeProfiler()
        pipeline = build_tracking_pipeline(config.tracking_args(), profiler=profiler)
        health.initialized = True
        delay = min(1.0, config.reconnect_max_s)
        last_maintenance = last_report = last_metrics = 0.0
        camera_directory = config.output_dir / config.camera_id
        replay = AttributeEventWriter(
            camera_directory, max_bytes=config.rotation_bytes, max_seconds=config.rotation_seconds
        )
        resources.callback(replay.close)
        try:
            while not stop.is_set():
                lease.check()
                health.heartbeat = time.monotonic()
                export_events(journal, replay, on_export=health.observe_attribute_event)
                capture = NetworkCapture(
                    url,
                    config.capture_timeout_s,
                    max_frame_bytes=config.max_frame_bytes,
                    timestamp_source=config.timestamp_source,
                )
                enricher = None
                tracks = None
                previous_index = -1
                healthy_since = time.monotonic()
                try:
                    while not stop.is_set():
                        lease.check()
                        health.heartbeat = time.monotonic()
                        captured = capture.next()
                        if stop.is_set():
                            break
                        if captured is None:
                            export_events(journal, replay, on_export=health.observe_attribute_event)
                            continue
                        if enricher is None:
                            enricher = AttributeEnricher(
                                attributes,
                                ownership=lease,
                                max_pending_jobs=config.max_pending_jobs,
                                max_crop_bytes=config.max_crop_bytes,
                            )
                            pipeline.attributes = enricher
                            pipeline.reset()
                            directory = camera_directory / enricher.run_id
                            tracks = RotatingJsonl(
                                directory,
                                "tracks",
                                max_bytes=config.rotation_bytes,
                                max_seconds=config.rotation_seconds,
                            )
                        if stop.is_set():
                            break
                        health.connected = True
                        health.dropped_frames += max(0, captured.index - previous_index - 1)
                        previous_index = captured.index
                        sequence_id = f"{config.camera_id}:{enricher.run_id}"
                        frame = frame_from_bgr(
                            captured.image,
                            sample_id=f"{sequence_id}:{captured.index}",
                            sequence_id=sequence_id,
                            frame_index=captured.index,
                            timestamp_s=captured.timestamp_s,
                            source_uri=f"camera:{config.camera_id}",
                        )
                        started = time.monotonic()
                        try:
                            with profiler.activate((frame.sample_id,)):
                                profiler.begin_pipeline(frame.sample_id)
                                try:
                                    result = pipeline.step(frame)
                                finally:
                                    profiler.end_pipeline(frame.sample_id)
                            lease.check()
                            # The DB fence in update precedes publication; session IDs
                            # distinguish any physical file writes by an obsolete owner.
                            tracks.write(
                                {
                                    "camera_id": config.camera_id,
                                    "session_id": enricher.run_id,
                                    "generation": 1,
                                    "sample_id": frame.sample_id,
                                    "frame_index": frame.frame_index,
                                    "timestamp_s": frame.timestamp_s,
                                    "tracks": _json_tracks(result, include_attributes=False),
                                    "captured_at_unix_s": captured.captured_wall_time,
                                }
                            )
                            tracks.sync()
                            export_events(journal, replay, on_export=health.observe_attribute_event)
                        finally:
                            profiler.finish_sample(frame.sample_id, (time.monotonic() - started) * 1000)
                            timings.append(profiler.release_sample(frame.sample_id))
                        health.last_frame = time.monotonic()
                        health.frames += 1
                        health.processing_seconds += time.monotonic() - started
                        health.mean_frame_ms = sum(row["overall"] for row in timings) / len(timings)
                        health.heartbeat = time.monotonic()
                        if time.monotonic() - healthy_since > 30:
                            delay = min(1.0, config.reconnect_max_s)
                        if time.monotonic() - last_metrics > 5:
                            health.observe_queue(enricher.queue_metrics)
                            last_metrics = time.monotonic()
                        if time.monotonic() - last_report > 30:
                            LOGGER.info(f"Stream {config.camera_id} status: {health.snapshot()}")
                            last_report = time.monotonic()
                        if time.monotonic() - last_maintenance > 3600:
                            replay.close()
                            if not journal.read(1):
                                prune_outputs(
                                    camera_directory, config.retention_days, active=(tracks.path, replay.path)
                                )
                            last_maintenance = time.monotonic()
                except CaptureDisconnected:
                    health.reconnects += 1
                    LOGGER.warning(f"Camera {config.camera_id} disconnected; starting a new session after reconnect.")
                finally:
                    health.connected = False
                    health.observe_queue({"pending_batches": 0, "queued_crop_bytes": 0, "oldest_batch_age_s": 0.0})
                    last_metrics = 0.0
                    capture.close()
                    # Cleanup also has a ceiling during ordinary reconnections.
                    cleanup = threading.Timer(20, os._exit, args=(1,))
                    cleanup.daemon = True
                    cleanup.start()
                    try:
                        with ExitStack() as closing:
                            if tracks is not None:
                                closing.callback(tracks.close)
                            if enricher is not None:
                                enricher.close()
                                pipeline.attributes = None
                                journal.end_session(enricher.run_id)
                                # Bounded pages; later pages remain durable for replay.
                                export_events(journal, replay, on_export=health.observe_attribute_event)
                    finally:
                        cleanup.cancel()
                _wait(stop, lease, health, delay * random.uniform(0.8, 1.0))
                delay = min(config.reconnect_max_s, delay * 2)
        finally:
            health.stopping = True
            health.owned = False
