"""Small independent health endpoint with distinct live and ready conditions."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StreamHealth:
    """Liveness follows supervision; readiness additionally requires fresh frames."""

    def __init__(self, ready_timeout_s: float = 30) -> None:
        self.ready_timeout_s = ready_timeout_s
        self.heartbeat = time.monotonic()
        self.last_frame = 0.0
        self.owned = False
        self.initialized = False
        self.connected = False
        self.stopping = False
        self.frames = 0
        self.dropped_frames = 0
        self.reconnects = 0
        self.pending_batches = 0
        self.queued_crop_bytes = 0
        self.oldest_batch_age_s = 0.0
        self.mean_frame_ms = 0.0
        self.processing_seconds = 0.0
        self.attribute_events = dict.fromkeys(("completed", "failed", "cancelled", "ineligible"), 0)
        self._queue_observed_at = 0.0
        self.ownership_lost: threading.Event | None = None

    def snapshot(self) -> dict[str, object]:
        now = time.monotonic()
        live = not self.stopping and now - self.heartbeat < max(60, self.ready_timeout_s)
        owned = self.owned and (self.ownership_lost is None or not self.ownership_lost.is_set())
        ready = live and owned and self.connected and now - self.last_frame < self.ready_timeout_s
        return {
            "live": live,
            "ready": ready,
            "frames": self.frames,
            "dropped_frames": self.dropped_frames,
            "reconnects": self.reconnects,
            "pending_batches": self.pending_batches,
            "queued_crop_bytes": self.queued_crop_bytes,
            "oldest_batch_age_s": (
                self.oldest_batch_age_s + max(0, now - self._queue_observed_at)
                if self.pending_batches and self._queue_observed_at
                else self.oldest_batch_age_s
            ),
            "mean_frame_ms": self.mean_frame_ms,
        }

    def observe_queue(self, metrics: dict[str, int | float]) -> None:
        """Cache database aggregates; HTTP probes never query the database."""
        self.pending_batches = int(metrics["pending_batches"])
        self.queued_crop_bytes = int(metrics["queued_crop_bytes"])
        self.oldest_batch_age_s = float(metrics["oldest_batch_age_s"])
        self._queue_observed_at = time.monotonic()

    def observe_attribute_event(self, event: dict[str, object]) -> None:
        """Count acknowledged exports, including recovered events from old sessions."""
        status = event.get("status")
        if isinstance(status, str) and status in self.attribute_events:
            self.attribute_events[status] += 1

    def metrics(self) -> bytes:
        """Render bounded Prometheus text with no frame, session, or URL labels.

        Counters have this worker process's lifetime. Queue gauges describe the
        current session and are sampled by supervision, never by a scrape.
        """
        snapshot = self.snapshot()
        samples = (
            ("ready", "gauge", "Worker owns the camera and has fresh frames.", int(snapshot["ready"])),
            ("initialized", "gauge", "Tracking pipeline initialization completed.", int(self.initialized)),
            ("frames_total", "counter", "Frames successfully processed by this worker.", self.frames),
            ("dropped_frames_total", "counter", "Decoded frames replaced before processing.", self.dropped_frames),
            ("reconnects_total", "counter", "Capture disconnects triggering a new session.", self.reconnects),
            ("pending_batches", "gauge", "Current session attribute batches awaiting outcomes.", self.pending_batches),
            ("queued_crop_bytes", "gauge", "Current session attribute crop payload bytes.", self.queued_crop_bytes),
            (
                "oldest_batch_age_seconds",
                "gauge",
                "Age of the oldest pending attribute batch, extrapolated since the last sample.",
                snapshot["oldest_batch_age_s"],
            ),
            (
                "queue_sample_age_seconds",
                "gauge",
                "Seconds since the supervisor refreshed attribute queue metrics.",
                max(0, time.monotonic() - self._queue_observed_at) if self._queue_observed_at else 0,
            ),
            (
                "frame_processing_seconds_total",
                "counter",
                "Total processing time of successfully processed frames, including output writes.",
                self.processing_seconds,
            ),
            (
                "frame_processing_mean_seconds",
                "gauge",
                "Mean processing time over at most the last 300 frames.",
                self.mean_frame_ms / 1000,
            ),
        )
        lines = []
        for suffix, kind, help_text, value in samples:
            name = f"boxmot_stream_{suffix}"
            lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", f"{name} {value}"))
        name = "boxmot_stream_attribute_events_exported_total"
        lines.extend(
            (
                f"# HELP {name} Acknowledged attribute outcome exports, including recovery replay.",
                f"# TYPE {name} counter",
            )
        )
        lines.extend(f'{name}{{status="{status}"}} {count}' for status, count in self.attribute_events.items())
        return ("\n".join(lines) + "\n").encode()


class HealthServer:
    """Serve read-only health without importing the separate tracking HTTP service."""

    def __init__(self, state: StreamHealth, port: int = 8080) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/metrics":
                    body = state.metrics()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path not in ("/healthz", "/readyz", "/startupz"):
                    self.send_error(404)
                    return
                snapshot = state.snapshot()
                body = json.dumps(snapshot).encode()
                good = (
                    state.initialized
                    if self.path == "/startupz"
                    else snapshot["live" if self.path == "/healthz" else "ready"]
                )
                self.send_response(200 if good else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="boxmot-health")
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
