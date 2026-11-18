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
            "oldest_batch_age_s": self.oldest_batch_age_s,
            "mean_frame_ms": self.mean_frame_ms,
        }


class HealthServer:
    """Serve read-only health without importing the separate tracking HTTP service."""

    def __init__(self, state: StreamHealth, port: int = 8080) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
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
