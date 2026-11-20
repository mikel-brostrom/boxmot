"""Serve a JPEG sequence over MJPEG without requiring inference dependencies."""

from __future__ import annotations

import argparse
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def camera_server(frames: list[bytes], fps: float, address: tuple[str, int]) -> ThreadingHTTPServer:
    """Loop deterministic frames per connection, pacing against a monotonic clock."""
    if not frames or not math.isfinite(fps) or fps <= 0:
        raise ValueError("At least one frame and finite positive FPS are required")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            index = sum(self.path.encode()) % len(frames)
            deadline = time.monotonic()
            try:
                while True:
                    frame = frames[index % len(frames)]
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode()
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                    self.wfile.flush()
                    index += 1
                    deadline += 1 / fps
                    now = time.monotonic()
                    # Slow clients must not receive catch-up bursts that inflate
                    # their observed source FPS after socket backpressure clears.
                    if deadline <= now:
                        deadline = now + 1 / fps
                    time.sleep(deadline - now)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args: object) -> None:
            """Avoid unbounded access logs during soak tests."""

    return ThreadingHTTPServer(address, Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=Path("/inputs"))
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    frames = [path.read_bytes() for path in sorted(args.frames.glob("*.jpg"))]
    with camera_server(frames, args.fps, ("0.0.0.0", args.port)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
