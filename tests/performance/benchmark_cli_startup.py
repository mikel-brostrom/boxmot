"""Measure actual terminal output from a fresh CLI process, including imports.

Example: python -m tests.performance.benchmark_cli_startup --repeat 3 -- eval --help
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import pty
import re
import select
import statistics
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_MARKERS = {"setup": "Setup", "materialization": "Dataset Materialization", "evaluation": "Evaluation"}


def measure(command: list[str], output: Path) -> dict[str, float | int | None]:
    """Timestamp bytes read from a real PTY without pre-importing application code."""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    environment = {**os.environ, "TERM": "xterm-256color", "COLUMNS": "120", "LINES": "40"}
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=slave, stderr=slave, env=environment)
    os.close(slave)
    timings: dict[str, float | int | None] = {"first_visible": None, **dict.fromkeys(_MARKERS)}
    chunks: list[bytes] = []
    visible = ""
    try:
        while True:
            ready, _, _ = select.select([master], [], [], 1.0)
            if not ready:
                if process.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(master, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            elapsed = time.perf_counter() - started
            chunks.append(chunk)
            visible += _ANSI.sub("", chunk.decode("utf-8", errors="replace"))
            if timings["first_visible"] is None and visible.strip():
                timings["first_visible"] = elapsed
            for name, marker in _MARKERS.items():
                if timings[name] is None and marker in visible:
                    timings[name] = elapsed
        timings["returncode"] = process.wait()
        timings["invocation"] = time.perf_counter() - started
    finally:
        os.close(master)
        if process.poll() is None:
            process.terminate()
            process.wait()
    output.with_suffix(".terminal.log").write_bytes(b"".join(chunks))
    output.with_suffix(".json").write_text(json.dumps({"command": command, "timings_s": timings}, indent=2) + "\n")
    if timings["returncode"]:
        raise RuntimeError(f"CLI failed; see {output.with_suffix('.terminal.log')}")
    return timings


def main() -> None:
    """Run sequential terminal startup measurements and report median latencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--label", default="startup")
    parser.add_argument("--output", type=Path, default=Path("runs/replay-benchmark/startup"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.repeat < 1:
        parser.error("provide a CLI command after -- and a positive --repeat")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(args.repeat):
        result = measure(
            [sys.executable, "-m", "boxmot.engine.cli", *command],
            args.output / f"{args.label}-{index + 1}",
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    print(
        "median_s",
        json.dumps(
            {
                key: statistics.median(result[key] for result in results)
                for key in results[0]
                if key != "returncode" and all(result[key] is not None for result in results)
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
