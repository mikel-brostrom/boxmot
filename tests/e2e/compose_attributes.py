"""Exercise real Compose tracking and streaming; run from the repository root.

    python -m tests.e2e.compose_attributes --profile cpu

Requires Docker access and ffmpeg. Builds the selected BoxMOT image, uses a
dedicated Compose project/database, and preserves logs/results under runs/.
No inference or database backend is mocked. This is a smoke/recovery test, not
a performance benchmark or a replacement for the documented 24-hour soak.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
CAMERA = """from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time

image = Path('/inputs/person.jpg').read_bytes()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                self.wfile.write(b'--frame\\r\\nContent-Type: image/jpeg\\r\\nContent-Length: '
                                 + str(len(image)).encode() + b'\\r\\n\\r\\n' + image + b'\\r\\n')
                self.wfile.flush()
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError):
            pass
    def log_message(self, *_args):
        pass
ThreadingHTTPServer(('0.0.0.0', 8090), Handler).serve_forever()
"""


def available_port() -> int:
    """Choose an unused loopback port for this isolated test project."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for(check: Callable[[], Any], description: str, timeout: float = 90) -> Any:
    """Poll a bounded real-service assertion without hiding unexpected errors."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(1)
    raise TimeoutError(description)


class ComposeTest:
    """Own only the containers and artifacts created by one test invocation."""

    def __init__(self, profile: str, image: str | None = None) -> None:
        self.project = os.environ.get("BOXMOT_TEST_PROJECT", f"boxmot-e2e-{uuid4().hex[:10]}")
        if not re.fullmatch(r"boxmot-e2e-[a-z0-9-]+", self.project):
            raise ValueError(
                "BOXMOT_TEST_PROJECT must start with boxmot-e2e- and use lowercase letters/digits/hyphens."
            )
        self.directory = ROOT / "runs" / self.project
        self.inputs = self.directory / "inputs"
        self.inputs.mkdir(parents=True)
        self.profile = profile
        self.image = image or f"boxmot/boxmot:{self.project}"
        self.server = f"vllm-{profile}"
        self.cache_volume = os.environ.get("BOXMOT_TEST_VLLM_CACHE_VOLUME", "boxmot-e2e-vllm-models")
        self.health_port = available_port()
        self.env = {
            **os.environ,
            "BOXMOT_UID": str(os.getuid()),
            "BOXMOT_GID": str(os.getgid()),
            "BOXMOT_INPUT_DIR": str(self.inputs),
            "BOXMOT_RUN_NAME": f"{self.project}/finite",
            "BOXMOT_POSTGRES_PORT": str(available_port()),
            "BOXMOT_POSTGRES_USER": "boxmot",
            "BOXMOT_POSTGRES_PASSWORD": uuid4().hex,
            "BOXMOT_POSTGRES_DB": "boxmot",
            "VLLM_PORT": str(available_port()),
        }
        # Include profiled services in logs, ps, and down as well as explicit up/run.
        self.compose = [
            "docker",
            "compose",
            "--profile",
            "*",
            "-p",
            self.project,
            "-f",
            str(ROOT / "docker/compose.attributes.yaml"),
        ]
        if profile == "gpu":
            self.compose += ["-f", str(ROOT / "docker/compose.attributes.gpu.yaml")]
        self.compose += ["-f", str(self.directory / "override.json")]
        self.commands: list[dict[str, Any]] = []
        self.passed: list[str] = []
        self.started = False

    def command(self, args: list[str], *, timeout: float = 120, check: bool = True) -> subprocess.CompletedProcess:
        """Retain exact commands, output, status, and duration outside version control."""
        index = len(self.commands)
        record: dict[str, Any] = {"command": shlex.join(args)}
        self.commands.append(record)
        log = self.directory / f"{index:03d}.log"
        start = time.monotonic()
        try:
            with log.open("w") as output:
                result = subprocess.run(
                    args, cwd=ROOT, env=self.env, stdout=output, stderr=subprocess.STDOUT, text=True, timeout=timeout
                )
            result.stdout, result.stderr = log.read_text(), ""
        except subprocess.TimeoutExpired:
            record.update(error="timeout", seconds=time.monotonic() - start)
            raise
        record.update(exit_code=result.returncode, seconds=time.monotonic() - start, log=log.name)
        if check and result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
        return result

    def dc(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess:
        return self.command([*self.compose, *args], **kwargs)

    def sql(self, query: str) -> str:
        return self.dc(
            "exec", "-T", "postgres", "psql", "-U", "boxmot", "-d", "boxmot", "-At", "-c", query
        ).stdout.strip()

    def sql_rows(self, query: str) -> list[dict[str, Any]]:
        """Read structured rows without ambiguous psql field delimiters."""
        return json.loads(self.sql(f"SELECT COALESCE(jsonb_agg(result), '[]') FROM ({query}) result"))

    def active_run_id(self) -> str:
        """Require exactly one current session for the fixture camera."""
        rows = self.sql_rows(
            "SELECT run_id FROM boxmot_attributes.runs WHERE camera_id='e2e-camera' AND state='active'"
        )
        assert len(rows) == 1, f"Expected one active camera session, found {rows}"
        return rows[0]["run_id"]

    def ownership(self) -> list[dict[str, Any]]:
        """Snapshot fencing identity without the routinely changing expiry time."""
        rows = self.sql_rows("SELECT owner_id, epoch FROM boxmot_attributes.camera_leases WHERE camera_id='e2e-camera'")
        assert len(rows) == 1 and rows[0]["owner_id"], "Camera has no current owner"
        return rows

    def health(self, path: str) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.health_port}/{path}", timeout=2) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def container_id(self) -> str:
        """Find only the worker owned by this isolated project."""
        container = self.dc("ps", "-a", "-q", "stream").stdout.strip()
        if not container:
            raise RuntimeError("Stream container disappeared unexpectedly.")
        return container

    def start_worker(self) -> None:
        """Restart only the worker; Compose start also starts dependencies."""
        self.command(["docker", "start", self.container_id()])

    def container_state(self) -> dict[str, Any]:
        """Inspect the isolated worker, including its final exit and OOM state."""
        result = self.command(["docker", "inspect", "--format", "{{json .State}}", self.container_id()])
        return json.loads(result.stdout)

    def exited(self) -> bool:
        return not self.container_state()["Running"]

    def mark(self, phase: str) -> None:
        self.passed.append(phase)
        print(f"PASS: {phase}", flush=True)

    def prepare(self) -> None:
        """Use one real pedestrian to keep CPU attribute requests bounded."""
        # The crop isolates the foreground pedestrian in this bundled fixture.
        source = ROOT / "assets/MOT17-mini/train/MOT17-02-FRCNN/img1/000001.jpg"
        self.command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                "crop=200:360:540:400",
                "-frames:v",
                "1",
                "-threads",
                "1",
                str(self.inputs / "person.jpg"),
            ]
        )
        self.command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-i",
                str(self.inputs / "person.jpg"),
                "-frames:v",
                "24",
                "-r",
                "10",
                "-c:v",
                "mpeg4",
                "-q:v",
                "2",
                str(self.inputs / "video.mp4"),
            ]
        )
        (self.inputs / "camera.py").write_text(CAMERA)
        # JSON is also valid YAML; use exactly the normal authored profile.
        (self.inputs / "stream.json").write_text(
            json.dumps(
                {
                    "camera_id": "e2e-camera",
                    "source_url_env": "BOXMOT_CAMERA_URL",
                    "attributes": "person-appearance-vllm-cpu",
                    "attributes_host": f"http://{self.server}:8000",
                    "output_dir": f"/opt/boxmot/runs/{self.project}/streams",
                    "tracking": {
                        "detector": "yolo26n",
                        "tracker": "bytetrack" if self.profile == "cpu" else "botsort",
                        "device": "cpu" if self.profile == "cpu" else "0",
                        "classes": [0],
                    },
                }
            )
        )
        image_name = self.image
        override = {
            # Reuse only model downloads across isolated projects, never their databases.
            "volumes": {"vllm-models": {"name": self.cache_volume, "external": True}},
            "services": {
                "boxmot": {"image": image_name},
                "fixture-camera": {
                    "image": image_name,
                    "init": True,
                    "entrypoint": ["python", "/inputs/camera.py"],
                    "volumes": [f"{self.inputs}:/inputs:ro"],
                },
                "stream": {
                    "image": image_name,
                    "entrypoint": ["boxmot", "stream"],
                    "command": ["--config", "/inputs/stream.json"],
                    "environment": {"BOXMOT_CAMERA_URL": "http://fixture-camera:8090/feed"},
                    "ports": [f"127.0.0.1:{self.health_port}:8080"],
                    "stop_signal": "SIGTERM",
                },
            }
        }
        (self.directory / "override.json").write_text(json.dumps(override, indent=2))
        for name in ("models", "runs"):
            (ROOT / name).mkdir(exist_ok=True)
        # Cross-file `extends: {service: boxmot}` sees the image-only override,
        # not the merged base/GPU service. Copy Compose's fully resolved service
        # so the worker receives the same mounts, credentials, and GPU requests.
        resolved = json.loads(self.dc("config", "--format", "json").stdout)
        stream = resolved["services"]["boxmot"]
        stream.pop("build", None)
        stream.update(
            entrypoint=["boxmot", "stream"],
            command=["--config", "/inputs/stream.json"],
            ports=[f"127.0.0.1:{self.health_port}:8080"],
            stop_signal="SIGTERM",
        )
        stream["environment"]["BOXMOT_CAMERA_URL"] = "http://fixture-camera:8090/feed"
        override["services"]["stream"] = stream
        (self.directory / "override.json").write_text(json.dumps(override, indent=2))
        self.dc("config", "--quiet")
        self.mark("fixture and Compose configuration")

    def run(self, *, skip_build: bool = False) -> None:
        self.command(["docker", "info", "--format", "{{.ServerVersion}}"])
        self.command(["docker", "volume", "create", self.cache_volume])
        if not skip_build:
            print("Building BoxMOT; this can take several minutes.", flush=True)
            self.dc("build", "boxmot", timeout=3600)
        self.started = True
        print("Starting PostgreSQL and the real VLM; waiting for model loading.", flush=True)
        self.dc("up", "-d", "--wait", "--wait-timeout", "1800", "postgres", self.server, timeout=1860)
        assert self.sql("SELECT 1") == "1"
        self.dc("images", "--format", "json")
        self.mark("database and VLM startup")

        print("Running finite tracking and waiting for actual attribute inference.", flush=True)
        self.dc("run", "--rm", "--no-deps", "boxmot", "--source", "/inputs/video.mp4", "--classes", "0", timeout=2400)
        tracks = read_jsonl(self.directory / "finite", "tracks*.jsonl")
        events = read_jsonl(self.directory / "finite", "attributes*.jsonl")
        assert any(row.get("tracks") for row in tracks), "No real person tracks exported"
        completed = [row for row in events if row.get("status") == "completed"]
        assert completed and all(isinstance(row.get("values"), dict) and row["values"] for row in completed), events
        assert not any(row.get("status") == "failed" for row in events), events
        before = int(self.sql("SELECT count(*) FROM boxmot_attributes.events WHERE payload->>'status'='completed'"))
        assert before >= len(completed)
        self.mark("real video → detector → tracker → VLM → PostgreSQL → JSONL")

        self.dc("up", "-d", "fixture-camera", "stream")
        wait_for(lambda: self.health("readyz"), "stream never became ready", 180)
        wait_for(
            lambda: (
                int(
                    self.sql(
                        "SELECT count(*) FROM boxmot_attributes.events e JOIN boxmot_attributes.runs r USING(run_id) "
                        "WHERE r.camera_id='e2e-camera' AND e.payload->>'status'='completed' "
                        "AND e.delivered_at IS NOT NULL"
                    )
                )
                > 0
            ),
            "stream never exported a completed real attribute result",
            1800,
        )
        self.mark("network camera capture and real streaming attributes")

        initial_run = self.active_run_id()
        original_owner = self.ownership()
        original_runs = self.sql_rows(
            "SELECT run_id FROM boxmot_attributes.runs WHERE camera_id='e2e-camera' ORDER BY run_id"
        )
        competitor = self.dc(
            "run",
            "--rm",
            "--no-deps",
            "--entrypoint",
            "boxmot",
            "-e",
            "BOXMOT_CAMERA_URL=http://fixture-camera:8090/feed",
            "boxmot",
            "stream",
            "--config",
            "/inputs/stream.json",
            check=False,
        )
        assert competitor.returncode != 0, "Competing camera owner was admitted"
        assert "Stream stopped (OwnershipLost)" in competitor.stdout, competitor.stdout
        assert self.ownership() == original_owner, "Rejected competitor changed the ownership token"
        assert self.active_run_id() == initial_run, "Rejected competitor replaced the active session"
        assert (
            self.sql_rows("SELECT run_id FROM boxmot_attributes.runs WHERE camera_id='e2e-camera' ORDER BY run_id")
            == original_runs
        ), "Rejected competitor persisted a session"
        self.mark("competing camera owner rejected without changing ownership or sessions")

        self.dc("stop", "fixture-camera")
        wait_for(lambda: not self.health("readyz"), "camera outage did not make stream unready", 40)
        assert self.health("healthz"), "Camera outage killed supervision"
        self.dc("start", "fixture-camera")
        wait_for(lambda: self.health("readyz"), "camera did not reconnect", 90)
        reconnected_run = self.active_run_id()
        assert reconnected_run != initial_run, "Camera reconnect reused its previous session"
        assert self.ownership() == original_owner, "Camera reconnect unexpectedly replaced the owner"
        self.mark("camera disconnect/reconnect creates a fresh session")

        # Hold real VLM requests until cancellation has been observed. Restart
        # the worker while the server is paused so a response sent before the
        # pause cannot race with this session's shutdown assertion.
        self.dc("pause", self.server)
        try:
            self.dc("stop", "-t", "25", "stream")
            self.start_worker()
            wait_for(lambda: self.health("readyz"), "stream did not restart for shutdown check", 180)
            shutdown_run = self.active_run_id()
            assert shutdown_run != reconnected_run
            run_literal = sql_literal(shutdown_run)
            running = wait_for(
                lambda: self.sql_rows(
                    "SELECT j.run_id,j.generation,j.track_id,"
                    "(SELECT jsonb_agg(c.sample ORDER BY c.id) FROM boxmot_attributes.crops c "
                    "WHERE c.run_id=j.run_id AND c.job_id=j.id) AS samples "
                    f"FROM boxmot_attributes.jobs j WHERE j.run_id={run_literal} AND j.state='running' "
                    "ORDER BY j.id LIMIT 1"
                ),
                "No inference in flight for shutdown check",
                600,
            )[0]
            start = time.monotonic()
            self.dc("kill", "-s", "SIGTERM", "stream")
            wait_for(self.exited, "SIGTERM shutdown exceeded application deadline", 22)
            assert time.monotonic() - start < 22
            state = self.container_state()
            assert state["ExitCode"] == 0 and not state["OOMKilled"], state
            assert (
                self.sql(
                    "SELECT count(*) FROM boxmot_attributes.jobs j JOIN boxmot_attributes.runs r USING(run_id) "
                    "WHERE r.camera_id='e2e-camera'"
                )
                == "0"
            ), "Shutdown retained outstanding camera jobs"
            outcomes = self.sql_rows(f"SELECT payload FROM boxmot_attributes.events WHERE run_id={run_literal}")
            assert any(
                row["payload"].get("status") == "cancelled"
                and all(row["payload"].get(key) == value for key, value in running.items())
                for row in outcomes
            ), "The observed in-flight inference batch has no explicit cancellation outcome"
            self.mark("SIGTERM cancels the observed inference batch and exits cleanly within deadline")
        finally:
            self.dc("unpause", self.server, check=False)

        self.start_worker()
        wait_for(lambda: self.health("readyz"), "stream did not restart", 180)
        before_loss_run = self.active_run_id()
        assert before_loss_run != shutdown_run
        completed_history = self.sql_rows(
            "SELECT run_id,id,payload FROM boxmot_attributes.events WHERE payload->>'status'='completed' ORDER BY id"
        )
        assert completed_history, "Database restart check has no completed history to preserve"
        self.dc("stop", "postgres")
        wait_for(self.exited, "database loss did not stop stream", 40)
        state = self.container_state()
        assert state["ExitCode"] != 0 and not state["OOMKilled"], state
        self.dc("start", "postgres")
        # The lease uses database time; let the last renewal expire after loss.
        time.sleep(31)
        self.start_worker()
        wait_for(lambda: self.health("readyz"), "stream did not recover after database restart", 180)
        recovered_run = self.active_run_id()
        assert recovered_run != before_loss_run, "Database recovery reused an obsolete session"
        wait_for(
            lambda: (
                int(
                    self.sql(
                        "SELECT count(*) FROM boxmot_attributes.events "
                        f"WHERE run_id={sql_literal(recovered_run)} "
                        "AND payload->>'status'='completed' AND delivered_at IS NOT NULL"
                    )
                )
                > 0
            ),
            "Recovered session never exported a new completed attribute result",
            1800,
        )
        recovered_history = {
            (row["run_id"], row["id"]): row["payload"]
            for row in self.sql_rows(
                "SELECT run_id,id,payload FROM boxmot_attributes.events WHERE payload->>'status'='completed'"
            )
        }
        assert all(recovered_history.get((row["run_id"], row["id"])) == row["payload"] for row in completed_history), (
            "Database restart lost or changed a completed event"
        )
        self.mark("database loss stops the worker; recovery preserves history and exports a new result")

        self.dc("stop", "-t", "25", "stream")
        events = read_jsonl(self.directory / "streams", "attributes*.jsonl*")
        assert events and all(row.get("event_id") for row in events)
        exported: dict[str, dict[str, Any]] = {}
        for event in events:
            previous = exported.setdefault(event["event_id"], event)
            assert previous == event, f"Duplicate event ID has inconsistent payloads: {event['event_id']}"
        persisted = self.sql_rows(
            "SELECT e.run_id,e.id,e.payload FROM boxmot_attributes.events e "
            "JOIN boxmot_attributes.runs r USING(run_id) WHERE r.camera_id='e2e-camera' AND e.delivered_at IS NOT NULL"
        )
        assert persisted, "No durable acknowledgments exist to verify"
        for row in persisted:
            event_id = f"{row['run_id']}:{row['id']}"
            assert event_id in exported, f"Acknowledged event is missing from synchronized JSONL: {event_id}"
            event = exported[event_id]
            assert event["camera_id"] == "e2e-camera" and event["session_id"] == row["run_id"], event
            assert {
                key: value
                for key, value in event.items()
                if key not in {"event_id", "camera_id", "session_id", "created_at"}
            } == row["payload"], f"Exported event differs from its database payload: {event_id}"
        track_sessions = {
            row["session_id"]
            for row in read_jsonl(self.directory / "streams", "tracks*.jsonl*")
            if row.get("camera_id") == "e2e-camera"
        }
        assert {initial_run, reconnected_run, shutdown_run, before_loss_run, recovered_run} <= track_sessions
        self.mark("acknowledged events match JSONL payloads, with consistent duplicate IDs and fresh track sessions")

    def finish(self, error: str | None) -> None:
        """Attempt every cleanup step and retain failures without masking the test error."""
        cleanup_errors: list[str] = []
        if self.started:
            # Each operation is independent: a failed log collection must not
            # prevent teardown. Preserve the volume and unrelated projects.
            for action, args in (
                ("collect logs", ("logs", "--no-color")),
                ("collect container state", ("ps", "-a", "--format", "json")),
                ("stop isolated project", ("down", "--timeout", "30")),
            ):
                try:
                    result = self.dc(*args, check=False, timeout=120)
                    if result.returncode:
                        cleanup_errors.append(f"{action}: exit status {result.returncode}")
                except Exception as exc:
                    cleanup_errors.append(f"{action}: {type(exc).__name__}: {exc}")
        report = {
            "project": self.project,
            "profile": self.profile,
            "passed": self.passed,
            "error": error or ("Cleanup failed: " + "; ".join(cleanup_errors) if cleanup_errors else None),
            "cleanup_errors": cleanup_errors,
            "commands": self.commands,
            "not_covered": [
                "24-hour soak",
                "live physical camera",
                "Kubernetes scheduling/PVCs",
                "seven-day wall-clock aging",
                "output failure injection",
                "crash between event synchronization and acknowledgment",
                "blocked capture backend",
                "queue capacity and rotation limits",
                "late inference response after ownership takeover",
                "bounded-memory soak",
            ],
        }
        try:
            (self.directory / "result.json").write_text(json.dumps(report, indent=2))
        except Exception as exc:
            cleanup_errors.append(f"write result.json: {type(exc).__name__}: {exc}")
        print(f"Evidence: {self.directory}", flush=True)
        if cleanup_errors:
            message = "Cleanup failed: " + "; ".join(cleanup_errors)
            if error is not None:
                # main() is already propagating the primary failure. Report
                # secondary failures visibly without replacing its traceback.
                print(message, file=sys.stderr, flush=True)
            else:
                raise RuntimeError(message)


def sql_literal(value: str) -> str:
    """Quote database-returned identifiers used as values in psql queries."""
    return "'" + value.replace("'", "''") + "'"


def read_jsonl(directory: Path, pattern: str) -> list[dict[str, Any]]:
    """Require complete, finite JSON records in the small smoke-test outputs."""

    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")

    rows = []
    for path in directory.rglob(pattern):
        contents = path.read_text()
        assert not contents or contents.endswith("\n"), f"Incomplete JSONL record in {path}"
        for line in contents.splitlines():
            value = json.loads(line, parse_constant=reject)
            json.dumps(value, allow_nan=False)
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--image", help="Reuse an already built BoxMOT image instead of building this checkout.")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Generate and validate fixtures without starting Docker."
    )
    args = parser.parse_args()
    test = ComposeTest(args.profile, image=args.image)
    error = None
    try:
        test.prepare()
        if not args.prepare_only:
            test.run(skip_build=bool(args.image))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        test.finish(error)


if __name__ == "__main__":
    main()
