"""Install, upgrade, and recover the real Helm deployment in an isolated namespace.

Requires an existing Kubernetes context, Helm, kubectl, ffmpeg, PyYAML, and
immutable images containing ``boxmot stream``. No inference is mocked. The
namespace and its test data are removed after evidence is collected. Existing
namespaces are never adopted. Run as ``python -m tests.e2e.helm_streaming --help``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import yaml

from tests.e2e.compose_attributes import CAMERA

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy/helm/boxmot"
RELEASE = "smoke"
CAMERA_ID = "camera-01"
OWNER_LABEL = "boxmot.io/helm-test-owner"


def wait_for(check: Callable[[], Any], description: str, timeout: float = 120) -> Any:
    """Retry an observable service condition within one bounded deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(5)
    raise TimeoutError(description)


class HelmTest:
    """Own one test namespace; never use production databases or camera inputs."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.owner = uuid4().hex
        self.namespace = args.namespace or f"boxmot-helm-e2e-{self.owner[:10]}"
        if not re.fullmatch(r"boxmot-helm-e2e-[a-z0-9-]{1,35}", self.namespace):
            raise ValueError("Test namespaces must start with boxmot-helm-e2e- (maximum 51 characters).")
        self.directory = Path(args.output or ROOT / "runs" / self.namespace).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.commands: list[dict[str, Any]] = []
        self.passed: list[str] = []
        self.owned = False
        self.runtime_version: str | None = None
        self.expected_sessions: set[str] = set()
        self.values: dict[str, Any] = {}
        self.defaults = yaml.safe_load((CHART / "values.yaml").read_text())
        self.chart = yaml.safe_load((CHART / "Chart.yaml").read_text())
        for image in (args.image, args.vllm_image or self.defaults["vllm"]["image"]):
            if not re.fullmatch(r"[^\s@]+@sha256:[a-fA-F0-9]{64}", image):
                raise ValueError("Images must be repository@sha256:<64 hexadecimal digits>.")

    def command(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        timeout: float = 120,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Record commands and public output, never stdin containing credentials."""
        index = len(self.commands)
        record: dict[str, Any] = {"command": shlex.join(args)}
        self.commands.append(record)
        path = self.directory / f"{index:04d}.log"
        start = time.monotonic()
        try:
            with path.open("w") as output:
                result = subprocess.run(
                    args,
                    cwd=ROOT,
                    input=input_text,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                )
            result.stdout = path.read_text()
        except subprocess.TimeoutExpired:
            record.update(error="timeout", seconds=time.monotonic() - start)
            raise
        record.update(exit_code=result.returncode, seconds=time.monotonic() - start, log=path.name)
        if check and result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); see {path}")
        return result

    def kubectl(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Always bind kubectl to the explicitly selected context and namespace."""
        return self.command(["kubectl", "--context", self.args.context, "--namespace", self.namespace, *args], **kwargs)

    def apply(self, *objects: dict[str, Any]) -> None:
        """Pass manifests over stdin so Secrets never enter logs or process arguments."""
        self.kubectl(
            "apply", "-f", "-", input_text=json.dumps({"apiVersion": "v1", "kind": "List", "items": list(objects)})
        )

    def mark(self, text: str) -> None:
        self.passed.append(text)
        print(f"PASS: {text}", flush=True)

    def install(self, *, first: bool = False) -> None:
        """Use the same explicit values for every install/upgrade; never reuse stale values."""
        values_path = self.directory / "values.json"
        values_path.write_text(json.dumps(self.values, indent=2) + "\n")
        command = [
            "helm",
            "upgrade",
            "--install",
            RELEASE,
            str(CHART),
            "--kube-context",
            self.args.context,
            "--namespace",
            self.namespace,
            "--values",
            str(values_path),
            "--timeout",
            "30m" if first else "5m",
        ]
        # WaitForFirstConsumer storage cannot bind suspended cameras' PVCs.
        # Wait for the active dependencies explicitly, then let the first
        # camera upgrade bind its own claims through the ordinary Helm wait.
        if not first:
            command.append("--wait")
        self.command(command, timeout=1860 if first else 360)
        if first:
            for name in (
                f"{RELEASE}-postgres",
                "fixture",
                *([f"{RELEASE}-vllm"] if not self.args.external_vllm_url else []),
            ):
                self.kubectl("rollout", "status", f"deployment/{name}", "--timeout=1800s", timeout=1860)

    def prepare(self) -> None:
        """Crop the bundled fixture and render the chart before any cluster mutation."""
        fixture = ROOT / "assets/MOT17-mini/train/MOT17-02-FRCNN/img1/000001.jpg"
        self.command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(fixture),
                "-vf",
                "crop=200:360:540:400",
                "-frames:v",
                "1",
                "-threads",
                "1",
                str(self.directory / "person.jpg"),
            ]
        )
        self.values = {
            "image": self.args.image,
            "database": {"existingSecret": "test-database", "urlKey": "url"},
            "postgres": {
                "enabled": True,
                "persistence": {"size": "1Gi", "storageClass": self.args.storage_class},
                "resources": {
                    "requests": {"cpu": "100m", "memory": "128Mi"},
                    "limits": {"cpu": "1", "memory": "512Mi"},
                },
            },
            "vllm": {
                "image": self.args.vllm_image or self.defaults["vllm"]["image"],
                "resources": {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"cpu": "4", "memory": "8Gi"}},
                "persistence": {"size": "4Gi", "storageClass": self.args.storage_class},
            },
            "cameraDefaults": {
                "suspend": True,
                "config": {"tracking": {"classes": [0]}},
                "resources": {"requests": {"cpu": "1", "memory": "1Gi"}, "limits": {"cpu": "2", "memory": "2Gi"}},
                "models": {"size": "1Gi", "storageClass": self.args.storage_class},
                "outputs": {"size": "1Gi", "storageClass": self.args.storage_class, "accessModes": ["ReadWriteOnce"]},
            },
            "cameras": {CAMERA_ID: {"sourceSecret": {"name": "test-camera", "key": "url"}}},
            # Run maintenance deliberately at the end, with no scheduler race.
            "maintenance": {"suspend": True},
        }
        if self.args.external_vllm_url:
            self.values["vllm"].update(enabled=False, externalEndpoint=self.args.external_vllm_url)
        if self.args.detector_weights:
            self.values["cameraDefaults"]["config"]["tracking"]["detector"] = "/opt/boxmot/models/yolo26n.pt"
        path = self.directory / "values.json"
        path.write_text(json.dumps(self.values, indent=2) + "\n")
        self.command(["helm", "template", RELEASE, str(CHART), "--namespace", self.namespace, "--values", str(path)])
        self.mark("fixture and Helm rendering")

    def fixture(self) -> None:
        """Expose only the repeatable, bundled pedestrian image inside the test namespace."""
        self.apply(
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "fixture"},
                "data": {"camera.py": CAMERA},
                "binaryData": {"person.jpg": base64.b64encode((self.directory / "person.jpg").read_bytes()).decode()},
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": "fixture"},
                "spec": {"selector": {"app": "fixture"}, "ports": [{"port": 8090, "targetPort": 8090}]},
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "fixture"},
                "spec": {
                    "replicas": 1,
                    "strategy": {"type": "Recreate"},
                    "selector": {"matchLabels": {"app": "fixture"}},
                    "template": {
                        "metadata": {"labels": {"app": "fixture"}},
                        "spec": {
                            "automountServiceAccountToken": False,
                            "enableServiceLinks": False,
                            "securityContext": {"runAsNonRoot": True, "runAsUser": 10001, "runAsGroup": 10001},
                            "containers": [
                                {
                                    "name": "fixture",
                                    "image": self.args.image,
                                    "imagePullPolicy": "IfNotPresent",
                                    "command": ["python", "/inputs/camera.py"],
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "capabilities": {"drop": ["ALL"]},
                                    },
                                    "readinessProbe": {"tcpSocket": {"port": 8090}},
                                    "resources": {
                                        "requests": {"cpu": "50m", "memory": "32Mi"},
                                        "limits": {"cpu": "200m", "memory": "128Mi"},
                                    },
                                    "volumeMounts": [{"name": "fixture", "mountPath": "/inputs", "readOnly": True}],
                                }
                            ],
                            "volumes": [{"name": "fixture", "configMap": {"name": "fixture"}}],
                        },
                    },
                },
            },
        )

    def volume_pod(self, name: str, claim: str, mount: str) -> None:
        """Mount this test's retained PVC as the same unprivileged application UID."""
        self.apply(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": name},
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "enableServiceLinks": False,
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 10001,
                        "runAsGroup": 10001,
                        "fsGroup": 10001,
                    },
                    "containers": [
                        {
                            "name": "reader",
                            "image": self.args.image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python", "-c", "import time; time.sleep(3600)"],
                            "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                            "resources": {
                                "requests": {"cpu": "50m", "memory": "64Mi"},
                                "limits": {"cpu": "500m", "memory": "256Mi"},
                            },
                            "volumeMounts": [{"name": "data", "mountPath": mount}],
                        }
                    ],
                    "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": claim}}],
                },
            }
        )
        self.kubectl("wait", "--for=condition=Ready", f"pod/{name}", "--timeout=180s", timeout=200)

    def sql(self, query: str) -> Any:
        """Read only the isolated database, returning structured JSON rows."""
        result = self.kubectl(
            "exec",
            f"deployment/{RELEASE}-postgres",
            "--",
            "psql",
            "-U",
            "boxmot",
            "-d",
            "boxmot",
            "-At",
            "-c",
            f"SELECT COALESCE(jsonb_agg(result), '[]') FROM ({query}) result",
        )
        return json.loads(result.stdout.strip())

    def run_id(self) -> str | None:
        rows = self.sql(f"SELECT run_id FROM boxmot_attributes.runs WHERE camera_id='{CAMERA_ID}' AND state='active'")
        assert len(rows) <= 1, "More than one active camera session"
        return rows[0]["run_id"] if rows else None

    def completed(self, run_id: str) -> bool:
        assert re.fullmatch(r"[a-f0-9]{32}", run_id)
        rows = self.sql(
            f"SELECT payload FROM boxmot_attributes.events WHERE run_id='{run_id}' "
            "AND payload->>'status'='completed' AND delivered_at IS NOT NULL"
        )
        return bool(rows) and all(
            isinstance(row["payload"].get("values"), dict) and row["payload"]["values"] for row in rows
        )

    def ready(self) -> bool:
        result = self.kubectl("get", "deployment", f"{RELEASE}-{CAMERA_ID}", "-o", "json")
        return json.loads(result.stdout).get("status", {}).get("readyReplicas", 0) == 1

    def camera_pods(self) -> list[dict[str, Any]]:
        """Select the camera through its deployment selector, not name substrings."""
        deployment = json.loads(self.kubectl("get", "deployment", f"{RELEASE}-{CAMERA_ID}", "-o", "json").stdout)
        selector = ",".join(f"{key}={value}" for key, value in deployment["spec"]["selector"]["matchLabels"].items())
        return json.loads(self.kubectl("get", "pods", "-l", selector, "-o", "json").stdout)["items"]

    def pvc_ids(self) -> dict[str, str]:
        items = json.loads(self.kubectl("get", "pvc", "-o", "json").stdout)["items"]
        assert not any(item["metadata"].get("deletionTimestamp") for item in items), "Retained PVC is being deleted"
        return {item["metadata"]["name"]: item["metadata"]["uid"] for item in items}

    def run(self) -> None:
        """Require attributes, data continuity, source recovery, and pod/DB recovery."""
        # A successful lookup of an existing namespace is always a refusal.
        result = self.kubectl("get", "namespace", self.namespace, "--ignore-not-found", "-o", "name")
        if result.stdout.strip():
            raise RuntimeError(f"Refusing to adopt existing namespace {self.namespace}")
        namespace = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": self.namespace, "labels": {OWNER_LABEL: self.owner}},
        }
        self.kubectl("create", "-f", "-", input_text=json.dumps(namespace))
        self.owned = True
        password = secrets.token_hex(32)
        self.apply(
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": "test-database"},
                "stringData": {
                    "password": password,
                    "url": f"postgresql://boxmot:{password}@{RELEASE}-postgres:5432/boxmot",
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": "test-camera"},
                "stringData": {"url": "http://fixture:8090/feed"},
            },
        )
        self.fixture()
        self.install(first=True)
        if self.args.detector_weights:
            self.volume_pod("seed-models", f"{RELEASE}-{CAMERA_ID}-models", "/models")
            self.kubectl("cp", str(Path(self.args.detector_weights).resolve()), "seed-models:/models/yolo26n.pt")
            self.kubectl("delete", "pod", "seed-models", "--wait=true")
        self.values["cameraDefaults"]["suspend"] = False
        self.install()
        self.runtime_version = self.kubectl(
            "exec",
            f"deployment/{RELEASE}-{CAMERA_ID}",
            "--",
            "python",
            "-c",
            "from importlib.metadata import version; print(version('boxmot'))",
        ).stdout.strip()
        assert self.runtime_version == str(self.chart["appVersion"]), (
            f"Image package version {self.runtime_version} differs from chart appVersion {self.chart['appVersion']}"
        )
        run_id = wait_for(self.run_id, "No active camera session")
        wait_for(lambda: self.completed(run_id), "No completed real attributes in PostgreSQL and export", 1800)
        self.mark("Helm install → network fixture → detector/tracker → real VLM → PostgreSQL/export")
        claims = self.pvc_ids()
        before = self.sql(
            "SELECT id,payload,delivered_at FROM boxmot_attributes.events WHERE payload->>'status'='completed'"
        )
        pod_uid = self.camera_pods()[0]["metadata"]["uid"]

        self.values["cameraDefaults"]["config"]["ready_timeout_s"] = 31
        self.install()
        upgraded_run = wait_for(
            lambda: current if (current := self.run_id()) and current != run_id else None,
            "Config upgrade did not start a new session",
        )
        assert self.camera_pods()[0]["metadata"]["uid"] != pod_uid
        assert self.pvc_ids() == claims, "Upgrade replaced persistent storage"
        after = self.sql(
            "SELECT id,payload,delivered_at FROM boxmot_attributes.events WHERE payload->>'status'='completed'"
        )
        assert all(row in after for row in before), "Upgrade changed existing attribute events"
        wait_for(lambda: self.completed(upgraded_run), "No real attributes after Helm upgrade", 1800)
        self.mark("Helm config upgrade restarts camera, preserves PVCs/history, and completes new attributes")

        self.kubectl("scale", "deployment/fixture", "--replicas=0")
        wait_for(lambda: not self.ready(), "Source outage did not withdraw readiness", 90)
        self.kubectl(
            "exec",
            f"deployment/{RELEASE}-{CAMERA_ID}",
            "--",
            "python",
            "-c",
            "import urllib.request; "
            "assert urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=5).status == 200",
        )
        self.kubectl("scale", "deployment/fixture", "--replicas=1")
        wait_for(self.ready, "Camera failed to reconnect", 180)
        reconnected_run = wait_for(
            lambda: current if (current := self.run_id()) and current != upgraded_run else None,
            "Reconnect did not create a fresh session",
        )
        self.mark("Camera disconnect/reconnect changes session without killing supervision")

        pods = self.camera_pods()
        assert len(pods) == 1
        self.kubectl("delete", "pod", pods[0]["metadata"]["name"], "--wait=true")
        wait_for(self.ready, "Camera pod replacement failed", 180)
        replacement_run = wait_for(
            lambda: current if (current := self.run_id()) and current != reconnected_run else None,
            "Pod replacement reused old session",
        )
        self.mark("Camera pod replacement recovers ownership with a fresh session")

        self.kubectl("scale", f"deployment/{RELEASE}-postgres", "--replicas=0")
        wait_for(lambda: not self.ready(), "Database loss did not withdraw camera readiness", 90)
        self.kubectl("scale", f"deployment/{RELEASE}-postgres", "--replicas=1")
        self.kubectl("rollout", "status", f"deployment/{RELEASE}-postgres", "--timeout=180s", timeout=200)
        wait_for(self.ready, "Camera did not recover after database restart", 240)
        recovered_run = wait_for(
            lambda: current if (current := self.run_id()) and current != replacement_run else None,
            "Database recovery did not create a new session",
        )
        wait_for(lambda: self.completed(recovered_run), "No completed attributes after recovery", 1800)
        self.expected_sessions = {run_id, upgraded_run, recovered_run}
        assert self.pvc_ids() == claims
        self.mark("Database restart preserves storage and resumes real attribute generation")

        self.values["cameraDefaults"]["suspend"] = True
        self.install()
        wait_for(lambda: not self.camera_pods(), "Camera did not stop before output verification", 90)
        for source, name in (
            (f"{RELEASE}-{CAMERA_ID}-out-gc", "output-maintenance"),
            (f"{RELEASE}-db-gc", "database-maintenance"),
        ):
            self.kubectl("create", "job", name, f"--from=cronjob/{source}")
            self.kubectl("wait", "--for=condition=complete", f"job/{name}", "--timeout=300s", timeout=320)
        self.verify_outputs()
        self.mark("Maintenance jobs complete; exported tracks/attributes match persisted outcomes")
        self.command(
            [
                "helm",
                "uninstall",
                RELEASE,
                "--kube-context",
                self.args.context,
                "--namespace",
                self.namespace,
                "--wait",
                "--timeout",
                "3m",
            ],
            timeout=200,
        )
        assert self.pvc_ids() == claims, "Uninstall deleted retained PVCs"
        self.mark("Helm uninstall retains every data/model PVC")

    def verify_outputs(self) -> None:
        """Require exact JSONL/database correspondence, allowing consistent replay IDs."""
        events = self.sql("SELECT run_id,id,payload,delivered_at FROM boxmot_attributes.events ORDER BY id")
        self.volume_pod("output-reader", f"{RELEASE}-{CAMERA_ID}-outputs", "/outputs")
        script = (
            "from pathlib import Path; import json; print(json.dumps([json.loads(line) "
            "for path in Path('/outputs').rglob('*.jsonl*') "
            "for line in path.read_text().splitlines() if line.strip()]))"
        )
        rows = json.loads(self.kubectl("exec", "output-reader", "--", "python", "-c", script).stdout)
        self.kubectl("delete", "pod", "output-reader", "--wait=true")
        tracks = [row for row in rows if "tracks" in row]
        assert any(row["tracks"] for row in tracks), "No detector/tracker output"
        assert self.expected_sessions <= {row["session_id"] for row in tracks if row["tracks"]}, (
            "Missing tracks for a successful attribute session"
        )
        exported: dict[str, dict[str, Any]] = {}
        for row in rows:
            if "event_id" not in row:
                continue
            event_id = row["event_id"]
            assert event_id not in exported or exported[event_id] == row, "Conflicting event replay"
            exported[event_id] = row
        completed = [row for row in events if row["payload"].get("status") == "completed"]
        assert completed, "No completed persisted attribute events"
        for event in events:
            assert event["delivered_at"] is not None, "Stopped fixture retained unexported events"
            event_id = f"{event['run_id']}:{event['id']}"
            assert event_id in exported, f"Missing event {event_id}"
            row = exported[event_id]
            assert row["camera_id"] == CAMERA_ID and row["session_id"] == event["run_id"]
            payload = {
                key: value
                for key, value in row.items()
                if key not in {"event_id", "camera_id", "session_id", "created_at"}
            }
            assert payload == event["payload"], f"Changed event {event_id}"
        assert set(exported) == {f"{event['run_id']}:{event['id']}" for event in events}
        summary = {
            "track_frames": len(tracks),
            "persisted_events": len(events),
            "completed_events": len(completed),
            "verified_events": len(exported),
            "session_ids": sorted({row["session_id"] for row in tracks}),
        }
        (self.directory / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")

    def finish(self, error: str | None) -> None:
        """Collect public evidence and delete only the namespace bearing our owner token."""
        cleanup_error = None
        diagnostic_errors = []
        if self.owned:
            for command in (
                ("get", "pods,pvc,deployments,jobs,events", "-o", "json"),
                (
                    "logs",
                    "-l",
                    "app.kubernetes.io/instance=smoke",
                    "--all-containers",
                    "--prefix",
                    "--tail=300",
                    "--max-log-requests=10",
                ),
            ):
                try:
                    self.kubectl(*command, check=False)
                except Exception as exc:
                    diagnostic_errors.append(f"{type(exc).__name__}: {exc}")
        try:
            if self.owned:
                namespace = json.loads(self.kubectl("get", "namespace", self.namespace, "-o", "json").stdout)
                assert namespace["metadata"]["labels"].get(OWNER_LABEL) == self.owner, (
                    "Namespace owner changed; refusing cleanup"
                )
                self.kubectl("delete", "namespace", self.namespace, "--wait=true", "--timeout=180s", timeout=200)
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
        result = {
            "passed": self.passed,
            "error": error,
            "cleanup_error": cleanup_error,
            "diagnostic_errors": diagnostic_errors,
            "namespace": self.namespace,
            "context": self.args.context,
            "commands": self.commands,
        }
        (self.directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if not error and not cleanup_error and not self.args.prepare_only:
            evidence = {
                "schema_version": 1,
                "camera_image": self.args.image,
                "vllm_image": self.values["vllm"]["image"],
                "model_revision": self.defaults["vllm"]["revision"],
                "chart_version": self.chart["version"],
                "app_version": self.chart["appVersion"],
                "runtime_app_version": self.runtime_version,
                "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "validation_run_id": os.environ.get("GITHUB_RUN_ID", ""),
                "validation_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
                "external_vllm": bool(self.args.external_vllm_url),
            }
            (self.directory / "deployment-images.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"Evidence: {self.directory}", flush=True)
        if cleanup_error:
            raise RuntimeError(f"Test namespace cleanup failed: {cleanup_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--image", required=True, help="Immutable BoxMOT CPU image to test.")
    parser.add_argument("--vllm-image", help="Immutable vLLM CPU image; defaults to chart image.")
    parser.add_argument("--namespace", help="New boxmot-helm-e2e-* namespace (never an existing one).")
    parser.add_argument("--output", help="New artifact directory; defaults to runs/<namespace>.")
    parser.add_argument("--storage-class", default="", help="Cluster StorageClass (default: cluster default).")
    parser.add_argument(
        "--detector-weights", type=Path, help="Seed local yolo26n.pt into the isolated camera model claim."
    )
    parser.add_argument(
        "--external-vllm-url",
        help="Reuse an existing endpoint for local chart testing; evidence is ineligible for image promotion.",
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Generate fixture/values and render without accessing Kubernetes."
    )
    args = parser.parse_args()
    if args.detector_weights and not args.detector_weights.is_file():
        parser.error("--detector-weights must be an existing YOLO26n weights file")
    test = HelmTest(args)
    error = None

    def terminate(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("Terminated; collecting evidence and removing test namespace")

    signal.signal(signal.SIGTERM, terminate)
    try:
        test.prepare()
        if not args.prepare_only:
            test.run()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        test.finish(error)


if __name__ == "__main__":
    main()
