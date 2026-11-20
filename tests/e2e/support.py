"""Shared isolated-cluster fixture, command recording, and evidence lifecycle."""

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


class HelmHarness:
    """Own one disposable namespace shared by independent E2E scenarios."""

    promotable = False
    scenario = "support"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        weights = getattr(args, "detector_weights", None)
        if weights and not Path(weights).is_file():
            raise ValueError("--detector-weights must be an existing YOLO26n weights file")
        self.owner = uuid4().hex
        self.namespace = args.namespace or f"boxmot-helm-e2e-{self.owner[:10]}"
        if not re.fullmatch(r"boxmot-helm-e2e-[a-z0-9-]{1,35}", self.namespace):
            raise ValueError("Test namespaces must start with boxmot-helm-e2e- (maximum 51 characters).")
        self.directory = Path(args.output or ROOT / "runs" / self.namespace).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.commands: list[dict[str, Any]] = []
        self.passed: list[str] = []
        self.owned = False
        self.scenario_completed = False
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
        """Apply owned manifests without duplicating large fixtures in annotations."""
        self.kubectl(
            "apply", "--server-side", "--field-manager=boxmot-e2e", "-f", "-",
            input_text=json.dumps({"apiVersion": "v1", "kind": "List", "items": list(objects)}),
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
            "cameras": {CAMERA_ID: {"sourceSecret": {"name": "test-camera", "key": CAMERA_ID}}},
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

    def fixture_images(self) -> list[Path]:
        """Return the bounded JPEG inputs mounted by this scenario."""
        return [self.directory / "person.jpg"]

    def fixture(self) -> None:
        """Expose only the repeatable, bundled pedestrian image inside the test namespace."""
        self.apply(
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "fixture"},
                "data": {"camera.py": (Path(__file__).parent / "fixtures/camera.py").read_text()},
                "binaryData": {
                    path.name: base64.b64encode(path.read_bytes()).decode() for path in self.fixture_images()
                },
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

    def pvc_ids(self) -> dict[str, str]:
        items = json.loads(self.kubectl("get", "pvc", "-o", "json").stdout)["items"]
        assert not any(item["metadata"].get("deletionTimestamp") for item in items), "Retained PVC is being deleted"
        return {item["metadata"]["name"]: item["metadata"]["uid"] for item in items}

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
            "scenario": self.scenario,
            "completed": self.scenario_completed,
            "passed": self.passed,
            "error": error,
            "cleanup_error": cleanup_error,
            "diagnostic_errors": diagnostic_errors,
            "namespace": self.namespace,
            "context": self.args.context,
            "commands": self.commands,
        }
        (self.directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if self.promotable and self.scenario_completed and self.owned and not error and not cleanup_error:
            evidence = {
                "schema_version": 1,
                "scenario": self.scenario,
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

    def start(self) -> None:
        """Create this run's namespace, dependencies, and camera workers."""
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
                "stringData": {camera: f"http://fixture:8090/{camera}" for camera in self.values["cameras"]},
            },
        )
        self.fixture()
        self.install(first=True)
        if self.args.detector_weights:
            for camera in self.values["cameras"]:
                self.volume_pod("seed-models", f"{RELEASE}-{camera}-models", "/models")
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


def argument_parser(description: str | None) -> argparse.ArgumentParser:
    """Common immutable-image, storage, and namespace options for E2E scenarios."""
    parser = argparse.ArgumentParser(description=description)
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
    return parser


def run_scenario(test: HelmHarness) -> None:
    """Always collect evidence and clean owned resources, including on interruption."""
    error = None

    def terminate(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("Terminated; collecting evidence and removing test namespace")

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        test.prepare()
        if not test.args.prepare_only:
            test.run()
            test.scenario_completed = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            test.finish(error)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
