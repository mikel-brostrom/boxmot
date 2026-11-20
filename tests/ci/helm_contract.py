"""Render the actual chart and check deployment invariants without a cluster.

Run ``python -m tests.ci.helm_contract`` from the repository root with Helm,
PyYAML and platformdirs. Synthetic digests are never used for deployment.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from boxmot.attributes.config import resolve_attribute_config
from boxmot.engine.config.streams import load_stream_config

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy/helm/boxmot"
IMAGE = "registry.example/boxmot@sha256:" + "a" * 64
GPU_IMAGE = "registry.example/boxmot-gpu@sha256:" + "b" * 64
VLLM_GPU_IMAGE = "registry.example/vllm-gpu@sha256:" + "c" * 64


def render(
    values: dict[str, Any] | None = None,
    *,
    release: str = "test",
    files: tuple[Path, ...] = (),
    valid: bool = True,
) -> list[dict[str, Any]]:
    """Exercise Helm's real merge, schema validation, and template engine."""
    with tempfile.TemporaryDirectory(prefix="boxmot-helm-contract-") as directory:
        override = Path(directory) / "values.json"
        override.write_text(json.dumps({"image": IMAGE, **(values or {})}))
        command = ["helm", "template", release, str(CHART), "--namespace", "test-streams"]
        for file in files:
            command.extend(["--values", str(file)])
        command.extend(["--values", str(override)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if not valid:
        assert result.returncode != 0, f"Helm accepted invalid configuration: {values}"
        assert "Error" in result.stderr, result.stderr
        return []
    assert result.returncode == 0, result.stderr
    objects = [obj for obj in yaml.safe_load_all(result.stdout) if obj]
    check_stream_configs(objects)
    return objects


def check_stream_configs(objects: list[dict[str, Any]]) -> None:
    """Load generated YAML through the worker's real, lightweight config boundary."""
    with tempfile.TemporaryDirectory(prefix="boxmot-stream-config-") as directory:
        for index, obj in enumerate(objects):
            if obj["kind"] != "ConfigMap" or "stream.yaml" not in obj.get("data", {}):
                continue
            path = Path(directory) / f"camera-{index}.yaml"
            path.write_text(obj["data"]["stream.yaml"])
            config = load_stream_config(path)
            args = config.tracking_args()
            assert not args.show and not args.save and not args.save_json and not args.save_txt
            # Exercise profile/inline schema validation and endpoint substitution
            # exactly as run_stream does, without loading models or opening sockets.
            attributes = resolve_attribute_config(config.attributes)
            replace(attributes, host=config.attributes_host)


def pod_spec(obj: dict[str, Any]) -> dict[str, Any]:
    """Select the embedded workload pod for deployments and maintenance jobs."""
    spec = obj["spec"]
    if obj["kind"] == "CronJob":
        spec = spec["jobTemplate"]["spec"]
    return spec["template"]["spec"]


def check_workloads(objects: list[dict[str, Any]]) -> None:
    """Preserve ownership, resource, probe, credential, and persistence contracts."""
    keys = [(obj["kind"], obj["metadata"]["name"]) for obj in objects]
    assert len(keys) == len(set(keys)), "Generated resource names collide"
    assert not any(obj["kind"] in ("Secret", "HorizontalPodAutoscaler", "Namespace") for obj in objects)
    for obj in objects:
        assert len(obj["metadata"]["name"]) <= 63
        if obj["kind"] == "PersistentVolumeClaim":
            assert obj["metadata"]["annotations"]["helm.sh/resource-policy"] == "keep"
        if obj["kind"] not in ("Deployment", "StatefulSet", "Job", "CronJob"):
            continue
        pod = pod_spec(obj)
        assert pod["automountServiceAccountToken"] is False
        assert pod["enableServiceLinks"] is False
        for container in pod["containers"]:
            assert re.fullmatch(r"\S+@sha256:[a-fA-F0-9]{64}", container["image"])
            assert "envFrom" not in container, "Secret keys must be selected individually"
            for bound in ("requests", "limits"):
                assert {"cpu", "memory"} <= container["resources"][bound].keys()
        if obj["kind"] == "Deployment":
            assert obj["spec"]["replicas"] in (0, 1)
            assert obj["spec"]["strategy"]["type"] == "Recreate"
            labels = obj["spec"]["template"]["metadata"]["labels"]
            assert all(labels.get(key) == value for key, value in obj["spec"]["selector"]["matchLabels"].items())
            for container in pod["containers"]:
                assert all(probe in container for probe in ("startupProbe", "readinessProbe", "livenessProbe"))
                if container["name"] == "camera":
                    assert pod["securityContext"]["runAsNonRoot"] is True
                    for probe, path in (
                        ("startupProbe", "/startupz"),
                        ("readinessProbe", "/readyz"),
                        ("livenessProbe", "/healthz"),
                    ):
                        assert container[probe]["httpGet"]["path"] == path
        if obj["kind"] == "CronJob":
            assert len(obj["metadata"]["name"]) <= 52
            assert obj["spec"]["concurrencyPolicy"] == "Forbid"
            assert obj["spec"]["jobTemplate"]["spec"]["activeDeadlineSeconds"] > 0


def by_name(objects: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    """Require exactly one named resource."""
    matches = [obj for obj in objects if obj["kind"] == kind and obj["metadata"]["name"] == name]
    assert len(matches) == 1, (kind, name, matches)
    return matches[0]


def check() -> None:
    """Check independent hardware/service choices, upgrades, and rejected inputs."""
    subprocess.run(["helm", "lint", "--strict", str(CHART), "--set-string", f"image={IMAGE}"], check=True, timeout=30)
    default = render()
    check_workloads(default)
    assert not any(obj["metadata"]["name"] == "test-postgres" for obj in default)
    assert not any(obj["kind"] == "Job" for obj in default), "Ordinary upgrades must not migrate the database"
    vllm = pod_spec(by_name(default, "Deployment", "test-vllm"))["containers"][0]
    assert "--revision" in vllm["args"]
    assert re.fullmatch(r"[a-f0-9]{40}", vllm["args"][vllm["args"].index("--revision") + 1])
    configs = [obj for obj in default if obj["kind"] == "ConfigMap" and "stream.yaml" in obj.get("data", {})]
    assert len(configs) == 1
    stream = yaml.safe_load(configs[0]["data"]["stream.yaml"])
    assert stream["camera_id"] == "camera-01"
    assert stream["attributes_host"] == "http://test-vllm:8000"
    assert stream["source_url_env"] == "BOXMOT_CAMERA_URL"
    assert stream["timestamp_source"] == "capture"
    assert not any(obj["kind"] in {"PodMonitor", "PrometheusRule"} for obj in default)

    multiple = render({"cameras": {"camera-02": {"sourceSecret": {"name": "camera-two", "key": "source"}}}})
    check_workloads(multiple)
    assert len([obj for obj in multiple if obj["kind"] == "CronJob"]) == 3
    database_job = pod_spec(by_name(multiple, "CronJob", "test-db-gc"))["containers"][0]
    assert database_job["command"][:3] == ["boxmot", "stream-db", "maintain"]
    for camera in ("camera-01", "camera-02"):
        output_job = pod_spec(by_name(multiple, "CronJob", f"test-{camera}-out-gc"))["containers"][0]
        assert output_job["command"][:3] == ["boxmot", "stream-outputs", "maintain"]
        assert output_job["command"][-2:] == ["--camera-id", camera]
    second = pod_spec(by_name(multiple, "Deployment", "test-camera-02"))
    secrets = {
        item["name"]: item["valueFrom"]["secretKeyRef"]
        for item in second["containers"][0]["env"]
        if "valueFrom" in item
    }
    assert secrets["BOXMOT_CAMERA_URL"] == {"name": "camera-two", "key": "source"}
    assert secrets["BOXMOT_ATTRIBUTES_DATABASE_URL"] == {"name": "attributes-database", "key": "url"}
    claims = {
        item["persistentVolumeClaim"]["claimName"] for item in second["volumes"] if "persistentVolumeClaim" in item
    }
    assert claims == {"test-camera-02-models", "test-camera-02-outputs"}

    external = render(
        {
            "vllm": {"enabled": False, "externalEndpoint": "https://models.example.test"},
            "database": {"existingSecret": "managed-postgres", "urlKey": "dsn"},
            "cameraDefaults": {
                "models": {"existingClaim": "preloaded-models"},
                "outputs": {"existingClaim": "durable-outputs"},
            },
        }
    )
    check_workloads(external)
    assert not any(obj["kind"] == "PersistentVolumeClaim" for obj in external)
    assert not any(obj["metadata"]["name"] == "test-vllm" for obj in external)

    mixed_values = {
        "vllm": {
            "image": VLLM_GPU_IMAGE,
            "env": {"VLLM_CPU_OMP_THREADS_BIND": None, "VLLM_CPU_NUM_OF_RESERVED_CPU": None},
            "resources": {"limits": {"nvidia.com/gpu": 1}},
            "nodeSelector": {"accelerator": "nvidia"},
        }
    }
    mixed = render(mixed_values)
    check_workloads(mixed)
    assert (
        "nvidia.com/gpu"
        not in pod_spec(by_name(mixed, "Deployment", "test-camera-01"))["containers"][0]["resources"]["limits"]
    )
    assert (
        pod_spec(by_name(mixed, "Deployment", "test-vllm"))["containers"][0]["resources"]["limits"]["nvidia.com/gpu"]
        == 1
    )
    both_gpu = render(
        {
            **mixed_values,
            "image": GPU_IMAGE,
            "cameraDefaults": {
                "config": {"tracking": {"device": "0"}},
                "resources": {"limits": {"nvidia.com/gpu": 1}},
                "nodeSelector": {"accelerator": "nvidia"},
            },
        }
    )
    check_workloads(both_gpu)
    for component in ("camera-01", "vllm"):
        pod = pod_spec(by_name(both_gpu, "Deployment", f"test-{component}"))
        assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 1
        assert pod["nodeSelector"] == {"accelerator": "nvidia"}

    local = render({"postgres": {"enabled": True}})
    check_workloads(local)
    assert by_name(local, "Service", "test-postgres")["spec"]["type"] == "ClusterIP"

    secured = render(
        {
            "postgres": {"enabled": True},
            "networkPolicy": {"enabled": True},
            "monitoring": {
                "serviceMonitor": {"enabled": True, "labels": {"release": "prometheus"}},
                "podMonitor": {"enabled": True, "labels": {"release": "prometheus"}},
                "rules": {"enabled": True, "labels": {"release": "prometheus"}},
            },
        }
    )
    check_workloads(secured)
    policy = by_name(secured, "NetworkPolicy", "test-postgres")["spec"]
    assert policy["policyTypes"] == ["Ingress"]
    assert policy["ingress"][0]["ports"] == [{"protocol": "TCP", "port": 5432}]
    clients = policy["ingress"][0]["from"][0]["podSelector"]
    assert clients["matchLabels"]["app.kubernetes.io/instance"] == "test"
    assert set(clients["matchExpressions"][0]["values"]) == {"camera", "maintenance", "output-maintenance", "migration"}
    monitor = by_name(secured, "ServiceMonitor", "test-vllm")
    service = by_name(secured, "Service", "test-vllm")
    assert all(
        service["metadata"]["labels"].get(key) == value
        for key, value in monitor["spec"]["selector"]["matchLabels"].items()
    )
    assert monitor["spec"]["endpoints"][0]["path"] == "/metrics"
    assert monitor["spec"]["endpoints"][0]["port"] == service["spec"]["ports"][0]["name"]
    camera_monitor = by_name(secured, "PodMonitor", "test-cameras")
    camera_pod = by_name(secured, "Deployment", "test-camera-01")["spec"]["template"]
    assert all(
        camera_pod["metadata"]["labels"].get(key) == value
        for key, value in camera_monitor["spec"]["selector"]["matchLabels"].items()
    )
    endpoint = camera_monitor["spec"]["podMetricsEndpoints"][0]
    assert endpoint["port"] == camera_pod["spec"]["containers"][0]["ports"][0]["name"]
    assert endpoint["path"] == "/metrics"
    rules = by_name(secured, "PrometheusRule", "test-cameras")["spec"]["groups"][0]["rules"]
    assert {item["alert"] for item in rules} == {
        "BoxmotCameraUnavailable", "BoxmotCameraScrapeFailed", "BoxmotAttributeBacklogOld",
        "BoxmotCameraFrameDropsHigh", "BoxmotAttributeFailures",
    }
    assert all('namespace="test-streams",boxmot_release="test"' in item["expr"] for item in rules)

    # Filesystem retention does not trigger database cleanup and can differ per camera.
    independent_retention = render({"cameraDefaults": {"config": {"retention_days": 1, "timestamp_source": "media"}}})
    output_job = pod_spec(by_name(independent_retention, "CronJob", "test-camera-01-out-gc"))["containers"][0]
    assert output_job["command"][output_job["command"].index("--retention-days") + 1] == "1"

    attributes = {
        "backend": "vllm",
        "model": "Qwen/Qwen3.5-9B",
        "api_key_env": "VLM_API_KEY",
        "prompt": "Describe visible clothing colors.",
        "schema": {"type": "object", "properties": {"color": {"type": "string"}}},
    }
    authenticated = render(
        {
            "vllm": {
                "model": attributes["model"],
                "revision": "d" * 40,
                "secretEnv": {"VLLM_API_KEY": {"name": "model-auth", "key": "api-key"}},
            },
            "cameraDefaults": {
                "config": {"attributes": attributes},
                "secretEnv": {"VLM_API_KEY": {"name": "model-auth", "key": "api-key"}},
            },
        }
    )
    check_workloads(authenticated)
    for component, variable in (("camera-01", "VLM_API_KEY"), ("vllm", "VLLM_API_KEY")):
        env = pod_spec(by_name(authenticated, "Deployment", f"test-{component}"))["containers"][0]["env"]
        entry = next(item for item in env if item["name"] == variable)
        assert entry == {"name": variable, "valueFrom": {"secretKeyRef": {"name": "model-auth", "key": "api-key"}}}

    maintenance = render(
        {
            "cameraDefaults": {"suspend": True},
            "maintenance": {"suspend": True},
            "migration": {"enabled": True, "id": "v2"},
        }
    )
    check_workloads(maintenance)
    migration = by_name(maintenance, "Job", "test-db-upgrade-v2")
    assert "helm.sh/hook" not in migration["metadata"].get("annotations", {})
    assert pod_spec(migration)["containers"][0]["command"] == ["boxmot", "stream-db", "upgrade"]
    assert migration["spec"]["activeDeadlineSeconds"] == 900

    original = by_name(default, "Deployment", "test-camera-01")["spec"]
    changed = render({"cameraDefaults": {"config": {"tracking": {"imgsz": [480, 480]}}}})
    updated = by_name(changed, "Deployment", "test-camera-01")["spec"]
    assert original["selector"] == updated["selector"]
    assert original["template"]["metadata"]["annotations"] != updated["template"]["metadata"]["annotations"]
    rotated = render({"secretRevision": "rotation-2"})
    assert (
        original["template"]["metadata"]["annotations"]
        != by_name(rotated, "Deployment", "test-camera-01")["spec"]["template"]["metadata"]["annotations"]
    )

    # Releases must not target each other's pods or claims, even for the same camera ID.
    other = render(release="another")
    other_camera = by_name(other, "Deployment", "another-camera-01")["spec"]
    assert other_camera["selector"] != original["selector"]
    assert other_camera["template"]["spec"]["volumes"] != original["template"]["spec"]["volumes"]
    # CronJob names have a stricter API limit than other resource names.
    boundary = {"cameras": {"camera-01": None, "c" * 20: {"sourceSecret": {"name": "camera", "key": "url"}}}}
    check_workloads(render(boundary, release="r" * 24))
    render(boundary, release="r" * 25, valid=False)

    invalid = [
        {"image": "boxmot/boxmot:latest"},
        {"image": ""},
        {"image": IMAGE + "garbage"},
        {"vllm": {"image": "vllm:latest"}},
        {"vllm": {"revision": "main"}},
        {"vllm": {"enabled": False, "externalEndpoint": ""}},
        {"vllm": {"externalEndpoint": "https://two-servers.example"}},
        {"cameras": {"BAD_CAMERA": {"sourceSecret": {"name": "camera", "key": "url"}}}},
        {"cameras": {"camera-01": {"replicas": 2}}},
        {"cameras": {"camera-01": {"sourceSecret": {"name": "", "key": "url"}}}},
        {"cameraDefaults": {"config": {"camera_id": "override"}}},
        {"cameraDefaults": {"config": {"max_pending_jobs": 0}}},
        {"migration": {"enabled": True}},
        {"cameraDefaults": {"suspend": True}, "migration": {"enabled": True}},
        {
            "cameraDefaults": {"suspend": True},
            "maintenance": {"suspend": True},
            "migration": {"enabled": True},
            "cameras": {"camera-01": {"suspend": False}},
        },
        {"cameraDefaults": {"config": {"timestamp_source": "fps"}}},
        {"cameras": {"vllm": {"sourceSecret": {"name": "camera", "key": "url"}}}},
        {"cameraDefaults": {"podLabels": {"app.kubernetes.io/instance": "another"}}},
        {"cameraDefaults": {"env": {"BOXMOT_ATTRIBUTES_DATABASE_URL": "postgresql://unsafe"}}},
        {"vllm": {"args": ["--revision=main"]}},
        {"vllm": {"args": ["--rev=main"]}},
        {"vllm": {"args": ["--served_model_name=alias"]}},
        {"vllm": {"args": ["--tokenizer-revision", "main"]}},
        {"vllm": {"args": ["--port", "9999"]}},
        {"vllm": {"args": ["--served-model-name", "different"]}},
        {"vllm": {"model": "Qwen/Qwen3.5-9B"}},
        {"cameraDefaults": {"config": {"attributes": "person-appearance-ollama"}}},
        {"cameraDefaults": {"config": {"attributes": attributes}}},
        {"cameraDefaults": {"secretEnv": {"BOXMOT_CAMERA_URL": {"name": "override", "key": "url"}}}},
        {
            "cameraDefaults": {
                "env": {"VLM_API_KEY": "literal"},
                "secretEnv": {"VLM_API_KEY": {"name": "model-auth", "key": "api-key"}},
            }
        },
        {
            "monitoring": {"serviceMonitor": {"enabled": True}},
            "vllm": {"enabled": False, "externalEndpoint": "https://models.example.test"},
        },
        {"monitoring": {"serviceMonitor": {"enabled": True, "scrapeTimeout": "60s", "interval": "30s"}}},
        {"monitoring": {"podMonitor": {"enabled": True, "scrapeTimeout": "60s", "interval": "30s"}}},
        {"monitoring": {"rules": {"enabled": True}}},
        {"monitoring": {"rules": {"maxDroppedFrameRatio": 2}}},
        {
            "cameraDefaults": {"serviceAccount": {"create": True, "name": "duplicate"}},
            "vllm": {"serviceAccount": {"create": True, "name": "duplicate"}},
        },
    ]
    for values in invalid:
        render(values, valid=False)
    examples = sorted((ROOT / "deploy/helm/examples").glob("*.yaml"))
    for example in examples:
        # GPU examples require an operator-supplied serving image. Use a fake
        # digest only for rendering; this check never pulls or starts it.
        values = yaml.safe_load(example.read_text())
        overrides = {"vllm": {"image": VLLM_GPU_IMAGE}} if values.get("vllm", {}).get("image") == "" else {}
        check_workloads(render(overrides, files=(example,)))
    print("Helm deployment contracts passed.")


if __name__ == "__main__":
    check()
