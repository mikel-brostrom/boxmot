"""Rendered deployment contracts; kubectl is required for structural checks."""

import shutil

import pytest
import yaml

from deploy.kubernetes.render import render

DIGEST = "registry.example/boxmot@sha256:" + "a" * 64
VLM = "registry.example/vllm@sha256:" + "b" * 64


def test_renderer_rejects_mutable_images():
    with pytest.raises(ValueError, match="sha256"):
        render("cpu", "boxmot:latest", VLM)


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl with Kustomize is required")
@pytest.mark.parametrize("profile", ["cpu", "gpu"])
def test_overlay_ownership_storage_images_and_gpu_isolation(profile):
    objects = list(yaml.safe_load_all(render(profile, DIGEST, VLM)))
    deployments = {obj["metadata"]["name"]: obj for obj in objects if obj["kind"] == "Deployment"}
    assert set(deployments) == {"camera-01", "vllm"}
    for name, deployment in deployments.items():
        spec = deployment["spec"]
        assert spec["replicas"] == 1 and spec["strategy"]["type"] == "Recreate"
        pod = spec["template"]["spec"]
        assert pod["terminationGracePeriodSeconds"] == 30
        container = pod["containers"][0]
        assert container["image"] == (DIGEST if name == "camera-01" else VLM)
        assert "startupProbe" in container and "readinessProbe" in container
        assert container["resources"]["limits"].get("nvidia.com/gpu", 0) == (1 if profile == "gpu" else 0)
        if name == "camera-01":
            assert pod["securityContext"]["runAsNonRoot"]
        elif profile == "gpu":
            assert not any(e["name"].startswith("VLLM_CPU") for e in container["env"])
    assert len([o for o in objects if o["kind"] == "CronJob"]) == 2
    assert not any(o["kind"] in ("StatefulSet", "Secret") for o in objects)
    upgrade = list(yaml.safe_load_all(render(profile, DIGEST, VLM, upgrade=True)))
    assert len(upgrade) == 1 and upgrade[0]["kind"] == "Job"
    assert upgrade[0]["spec"]["activeDeadlineSeconds"] == 900
    pod = upgrade[0]["spec"]["template"]["spec"]
    assert pod["containers"][0]["command"] == ["boxmot", "stream-db", "upgrade"]
    assert "volumes" not in pod
