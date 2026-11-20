"""Validate immutable deployment releases without importing the application.

Run from the repository root with ``python -m tests.ci.deployment_release``.
Only chart metadata inspection needs PyYAML; image tag checks use the stdlib.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

STABLE_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
COMMIT = re.compile(r"[0-9a-f]{40}")
IMAGE = re.compile(r"[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}")


def latest_stable_release(release_tag: str, tags: list[str]) -> bool:
    """Only the numerically newest existing stable tag may update latest."""
    versions: dict[str, tuple[int, ...]] = {}
    for tag in tags:
        if tag.startswith("v") and STABLE_VERSION.fullmatch(tag[1:]):
            versions[tag] = tuple(map(int, tag[1:].split(".")))
    return bool(versions) and release_tag in versions and versions[release_tag] == max(versions.values())


def git(*arguments: str) -> str:
    """Read the current checkout; callers provide validated revision names."""
    return subprocess.check_output(["git", *arguments], text=True).strip()


def validate_chart_release(tag: str, source_sha: str, default_branch: str, chart: Path) -> dict[str, str]:
    """Tie a chart version to an existing tag and a commit on the default branch."""
    import yaml

    prefix = "boxmot-chart-"
    if not tag.startswith(prefix) or not STABLE_VERSION.fullmatch(tag[len(prefix) :]):
        raise ValueError("Chart tag must be boxmot-chart-X.Y.Z with a stable semantic version")
    if not COMMIT.fullmatch(source_sha):
        raise ValueError("source_sha must be a full lowercase commit SHA")
    if not default_branch or default_branch.startswith("-"):
        raise ValueError("A default branch name is required")
    if git("rev-parse", "HEAD") != source_sha:
        raise ValueError("The checked-out commit does not match source_sha")
    if git("rev-parse", f"refs/tags/{tag}^{{commit}}") != source_sha:
        raise ValueError("The chart tag does not point at source_sha")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_sha, f"refs/remotes/origin/{default_branch}"],
        check=True,
    )
    metadata = yaml.safe_load((chart / "Chart.yaml").read_text())
    version = tag[len(prefix) :]
    if metadata["name"] != "boxmot" or metadata["version"] != version:
        raise ValueError("Chart.yaml name/version does not match the requested release")
    return {"chart_version": version, "app_version": str(metadata["appVersion"]), "source_sha": source_sha}


def require_unused_chart_version(owner: str, version: str, username: str, password: str) -> None:
    """Fail closed on registry errors and reject existing GHCR chart versions."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", owner) or not STABLE_VERSION.fullmatch(version):
        raise ValueError("Invalid chart owner/version")
    if not username or not password:
        raise ValueError("GHCR credentials are required to check chart publication")
    repository = f"{owner}/charts/boxmot"
    query = urllib.parse.urlencode({"service": "ghcr.io", "scope": f"repository:{repository}:pull,push"})
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    request = urllib.request.Request(
        f"https://ghcr.io/token?{query}", headers={"Authorization": f"Basic {credentials}"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        token = json.load(response)["token"]
    request = urllib.request.Request(
        f"https://ghcr.io/v2/{repository}/manifests/{version}",
        method="HEAD",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.oci.image.manifest.v1+json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            pass
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return
        raise
    raise ValueError(f"Chart {repository}:{version} already exists; bump the chart version")


def validate_promotion(
    manifest: dict[str, Any], *, run_id: str, run_attempt: str, source_sha: str
) -> dict[str, Any]:
    """Accept only the immutable images recorded by the successful CPU Helm run."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported deployment evidence schema")
    if manifest.get("scenario") != "recovery":
        raise ValueError("Promotion requires successful recovery scenario evidence")
    for key, expected in (
        ("validation_run_id", run_id),
        ("validation_run_attempt", run_attempt),
        ("source_sha", source_sha),
    ):
        if str(manifest.get(key, "")) != expected:
            raise ValueError(f"Deployment evidence does not match {key}")
    if not COMMIT.fullmatch(source_sha) or not COMMIT.fullmatch(str(manifest.get("model_revision", ""))):
        raise ValueError("Chart source and model revision must be immutable commit SHAs")
    if manifest.get("external_vllm") is not False:
        raise ValueError("Promotion requires validation of the bundled vLLM image")
    camera_image = str(manifest.get("camera_image", ""))
    vllm_image = str(manifest.get("vllm_image", ""))
    if not IMAGE.fullmatch(camera_image) or not camera_image.startswith("boxmot/boxmot@sha256:"):
        raise ValueError("CPU promotion requires a tested boxmot/boxmot@sha256 image")
    if not IMAGE.fullmatch(vllm_image):
        raise ValueError("vLLM image must include its sha256 digest")
    if not STABLE_VERSION.fullmatch(str(manifest.get("chart_version", ""))):
        raise ValueError("Deployment evidence is missing the stable chart version")
    if not manifest.get("app_version"):
        raise ValueError("Deployment evidence is missing app_version")
    if manifest.get("runtime_app_version") != manifest["app_version"]:
        raise ValueError("The tested image package version does not match chart appVersion")
    return manifest


def main() -> None:
    """Expose the checks shared by chart publication and image promotion CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    latest = commands.add_parser("latest")
    latest.add_argument("--release-tag", required=True)
    chart = commands.add_parser("chart")
    chart.add_argument("--tag", required=True)
    chart.add_argument("--source-sha", required=True)
    chart.add_argument("--default-branch", required=True)
    chart.add_argument("--chart", type=Path, default=Path("deploy/helm/boxmot"))
    chart.add_argument("--output", type=Path, required=True)
    available = commands.add_parser("unused-chart")
    available.add_argument("--owner", required=True)
    available.add_argument("--version", required=True)
    promotion = commands.add_parser("promotion")
    promotion.add_argument("--manifest", type=Path, required=True)
    promotion.add_argument("--run-id", required=True)
    promotion.add_argument("--run-attempt", required=True)
    promotion.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    if args.command == "latest":
        print(str(latest_stable_release(args.release_tag, git("tag", "--list").splitlines())).lower())
    elif args.command == "chart":
        metadata = validate_chart_release(args.tag, args.source_sha, args.default_branch, args.chart)
        args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    elif args.command == "unused-chart":
        require_unused_chart_version(args.owner, args.version, os.environ["GHCR_USERNAME"], os.environ["GHCR_TOKEN"])
    elif args.command == "promotion":
        manifest = json.loads(args.manifest.read_text())
        validate_promotion(manifest, run_id=args.run_id, run_attempt=args.run_attempt, source_sha=args.source_sha)
        print(manifest["camera_image"])


if __name__ == "__main__":
    main()
