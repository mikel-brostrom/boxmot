"""Render deployable YAML only after supplying immutable image references."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


def render(profile: str, camera_image: str, vlm_image: str, *, upgrade: bool = False) -> str:
    """Require real digests and render a separate explicit schema-upgrade Job."""
    for value in (camera_image, vlm_image):
        if not re.fullmatch(r"[^\s@]+@sha256:[a-fA-F0-9]{64}", value):
            raise ValueError("Images must be repository@sha256:<64 hex digits> references.")
    path = Path(__file__).parent / "overlays" / profile
    rendered = subprocess.run(["kubectl", "kustomize", str(path)], check=True, capture_output=True, text=True).stdout
    objects = list(yaml.safe_load_all(rendered))
    replacements = {"boxmot-runtime": camera_image, "vllm-runtime": vlm_image}

    def replace_images(value: object) -> None:
        if isinstance(value, dict):
            if "image" in value:
                value["image"] = replacements.get(value["image"], value["image"])
            for child in value.values():
                replace_images(child)
        elif isinstance(value, list):
            for child in value:
                replace_images(child)

    for obj in objects:
        replace_images(obj)
    if upgrade:
        maintenance = next(
            obj
            for obj in objects
            if obj.get("kind") == "CronJob" and obj["metadata"]["name"] == "attribute-maintenance"
        )
        spec = maintenance["spec"]["jobTemplate"]["spec"]
        spec["activeDeadlineSeconds"] = 900
        spec["template"]["spec"]["containers"][0]["command"] = ["boxmot", "stream-db", "upgrade"]
        objects = [
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": "attribute-schema-upgrade", "namespace": "boxmot-streams"},
                "spec": spec,
            }
        ]
    # JSON documents are valid YAML and preserve every scalar without quoting surprises.
    return "\n---\n".join(json.dumps(obj, indent=2) for obj in objects) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--camera-image", required=True)
    parser.add_argument("--vlm-image", required=True)
    parser.add_argument("--upgrade", action="store_true")
    args = parser.parse_args()
    sys.stdout.write(render(args.profile, args.camera_image, args.vlm_image, upgrade=args.upgrade))


if __name__ == "__main__":
    main()
