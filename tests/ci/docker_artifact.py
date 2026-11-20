"""Bind a saved, smoke-tested Docker image to its exact release source and bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def archive_sha256(path: Path) -> str:
    """Hash an image archive without reading its potentially large layers into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_identity(source_sha: str, version: str, target: str, image: str) -> dict[str, str | int]:
    """Describe the release identity expected by both the gate and publisher."""
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("Candidate source must be a full Git SHA.")
    return {"schema_version": 1, "source_sha": source_sha, "version": version, "target": target, "image": image}


def record_candidate(directory: Path, identity: dict[str, str | int], image_id: str) -> None:
    """Record provenance only after the workflow has passed every image smoke test."""
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("Invalid Docker image ID.")
    payload = {**identity, "image_id": image_id, "archive_sha256": archive_sha256(directory / "image.tar.gz")}
    (directory / "candidate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify_candidate(directory: Path, identity: dict[str, str | int]) -> str:
    """Reject different revisions, targets, versions, or bytes before Docker loads them."""
    payload = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {*identity, "image_id", "archive_sha256"}:
        raise ValueError("Invalid Docker candidate manifest.")
    for key, expected in identity.items():
        if payload[key] != expected:
            raise ValueError(f"Docker candidate {key} does not match the requested release.")
    image_id = payload["image_id"]
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("Invalid Docker candidate image ID.")
    if archive_sha256(directory / "image.tar.gz") != payload["archive_sha256"]:
        raise ValueError("Docker candidate archive checksum mismatch.")
    return image_id


def main() -> None:
    """Record or verify a workflow artifact using only Python's standard library."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "verify"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-id")
    args = parser.parse_args()
    identity = candidate_identity(args.source_sha, args.version, args.target, args.image)
    if args.command == "record":
        if args.image_id is None:
            parser.error("record requires --image-id")
        record_candidate(args.directory, identity, args.image_id)
    else:
        print(verify_candidate(args.directory, identity))


if __name__ == "__main__":
    main()
