"""Lightweight provenance records for reproducible ReID training runs."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from boxmot.reid.core.artifacts import file_sha256, reid_code_sha256

_PRETRAINED_PROVENANCE_ATTRS = {
    "url": "pretrained_url",
    "sha256": "pretrained_sha256",
    "required_tensor_count": "pretrained_backbone_required_tensor_count",
    "matched_tensor_count": "pretrained_backbone_matched_tensor_count",
    "tensor_coverage": "pretrained_backbone_tensor_coverage",
    "required_numel": "pretrained_backbone_required_numel",
    "matched_numel": "pretrained_backbone_matched_numel",
    "numel_coverage": "pretrained_backbone_numel_coverage",
}

_ANATOMICAL_METADATA_SCHEMA = "anatomical-metadata-content-v1"
_ANATOMICAL_REFERENCED_ASSET_FIELDS = ("person_mask", "bag_mask")


def _configured_path_identity(path: str | Path | None) -> str | None:
    """Return a stable identity for one configured data root.

    Relative paths deliberately remain relative so the same recipe has the
    same identity on another machine. The configured identity, rather than
    only the resolved file bytes, is part of the contract: moving supervision
    data to a different root is an explicit resume decision.
    """
    if path is None or not str(path).strip():
        return None
    return Path(path).expanduser().as_posix()


def _content_sha256(path: Path) -> str | None:
    """Hash a file, returning ``None`` for a missing or unreadable asset."""
    try:
        return file_sha256(path) if path.is_file() else None
    except OSError:
        return None


def _update_manifest_digest(
    digest: Any,
    *values: Any,
) -> None:
    """Add one unambiguous canonical record to a streaming manifest hash."""
    payload = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def anatomical_metadata_provenance(
    metadata_dir: str | Path | None,
    person_mask_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fingerprint the complete anatomical-supervision input contract.

    The digest covers the configured roots, raw ``metadata.json`` bytes, every
    ``person_mask``/``bag_mask`` referenced by the manifest, and every ``.png``
    that can be selected from the optional external person-mask root. Hashing
    the full external candidate set is conservative, but lets ablation run
    discovery bind the inputs before the training dataset is constructed.

    Missing or malformed inputs are represented in the digest instead of
    raising here. Training's target-provider validation remains responsible
    for the user-facing error, while dry-run contract construction stays
    deterministic and checkpoint metadata can still explain the failure.
    """
    metadata_identity = _configured_path_identity(metadata_dir)
    person_mask_identity = _configured_path_identity(person_mask_dir)
    metadata_root = (
        None
        if metadata_identity is None
        else Path(metadata_dir).expanduser().resolve()
    )
    external_root = (
        None
        if person_mask_identity is None
        else Path(person_mask_dir).expanduser().resolve()
    )
    manifest_path = (
        None if metadata_root is None else metadata_root / "metadata.json"
    )
    try:
        manifest_bytes = (
            None
            if manifest_path is None or not manifest_path.is_file()
            else manifest_path.read_bytes()
        )
    except OSError:
        manifest_bytes = None
    manifest_sha256 = (
        None
        if manifest_bytes is None
        else hashlib.sha256(manifest_bytes).hexdigest()
    )

    digest = hashlib.sha256()
    _update_manifest_digest(
        digest,
        _ANATOMICAL_METADATA_SCHEMA,
        metadata_identity,
        person_mask_identity,
    )
    _update_manifest_digest(
        digest,
        "metadata.json",
        manifest_sha256,
    )

    referenced_assets: set[tuple[str, str]] = set()
    manifest_valid = False
    if manifest_bytes is not None:
        try:
            payload = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, TypeError):
            pass
        else:
            images = (
                payload.get("images")
                if isinstance(payload, dict)
                else None
            )
            manifest_valid = isinstance(images, dict)
            if manifest_valid:
                for record in images.values():
                    if not isinstance(record, dict):
                        continue
                    for field in _ANATOMICAL_REFERENCED_ASSET_FIELDS:
                        value = record.get(field)
                        if isinstance(value, str) and value.strip():
                            referenced_assets.add(
                                (field, Path(value).as_posix())
                            )
    _update_manifest_digest(digest, "manifest_valid", manifest_valid)

    missing_referenced_asset_count = 0
    for field, configured_asset in sorted(referenced_assets):
        asset_path = Path(configured_asset).expanduser()
        if not asset_path.is_absolute() and metadata_root is not None:
            asset_path = metadata_root / asset_path
        asset_sha256 = _content_sha256(asset_path)
        if asset_sha256 is None:
            missing_referenced_asset_count += 1
        _update_manifest_digest(
            digest,
            "referenced_asset",
            field,
            configured_asset,
            asset_sha256,
        )

    external_masks: list[Path] = []
    if external_root is not None and external_root.is_dir():
        try:
            external_masks = sorted(
                path
                for path in external_root.rglob("*.png")
                if path.is_file()
            )
        except OSError:
            external_masks = []
    for mask_path in external_masks:
        try:
            relative = mask_path.relative_to(external_root).as_posix()
        except ValueError:
            relative = mask_path.as_posix()
        _update_manifest_digest(
            digest,
            "external_person_mask",
            relative,
            _content_sha256(mask_path),
        )

    return {
        "schema": _ANATOMICAL_METADATA_SCHEMA,
        "sha256": digest.hexdigest(),
        "metadata_dir": metadata_identity,
        "person_mask_dir": person_mask_identity,
        "manifest_sha256": manifest_sha256,
        "manifest_valid": manifest_valid,
        "referenced_asset_count": len(referenced_assets),
        "missing_referenced_asset_count": (
            missing_referenced_asset_count
        ),
        "external_person_mask_count": len(external_masks),
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_revision(root: Path) -> str | None:
    """Return the current commit without making provenance collection fatal."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


@lru_cache(maxsize=1)
def source_provenance() -> dict[str, Any]:
    """Describe the executable source and dependency lock used by a run."""
    root = _repository_root()
    lock_path = root / "uv.lock"
    return {
        "git_commit": _git_revision(root),
        # This digest includes untracked ReID Python files, unlike git status.
        "reid_code_sha256": reid_code_sha256(),
        "uv_lock_sha256": file_sha256(lock_path) if lock_path.is_file() else None,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def _portable_sample_path(path: str | Path, root: Path | None) -> str:
    resolved = Path(path).expanduser().resolve()
    if root is not None:
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            pass
    # Avoid persisting machine-specific absolute paths while retaining enough
    # hierarchy to disambiguate common ReID split directories.
    return "/".join(resolved.parts[-3:])


def dataset_manifest(dataset: Any) -> dict[str, Any]:
    """Hash the ordered split membership without reading every image payload.

    The manifest covers relative path, identity, camera, source, and file size.
    It is intentionally cheap enough to compute at every run start; checkpoint
    bytes are separately protected by SHA-256.
    """
    digest = hashlib.sha256()
    split_counts: dict[str, int] = {}
    dataset_root = getattr(dataset, "root", None)
    root = Path(dataset_root).expanduser().resolve() if dataset_root else None
    for split_name in ("train", "query", "gallery"):
        split = getattr(dataset, split_name, None)
        samples = tuple(getattr(split, "samples", ()) or ())
        split_counts[split_name] = len(samples)
        for sample in samples:
            sample_path = Path(sample.img_path).expanduser()
            try:
                size = sample_path.stat().st_size
            except OSError:
                size = -1
            record = (
                split_name,
                _portable_sample_path(sample_path, root),
                int(sample.pid),
                int(sample.camid),
                str(getattr(sample, "source", "")),
                int(size),
            )
            digest.update(repr(record).encode("utf-8"))
            digest.update(b"\n")
    return {
        "schema": "reid-split-path-pid-camid-size-v1",
        "sha256": digest.hexdigest(),
        "splits": split_counts,
    }


def model_pretrained_provenance(model: Any) -> dict[str, Any] | None:
    """Return the verified pretrained source recorded by a backbone loader."""
    while hasattr(model, "module"):
        model = model.module
    sha256 = getattr(model, "pretrained_sha256", None)
    url = getattr(model, "pretrained_url", None)
    if not sha256 and not url:
        return None
    return {
        "url": url,
        "sha256": sha256,
        "required_tensor_count": getattr(
            model,
            "pretrained_backbone_required_tensor_count",
            None,
        ),
        "matched_tensor_count": getattr(
            model,
            "pretrained_backbone_matched_tensor_count",
            None,
        ),
        "tensor_coverage": getattr(
            model,
            "pretrained_backbone_tensor_coverage",
            None,
        ),
        "required_numel": getattr(
            model,
            "pretrained_backbone_required_numel",
            None,
        ),
        "matched_numel": getattr(
            model,
            "pretrained_backbone_matched_numel",
            None,
        ),
        "numel_coverage": getattr(
            model,
            "pretrained_backbone_numel_coverage",
            None,
        ),
    }


def _validate_pretrained_provenance(value: Any) -> dict[str, Any] | None:
    """Validate and normalize a persisted pretrained-source record."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("checkpoint pretrained provenance must be a dictionary or null")
    unknown = sorted(set(value).difference(_PRETRAINED_PROVENANCE_ATTRS))
    if unknown:
        raise ValueError(f"unknown checkpoint pretrained provenance fields: {unknown}")

    normalized = {key: value.get(key) for key in _PRETRAINED_PROVENANCE_ATTRS}
    url = normalized["url"]
    if url is not None and (not isinstance(url, str) or not url.strip()):
        raise ValueError("checkpoint pretrained provenance URL must be a non-empty string")
    if isinstance(url, str):
        normalized["url"] = url.strip()

    sha256 = normalized["sha256"]
    if sha256 is not None:
        if not isinstance(sha256, str):
            raise ValueError("checkpoint pretrained provenance SHA-256 must be a string")
        sha256 = sha256.lower()
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError("checkpoint pretrained provenance SHA-256 must contain 64 hexadecimal characters")
        normalized["sha256"] = sha256
    if normalized["url"] is None and normalized["sha256"] is None:
        raise ValueError("checkpoint pretrained provenance must include a URL or SHA-256")

    for prefix in ("tensor", "numel"):
        required_key = f"required_{prefix}_count" if prefix == "tensor" else "required_numel"
        matched_key = f"matched_{prefix}_count" if prefix == "tensor" else "matched_numel"
        coverage_key = f"{prefix}_coverage"
        required = normalized[required_key]
        matched = normalized[matched_key]
        coverage = normalized[coverage_key]
        if (required is None) != (matched is None):
            raise ValueError(
                f"checkpoint pretrained provenance requires both {required_key} and {matched_key}"
            )
        if required is not None:
            if (
                isinstance(required, bool)
                or isinstance(matched, bool)
                or not isinstance(required, int)
                or not isinstance(matched, int)
                or required < 0
                or matched < 0
                or matched > required
            ):
                raise ValueError(
                    f"checkpoint pretrained provenance has invalid {prefix} matched/required counts"
                )
        if coverage is not None:
            if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
                raise ValueError(f"checkpoint pretrained provenance {coverage_key} must be numeric")
            coverage = float(coverage)
            if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
                raise ValueError(
                    f"checkpoint pretrained provenance {coverage_key} must be finite and in [0, 1]"
                )
            normalized[coverage_key] = coverage
        if required is not None and coverage is not None:
            expected = matched / required if required else 0.0
            if not math.isclose(coverage, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"checkpoint pretrained provenance {coverage_key}={coverage} "
                    f"does not match {matched}/{required}"
                )
    return normalized


def checkpoint_pretrained_provenance(checkpoint: Any) -> dict[str, Any] | None:
    """Return one validated pretrained record from a training checkpoint."""
    if not isinstance(checkpoint, dict):
        raise ValueError("training checkpoint must be a dictionary")
    top_level = checkpoint.get("pretrained")
    model_metadata = checkpoint.get("model")
    nested = model_metadata.get("pretrained") if isinstance(model_metadata, dict) else None
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError("checkpoint has conflicting top-level and model pretrained provenance")
    return _validate_pretrained_provenance(nested if nested is not None else top_level)


def restore_model_pretrained_provenance(
    model: Any,
    provenance: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Restore validated non-state pretrained attributes onto one live model."""
    normalized = _validate_pretrained_provenance(provenance)
    if normalized is None:
        return None
    while hasattr(model, "module"):
        model = model.module
    for key, attribute in _PRETRAINED_PROVENANCE_ATTRS.items():
        setattr(model, attribute, normalized[key])
    return normalized


def build_run_provenance(
    model: Any,
    dataset: Any,
    *,
    anatomical_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete compact provenance block persisted in hparams."""
    return {
        "source": source_provenance(),
        "dataset_manifest": dataset_manifest(dataset),
        "pretrained": model_pretrained_provenance(model),
        "anatomical_metadata": anatomical_metadata,
        "executable": sys.executable,
    }


__all__ = [
    "anatomical_metadata_provenance",
    "build_run_provenance",
    "checkpoint_pretrained_provenance",
    "dataset_manifest",
    "model_pretrained_provenance",
    "restore_model_pretrained_provenance",
    "source_provenance",
]
