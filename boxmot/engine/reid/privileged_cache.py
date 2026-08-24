"""Build and validate offline HP-GRD privileged-teacher caches.

This standalone entrypoint consumes precomputed tensors only; it never
imports or downloads a teacher model.  Example::

    uv run python -m boxmot.engine.reid.privileged_cache build \
      --tensor-input teacher-signals.pt \
      --dataset-index train-samples.json \
      --teacher-provenance teacher-provenance.json \
      --part-names head torso left_arm right_arm left_leg right_leg \
      --output hpgrd-cache.pt

The dataset index is JSON/JSONL with rows containing ``index``, ``img_path``,
``pid``, and ``camid``.  The tensor input is a ``torch.save`` dictionary with
global/part descriptors, visibility/confidence, and optional global confidence
or leave-part-out descriptors. ``sample_indices`` may be included; otherwise
the dataset-index row order supplies them. Teacher extractors may wrap tensors
as ``{"part_names": [...], "tensors": {...}}``; otherwise ``--part-names`` is
required. The exact ordered names are signed into cache schema v2.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from boxmot.reid.training.trainer_components.privileged_graph import (
    TEACHER_CACHE_OPTIONAL_FIELDS,
    TEACHER_CACHE_REQUIRED_FIELDS,
    DatasetSampleProvenance,
    PrivilegedGraphTeacherCache,
    dataset_samples_sha256,
    sha256_file,
    validate_part_names,
)

DATASET_INDEX_SCHEMA = "boxmot_reid_dataset_index_v1"
_RESERVED_EXTRA_FIELDS = frozenset(
    {
        "builder",
        "dataset_index_file_sha256",
        "tensor_input_sha256",
        "part_names",
    }
)


@dataclass(frozen=True)
class PrivilegedCacheBuildResult:
    """Output path and signed manifest produced by a successful build."""

    output_path: Path
    manifest: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable command result."""

        return {
            "cache": str(self.output_path),
            "manifest_sha256": self.manifest["manifest_sha256"],
            "dataset_sha256": self.manifest["dataset_sha256"],
            "teacher_sha256": self.manifest["teacher_sha256"],
            "payload_sha256": self.manifest["payload_sha256"],
            "sample_count": self.manifest["sample_count"],
            "part_count": self.manifest["part_count"],
            "part_names": self.manifest["part_names"],
        }


@dataclass(frozen=True)
class PrivilegedCacheValidationResult:
    """Validated cache identity and shape summary."""

    cache_path: Path
    manifest: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable command result."""

        return {
            "cache": str(self.cache_path),
            "valid": True,
            "manifest_sha256": self.manifest["manifest_sha256"],
            "dataset_sha256": self.manifest["dataset_sha256"],
            "teacher_sha256": self.manifest["teacher_sha256"],
            "payload_sha256": self.manifest["payload_sha256"],
            "sample_count": self.manifest["sample_count"],
            "part_count": self.manifest["part_count"],
            "part_names": self.manifest["part_names"],
            "global_dim": self.manifest["global_dim"],
            "part_dim": self.manifest["part_dim"],
            "leave_part_out_dim": self.manifest["leave_part_out_dim"],
        }


@dataclass(frozen=True)
class DatasetIndexExportResult:
    """Stable training-index artifact produced from a registered dataset."""

    output_path: Path
    dataset_name: str
    sample_count: int
    dataset_sha256: str

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable command result."""

        return {
            "dataset_index": str(self.output_path),
            "dataset": self.dataset_name,
            "sample_count": self.sample_count,
            "dataset_sha256": self.dataset_sha256,
        }


def export_dataset_index(
    *,
    dataset_name: str,
    data_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
    overwrite: bool = False,
) -> DatasetIndexExportResult:
    """Atomically export the exact stable mapping used by live training."""

    from boxmot.reid.datasets import build_dataset

    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset index: {destination}")
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"Dataset-index output must be a file: {destination}")

    dataset = build_dataset(str(dataset_name), str(data_dir))
    samples = tuple(dataset.train.samples)
    if not samples:
        raise ValueError(f"Registered dataset {dataset_name!r} has no training samples")
    rows = [
        {
            "index": index,
            "img_path": os.fspath(sample.img_path),
            "pid": int(sample.pid),
            "camid": int(sample.camid),
        }
        for index, sample in enumerate(samples)
    ]
    dataset_hash = dataset_samples_sha256(samples)
    if dataset_samples_sha256(rows) != dataset_hash:
        raise RuntimeError("Exported dataset index does not reproduce the live training mapping")
    payload = {"schema": DATASET_INDEX_SCHEMA, "samples": rows}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        load_dataset_index(temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return DatasetIndexExportResult(
        output_path=destination,
        dataset_name=str(dataset_name),
        sample_count=len(samples),
        dataset_sha256=dataset_hash,
    )


def load_dataset_index(path: str | os.PathLike[str]) -> tuple[DatasetSampleProvenance, ...]:
    """Load strict JSON/JSONL sample provenance in tensor-row order."""

    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows: list[object] = []
        with source.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on dataset-index line {line_number}: {error.msg}") from error
    else:
        with source.open("r", encoding="utf-8") as file:
            try:
                payload = json.load(file)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid dataset-index JSON: {error.msg}") from error
        if isinstance(payload, dict):
            unknown = set(payload) - {"schema", "samples"}
            if unknown or "samples" not in payload:
                raise ValueError("Dataset-index object may contain only 'schema' and required 'samples'")
            if payload.get("schema", DATASET_INDEX_SCHEMA) != DATASET_INDEX_SCHEMA:
                raise ValueError(f"Unsupported dataset-index schema {payload.get('schema')!r}")
            rows = payload["samples"]
        else:
            rows = payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset index must contain a non-empty JSON list of samples")

    samples = tuple(_parse_dataset_row(row, row_number) for row_number, row in enumerate(rows, start=1))
    dataset_samples_sha256(samples)
    return samples


@dataclass(frozen=True)
class PrecomputedTeacherSignalBundle:
    """Validated extractor tensors and optional ordered part-axis metadata."""

    tensors: Mapping[str, torch.Tensor]
    part_names: tuple[str, ...] | None


def load_precomputed_teacher_signal_bundle(
    path: str | os.PathLike[str],
) -> PrecomputedTeacherSignalBundle:
    """Load safe precomputed signals, including signed part-name provenance."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Precomputed tensor input must be a dictionary")
    part_names: object | None = None
    if "tensors" in payload:
        if set(payload) not in ({"tensors"}, {"tensors", "part_names"}):
            raise ValueError("Wrapped precomputed signals may contain only 'tensors' and 'part_names'")
        part_names = payload.get("part_names")
        payload = payload["tensors"]
    if not isinstance(payload, dict):
        raise TypeError("Precomputed 'tensors' field must be a dictionary")

    allowed = set(TEACHER_CACHE_REQUIRED_FIELDS) | set(TEACHER_CACHE_OPTIONAL_FIELDS)
    required_without_indices = set(TEACHER_CACHE_REQUIRED_FIELDS) - {"sample_indices"}
    fields = set(payload)
    missing = required_without_indices - fields
    unknown = fields - allowed
    if missing or unknown:
        raise ValueError(f"Precomputed tensor fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if not all(torch.is_tensor(value) for value in payload.values()):
        raise TypeError("Every precomputed teacher value must be a tensor")
    tensors = {name: value.detach().to(device="cpu") for name, value in payload.items()}
    part_descriptors = tensors["part_descriptors"]
    if part_descriptors.ndim != 3 or part_descriptors.shape[1] < 1:
        raise ValueError("Precomputed part_descriptors must have shape [N,P,D] with P > 0")
    normalized_names = None
    if part_names is not None:
        normalized_names = validate_part_names(
            part_names,
            int(part_descriptors.shape[1]),
        )
    return PrecomputedTeacherSignalBundle(tensors=tensors, part_names=normalized_names)


def load_precomputed_teacher_tensors(path: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    """Load only tensors while preserving the established public helper API."""

    return dict(load_precomputed_teacher_signal_bundle(path).tensors)


def build_privileged_cache(
    *,
    tensor_input: str | os.PathLike[str],
    dataset_index: str | os.PathLike[str],
    teacher_provenance: str | os.PathLike[str],
    output: str | os.PathLike[str],
    part_names: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> PrivilegedCacheBuildResult:
    """Build, self-validate, and atomically publish a strict trainer cache."""

    tensor_path = _required_file(tensor_input, "tensor_input")
    index_path = _required_file(dataset_index, "dataset_index")
    provenance_path = _required_file(teacher_provenance, "teacher_provenance")
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing privileged cache: {destination}")
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"Privileged cache output must be a file: {destination}")

    dataset_samples = load_dataset_index(index_path)
    dataset_hash = dataset_samples_sha256(dataset_samples)
    dataset_indices = torch.tensor([sample.index for sample in dataset_samples], dtype=torch.long)
    signal_bundle = load_precomputed_teacher_signal_bundle(tensor_path)
    tensors = dict(signal_bundle.tensors)
    if "sample_indices" in tensors:
        _validate_tensor_index_set(tensors["sample_indices"], dataset_indices)
    else:
        tensors["sample_indices"] = dataset_indices

    tensor_part_count = int(tensors["part_descriptors"].shape[1])
    explicit_part_names = None if part_names is None else validate_part_names(part_names, tensor_part_count)
    if explicit_part_names is not None and signal_bundle.part_names is not None:
        if explicit_part_names != signal_bundle.part_names:
            raise ValueError(
                "--part-names do not match the ordered names embedded by the teacher extractor: "
                f"explicit={list(explicit_part_names)!r}, "
                f"embedded={list(signal_bundle.part_names)!r}"
            )
    resolved_part_names = explicit_part_names or signal_bundle.part_names
    if resolved_part_names is None:
        raise ValueError(
            "Ordered semantic part names are required; pass part_names/--part-names "
            "or use a named teacher-extractor bundle"
        )

    cache = PrivilegedGraphTeacherCache(
        part_names=resolved_part_names,
        sample_indices=tensors["sample_indices"],
        global_descriptors=tensors["global_descriptors"],
        part_descriptors=tensors["part_descriptors"],
        part_visibility=tensors["part_visibility"],
        part_confidence=tensors["part_confidence"],
        global_confidence=tensors.get("global_confidence"),
        leave_part_out_descriptors=tensors.get("leave_part_out_descriptors"),
    )
    reserved_extra = {
        "builder": "boxmot.engine.reid.privileged_cache",
        "dataset_index_file_sha256": sha256_file(index_path),
        "tensor_input_sha256": sha256_file(tensor_path),
    }
    if extra:
        collisions = set(extra) & _RESERVED_EXTRA_FIELDS
        if collisions:
            raise ValueError(f"extra cannot replace reserved builder fields: {sorted(collisions)}")
        reserved_extra["user"] = dict(extra)
    teacher_hash = sha256_file(provenance_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        manifest = cache.save(
            temporary_path,
            dataset_sha256=dataset_hash,
            teacher_sha256=teacher_hash,
            extra=reserved_extra,
        )
        PrivilegedGraphTeacherCache.load(
            temporary_path,
            expected_dataset_sha256=dataset_hash,
            expected_teacher_sha256=teacher_hash,
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_part_names=resolved_part_names,
        )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return PrivilegedCacheBuildResult(destination, manifest)


def validate_privileged_cache(
    *,
    cache_path: str | os.PathLike[str],
    dataset_index: str | os.PathLike[str],
    teacher_provenance: str | os.PathLike[str],
    expected_manifest_sha256: str | None = None,
    expected_part_names: Sequence[str] | None = None,
    require_exact_index_file: bool = False,
) -> PrivilegedCacheValidationResult:
    """Validate semantic dataset, teacher, manifest, payload, and shapes."""

    source = _required_file(cache_path, "cache_path")
    index_path = _required_file(dataset_index, "dataset_index")
    provenance_path = _required_file(teacher_provenance, "teacher_provenance")
    dataset_hash = dataset_samples_sha256(load_dataset_index(index_path))
    teacher_hash = sha256_file(provenance_path)
    cache = PrivilegedGraphTeacherCache.load(
        source,
        expected_dataset_sha256=dataset_hash,
        expected_teacher_sha256=teacher_hash,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_part_names=expected_part_names,
    )
    manifest = cache.manifest
    if manifest is None:
        raise RuntimeError("Loaded privileged cache did not retain its validated manifest")
    if require_exact_index_file:
        expected_file_hash = manifest["extra"].get("dataset_index_file_sha256")
        if not isinstance(expected_file_hash, str):
            raise ValueError("Cache does not record an exact dataset-index file hash")
        actual_file_hash = sha256_file(index_path)
        if not hmac.compare_digest(expected_file_hash, actual_file_hash):
            raise ValueError(
                "Privileged cache dataset-index file SHA-256 mismatch: "
                f"expected {expected_file_hash}, got {actual_file_hash}"
            )
    return PrivilegedCacheValidationResult(source, manifest)


def _required_file(path: str | os.PathLike[str], name: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a file: {resolved}")
    return resolved


def _parse_dataset_row(row: object, row_number: int) -> DatasetSampleProvenance:
    if isinstance(row, dict):
        allowed = {"index", "sample_index", "img_path", "pid", "camid"}
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"Dataset-index row {row_number} has unknown keys: {sorted(unknown)}")
        if "index" in row and "sample_index" in row and row["index"] != row["sample_index"]:
            raise ValueError(f"Dataset-index row {row_number} has conflicting index fields")
        index = row.get("index", row.get("sample_index"))
        missing = [name for name in ("img_path", "pid", "camid") if name not in row]
        if index is None:
            missing.insert(0, "index")
        if missing:
            raise ValueError(f"Dataset-index row {row_number} is missing fields: {missing}")
        sample = DatasetSampleProvenance(index=index, img_path=row["img_path"], pid=row["pid"], camid=row["camid"])
    elif isinstance(row, list) and len(row) == 4:
        sample = DatasetSampleProvenance(index=row[0], img_path=row[1], pid=row[2], camid=row[3])
    else:
        raise TypeError(f"Dataset-index row {row_number} must be an object or [index, img_path, pid, camid]")
    dataset_samples_sha256((sample,))
    return sample


def _validate_tensor_index_set(tensor_indices: torch.Tensor, dataset_indices: torch.Tensor) -> None:
    if tensor_indices.ndim != 1:
        raise ValueError("Precomputed sample_indices must have shape [N]")
    if tensor_indices.dtype == torch.bool or tensor_indices.dtype.is_floating_point or tensor_indices.is_complex():
        raise TypeError("Precomputed sample_indices must use an integer dtype")
    tensor_indices = tensor_indices.detach().to(device="cpu", dtype=torch.long)
    if tensor_indices.numel() != dataset_indices.numel():
        raise ValueError(
            "Precomputed sample_indices and dataset index have different row counts: "
            f"{tensor_indices.numel()} != {dataset_indices.numel()}"
        )
    sorted_tensor = tensor_indices.sort().values
    sorted_dataset = dataset_indices.sort().values
    if not torch.equal(sorted_tensor, sorted_dataset):
        tensor_only = sorted(set(sorted_tensor.tolist()) - set(sorted_dataset.tolist()))
        dataset_only = sorted(set(sorted_dataset.tolist()) - set(sorted_tensor.tolist()))
        raise ValueError(
            "Precomputed sample_indices do not match the dataset index: "
            f"tensor_only={tensor_only}, dataset_only={dataset_only}"
        )


def _parse_extra_json(value: str | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"--extra-json is invalid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError("--extra-json must decode to a JSON object")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser(
        "index",
        help="Export the exact stable training mapping from a registered dataset",
    )
    index.add_argument("--dataset", required=True, help="Registered ReID dataset name")
    index.add_argument("--data-dir", type=Path, required=True, help="Dataset root used by training")
    index.add_argument("--output", type=Path, required=True, help="Destination JSON index")
    index.add_argument("--overwrite", action="store_true", help="Atomically replace an existing output file")

    build = commands.add_parser("build", help="Build and self-validate a cache from precomputed tensors")
    build.add_argument("--tensor-input", type=Path, required=True, help="torch.save tensor dictionary")
    build.add_argument("--dataset-index", type=Path, required=True, help="JSON/JSONL stable sample index")
    build.add_argument("--teacher-provenance", type=Path, required=True, help="Teacher artifact/config to hash")
    build.add_argument(
        "--part-names",
        nargs="+",
        help="Ordered semantic part names (checked against extractor metadata when present)",
    )
    build.add_argument("--output", type=Path, required=True, help="Destination .pt cache")
    build.add_argument("--extra-json", help="Optional JSON object stored under manifest extra.user")
    build.add_argument("--overwrite", action="store_true", help="Atomically replace an existing output file")

    validate = commands.add_parser("validate", help="Validate cache payload and external provenance")
    validate.add_argument("--cache", type=Path, required=True, help="Cache produced by the build command")
    validate.add_argument("--dataset-index", type=Path, required=True, help="JSON/JSONL stable sample index")
    validate.add_argument("--teacher-provenance", type=Path, required=True, help="Teacher artifact/config to hash")
    validate.add_argument("--manifest-sha256", help="Optional pinned manifest digest")
    validate.add_argument("--part-names", nargs="+", help="Require this exact ordered semantic part axis")
    validate.add_argument(
        "--require-exact-index-file",
        action="store_true",
        help="Also require byte-for-byte identity of the original index file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone cache builder/validator command."""

    args = _build_parser().parse_args(argv)
    if args.command == "index":
        result: DatasetIndexExportResult | PrivilegedCacheBuildResult | PrivilegedCacheValidationResult = (
            export_dataset_index(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                output=args.output,
                overwrite=args.overwrite,
            )
        )
    elif args.command == "build":
        result = build_privileged_cache(
            tensor_input=args.tensor_input,
            dataset_index=args.dataset_index,
            teacher_provenance=args.teacher_provenance,
            output=args.output,
            part_names=args.part_names,
            extra=_parse_extra_json(args.extra_json),
            overwrite=args.overwrite,
        )
    else:
        result = validate_privileged_cache(
            cache_path=args.cache,
            dataset_index=args.dataset_index,
            teacher_provenance=args.teacher_provenance,
            expected_manifest_sha256=args.manifest_sha256,
            expected_part_names=args.part_names,
            require_exact_index_file=args.require_exact_index_file,
        )
    print(json.dumps(result.summary(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
