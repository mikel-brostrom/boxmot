"""Training-only privileged relational supervision for person ReID.

The deployed descriptor is never changed by this module.  Expensive human
parsing, pose, or foundation-model teachers are expected to run offline and
their outputs are addressed by a stable dataset sample index.  During
training, this module transfers only the teacher's relational geometry to the
student descriptor and optional training-only part/intervention descriptors.

The module intentionally has no dependency on a particular head or trainer.
See :func:`privileged_graph_integration_hooks` for the small integration
contract expected from those components.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import operator
import os
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_FORMAT = "boxmot_privileged_graph"
CACHE_VERSION = 2

DEPLOYED_DESCRIPTOR_KEY = "norm_concat_bn"
# Generic training-packet key. The canonical 7M integration fills it via fixed
# mask pooling; no corresponding module or export surface is deployed.
PART_DESCRIPTOR_KEY = "_anatomical_student_tokens"
PART_RELIABILITY_KEY = "_hpgrd_student_part_reliability"
BACKGROUND_DESCRIPTOR_KEY = "_privileged_graph_background_descriptors"
BACKGROUND_INDICES_KEY = "_privileged_graph_background_indices"
BACKGROUND_CONFIDENCE_KEY = "_privileged_graph_background_confidence"
SEMANTIC_DROP_DESCRIPTOR_KEY = "_privileged_graph_semantic_drop_descriptors"
SEMANTIC_DROP_INDICES_KEY = "_privileged_graph_semantic_drop_indices"
SEMANTIC_DROP_PARTS_KEY = "_privileged_graph_semantic_drop_parts"
SEMANTIC_DROP_CONFIDENCE_KEY = "_privileged_graph_semantic_drop_confidence"

TEACHER_CACHE_REQUIRED_FIELDS = (
    "sample_indices",
    "global_descriptors",
    "part_descriptors",
    "part_visibility",
    "part_confidence",
)
TEACHER_CACHE_OPTIONAL_FIELDS = (
    "global_confidence",
    "leave_part_out_descriptors",
)
_TEACHER_TENSOR_NAMES = TEACHER_CACHE_REQUIRED_FIELDS + TEACHER_CACHE_OPTIONAL_FIELDS
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class PrivilegedGraphTeacherBatch:
    """Frozen teacher signals aligned to one student batch."""

    sample_indices: torch.Tensor
    global_descriptors: torch.Tensor
    part_descriptors: torch.Tensor
    part_visibility: torch.Tensor
    part_confidence: torch.Tensor
    global_confidence: torch.Tensor | None = None
    leave_part_out_descriptors: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _validate_teacher_tensors(self.as_mapping())

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""

        return int(self.sample_indices.numel())

    @property
    def part_count(self) -> int:
        """Number of semantic parts in the cache schema."""

        return int(self.part_descriptors.shape[1])

    def as_mapping(self) -> dict[str, torch.Tensor | None]:
        """Return the batch in the serialization/integration schema."""

        return {
            "sample_indices": self.sample_indices,
            "global_descriptors": self.global_descriptors,
            "part_descriptors": self.part_descriptors,
            "part_visibility": self.part_visibility,
            "part_confidence": self.part_confidence,
            "global_confidence": self.global_confidence,
            "leave_part_out_descriptors": self.leave_part_out_descriptors,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, torch.Tensor | None]) -> PrivilegedGraphTeacherBatch:
        """Build a validated batch from a trainer/cache dictionary."""

        required = (
            "sample_indices",
            "global_descriptors",
            "part_descriptors",
            "part_visibility",
            "part_confidence",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise KeyError(f"Teacher packet is missing required keys: {missing}")
        return cls(
            sample_indices=_require_tensor(values["sample_indices"], "sample_indices"),
            global_descriptors=_require_tensor(values["global_descriptors"], "global_descriptors"),
            part_descriptors=_require_tensor(values["part_descriptors"], "part_descriptors"),
            part_visibility=_require_tensor(values["part_visibility"], "part_visibility"),
            part_confidence=_require_tensor(values["part_confidence"], "part_confidence"),
            global_confidence=_optional_tensor(values.get("global_confidence"), "global_confidence"),
            leave_part_out_descriptors=_optional_tensor(
                values.get("leave_part_out_descriptors"),
                "leave_part_out_descriptors",
            ),
        )


@dataclass(frozen=True)
class RelationalLossResult:
    """A balanced relation loss and its pair-group diagnostics."""

    loss: torch.Tensor
    positive_loss: torch.Tensor
    negative_loss: torch.Tensor
    positive_pairs: int
    negative_pairs: int


@dataclass(frozen=True)
class PartRelationalLossResult:
    """Mean relation loss across semantic parts with usable pairs."""

    loss: torch.Tensor
    active_parts: int
    positive_pairs: int
    negative_pairs: int


@dataclass(frozen=True)
class PrivilegedGraphLossResult:
    """Weighted auxiliary objective with unweighted component reporting."""

    total: torch.Tensor
    components: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, float | int]


@dataclass(frozen=True)
class GradientBudgetResult:
    """Auxiliary loss after limiting its gradient norm relative to the base loss."""

    scaled_loss: torch.Tensor
    scale: torch.Tensor
    base_grad_norm: torch.Tensor
    auxiliary_grad_norm: torch.Tensor


@dataclass(frozen=True)
class PrivilegedGraphIntegrationHook:
    """One explicit integration obligation outside this isolated module."""

    owner: str
    contract: str


@dataclass(frozen=True)
class DatasetSampleProvenance:
    """Canonical dataset row used to bind a cache to its training samples."""

    index: int
    img_path: str
    pid: int
    camid: int


def privileged_graph_integration_hooks() -> tuple[PrivilegedGraphIntegrationHook, ...]:
    """Describe the external hooks needed without modifying inference code."""

    return (
        PrivilegedGraphIntegrationHook(
            "dataset",
            "Return a deterministic stable sample index with every image; "
            "augmentation order must not define the index.",
        ),
        PrivilegedGraphIntegrationHook(
            "training setup",
            "Validate dataset/teacher hashes once, load PrivilegedGraphTeacherCache, "
            "and look up each batch by stable index.",
        ),
        PrivilegedGraphIntegrationHook(
            "training head packet",
            f"Expose {DEPLOYED_DESCRIPTOR_KEY!r} and, when part loss is enabled, {PART_DESCRIPTOR_KEY!r}.",
        ),
        PrivilegedGraphIntegrationHook(
            "intervention forward",
            "Optionally emit background-perturbed and semantic-part-drop descriptors plus their base-row indices.",
        ),
        PrivilegedGraphIntegrationHook(
            "training loop",
            "Apply the epoch schedule to result.total, then optionally constrain it with the gradient-budget helper.",
        ),
        PrivilegedGraphIntegrationHook(
            "export/eval",
            "Disable and prune training-only part/intervention branches; "
            "inference consumes only the deployed descriptor.",
        ),
    )


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return a streaming SHA-256 digest for a dataset index or teacher artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_samples_sha256(samples: Sequence[object]) -> str:
    """Hash stable ReID sample metadata independently of input row order.

    Ordinary ``ReIDSample`` objects receive their sequence position as the
    stable index.  Cache-builder rows may instead provide explicit
    ``index``/``sample_index`` fields, either as mappings or as
    ``(index, img_path, pid, camid)`` sequences.  Paths are normalized
    lexically to POSIX separators but are not resolved against the current
    machine, keeping the function free of filesystem state.
    """

    canonical_rows: list[list[int | str]] = []
    seen_indices: set[int] = set()
    for fallback_index, sample in enumerate(samples):
        index, img_path, pid, camid = _dataset_sample_fields(sample, fallback_index)
        if index in seen_indices:
            raise ValueError(f"Dataset samples contain duplicate stable index {index}")
        seen_indices.add(index)
        canonical_rows.append(
            [
                index,
                _normalize_dataset_image_path(img_path),
                pid,
                camid,
            ]
        )
    canonical_rows.sort(key=lambda row: int(row[0]))
    payload = {
        "schema": "boxmot_reid_dataset_samples_v1",
        "samples": canonical_rows,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


class PrivilegedGraphTeacherCache:
    """Immutable CPU cache of offline privileged teacher signals.

    Cache rows are sorted by ``sample_indices`` at construction, making lookup
    independent of dataloader ordering while preserving the requested batch
    order (including duplicate requests).
    """

    def __init__(
        self,
        *,
        part_names: Sequence[str],
        sample_indices: torch.Tensor,
        global_descriptors: torch.Tensor,
        part_descriptors: torch.Tensor,
        part_visibility: torch.Tensor,
        part_confidence: torch.Tensor,
        global_confidence: torch.Tensor | None = None,
        leave_part_out_descriptors: torch.Tensor | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        batch = PrivilegedGraphTeacherBatch(
            sample_indices=sample_indices,
            global_descriptors=global_descriptors,
            part_descriptors=part_descriptors,
            part_visibility=part_visibility,
            part_confidence=part_confidence,
            global_confidence=global_confidence,
            leave_part_out_descriptors=leave_part_out_descriptors,
        )
        if batch.batch_size == 0:
            raise ValueError("Privileged teacher cache cannot be empty")
        if sample_indices.dtype == torch.bool or sample_indices.dtype.is_floating_point:
            raise TypeError("sample_indices must use an integer dtype")
        self._part_names = validate_part_names(part_names, batch.part_count)

        cpu_indices = sample_indices.detach().to(device="cpu", dtype=torch.long).clone()
        sorted_indices, order = torch.sort(cpu_indices)
        if bool((sorted_indices[1:] == sorted_indices[:-1]).any()):
            duplicates = sorted_indices[1:][sorted_indices[1:] == sorted_indices[:-1]].unique().tolist()
            raise ValueError(f"sample_indices must be unique; duplicates: {duplicates}")

        self._tensors: dict[str, torch.Tensor] = {}
        for name, value in batch.as_mapping().items():
            if value is None:
                continue
            cpu_value = value.detach().to(device="cpu").clone()
            if name == "sample_indices":
                cpu_value = sorted_indices
            else:
                cpu_value = cpu_value.index_select(0, order)
            cpu_value.requires_grad_(False)
            self._tensors[name] = cpu_value
        self._manifest = dict(manifest) if manifest is not None else None

    def __len__(self) -> int:
        return int(self._tensors["sample_indices"].numel())

    @property
    def manifest(self) -> Mapping[str, Any] | None:
        """A copy of the validated loaded/saved manifest, if available."""

        return None if self._manifest is None else dict(self._manifest)

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        """Copies of the frozen CPU tensors for inspection/diagnostics."""

        return {name: value.clone() for name, value in self._tensors.items()}

    @property
    def part_names(self) -> tuple[str, ...]:
        """Ordered semantic identity of the part-descriptor axis."""

        return self._part_names

    def lookup(
        self,
        sample_indices: torch.Tensor | Sequence[int],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PrivilegedGraphTeacherBatch:
        """Look up teacher rows by stable index in the exact requested order."""

        requested = torch.as_tensor(sample_indices)
        if requested.ndim != 1:
            raise ValueError(f"Requested sample_indices must have shape [B], got {tuple(requested.shape)}")
        if requested.dtype == torch.bool or requested.dtype.is_floating_point:
            raise TypeError("Requested sample_indices must use an integer dtype")
        requested_cpu = requested.detach().to(device="cpu", dtype=torch.long)
        sorted_indices = self._tensors["sample_indices"]
        positions = torch.searchsorted(sorted_indices, requested_cpu)
        safe_positions = positions.clamp(max=len(self) - 1)
        found = (positions < len(self)) & (sorted_indices.index_select(0, safe_positions) == requested_cpu)
        if not bool(found.all()):
            missing = requested_cpu[~found].unique().tolist()
            raise KeyError(f"Privileged teacher cache has no rows for stable sample indices {missing}")

        target_device = device if device is not None else requested.device
        values: dict[str, torch.Tensor | None] = {}
        for name in _TEACHER_TENSOR_NAMES:
            value = self._tensors.get(name)
            if value is None:
                values[name] = None
                continue
            selected = value.index_select(0, safe_positions)
            target_dtype = dtype if dtype is not None and selected.dtype.is_floating_point else selected.dtype
            values[name] = selected.to(device=target_device, dtype=target_dtype).detach()
        return PrivilegedGraphTeacherBatch.from_mapping(values)

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        dataset_sha256: str,
        teacher_sha256: str,
        extra: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Save tensors plus a self-validating provenance/shape manifest."""

        _validate_sha256(dataset_sha256, "dataset_sha256")
        _validate_sha256(teacher_sha256, "teacher_sha256")
        global_descriptors = self._tensors["global_descriptors"]
        part_descriptors = self._tensors["part_descriptors"]
        leave_part_out = self._tensors.get("leave_part_out_descriptors")
        manifest: dict[str, Any] = {
            "format": CACHE_FORMAT,
            "version": CACHE_VERSION,
            "sample_count": len(self),
            "part_count": int(part_descriptors.shape[1]),
            "part_names": list(self._part_names),
            "global_dim": int(global_descriptors.shape[1]),
            "part_dim": int(part_descriptors.shape[2]),
            "leave_part_out_dim": None if leave_part_out is None else int(leave_part_out.shape[2]),
            "dataset_sha256": dataset_sha256,
            "teacher_sha256": teacher_sha256,
            "payload_sha256": _tensor_payload_sha256(self._tensors),
            "extra": dict(extra or {}),
        }
        manifest["manifest_sha256"] = _manifest_sha256(manifest)
        _validate_manifest(manifest)

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"manifest": manifest, "tensors": self._tensors}, destination)
        self._manifest = dict(manifest)
        return dict(manifest)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_dataset_sha256: str | None = None,
        expected_teacher_sha256: str | None = None,
        expected_manifest_sha256: str | None = None,
        expected_part_names: Sequence[str] | None = None,
    ) -> PrivilegedGraphTeacherCache:
        """Load a cache and reject provenance, manifest, or payload drift."""

        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {"manifest", "tensors"}:
            raise ValueError("Privileged cache must contain exactly 'manifest' and 'tensors'")
        manifest = payload["manifest"]
        tensors = payload["tensors"]
        if not isinstance(manifest, dict) or not isinstance(tensors, dict):
            raise ValueError("Privileged cache manifest and tensors must be dictionaries")
        _validate_manifest(manifest)

        if expected_dataset_sha256 is not None:
            _validate_sha256(expected_dataset_sha256, "expected_dataset_sha256")
            _require_matching_hash(manifest["dataset_sha256"], expected_dataset_sha256, "dataset")
        if expected_teacher_sha256 is not None:
            _validate_sha256(expected_teacher_sha256, "expected_teacher_sha256")
            _require_matching_hash(manifest["teacher_sha256"], expected_teacher_sha256, "teacher")
        if expected_manifest_sha256 is not None:
            _validate_sha256(expected_manifest_sha256, "expected_manifest_sha256")
            _require_matching_hash(manifest["manifest_sha256"], expected_manifest_sha256, "manifest")
        if expected_part_names is not None:
            expected_names = validate_part_names(expected_part_names, manifest["part_count"])
            if tuple(manifest["part_names"]) != expected_names:
                raise ValueError(
                    "Privileged cache ordered part names mismatch: "
                    f"expected {list(expected_names)!r}, got {manifest['part_names']!r}"
                )

        tensor_names = set(tensors)
        required_names = set(_TEACHER_TENSOR_NAMES[:5])
        allowed_names = set(_TEACHER_TENSOR_NAMES)
        if not required_names.issubset(tensor_names) or not tensor_names.issubset(allowed_names):
            raise ValueError("Privileged cache tensor fields do not match the supported schema")
        if not all(torch.is_tensor(value) for value in tensors.values()):
            raise TypeError("All privileged cache payload values must be tensors")
        actual_payload_hash = _tensor_payload_sha256(tensors)
        _require_matching_hash(manifest["payload_sha256"], actual_payload_hash, "payload")

        cache = cls(
            part_names=manifest["part_names"],
            sample_indices=tensors["sample_indices"],
            global_descriptors=tensors["global_descriptors"],
            part_descriptors=tensors["part_descriptors"],
            part_visibility=tensors["part_visibility"],
            part_confidence=tensors["part_confidence"],
            global_confidence=tensors.get("global_confidence"),
            leave_part_out_descriptors=tensors.get("leave_part_out_descriptors"),
            manifest=manifest,
        )
        _validate_manifest_shapes(manifest, cache._tensors)
        return cache


def fuse_privileged_confidence(
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    visibility_weight: float = 0.5,
    confidence_weight: float = 0.5,
    minimum: float = 0.0,
) -> torch.Tensor:
    """Fuse independent reliability signals without multiplicative collapse.

    A weighted arithmetic mean preserves a useful signal when one calibrated
    teacher is conservative.  Exact-zero/missing signals are still excluded
    by ``valid`` (or by the default positive-visibility mask), so the mean
    cannot resurrect an absent body part.  A zero secondary confidence does
    not by itself erase a visible part; callers can pass an explicit ``valid``
    mask when zero means an invalid rather than merely uncertain observation.
    """

    if visibility.shape != confidence.shape:
        raise ValueError("visibility and confidence must have identical shapes")
    if not visibility.dtype.is_floating_point or not confidence.dtype.is_floating_point:
        raise TypeError("visibility and confidence must be floating-point tensors")
    if not math.isfinite(visibility_weight) or not math.isfinite(confidence_weight):
        raise ValueError("confidence fusion weights must be finite")
    if visibility_weight < 0 or confidence_weight < 0 or visibility_weight + confidence_weight <= 0:
        raise ValueError("confidence fusion weights must be non-negative with a positive sum")
    if not 0 <= minimum <= 1:
        raise ValueError("minimum must be in [0, 1]")
    _validate_unit_interval(visibility, "visibility")
    _validate_unit_interval(confidence, "confidence")

    confidence = confidence.to(device=visibility.device, dtype=visibility.dtype)
    if valid is None:
        valid_mask = visibility > 0
    else:
        if valid.shape != visibility.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean tensor with the same shape as visibility")
        valid_mask = valid.to(device=visibility.device)
    fused = (visibility * visibility_weight + confidence * confidence_weight) / (visibility_weight + confidence_weight)
    fused = fused.clamp(0, 1)
    if minimum > 0:
        fused = fused.clamp_min(minimum)
    return torch.where(valid_mask, fused, torch.zeros_like(fused))


def balanced_identity_relational_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    pids: torch.Tensor,
    reliability: torch.Tensor | None = None,
) -> RelationalLossResult:
    """Match cosine-relation graphs with identity-balanced pair groups.

    Descriptor dimensions may differ: only their ``B x B`` cosine matrices
    are compared. Every non-self pair with the same identity is a positive and
    every different-identity pair is a negative. Camera metadata deliberately
    plays no role in the target geometry: the deployed tracker consumes only
    embeddings, so its metric must encode identity rather than camera origin.
    The two groups are averaged so the larger negative graph cannot drown the
    identity-preserving signal. Pair counts are directed.
    """

    if student.ndim != 2 or teacher.ndim != 2:
        raise ValueError("student and teacher descriptors must have shape [B,D]")
    if student.shape[0] != teacher.shape[0]:
        raise ValueError("student and teacher batch dimensions must match")
    batch_size = student.shape[0]
    if pids.shape != (batch_size,):
        raise ValueError("pids must have shape [B]")
    if reliability is None:
        reliability = student.new_ones(batch_size)
    if reliability.shape != (batch_size,):
        raise ValueError("reliability must have shape [B]")
    _validate_descriptor(student, "student")
    _validate_descriptor(teacher, "teacher")
    _validate_unit_interval(reliability, "reliability")

    loss_dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    normalized_student = F.normalize(student.to(dtype=loss_dtype), p=2, dim=1)
    normalized_teacher = F.normalize(
        teacher.detach().to(device=student.device, dtype=loss_dtype),
        p=2,
        dim=1,
    )
    reliability = reliability.detach().to(device=student.device, dtype=loss_dtype)
    pids = pids.to(device=student.device)

    not_self = ~torch.eye(batch_size, device=student.device, dtype=torch.bool)
    same_identity = pids[:, None] == pids[None, :]
    has_support = reliability > 0
    valid = has_support[:, None] & has_support[None, :] & not_self
    pair_weights = torch.sqrt(reliability[:, None] * reliability[None, :])
    errors = (
        normalized_student @ normalized_student.transpose(0, 1)
        - normalized_teacher @ normalized_teacher.transpose(0, 1)
    ).square()
    differentiable_zero = normalized_student.sum() * 0

    positive_mask = valid & same_identity
    negative_mask = valid & ~same_identity
    positive_loss = _weighted_pair_mean(errors, pair_weights, positive_mask, differentiable_zero)
    negative_loss = _weighted_pair_mean(errors, pair_weights, negative_mask, differentiable_zero)
    active_losses = []
    if bool(positive_mask.any()):
        active_losses.append(positive_loss)
    if bool(negative_mask.any()):
        active_losses.append(negative_loss)
    loss = torch.stack(active_losses).mean() if active_losses else differentiable_zero
    return RelationalLossResult(
        loss=loss,
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        positive_pairs=int(positive_mask.sum().item()),
        negative_pairs=int(negative_mask.sum().item()),
    )


def part_relational_loss(
    student_parts: torch.Tensor,
    teacher_parts: torch.Tensor,
    pids: torch.Tensor,
    part_reliability: torch.Tensor,
) -> PartRelationalLossResult:
    """Average independent part-relation graphs, ignoring missing parts."""

    if student_parts.ndim != 3 or teacher_parts.ndim != 3:
        raise ValueError("student_parts and teacher_parts must have shape [B,P,D]")
    if student_parts.shape[:2] != teacher_parts.shape[:2]:
        raise ValueError("student and teacher part batch/part dimensions must match")
    if part_reliability.shape != student_parts.shape[:2]:
        raise ValueError("part_reliability must have shape [B,P]")
    _validate_descriptor(student_parts, "student_parts")
    _validate_descriptor(teacher_parts, "teacher_parts")
    _validate_unit_interval(part_reliability, "part_reliability")

    losses: list[torch.Tensor] = []
    positive_pairs = 0
    negative_pairs = 0
    for part_index in range(student_parts.shape[1]):
        relation = balanced_identity_relational_loss(
            student_parts[:, part_index],
            teacher_parts[:, part_index],
            pids,
            part_reliability[:, part_index],
        )
        positive_pairs += relation.positive_pairs
        negative_pairs += relation.negative_pairs
        if relation.positive_pairs + relation.negative_pairs > 0:
            losses.append(relation.loss)
    zero = student_parts.sum() * 0
    return PartRelationalLossResult(
        loss=torch.stack(losses).mean() if losses else zero,
        active_parts=len(losses),
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
    )


def cosine_consistency_loss(
    clean_descriptors: torch.Tensor,
    altered_descriptors: torch.Tensor,
    *,
    base_indices: torch.Tensor | None = None,
    reliability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep the deployed descriptor stable under background-only changes."""

    if clean_descriptors.ndim != 2 or altered_descriptors.ndim != 2:
        raise ValueError("clean and altered descriptors must have shape [N,D]")
    if clean_descriptors.shape[1] != altered_descriptors.shape[1]:
        raise ValueError("clean and altered descriptor dimensions must match")
    _validate_descriptor(clean_descriptors, "clean_descriptors")
    _validate_descriptor(altered_descriptors, "altered_descriptors")
    if base_indices is None:
        if clean_descriptors.shape[0] != altered_descriptors.shape[0]:
            raise ValueError("base_indices are required when clean and altered row counts differ")
        selected_clean = clean_descriptors
    else:
        base_indices = _validated_local_indices(base_indices, clean_descriptors.shape[0], "base_indices")
        if base_indices.numel() != altered_descriptors.shape[0]:
            raise ValueError("base_indices length must equal altered descriptor rows")
        selected_clean = clean_descriptors.index_select(0, base_indices.to(device=clean_descriptors.device))
    if reliability is None:
        reliability = altered_descriptors.new_ones(altered_descriptors.shape[0])
    if reliability.shape != (altered_descriptors.shape[0],):
        raise ValueError("reliability must have shape [N_altered]")
    _validate_unit_interval(reliability, "reliability")

    loss_dtype = torch.float64 if altered_descriptors.dtype == torch.float64 else torch.float32
    selected_clean = selected_clean.to(
        device=altered_descriptors.device,
        dtype=loss_dtype,
    )
    altered = altered_descriptors.to(dtype=loss_dtype)
    values = 1 - F.cosine_similarity(selected_clean, altered, dim=1)
    weights = reliability.detach().to(device=values.device, dtype=values.dtype)
    return _weighted_vector_mean(values, weights, altered.sum() * 0)


def semantic_drop_relational_loss(
    dropped_descriptors: torch.Tensor,
    teacher_leave_part_out: torch.Tensor | None,
    *,
    base_indices: torch.Tensor,
    dropped_parts: torch.Tensor,
    pids: torch.Tensor,
    part_reliability: torch.Tensor,
    intervention_confidence: torch.Tensor | None = None,
) -> PartRelationalLossResult:
    """Match leave-part-out teacher graphs for like-for-like interventions.

    Rows are grouped by the identity of the removed part before graph
    construction.  Comparing relations across different interventions would
    teach an ill-defined target and is deliberately disallowed here.
    """

    if dropped_descriptors.ndim != 2:
        raise ValueError("dropped_descriptors must have shape [N,D]")
    _validate_descriptor(dropped_descriptors, "dropped_descriptors")
    zero = dropped_descriptors.sum() * 0
    if teacher_leave_part_out is None:
        return PartRelationalLossResult(zero, 0, 0, 0)
    if teacher_leave_part_out.ndim != 3:
        raise ValueError("teacher_leave_part_out must have shape [B,P,D]")
    _validate_descriptor(teacher_leave_part_out, "teacher_leave_part_out")
    batch_size, part_count, _ = teacher_leave_part_out.shape
    if pids.shape != (batch_size,):
        raise ValueError("pids must align with teacher batch rows")
    if part_reliability.shape != (batch_size, part_count):
        raise ValueError("part_reliability must have shape [B,P]")
    _validate_unit_interval(part_reliability, "part_reliability")
    base_indices = _validated_local_indices(base_indices, batch_size, "base_indices")
    dropped_parts = _validated_local_indices(dropped_parts, part_count, "dropped_parts")
    row_count = dropped_descriptors.shape[0]
    if base_indices.shape != (row_count,) or dropped_parts.shape != (row_count,):
        raise ValueError("base_indices and dropped_parts must have one entry per dropped descriptor")
    if intervention_confidence is not None:
        if intervention_confidence.shape != (row_count,):
            raise ValueError("intervention_confidence must have shape [N]")
        _validate_unit_interval(intervention_confidence, "intervention_confidence")

    base_device = teacher_leave_part_out.device
    base_indices = base_indices.to(device=base_device)
    dropped_parts = dropped_parts.to(device=base_device)
    losses: list[torch.Tensor] = []
    positive_pairs = 0
    negative_pairs = 0
    for part_index_tensor in dropped_parts.unique(sorted=True):
        part_index = int(part_index_tensor.item())
        row_mask = dropped_parts == part_index
        row_positions = row_mask.nonzero(as_tuple=False).flatten()
        selected_base = base_indices.index_select(0, row_positions)
        selected_reliability = part_reliability.to(device=base_device)[selected_base, part_index]
        if intervention_confidence is not None:
            selected_intervention = intervention_confidence.to(device=base_device).index_select(0, row_positions)
            selected_reliability = fuse_privileged_confidence(
                selected_reliability,
                selected_intervention.to(dtype=selected_reliability.dtype),
                valid=(selected_reliability > 0) & (selected_intervention > 0),
            )
        relation = balanced_identity_relational_loss(
            dropped_descriptors.index_select(0, row_positions.to(device=dropped_descriptors.device)),
            teacher_leave_part_out[selected_base, part_index],
            pids.to(device=base_device).index_select(0, selected_base),
            selected_reliability,
        )
        positive_pairs += relation.positive_pairs
        negative_pairs += relation.negative_pairs
        if relation.positive_pairs + relation.negative_pairs > 0:
            losses.append(relation.loss)
    return PartRelationalLossResult(
        loss=torch.stack(losses).mean() if losses else zero,
        active_parts=len(losses),
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
    )


class PrivilegedGraphLoss(nn.Module):
    """Compose cached privileged and training-only intervention objectives."""

    def __init__(
        self,
        *,
        global_weight: float = 0.30,
        part_weight: float = 0.15,
        background_weight: float = 0.10,
        semantic_drop_weight: float = 0.05,
        visibility_weight: float = 0.5,
        confidence_weight: float = 0.5,
    ) -> None:
        super().__init__()
        weights = {
            "global_weight": global_weight,
            "part_weight": part_weight,
            "background_weight": background_weight,
            "semantic_drop_weight": semantic_drop_weight,
            "visibility_weight": visibility_weight,
            "confidence_weight": confidence_weight,
        }
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("All privileged loss/fusion weights must be finite and non-negative")
        if visibility_weight + confidence_weight <= 0:
            raise ValueError("At least one confidence fusion weight must be positive")
        self.global_weight = float(global_weight)
        self.part_weight = float(part_weight)
        self.background_weight = float(background_weight)
        self.semantic_drop_weight = float(semantic_drop_weight)
        self.visibility_weight = float(visibility_weight)
        self.confidence_weight = float(confidence_weight)

    def forward(
        self,
        student_packet: Mapping[str, torch.Tensor],
        teacher_packet: PrivilegedGraphTeacherBatch | Mapping[str, torch.Tensor | None],
        pids: torch.Tensor,
    ) -> PrivilegedGraphLossResult:
        """Evaluate available auxiliary signals and report each raw term."""

        teacher = (
            teacher_packet
            if isinstance(teacher_packet, PrivilegedGraphTeacherBatch)
            else PrivilegedGraphTeacherBatch.from_mapping(teacher_packet)
        )
        deployed = _packet_tensor(student_packet, DEPLOYED_DESCRIPTOR_KEY)
        if deployed.shape[0] != teacher.batch_size:
            raise ValueError("Student and privileged teacher batch sizes must match")
        if pids.shape != (teacher.batch_size,):
            raise ValueError("pids must have shape [B]")

        part_reliability = fuse_privileged_confidence(
            teacher.part_visibility.to(device=deployed.device),
            teacher.part_confidence.to(device=deployed.device),
            visibility_weight=self.visibility_weight,
            confidence_weight=self.confidence_weight,
        )
        student_part_reliability = _packet_optional_tensor(
            student_packet,
            PART_RELIABILITY_KEY,
        )
        if student_part_reliability is not None:
            if student_part_reliability.shape != part_reliability.shape:
                raise ValueError(f"{PART_RELIABILITY_KEY!r} must have shape {tuple(part_reliability.shape)}")
            student_part_reliability = student_part_reliability.to(
                device=deployed.device,
                dtype=part_reliability.dtype,
            )
            _validate_unit_interval(
                student_part_reliability,
                PART_RELIABILITY_KEY,
            )
            part_reliability = fuse_privileged_confidence(
                part_reliability,
                student_part_reliability,
                valid=(part_reliability > 0) & (student_part_reliability > 0),
            )
        if teacher.global_confidence is None:
            global_reliability = deployed.new_ones(teacher.batch_size)
        else:
            global_reliability = teacher.global_confidence.to(device=deployed.device, dtype=deployed.dtype)
        global_relation = balanced_identity_relational_loss(
            deployed,
            teacher.global_descriptors,
            pids,
            global_reliability,
        )

        zero = deployed.sum() * 0
        part_result = PartRelationalLossResult(zero, 0, 0, 0)
        if self.part_weight > 0:
            student_parts = _packet_tensor(student_packet, PART_DESCRIPTOR_KEY)
            part_result = part_relational_loss(
                student_parts,
                teacher.part_descriptors,
                pids,
                part_reliability,
            )

        background_loss = zero
        if BACKGROUND_DESCRIPTOR_KEY in student_packet:
            background = _packet_tensor(student_packet, BACKGROUND_DESCRIPTOR_KEY)
            background_indices = _packet_optional_tensor(student_packet, BACKGROUND_INDICES_KEY)
            background_confidence = _packet_optional_tensor(student_packet, BACKGROUND_CONFIDENCE_KEY)
            if background_indices is None:
                if background.shape[0] != teacher.batch_size:
                    raise KeyError(
                        f"{BACKGROUND_INDICES_KEY!r} is required for a partial background intervention batch"
                    )
                background_reliability = global_reliability
            else:
                checked_indices = _validated_local_indices(
                    background_indices,
                    teacher.batch_size,
                    BACKGROUND_INDICES_KEY,
                )
                background_reliability = global_reliability.index_select(
                    0,
                    checked_indices.to(device=global_reliability.device),
                )
            if background_confidence is not None:
                background_reliability = fuse_privileged_confidence(
                    background_reliability,
                    background_confidence.to(device=deployed.device, dtype=deployed.dtype),
                    valid=(background_reliability > 0) & (background_confidence.to(device=deployed.device) > 0),
                )
            background_loss = cosine_consistency_loss(
                deployed,
                background,
                base_indices=background_indices,
                reliability=background_reliability,
            )

        semantic_result = PartRelationalLossResult(zero, 0, 0, 0)
        if SEMANTIC_DROP_DESCRIPTOR_KEY in student_packet:
            dropped = _packet_tensor(student_packet, SEMANTIC_DROP_DESCRIPTOR_KEY)
            drop_indices = _packet_tensor(student_packet, SEMANTIC_DROP_INDICES_KEY)
            drop_parts = _packet_tensor(student_packet, SEMANTIC_DROP_PARTS_KEY)
            drop_confidence = _packet_optional_tensor(student_packet, SEMANTIC_DROP_CONFIDENCE_KEY)
            semantic_result = semantic_drop_relational_loss(
                dropped,
                teacher.leave_part_out_descriptors,
                base_indices=drop_indices,
                dropped_parts=drop_parts,
                pids=pids,
                part_reliability=part_reliability,
                intervention_confidence=drop_confidence,
            )

        components = {
            "global_relational": global_relation.loss,
            "part_relational": part_result.loss,
            "background_consistency": background_loss,
            "semantic_drop_relational": semantic_result.loss,
        }
        total = (
            self.global_weight * components["global_relational"]
            + self.part_weight * components["part_relational"]
            + self.background_weight * components["background_consistency"]
            + self.semantic_drop_weight * components["semantic_drop_relational"]
        )
        diagnostics: dict[str, float | int] = {
            "global_positive_pairs": global_relation.positive_pairs,
            "global_negative_pairs": global_relation.negative_pairs,
            "part_active_parts": part_result.active_parts,
            "part_positive_pairs": part_result.positive_pairs,
            "part_negative_pairs": part_result.negative_pairs,
            "semantic_drop_active_parts": semantic_result.active_parts,
            "semantic_drop_positive_pairs": semantic_result.positive_pairs,
            "semantic_drop_negative_pairs": semantic_result.negative_pairs,
            "mean_part_reliability": float(part_reliability.detach().mean().item()),
        }
        return PrivilegedGraphLossResult(total=total, components=components, diagnostics=diagnostics)


def gradient_budget_factor(
    base_grad_norm: torch.Tensor | float,
    auxiliary_grad_norm: torch.Tensor | float,
    *,
    max_ratio: float = 0.30,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return ``min(1, max_ratio * ||g_base|| / ||g_aux||)`` safely."""

    if not math.isfinite(max_ratio) or max_ratio < 0:
        raise ValueError("max_ratio must be finite and non-negative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if torch.is_tensor(base_grad_norm):
        base = base_grad_norm.detach()
        auxiliary = torch.as_tensor(auxiliary_grad_norm, device=base.device, dtype=base.dtype).detach()
    elif torch.is_tensor(auxiliary_grad_norm):
        auxiliary = auxiliary_grad_norm.detach()
        base = torch.as_tensor(base_grad_norm, device=auxiliary.device, dtype=auxiliary.dtype).detach()
    else:
        base = torch.tensor(float(base_grad_norm))
        auxiliary = torch.tensor(float(auxiliary_grad_norm))
    if base.numel() != 1 or auxiliary.numel() != 1:
        raise ValueError("Gradient norms must be scalar")
    if not bool(torch.isfinite(base)) or not bool(torch.isfinite(auxiliary)):
        raise ValueError("Gradient norms must be finite")
    if bool((base < 0) | (auxiliary < 0)):
        raise ValueError("Gradient norms must be non-negative")
    if bool(auxiliary <= epsilon):
        return torch.ones_like(auxiliary)
    return (max_ratio * base / auxiliary.clamp_min(epsilon)).clamp(max=1.0)


def scale_auxiliary_loss_to_gradient_budget(
    base_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: Sequence[nn.Parameter] | Sequence[torch.Tensor],
    *,
    max_ratio: float = 0.30,
    epsilon: float = 1e-12,
) -> GradientBudgetResult:
    """Limit auxiliary gradient pressure on representative shared tensors.

    Call before the final backward pass, then backpropagate
    ``base_loss + result.scaled_loss``.  The two ``autograd.grad`` probes are
    detached diagnostics and do not populate ``tensor.grad``. Callers may pass
    parameters or intermediate activations; using a shared late activation
    avoids an additional reverse traversal through the whole backbone.
    """

    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable:
        raise ValueError("parameters must contain at least one trainable tensor")
    base_norm = _loss_gradient_norm(base_loss, trainable)
    auxiliary_norm = _loss_gradient_norm(auxiliary_loss, trainable)
    scale = gradient_budget_factor(base_norm, auxiliary_norm, max_ratio=max_ratio, epsilon=epsilon)
    scale = scale.to(device=auxiliary_loss.device, dtype=auxiliary_loss.dtype)
    return GradientBudgetResult(
        scaled_loss=auxiliary_loss * scale,
        scale=scale,
        base_grad_norm=base_norm,
        auxiliary_grad_norm=auxiliary_norm,
    )


def _loss_gradient_norm(loss: torch.Tensor, parameters: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if loss.numel() != 1:
        raise ValueError("Losses used for gradient budgeting must be scalar")
    reference = parameters[0]
    if not loss.requires_grad:
        return torch.zeros((), device=reference.device, dtype=torch.float32)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    squared_norm = torch.zeros((), device=reference.device, dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.detach().float().square().sum()
    return squared_norm.sqrt()


def _weighted_pair_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    selected_weights = weights * mask.to(dtype=weights.dtype)
    denominator = selected_weights.sum()
    if not bool(denominator > 0):
        return zero
    return (values * selected_weights).sum() / denominator


def _weighted_vector_mean(values: torch.Tensor, weights: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    denominator = weights.sum()
    if not bool(denominator > 0):
        return zero
    return (values * weights).sum() / denominator


def _validated_local_indices(indices: torch.Tensor, upper_bound: int, name: str) -> torch.Tensor:
    if not torch.is_tensor(indices) or indices.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if indices.dtype == torch.bool or indices.dtype.is_floating_point:
        raise TypeError(f"{name} must use an integer dtype")
    indices = indices.to(dtype=torch.long)
    if indices.numel() and (bool((indices < 0).any()) or bool((indices >= upper_bound).any())):
        raise IndexError(f"{name} contains a value outside [0, {upper_bound})")
    return indices


def _packet_tensor(packet: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in packet:
        raise KeyError(f"Student training packet is missing required key {name!r}")
    return _require_tensor(packet[name], name)


def _packet_optional_tensor(packet: Mapping[str, torch.Tensor], name: str) -> torch.Tensor | None:
    value = packet.get(name)
    return _optional_tensor(value, name)


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    return value


def _optional_tensor(value: object, name: str) -> torch.Tensor | None:
    if value is None:
        return None
    return _require_tensor(value, name)


def _dataset_sample_fields(sample: object, fallback_index: int) -> tuple[int, object, int, int]:
    if isinstance(sample, DatasetSampleProvenance):
        raw_index = sample.index
        img_path = sample.img_path
        raw_pid = sample.pid
        raw_camid = sample.camid
    elif isinstance(sample, Mapping):
        if "index" in sample and "sample_index" in sample and sample["index"] != sample["sample_index"]:
            raise ValueError("Dataset row index and sample_index disagree")
        raw_index = sample.get("index", sample.get("sample_index", fallback_index))
        missing = [name for name in ("img_path", "pid", "camid") if name not in sample]
        if missing:
            raise KeyError(f"Dataset provenance row is missing keys: {missing}")
        img_path = sample["img_path"]
        raw_pid = sample["pid"]
        raw_camid = sample["camid"]
    elif isinstance(sample, Sequence) and not isinstance(sample, (str, bytes, bytearray)):
        if len(sample) != 4:
            raise ValueError("Dataset provenance sequences must be (index, img_path, pid, camid)")
        raw_index, img_path, raw_pid, raw_camid = sample
    elif all(hasattr(sample, name) for name in ("img_path", "pid", "camid")):
        raw_index = getattr(sample, "sample_index", fallback_index)
        img_path = getattr(sample, "img_path")
        raw_pid = getattr(sample, "pid")
        raw_camid = getattr(sample, "camid")
    else:
        raise TypeError("Dataset samples must expose img_path/pid/camid or be explicit provenance mappings/sequences")
    return (
        _strict_integer(raw_index, "sample index"),
        img_path,
        _strict_integer(raw_pid, "pid"),
        _strict_integer(raw_camid, "camid"),
    )


def _strict_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"Dataset {name} must be an integer, not bool")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"Dataset {name} must be an integer") from error


def _normalize_dataset_image_path(value: object) -> str:
    try:
        raw_path = os.fspath(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("Dataset img_path must be a string or path-like value") from error
    if isinstance(raw_path, bytes):
        raise TypeError("Dataset img_path must not be bytes")
    if not raw_path or "\x00" in raw_path:
        raise ValueError("Dataset img_path must be a non-empty path without NUL characters")
    return posixpath.normpath(raw_path.replace("\\", "/"))


def _validate_teacher_tensors(values: Mapping[str, torch.Tensor | None]) -> None:
    sample_indices = _require_tensor(values["sample_indices"], "sample_indices")
    global_descriptors = _require_tensor(values["global_descriptors"], "global_descriptors")
    part_descriptors = _require_tensor(values["part_descriptors"], "part_descriptors")
    part_visibility = _require_tensor(values["part_visibility"], "part_visibility")
    part_confidence = _require_tensor(values["part_confidence"], "part_confidence")
    if sample_indices.ndim != 1:
        raise ValueError("sample_indices must have shape [B]")
    batch_size = sample_indices.shape[0]
    if global_descriptors.ndim != 2 or global_descriptors.shape[0] != batch_size:
        raise ValueError("global_descriptors must have shape [B,D]")
    if part_descriptors.ndim != 3 or part_descriptors.shape[0] != batch_size:
        raise ValueError("part_descriptors must have shape [B,P,D]")
    if part_descriptors.shape[1] <= 0:
        raise ValueError("part_descriptors must contain at least one semantic part")
    if part_visibility.shape != part_descriptors.shape[:2]:
        raise ValueError("part_visibility must have shape [B,P]")
    if part_confidence.shape != part_descriptors.shape[:2]:
        raise ValueError("part_confidence must have shape [B,P]")
    _validate_descriptor(global_descriptors, "global_descriptors")
    _validate_descriptor(part_descriptors, "part_descriptors")
    _validate_unit_interval(part_visibility, "part_visibility")
    _validate_unit_interval(part_confidence, "part_confidence")
    global_confidence = values.get("global_confidence")
    if global_confidence is not None:
        if global_confidence.shape != (batch_size,):
            raise ValueError("global_confidence must have shape [B]")
        _validate_unit_interval(global_confidence, "global_confidence")
    global_reliability = (
        torch.ones(batch_size, device=global_descriptors.device, dtype=torch.bool)
        if global_confidence is None
        else global_confidence.to(device=global_descriptors.device) > 0
    )
    _validate_reliable_descriptor_norms(
        global_descriptors,
        global_reliability,
        "global_descriptors",
    )
    part_reliability = part_visibility.to(device=part_descriptors.device) > 0
    _validate_reliable_descriptor_norms(
        part_descriptors,
        part_reliability,
        "part_descriptors",
    )
    leave_part_out = values.get("leave_part_out_descriptors")
    if leave_part_out is not None:
        if leave_part_out.ndim != 3 or leave_part_out.shape[:2] != part_descriptors.shape[:2]:
            raise ValueError("leave_part_out_descriptors must have shape [B,P,D]")
        _validate_descriptor(leave_part_out, "leave_part_out_descriptors")
        _validate_reliable_descriptor_norms(
            leave_part_out,
            part_reliability.to(device=leave_part_out.device),
            "leave_part_out_descriptors",
        )


def _validate_descriptor(value: torch.Tensor, name: str) -> None:
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point")
    if value.ndim == 0 or value.shape[-1] <= 0:
        raise ValueError(f"{name} must have a positive descriptor dimension")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_reliable_descriptor_norms(
    descriptors: torch.Tensor,
    reliability: torch.Tensor,
    name: str,
) -> None:
    """Reject meaningless zero vectors wherever a teacher signal is active."""

    if reliability.shape != descriptors.shape[:-1]:
        raise ValueError(f"{name} reliability shape must match descriptor rows")
    reliable = reliability.to(device=descriptors.device, dtype=torch.bool)
    if not bool(reliable.any()):
        return
    norms = torch.linalg.vector_norm(descriptors.float(), dim=-1)
    invalid = reliable & (norms <= 1e-12)
    if bool(invalid.any()):
        coordinates = invalid.nonzero(as_tuple=False)[:8].detach().cpu().tolist()
        raise ValueError(f"{name} contains zero-norm rows with positive reliability at {coordinates}")


def validate_part_names(
    part_names: Sequence[str],
    expected_count: int,
) -> tuple[str, ...]:
    """Validate and freeze an exact ordered semantic part-axis contract."""

    if isinstance(part_names, (str, bytes)) or not isinstance(part_names, Sequence):
        raise TypeError("part_names must be an ordered sequence of strings")
    names = tuple(part_names)
    if len(names) != expected_count:
        raise ValueError(f"part_names length must match part_count {expected_count}, got {len(names)}")
    if not names:
        raise ValueError("part_names must be non-empty")
    for name in names:
        if not isinstance(name, str):
            raise TypeError("Every part name must be a string")
        if not name or not name.strip() or name != name.strip():
            raise ValueError("Every part name must be non-empty and have no surrounding whitespace")
    if len(set(names)) != len(names):
        raise ValueError("part_names must be unique while preserving semantic order")
    return names


def _validate_unit_interval(value: torch.Tensor, name: str) -> None:
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    if value.numel() and (bool((value < 0).any()) or bool((value > 1).any())):
        raise ValueError(f"{name} values must lie in [0, 1]")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Cache manifest fields must be finite JSON values") from error
    return encoded.encode("utf-8")


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _tensor_payload_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json({"shape": list(value.shape)}))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")


def _require_matching_hash(expected: str, actual: str, label: str) -> None:
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"Privileged cache {label} SHA-256 mismatch: expected {expected}, got {actual}")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "format",
        "version",
        "sample_count",
        "part_count",
        "part_names",
        "global_dim",
        "part_dim",
        "leave_part_out_dim",
        "dataset_sha256",
        "teacher_sha256",
        "payload_sha256",
        "extra",
        "manifest_sha256",
    }
    if set(manifest) != required:
        raise ValueError(f"Privileged cache manifest fields must be exactly {sorted(required)}")
    if manifest["format"] != CACHE_FORMAT or manifest["version"] != CACHE_VERSION:
        raise ValueError("Unsupported privileged cache format or version")
    for name in ("sample_count", "part_count", "global_dim", "part_dim"):
        if not isinstance(manifest[name], int) or isinstance(manifest[name], bool) or manifest[name] <= 0:
            raise ValueError(f"Manifest {name} must be a positive integer")
    if not isinstance(manifest["part_names"], list):
        raise TypeError("Manifest part_names must be an ordered JSON list")
    validate_part_names(manifest["part_names"], manifest["part_count"])
    leave_dim = manifest["leave_part_out_dim"]
    if leave_dim is not None and (not isinstance(leave_dim, int) or isinstance(leave_dim, bool) or leave_dim <= 0):
        raise ValueError("Manifest leave_part_out_dim must be null or a positive integer")
    if not isinstance(manifest["extra"], dict):
        raise ValueError("Manifest extra must be a dictionary")
    for name in ("dataset_sha256", "teacher_sha256", "payload_sha256", "manifest_sha256"):
        _validate_sha256(manifest[name], name)
    actual_manifest_hash = _manifest_sha256(manifest)
    _require_matching_hash(manifest["manifest_sha256"], actual_manifest_hash, "manifest")


def _validate_manifest_shapes(manifest: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> None:
    part_descriptors = tensors["part_descriptors"]
    leave_part_out = tensors.get("leave_part_out_descriptors")
    actual = {
        "sample_count": int(tensors["sample_indices"].numel()),
        "part_count": int(part_descriptors.shape[1]),
        "global_dim": int(tensors["global_descriptors"].shape[1]),
        "part_dim": int(part_descriptors.shape[2]),
        "leave_part_out_dim": None if leave_part_out is None else int(leave_part_out.shape[2]),
    }
    for name, value in actual.items():
        if manifest[name] != value:
            raise ValueError(f"Manifest {name}={manifest[name]!r} does not match tensor payload {value!r}")
