"""Tests for isolated training-only privileged graph supervision."""

from __future__ import annotations

import math

import pytest
import torch

from boxmot.reid.training.trainer_components.privileged_graph import (
    BACKGROUND_DESCRIPTOR_KEY,
    BACKGROUND_INDICES_KEY,
    DEPLOYED_DESCRIPTOR_KEY,
    PART_DESCRIPTOR_KEY,
    PART_RELIABILITY_KEY,
    SEMANTIC_DROP_DESCRIPTOR_KEY,
    SEMANTIC_DROP_INDICES_KEY,
    SEMANTIC_DROP_PARTS_KEY,
    PrivilegedGraphLoss,
    PrivilegedGraphTeacherBatch,
    PrivilegedGraphTeacherCache,
    balanced_identity_relational_loss,
    cosine_consistency_loss,
    fuse_privileged_confidence,
    gradient_budget_factor,
    part_relational_loss,
    privileged_graph_integration_hooks,
    scale_auxiliary_loss_to_gradient_budget,
    semantic_drop_relational_loss,
    validate_part_names,
)

PART_NAMES = ("upper", "lower")


def _teacher_tensors() -> dict[str, torch.Tensor]:
    sample_indices = torch.tensor([30, 10, 40, 20])
    global_descriptors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
        ]
    )
    part_descriptors = torch.stack((global_descriptors, global_descriptors.roll(1, dims=1)), dim=1)
    part_visibility = torch.tensor([[1.0, 0.8], [1.0, 0.0], [0.9, 1.0], [0.7, 0.6]])
    part_confidence = torch.tensor([[0.9, 0.7], [0.8, 0.0], [0.8, 0.9], [0.6, 0.5]])
    leave_part_out = torch.stack((global_descriptors.roll(1, dims=1), global_descriptors.roll(2, dims=1)), dim=1)
    return {
        "sample_indices": sample_indices,
        "global_descriptors": global_descriptors,
        "part_descriptors": part_descriptors,
        "part_visibility": part_visibility,
        "part_confidence": part_confidence,
        "global_confidence": torch.tensor([0.9, 0.8, 0.9, 0.7]),
        "leave_part_out_descriptors": leave_part_out,
    }


def _cache() -> PrivilegedGraphTeacherCache:
    return PrivilegedGraphTeacherCache(part_names=PART_NAMES, **_teacher_tensors())


def _relation_batch() -> tuple[torch.Tensor, torch.Tensor]:
    descriptors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
        ]
    )
    return descriptors, torch.tensor([0, 0, 1, 1])


def test_cache_round_trip_and_lookup_preserve_requested_stable_order(tmp_path) -> None:
    cache = _cache()
    destination = tmp_path / "privileged-cache.pt"
    dataset_hash = "a" * 64
    teacher_hash = "b" * 64

    manifest = cache.save(
        destination,
        dataset_sha256=dataset_hash,
        teacher_sha256=teacher_hash,
        extra={"teacher_family": "test"},
    )
    loaded = PrivilegedGraphTeacherCache.load(
        destination,
        expected_dataset_sha256=dataset_hash,
        expected_teacher_sha256=teacher_hash,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_part_names=PART_NAMES,
    )
    batch = loaded.lookup(torch.tensor([20, 30, 20]))
    source = _teacher_tensors()

    assert batch.sample_indices.tolist() == [20, 30, 20]
    assert torch.equal(batch.global_descriptors, source["global_descriptors"][[3, 0, 3]])
    assert torch.equal(batch.part_descriptors, source["part_descriptors"][[3, 0, 3]])
    assert torch.equal(batch.leave_part_out_descriptors, source["leave_part_out_descriptors"][[3, 0, 3]])
    assert all(not value.requires_grad for value in batch.as_mapping().values() if value is not None)
    assert loaded.manifest == manifest
    assert loaded.part_names == PART_NAMES
    assert manifest["part_names"] == list(PART_NAMES)


def test_cache_rejects_missing_indices_provenance_mismatch_and_payload_tampering(tmp_path) -> None:
    cache = _cache()
    destination = tmp_path / "privileged-cache.pt"
    cache.save(destination, dataset_sha256="a" * 64, teacher_sha256="b" * 64)

    with pytest.raises(KeyError, match="999"):
        cache.lookup([999])
    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        PrivilegedGraphTeacherCache.load(destination, expected_dataset_sha256="c" * 64)

    payload = torch.load(destination, map_location="cpu", weights_only=True)
    payload["tensors"]["global_descriptors"][0, 0] += 1
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        PrivilegedGraphTeacherCache.load(tampered)


def test_cache_rejects_duplicate_or_non_integer_stable_indices() -> None:
    values = _teacher_tensors()
    values["sample_indices"] = torch.tensor([10, 10, 20, 30])
    with pytest.raises(ValueError, match="unique"):
        PrivilegedGraphTeacherCache(part_names=PART_NAMES, **values)

    values["sample_indices"] = torch.tensor([10.0, 11.0, 20.0, 30.0])
    with pytest.raises(TypeError, match="integer"):
        PrivilegedGraphTeacherCache(part_names=PART_NAMES, **values)


def test_cache_part_names_are_unique_ordered_signed_schema(tmp_path) -> None:
    values = _teacher_tensors()
    with pytest.raises(ValueError, match="length must match"):
        PrivilegedGraphTeacherCache(part_names=("only",), **values)
    with pytest.raises(ValueError, match="unique"):
        PrivilegedGraphTeacherCache(part_names=("same", "same"), **values)
    with pytest.raises(ValueError, match="non-empty"):
        PrivilegedGraphTeacherCache(part_names=("upper", ""), **values)
    with pytest.raises(TypeError, match="ordered sequence"):
        validate_part_names("upper,lower", 2)

    destination = tmp_path / "named-cache.pt"
    manifest = _cache().save(
        destination,
        dataset_sha256="a" * 64,
        teacher_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="ordered part names mismatch"):
        PrivilegedGraphTeacherCache.load(
            destination,
            expected_part_names=tuple(reversed(PART_NAMES)),
        )

    payload = torch.load(destination, map_location="cpu", weights_only=True)
    payload["manifest"]["part_names"].reverse()
    tampered = tmp_path / "part-name-tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        PrivilegedGraphTeacherCache.load(tampered)
    assert manifest["version"] == 2


def test_cache_rejects_zero_width_and_reliable_zero_norm_teacher_rows() -> None:
    values = _teacher_tensors()
    values["global_descriptors"] = torch.empty(4, 0)
    with pytest.raises(ValueError, match="positive descriptor dimension"):
        PrivilegedGraphTeacherCache(part_names=PART_NAMES, **values)

    values = _teacher_tensors()
    values["global_descriptors"][0].zero_()
    with pytest.raises(ValueError, match="positive reliability"):
        PrivilegedGraphTeacherCache(part_names=PART_NAMES, **values)

    values = _teacher_tensors()
    values["part_descriptors"][0, 0].zero_()
    with pytest.raises(ValueError, match="positive reliability"):
        PrivilegedGraphTeacherCache(part_names=PART_NAMES, **values)

    # Zero vectors remain a valid compact sentinel for genuinely absent parts.
    values["part_visibility"][0, 0] = 0
    values["leave_part_out_descriptors"][0, 0].zero_()
    cache = PrivilegedGraphTeacherCache(part_names=PART_NAMES, **values)
    assert cache.part_names == PART_NAMES


def test_confidence_fusion_uses_mean_but_never_resurrects_missing_parts() -> None:
    visibility = torch.tensor([0.2, 0.0, 1.0, 0.4])
    confidence = torch.tensor([0.8, 0.9, 0.0, 0.6])

    fused = fuse_privileged_confidence(visibility, confidence)

    assert torch.allclose(fused, torch.tensor([0.5, 0.0, 0.5, 0.5]))
    assert not math.isclose(float(fused[0]), float(visibility[0] * confidence[0]))


def test_balanced_identity_relation_matches_geometry_with_different_dimensions() -> None:
    teacher, pids = _relation_batch()
    student = torch.cat((teacher, torch.zeros(4, 2)), dim=1).requires_grad_()

    result = balanced_identity_relational_loss(student, teacher, pids)

    assert result.positive_pairs == 4
    assert result.negative_pairs == 8
    assert result.loss.item() == pytest.approx(0.0, abs=1e-8)
    result.loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_relation_keeps_every_different_identity_pair_as_a_negative() -> None:
    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    student = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    pids = torch.tensor([0, 1])

    result = balanced_identity_relational_loss(student, teacher, pids)

    assert result.positive_pairs == 0
    assert result.negative_pairs == 2
    assert result.loss > 0
    result.loss.backward()
    assert student.grad is not None


def test_relation_keeps_every_non_self_same_identity_pair_as_a_positive() -> None:
    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    student = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    pids = torch.tensor([7, 7])

    result = balanced_identity_relational_loss(student, teacher, pids)

    assert result.positive_pairs == 2
    assert result.negative_pairs == 0
    assert result.loss > 0
    result.loss.backward()
    assert student.grad is not None


def test_relation_balances_positive_and_negative_groups_instead_of_pair_counts() -> None:
    teacher, pids = _relation_batch()
    student = teacher.clone()
    student[1] = torch.tensor([0.0, 0.0, 1.0])

    result = balanced_identity_relational_loss(student, teacher, pids)

    expected = (result.positive_loss + result.negative_loss) / 2
    assert result.loss.item() == pytest.approx(expected.item())
    assert result.positive_loss > 0
    assert result.negative_loss > 0


def test_part_relation_ignores_missing_parts_and_allows_different_feature_dims() -> None:
    teacher, pids = _relation_batch()
    student_parts = torch.stack((teacher[:, :2], teacher[:, :2]), dim=1).requires_grad_()
    teacher_parts = torch.stack(
        (
            torch.cat((teacher[:, :2], torch.zeros(4, 1)), dim=1),
            torch.full((4, 3), 1000.0),
        ),
        dim=1,
    )
    reliability = torch.tensor([[1.0, 0.0]]).expand(4, -1)

    result = part_relational_loss(student_parts, teacher_parts, pids, reliability)

    assert result.active_parts == 1
    assert result.loss.item() == pytest.approx(0.0, abs=1e-8)
    result.loss.backward()
    assert student_parts.grad is not None
    assert torch.equal(student_parts.grad[:, 1], torch.zeros_like(student_parts.grad[:, 1]))


def test_background_consistency_uses_explicit_base_rows_and_weights() -> None:
    clean = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    altered = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    indices = torch.tensor([1, 0])

    matching = cosine_consistency_loss(clean, altered, base_indices=indices)
    mismatching = cosine_consistency_loss(
        clean,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True),
        base_indices=indices,
        reliability=torch.tensor([1.0, 0.0]),
    )

    assert matching.item() == pytest.approx(0.0)
    assert mismatching.item() == pytest.approx(1.0)


def test_semantic_drop_relations_are_grouped_by_removed_part() -> None:
    base, pids = _relation_batch()
    leave_part_out = torch.stack((base, base.roll(1, dims=1)), dim=1)
    base_indices = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    dropped_parts = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    dropped = torch.cat((leave_part_out[:, 0], leave_part_out[:, 1]), dim=0).requires_grad_()

    result = semantic_drop_relational_loss(
        dropped,
        leave_part_out,
        base_indices=base_indices,
        dropped_parts=dropped_parts,
        pids=pids,
        part_reliability=torch.ones(4, 2),
    )
    unavailable = semantic_drop_relational_loss(
        dropped,
        None,
        base_indices=base_indices,
        dropped_parts=dropped_parts,
        pids=pids,
        part_reliability=torch.ones(4, 2),
    )

    assert result.active_parts == 2
    assert result.positive_pairs == 8
    assert result.negative_pairs == 16
    assert result.loss.item() == pytest.approx(0.0, abs=1e-8)
    assert unavailable.active_parts == 0
    assert unavailable.loss.item() == pytest.approx(0.0)


def test_composed_loss_consumes_existing_head_packet_and_reports_components() -> None:
    teacher, pids = _relation_batch()
    teacher_parts = torch.stack((teacher, teacher.roll(1, dims=1)), dim=1)
    teacher_batch = PrivilegedGraphTeacherBatch(
        sample_indices=torch.arange(4),
        global_descriptors=teacher,
        part_descriptors=teacher_parts,
        part_visibility=torch.ones(4, 2),
        part_confidence=torch.ones(4, 2),
    )
    deployed = teacher.clone().requires_grad_()
    student_packet = {
        DEPLOYED_DESCRIPTOR_KEY: deployed,
        PART_DESCRIPTOR_KEY: teacher_parts.clone().requires_grad_(),
        BACKGROUND_DESCRIPTOR_KEY: teacher[[1, 0]].clone().requires_grad_(),
        BACKGROUND_INDICES_KEY: torch.tensor([1, 0]),
    }
    objective = PrivilegedGraphLoss(
        global_weight=0.3,
        part_weight=0.2,
        background_weight=0.1,
        semantic_drop_weight=0.0,
    )

    result = objective(student_packet, teacher_batch, pids)

    assert set(result.components) == {
        "global_relational",
        "part_relational",
        "background_consistency",
        "semantic_drop_relational",
    }
    assert result.total.item() == pytest.approx(0.0, abs=1e-8)
    assert result.diagnostics["part_active_parts"] == 2
    result.total.backward()
    assert deployed.grad is not None


def test_composed_loss_requires_existing_anatomical_packet_only_when_weighted() -> None:
    teacher, pids = _relation_batch()
    teacher_batch = PrivilegedGraphTeacherBatch(
        sample_indices=torch.arange(4),
        global_descriptors=teacher,
        part_descriptors=teacher[:, None],
        part_visibility=torch.ones(4, 1),
        part_confidence=torch.ones(4, 1),
    )
    packet = {DEPLOYED_DESCRIPTOR_KEY: teacher.clone().requires_grad_()}

    with pytest.raises(KeyError, match=PART_DESCRIPTOR_KEY):
        PrivilegedGraphLoss(part_weight=0.1)(packet, teacher_batch, pids)
    result = PrivilegedGraphLoss(part_weight=0.0)(packet, teacher_batch, pids)
    assert result.components["part_relational"].item() == 0


def test_runtime_crop_reliability_suppresses_missing_student_parts() -> None:
    teacher, pids = _relation_batch()
    teacher_batch = PrivilegedGraphTeacherBatch(
        sample_indices=torch.arange(4),
        global_descriptors=teacher,
        part_descriptors=teacher[:, None],
        part_visibility=torch.ones(4, 1),
        part_confidence=torch.ones(4, 1),
    )
    packet = {
        DEPLOYED_DESCRIPTOR_KEY: teacher.clone().requires_grad_(),
        PART_DESCRIPTOR_KEY: teacher[:, None].clone().requires_grad_(),
        PART_RELIABILITY_KEY: torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
    }

    result = PrivilegedGraphLoss(
        global_weight=0.0,
        part_weight=1.0,
        background_weight=0.0,
        semantic_drop_weight=0.0,
    )(packet, teacher_batch, pids)

    assert result.diagnostics["part_positive_pairs"] == 0
    assert result.diagnostics["part_negative_pairs"] == 2
    assert result.diagnostics["mean_part_reliability"] == pytest.approx(0.5)


def test_gradient_budget_limits_auxiliary_norm_without_populating_grads() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    base_loss = 0.5 * parameter.square()
    auxiliary_loss = 10.0 * parameter

    result = scale_auxiliary_loss_to_gradient_budget(
        base_loss,
        auxiliary_loss,
        [parameter],
        max_ratio=0.25,
    )

    assert result.base_grad_norm.item() == pytest.approx(2.0)
    assert result.auxiliary_grad_norm.item() == pytest.approx(10.0)
    assert result.scale.item() == pytest.approx(0.05)
    assert parameter.grad is None
    assert gradient_budget_factor(2.0, 0.0, max_ratio=0.25).item() == 1.0


def test_integration_hooks_keep_privileged_branches_out_of_inference() -> None:
    hooks = privileged_graph_integration_hooks()
    contracts = " ".join(hook.contract for hook in hooks)

    assert "stable sample index" in contracts
    assert DEPLOYED_DESCRIPTOR_KEY in contracts
    assert PART_DESCRIPTOR_KEY in contracts
    assert "prune training-only" in contracts
    assert SEMANTIC_DROP_DESCRIPTOR_KEY.startswith("_privileged_graph")
    assert SEMANTIC_DROP_INDICES_KEY.startswith("_privileged_graph")
    assert SEMANTIC_DROP_PARTS_KEY.startswith("_privileged_graph")
