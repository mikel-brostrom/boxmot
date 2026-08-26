"""Focused tests for identity-only GlobalAP instance memory."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from boxmot.reid.training.trainer_components.global_ap import (
    IdentityGlobalAP,
)


def _metadata(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def test_memory_update_is_detached_normalized_and_instance_indexed() -> None:
    memory = IdentityGlobalAP(memory_size=6, feature_dim=3)
    descriptors = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        requires_grad=True,
    )

    memory.update(
        descriptors,
        _metadata(4, 1),
        _metadata(7, 9),
    )

    torch.testing.assert_close(
        memory.memory_features[4],
        torch.tensor([0.6, 0.8, 0.0]),
    )
    torch.testing.assert_close(
        memory.memory_features[1],
        torch.tensor([0.0, 0.0, 1.0]),
    )
    assert memory.memory_features.dtype == torch.float32
    assert memory.memory_features.requires_grad is False
    assert memory.memory_pids.tolist() == [-1, 9, -1, -1, 7, -1]
    assert memory.memory_valid.tolist() == [False, True, False, False, True, False]
    assert memory.num_valid == 2
    assert memory.memory_step.item() == 1


def test_same_identity_positive_drives_fp32_smooth_ap_gradient() -> None:
    memory = IdentityGlobalAP(
        memory_size=4,
        feature_dim=2,
        top_k=None,
        temperature=0.2,
    )
    memory.update(
        torch.tensor([[0.8, 0.6], [1.0, 0.0]]),
        _metadata(1, 2),
        _metadata(3, 8),
    )
    query = torch.tensor([[1.0, 0.2]], dtype=torch.float16, requires_grad=True)

    loss = memory(
        query,
        _metadata(0),
        _metadata(3),
    )
    loss.backward()

    normalized_query = F.normalize(query.detach().float(), dim=1)
    positive_score = (normalized_query @ memory.memory_features[1, :, None]).squeeze()
    negative_score = (normalized_query @ memory.memory_features[2, :, None]).squeeze()
    expected = 1.0 - 1.0 / (1.0 + torch.sigmoid((negative_score - positive_score) / 0.2))
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected)
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.count_nonzero(query.grad).item() > 0


def test_stable_sample_self_is_excluded_but_same_pid_non_self_is_positive() -> None:
    memory = IdentityGlobalAP(
        memory_size=5,
        feature_dim=2,
        top_k=None,
        temperature=0.1,
    )
    indices = _metadata(0, 1, 2)
    pids = _metadata(4, 4, 9)
    memory.update(
        torch.tensor([[0.0, 1.0], [0.7, 0.7], [0.9, 0.1]]),
        indices,
        pids,
    )
    query = torch.tensor([[1.0, 0.1]], requires_grad=True)
    query_args = (query, _metadata(0), _metadata(4))
    original = memory(*query_args)

    # Changing the query's own stale memory row cannot alter the loss.
    memory.update(
        torch.tensor([[-1.0, 0.0]]),
        _metadata(0),
        _metadata(4),
    )
    self_changed = memory(*query_args)
    torch.testing.assert_close(self_changed, original, rtol=0, atol=0)

    # The other sample of the same PID is a positive regardless of camera.
    memory.update(
        torch.tensor([[-1.0, 0.0]]),
        _metadata(1),
        _metadata(4),
    )
    positive_changed = memory(*query_args)
    assert positive_changed > original


def test_top_k_uses_only_the_hard_negative_neighborhood() -> None:
    memory = IdentityGlobalAP(
        memory_size=6,
        feature_dim=2,
        top_k=1,
        temperature=0.1,
    )
    memory.update(
        torch.tensor(
            [
                [0.8, 0.6],  # same-identity positive
                [0.95, 0.05],  # hard negative
                [-1.0, 0.0],  # easy negative outside negative top-k
            ]
        ),
        _metadata(1, 2, 3),
        _metadata(5, 8, 9),
    )
    query = torch.tensor([[1.0, 0.15]], requires_grad=True)
    args = (query, _metadata(0), _metadata(5))
    baseline = memory(*args)

    memory.update(
        torch.tensor([[-0.8, -0.6]]),
        _metadata(3),
        _metadata(9),
    )
    easy_negative_changed = memory(*args)

    torch.testing.assert_close(easy_negative_changed, baseline, rtol=0, atol=0)
    assert baseline.item() > 0


def test_positive_below_negative_top_k_still_produces_loss_and_gradient() -> None:
    memory = IdentityGlobalAP(
        memory_size=5,
        feature_dim=2,
        top_k=1,
        temperature=0.1,
        max_age=None,
    )
    memory.update(
        torch.tensor(
            [
                [0.8, 0.6],  # same-identity positive below the top-1 cutoff
                [1.0, 0.0],  # hardest negative
                [-0.8, 0.6],  # easy negative outside the negative top-k
            ]
        ),
        _metadata(1, 2, 3),
        _metadata(7, 8, 9),
    )
    query = torch.tensor([[1.0, 0.2]], requires_grad=True)

    loss = memory(
        query,
        _metadata(0),
        _metadata(7),
    )
    loss.backward()

    assert loss.item() > 0
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.count_nonzero(query.grad).item() > 0


def test_empty_same_identity_positive_returns_differentiable_zero() -> None:
    memory = IdentityGlobalAP(memory_size=4, feature_dim=2, top_k=2)
    memory.update(
        torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        _metadata(1, 2),
        _metadata(7, 8),
    )
    query = torch.tensor([[0.7, 0.3]], requires_grad=True)

    loss = memory(
        query,
        _metadata(0),
        _metadata(6),
    )
    loss.backward()

    assert loss.shape == torch.Size([])
    assert loss.dtype == torch.float32
    assert loss.item() == 0.0
    assert query.grad is not None
    torch.testing.assert_close(query.grad, torch.zeros_like(query.grad))


def test_max_age_excludes_stale_positives() -> None:
    memory = IdentityGlobalAP(
        memory_size=8,
        feature_dim=2,
        top_k=None,
        max_age=1,
    )
    memory.update(
        torch.tensor([[0.8, 0.6], [1.0, 0.0]]),
        _metadata(1, 2),
        _metadata(3, 8),
    )
    query = torch.tensor([[1.0, 0.1]], requires_grad=True)
    args = (query, _metadata(0), _metadata(3))
    assert memory(*args).item() > 0

    memory.update(
        torch.tensor([[0.0, 1.0]]),
        _metadata(6),
        _metadata(10),
    )
    assert memory(*args).item() > 0
    memory.update(
        torch.tensor([[-1.0, 0.0]]),
        _metadata(7),
        _metadata(11),
    )

    stale_loss = memory(*args)
    assert stale_loss.item() == 0.0


def test_staleness_decay_downweights_an_old_hard_negative() -> None:
    decayed = IdentityGlobalAP(
        memory_size=4,
        feature_dim=2,
        top_k=None,
        max_age=None,
        staleness_tau=0.5,
        temperature=0.1,
    )
    unweighted = IdentityGlobalAP(
        memory_size=4,
        feature_dim=2,
        top_k=None,
        max_age=None,
        staleness_tau=None,
        temperature=0.1,
    )
    for memory in (decayed, unweighted):
        memory.update(
            torch.tensor([[0.8, 0.6], [1.0, 0.0]]),
            _metadata(1, 2),
            _metadata(3, 8),
            step=1,
        )
        # Refresh only the positive.  The hard negative is now two steps old.
        memory.update(
            torch.tensor([[0.8, 0.6]]),
            _metadata(1),
            _metadata(3),
            step=3,
        )

    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    args = (query, _metadata(0), _metadata(3))
    assert 0 < decayed(*args).item() < unweighted(*args).item()


def test_memory_state_dict_round_trip_restores_loss_and_age_clock() -> None:
    source = IdentityGlobalAP(
        memory_size=5,
        feature_dim=3,
        top_k=3,
        max_age=7,
        staleness_tau=3.0,
        memory_momentum=0.25,
    )
    source.update(
        torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.9, 0.0, 0.1]]),
        _metadata(0, 2, 4),
        _metadata(1, 1, 7),
        step=11,
    )
    restored = IdentityGlobalAP(
        memory_size=5,
        feature_dim=3,
        top_k=3,
        max_age=7,
        staleness_tau=3.0,
        memory_momentum=0.25,
    )
    restored.load_state_dict(source.state_dict(), strict=True)
    query = torch.tensor([[1.0, 0.1, 0.0]], requires_grad=True)
    args = (query, _metadata(1), _metadata(1))

    assert restored.memory_step.item() == 11
    assert restored.num_valid == source.num_valid
    assert set(restored.state_dict()) == {
        "memory_features",
        "memory_pids",
        "memory_last_update",
        "memory_valid",
        "memory_step",
    }
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    torch.testing.assert_close(restored(*args), source(*args), rtol=0, atol=0)


def test_duplicate_indices_are_averaged_and_metadata_must_be_stable() -> None:
    memory = IdentityGlobalAP(memory_size=3, feature_dim=2)
    memory.update(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        _metadata(1, 1),
        _metadata(5, 5),
    )
    expected = F.normalize(torch.tensor([0.5, 0.5]), dim=0)
    torch.testing.assert_close(memory.memory_features[1], expected)

    with pytest.raises(ValueError, match="metadata changed"):
        memory.update(
            torch.tensor([[1.0, 0.0]]),
            _metadata(1),
            _metadata(6),
        )


def test_invalid_stable_index_and_backward_step_are_rejected() -> None:
    memory = IdentityGlobalAP(memory_size=2, feature_dim=2)
    with pytest.raises(TypeError, match="sample_indices must use an integer dtype"):
        memory.update(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0.5]),
            _metadata(1),
        )
    with pytest.raises(IndexError, match="stable sample indices"):
        memory.update(
            torch.tensor([[1.0, 0.0]]),
            _metadata(2),
            _metadata(1),
        )
    memory.update(
        torch.tensor([[1.0, 0.0]]),
        _metadata(0),
        _metadata(1),
        step=4,
    )
    with pytest.raises(ValueError, match="nondecreasing"):
        memory.update(
            torch.tensor([[1.0, 0.0]]),
            _metadata(0),
            _metadata(1),
            step=3,
        )
