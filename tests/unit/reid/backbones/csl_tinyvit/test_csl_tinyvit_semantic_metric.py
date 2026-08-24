"""Regression tests for the 7M semantic/metric ablation treatments."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.reid.training.config import ReIDTrainConfig, trainer_kwargs_from_args
from boxmot.reid.training.losses import AdaSPLoss
from boxmot.reid.training.trainer import ReIDTrainer


def _trainer(tmp_path, **kwargs) -> ReIDTrainer:
    values = {
        "model_name": "csl_tinyvit_7m",
        "dataset_name": "market1501",
        "data_dir": str(tmp_path),
        "device": "cpu",
        "pretrained": False,
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "spatial_conv_mode": "depthwise_separable",
        "feat_dim": 384,
        "neck_dim": 384,
        "head_parts": (1, 2, 4),
        "head_type": "standard",
        "part_pooling": "stripes",
        "scale_balanced_branches": True,
        "metric_feature": "raw_concat",
        "inference_feature": "norm_concat_bn",
        "anatomical_auxiliary": False,
        "anatomical_multiscale": False,
        "ema_decay": 0.0,
    }
    values.update(kwargs)
    return ReIDTrainer(**values)


def _adasp_reference(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Direct equal-K transcription of the authors' published implementation."""
    normalized = F.normalize(features, dim=1)
    identities = torch.unique(labels, sorted=True)
    groups = [torch.nonzero(labels == identity).flatten() for identity in identities]
    assert len({group.numel() for group in groups}) == 1
    similarities = normalized @ normalized.T / temperature
    hh_positive = []
    he_positive = []
    negatives = []
    for anchor in groups:
        within = similarities[anchor][:, anchor]
        hh_positive.append(1.0 / torch.exp(-within).sum())
        he_positive.append((1.0 / torch.exp(-within).sum(dim=1)).sum())
        negatives.append(
            torch.stack(
                [torch.exp(similarities[anchor][:, gallery]).sum() for gallery in groups]
            )
        )
    hh_positive = torch.stack(hh_positive)
    he_positive = torch.stack(he_positive)
    weight_hh = hh_positive.log().detach() * temperature
    weight_he = he_positive.log().detach() * temperature
    weights = 2 * weight_hh * weight_he / (weight_hh + weight_he)
    weights = torch.where(weight_hh < 0, torch.zeros_like(weights), weights)
    adaptive_positive = torch.exp(
        weights * hh_positive.log() + (1 - weights) * he_positive.log()
    )
    losses = []
    for index, positive in enumerate(adaptive_positive):
        denominator = positive + sum(
            negatives[index][other]
            for other in range(len(groups))
            if other != index
        )
        losses.append(-torch.log(positive / denominator))
    return torch.stack(losses).mean()


def test_adasp_matches_reference_and_is_batch_permutation_invariant():
    torch.manual_seed(7)
    features = torch.randn(12, 16, dtype=torch.float64, requires_grad=True)
    labels = torch.arange(4).repeat_interleave(3)
    loss = AdaSPLoss(temperature=0.2)(features, labels)
    reference = _adasp_reference(features, labels, temperature=0.2)

    assert torch.allclose(loss, reference, atol=1e-10, rtol=1e-10)
    permutation = torch.randperm(len(labels))
    permuted = AdaSPLoss(temperature=0.2)(
        features[permutation],
        labels[permutation],
    )
    assert torch.allclose(loss, permuted, atol=1e-10, rtol=1e-10)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_adasp_requires_two_ids_with_two_instances_each():
    criterion = AdaSPLoss()
    features = torch.randn(4, 8, requires_grad=True)
    loss = criterion(features, torch.tensor([0, 0, 1, 2]))

    assert loss.item() == 0
    loss.backward()
    assert features.grad is not None


def test_stripe_ce_can_be_removed_while_global_ce_is_preserved(tmp_path):
    trainer = _trainer(
        tmp_path,
        coarse_branch_ce_weight=0.0,
        fine_branch_ce_weight=0.0,
    )
    losses = [
        torch.tensor(1.0),
        torch.tensor(3.0),
        torch.tensor(5.0),
        torch.tensor(100.0),
        torch.tensor(100.0),
        torch.tensor(100.0),
        torch.tensor(100.0),
    ]

    actual = trainer._scale_balanced_classification_loss(losses, aux_weight=1.0)

    # Every stripe is excluded from CE; full-concat metric learning remains.
    assert actual.item() == pytest.approx(1.0)


def test_part_relation_only_trains_fine_parts_and_global(tmp_path):
    trainer = _trainer(
        tmp_path,
        coarse_branch_ce_weight=0.0,
        fine_branch_ce_weight=0.0,
        part_relation_weight=0.25,
        part_to_global_weight=0.1,
    )
    labels = torch.arange(4).repeat_interleave(3)
    student = {"global": torch.randn(12, 32, requires_grad=True)}
    teacher = {"global": torch.randn(12, 32)}
    for index, width in enumerate((16, 16, 8, 8, 8, 8)):
        student[f"part{index}"] = torch.randn(12, width, requires_grad=True)
        teacher[f"part{index}"] = torch.randn(12, width)

    relation, global_distill = trainer._part_relation_losses(student, teacher, labels)
    total = 0.25 * relation + 0.1 * global_distill
    total.backward()

    assert torch.isfinite(total)
    assert student["global"].grad is not None
    assert student["part0"].grad is None
    assert student["part1"].grad is None
    for index in range(2, 6):
        assert student[f"part{index}"].grad is not None


def test_part_relation_builds_training_only_ema_and_keeps_live_validation(tmp_path):
    trainer = _trainer(
        tmp_path,
        coarse_branch_ce_weight=0.0,
        fine_branch_ce_weight=0.0,
        part_relation_weight=0.25,
        part_to_global_weight=0.1,
    )

    models = trainer._build_model_bundle(num_classes=751)

    assert models.ema_model is not None
    assert models.val_model is models.model
    assert models.model.head.return_auxiliary_features is True
    assert models.ema_model.head.return_auxiliary_features is True
    assert all(not parameter.requires_grad for parameter in models.ema_model.parameters())
    teacher_features = trainer._ema_part_teacher_features(
        models.ema_model,
        torch.zeros(2, 3, 384, 128),
    )
    assert {f"part{index}" for index in range(6)} <= teacher_features.keys()
    assert models.ema_model.training is False
    assert models.ema_model.head.training is False
    models.model.eval()
    with torch.inference_mode():
        descriptor = models.model(torch.zeros(1, 3, 384, 128))
    assert descriptor.shape == (1, 1152)
    metadata = trainer._checkpoint_metadata(models.model)
    assert metadata["coarse_branch_ce_weight"] == 0
    assert metadata["fine_branch_ce_weight"] == 0
    assert metadata["part_relation_weight"] == 0.25
    assert metadata["part_to_global_weight"] == 0.1
    assert metadata["resume_contract"]["loss"]["part_relation_weight"] == 0.25


def test_adasp_replacement_does_not_build_or_compute_triplet(tmp_path):
    trainer = _trainer(
        tmp_path,
        metric_loss_weight=0.0,
        adasp_loss_weight=1.0,
        adasp_temperature=0.04,
        adasp_scale=0.1,
    )
    models = trainer._build_model_bundle(num_classes=751)

    losses = trainer._build_loss_bundle(models, num_classes=751)

    assert losses.criterion_metric is None
    assert isinstance(losses.criterion_adasp, AdaSPLoss)


def test_cli_propagates_semantic_metric_options(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.reid.trainer",
        SimpleNamespace(main=fake_main),
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--recipe",
            "csl_tinyvit_7m_v20",
            "--data-dir",
            ".",
            "--adasp-loss-weight",
            "0.5",
            "--coarse-branch-ce-weight",
            "0",
            "--fine-branch-ce-weight",
            "0",
            "--part-relation-weight",
            "0.25",
            "--part-to-global-weight",
            "0.1",
            "--no-anatomical-auxiliary",
            "--no-anatomical-multiscale",
            "--anatomical-distill-weight",
            "0",
            "--anatomical-attention-weight",
            "0",
            "--anatomical-visibility-weight",
            "0",
            "--anatomical-contrastive-weight",
            "0",
            "--anatomical-pose-teacher-weight",
            "0",
            "--anatomical-cross-scale-weight",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.adasp_loss_weight == 0.5
    assert args.adasp_temperature == 0.04
    assert args.adasp_scale == 0.1
    assert args.coarse_branch_ce_weight == 0
    assert args.fine_branch_ce_weight == 0
    assert args.part_relation_weight == 0.25
    assert args.part_to_global_weight == 0.1
    assert args.part_relation_teacher_momentum == 0.999
    assert args.part_relation_temperature == 0.07
    kwargs = trainer_kwargs_from_args(args, {})
    config = ReIDTrainConfig.from_flat_kwargs(**kwargs)
    trainer = ReIDTrainer.from_config(config)
    assert trainer.adasp_loss_weight == 0.5
    assert trainer.adasp_temperature == 0.04
    assert trainer.adasp_scale == 0.1
    assert trainer.coarse_branch_ce_weight == 0
    assert trainer.fine_branch_ce_weight == 0
    assert trainer.part_relation_weight == 0.25
    assert trainer.part_to_global_weight == 0.1
    assert trainer.return_auxiliary_features is True
