from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from boxmot.reid.backbones.families.csl_tinyvit.multilevel_suppression import (
    MULTILEVEL_SUPPRESSION_IMPLEMENTATION_VERSION,
)
from boxmot.reid.training.losses import CenterLoss, CrossEntropyLabelSmooth
from boxmot.reid.training.model_options import build_reid_model_kwargs
from boxmot.reid.training.resume import (
    contract_differences,
    contract_fingerprint,
)
from boxmot.reid.training.trainer import ReIDTrainer
from boxmot.reid.training.trainer_components.types import (
    DatasetBundle,
    LossBundle,
    ModelBundle,
    TrainMetrics,
)


def _suppression_trainer(tmp_path, **overrides) -> ReIDTrainer:
    values = {
        "model_name": "csl_tinyvit_7m",
        "dataset_name": "market1501",
        "data_dir": str(tmp_path),
        "epochs": 200,
        "warmup_epochs": 20,
        "pretrained": False,
        "classifier_loss": "ce",
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "spatial_conv_mode": "depthwise_separable",
        "feat_dim": 384,
        "neck_dim": 384,
        "metric_feature": "raw_concat",
        "inference_feature": "norm_concat_bn",
        "head_type": "standard",
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "scale_balanced_branches": True,
        "multilevel_suppression": True,
        "multilevel_suppression_ratio": 0.15,
        "multilevel_suppression_loss_weight": 0.20,
        "multilevel_suppression_start_epoch": 20,
        "multilevel_suppression_ramp_end_epoch": 50,
        "multilevel_suppression_decay_start_epoch": 140,
        "multilevel_suppression_decay_end_epoch": 170,
    }
    values.update(overrides)
    return ReIDTrainer(**values)


def test_multilevel_suppression_schedule_ramps_holds_and_decays(tmp_path):
    trainer = _suppression_trainer(tmp_path)

    assert trainer._multilevel_suppression_progress(20) == 0.0
    assert trainer._multilevel_suppression_progress(35) == 0.5
    assert trainer._multilevel_suppression_progress(50) == 1.0
    assert trainer._multilevel_suppression_progress(140) == 1.0
    assert trainer._multilevel_suppression_progress(155) == 0.5
    assert trainer._multilevel_suppression_progress(170) == 0.0


def test_multilevel_suppression_aux_ce_is_scale_balanced(tmp_path):
    trainer = _suppression_trainer(tmp_path)
    trainer._current_multilevel_suppression_progress = 1.0
    pids = torch.tensor([0, 1, 2])
    coarse = tuple(torch.randn(3, 3, requires_grad=True) for _ in range(2))
    fine = tuple(torch.randn(3, 3, requires_grad=True) for _ in range(4))
    features = {
        "_multilevel_suppression_logits": {
            "coarse": coarse,
            "fine": fine,
        },
        "_multilevel_suppression_active": {
            "coarse": torch.ones(3, 2, dtype=torch.bool),
            "fine": torch.ones(3, 4, dtype=torch.bool),
        },
    }

    actual = trainer._multilevel_suppression_loss(
        torch.nn.CrossEntropyLoss(),
        features,
        pids,
    )
    expected = 0.5 * (
        torch.stack(tuple(F.cross_entropy(logits, pids) for logits in coarse)).mean()
        + torch.stack(tuple(F.cross_entropy(logits, pids) for logits in fine)).mean()
    )

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(logits.grad is not None for logits in (*coarse, *fine))


def test_multilevel_suppression_aux_ce_skips_inactive_cam_samples(tmp_path):
    trainer = _suppression_trainer(tmp_path)
    trainer._current_multilevel_suppression_progress = 1.0
    pids = torch.tensor([0, 1, 2])
    coarse = tuple(torch.randn(3, 3, requires_grad=True) for _ in range(2))
    fine = tuple(torch.randn(3, 3, requires_grad=True) for _ in range(4))
    coarse_active = torch.tensor(
        [[True, False], [False, False], [True, True]],
    )
    fine_active = torch.tensor(
        [
            [True, False, False, True],
            [False, False, False, True],
            [True, False, False, False],
        ],
    )
    features = {
        "_multilevel_suppression_logits": {
            "coarse": coarse,
            "fine": fine,
        },
        "_multilevel_suppression_active": {
            "coarse": coarse_active,
            "fine": fine_active,
        },
    }

    actual = trainer._multilevel_suppression_loss(
        torch.nn.CrossEntropyLoss(),
        features,
        pids,
    )

    def masked_branch_loss(logits, active):
        if active.any():
            return F.cross_entropy(logits[active], pids[active])
        return logits.sum() * 0.0

    expected_coarse = torch.stack(
        [
            masked_branch_loss(logits, coarse_active[:, index])
            for index, logits in enumerate(coarse)
        ]
    ).mean()
    expected_fine = torch.stack(
        [
            masked_branch_loss(logits, fine_active[:, index])
            for index, logits in enumerate(fine)
        ]
    ).mean()
    expected = 0.5 * (expected_coarse + expected_fine)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(logits.grad is not None for logits in (*coarse, *fine))
    assert torch.count_nonzero(fine[1].grad).item() == 0
    assert torch.count_nonzero(fine[2].grad).item() == 0


def test_multilevel_suppression_rejects_malformed_activity_mask(tmp_path):
    trainer = _suppression_trainer(tmp_path)
    trainer._current_multilevel_suppression_progress = 1.0
    pids = torch.tensor([0, 1, 2])
    features = {
        "_multilevel_suppression_logits": {
            "coarse": tuple(torch.randn(3, 3) for _ in range(2)),
            "fine": tuple(torch.randn(3, 3) for _ in range(4)),
        },
        "_multilevel_suppression_active": {
            "coarse": torch.ones(3, 2),
            "fine": torch.ones(3, 4, dtype=torch.bool),
        },
    }

    with pytest.raises(RuntimeError, match="coarse activity must be a bool tensor"):
        trainer._multilevel_suppression_loss(
            torch.nn.CrossEntropyLoss(),
            features,
            pids,
        )


def test_multilevel_suppression_masked_label_smoothing_matches_subset_ce():
    criterion = CrossEntropyLabelSmooth(num_classes=3, epsilon=0.05)
    logits = torch.randn(4, 3, requires_grad=True)
    pids = torch.tensor([0, 1, 2, 0])
    active = torch.tensor([True, False, True, False])

    actual = ReIDTrainer._masked_multilevel_suppression_ce(
        criterion,
        logits,
        pids,
        active,
    )
    expected = criterion(logits[active], pids[active])

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"multilevel_suppression_ratio": 0.0}, "positive ratio"),
        ({"multilevel_suppression_loss_weight": 0.0}, "positive loss weight"),
        (
            {"multilevel_suppression_loss_weight": float("nan")},
            "loss_weight must be finite",
        ),
        (
            {"multilevel_suppression_loss_weight": float("inf")},
            "loss_weight must be finite",
        ),
        ({"model_name": "osnet_x0_25"}, "only for CSL-TinyViT"),
        ({"head_parts": (1, 2)}, "head_parts=\\(1, 2, 4\\)"),
        ({"scale_balanced_branches": False}, "head_parts=\\(1, 2, 4\\)"),
        ({"part_pooling": "overlap_stripes"}, "standard fixed-stripe"),
        ({"feature_fusion": "last3"}, "Stage-0 semantic-fine"),
        (
            {
                "backbone_freeze_epochs": 21,
                "multilevel_suppression_start_epoch": 20,
            },
            "start_epoch >= backbone_freeze_epochs",
        ),
        (
            {"reid_adapter_stages": (3,)},
            "requires reid_adapter_stages=\\(\\)",
        ),
        (
            {"multilevel_suppression_decay_end_epoch": 201},
            "decay_end_epoch <= epochs",
        ),
    ],
)
def test_multilevel_suppression_rejects_incompatible_training_paths(
    tmp_path,
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        _suppression_trainer(tmp_path, **overrides)


def test_multilevel_suppression_model_kwargs_are_architecture_complete(tmp_path):
    trainer = _suppression_trainer(tmp_path)

    kwargs = build_reid_model_kwargs(trainer)

    assert kwargs["multilevel_suppression"] is True
    assert kwargs["multilevel_suppression_ratio"] == 0.15


def test_multilevel_suppression_checkpoint_and_resume_contract(tmp_path):
    trainer = _suppression_trainer(tmp_path)
    metadata = trainer._checkpoint_metadata(torch.nn.Linear(2, 2))
    contract = trainer._resume_contract()
    expected_loss_contract = {
        "multilevel_suppression_loss_weight": 0.20,
        "multilevel_suppression_start_epoch": 20,
        "multilevel_suppression_ramp_end_epoch": 50,
        "multilevel_suppression_decay_start_epoch": 140,
        "multilevel_suppression_decay_end_epoch": 170,
    }

    assert metadata["multilevel_suppression"] is True
    assert metadata["multilevel_suppression_ratio"] == pytest.approx(0.15)
    assert metadata["multilevel_suppression_version"] == 2
    assert (
        metadata["model"]["transformer"]["multilevel_suppression"]["version"]
        == 2
    )
    for key, value in expected_loss_contract.items():
        assert metadata[key] == pytest.approx(value)
    assert contract["model"]["multilevel_suppression"] is True
    assert contract["model"]["multilevel_suppression_ratio"] == pytest.approx(0.15)
    assert contract["model"]["multilevel_suppression_version"] == 2
    assert {key: contract["loss"][key] for key in expected_loss_contract} == expected_loss_contract

    changed = _suppression_trainer(
        tmp_path,
        multilevel_suppression_ratio=0.20,
    )._resume_contract()
    assert contract_fingerprint(changed) != contract_fingerprint(contract)
    assert contract_differences(contract, changed) == ["model.multilevel_suppression_ratio: saved=0.15, requested=0.2"]

    checkpoint_path = tmp_path / "last.pt"
    trainer._assert_resume_compatible(
        {
            "resume_contract": contract,
            "epoch": 10,
            "epochs": 200,
        },
        checkpoint_path,
    )
    for saved_version in (None, 1):
        stale = json.loads(json.dumps(contract))
        if saved_version is None:
            stale["model"].pop("multilevel_suppression_version")
        else:
            stale["model"]["multilevel_suppression_version"] = saved_version
        with pytest.raises(
            ValueError,
            match="model.multilevel_suppression_version",
        ):
            trainer._assert_resume_compatible(
                {
                    "resume_contract": stale,
                    "epoch": 10,
                    "epochs": 200,
                },
                checkpoint_path,
            )

    assert (
        MULTILEVEL_SUPPRESSION_IMPLEMENTATION_VERSION
        == contract["model"]["multilevel_suppression_version"]
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_multilevel_suppression_version_is_persisted_only_when_enabled(
    tmp_path,
    enabled,
):
    trainer = _suppression_trainer(
        tmp_path,
        multilevel_suppression=enabled,
    )
    model = torch.nn.Linear(2, 2)
    recipe = trainer._training_recipe_for_family("transformer")
    models = ModelBundle(
        model=model,
        ema_model=None,
        val_model=model,
        is_transformer=True,
        training_family="transformer",
        recipe=recipe,
    )
    losses = LossBundle(
        criterion_id=torch.nn.Identity(),
        criterion_metric=None,
        criterion_center=CenterLoss(num_classes=2, feat_dim=2),
        label_smooth=0.05,
        soft_margin=True,
        metric_dim=2,
        classifier_dim=2,
    )
    data = DatasetBundle(
        dataset=None,
        num_classes=2,
        default_eval_name="market1501",
    )

    save_dir = tmp_path / str(enabled)
    trainer._write_hparams(save_dir, data, models, losses)
    hparams = json.loads((save_dir / "hparams.json").read_text())
    metadata = trainer._checkpoint_metadata(model)
    contract = trainer._resume_contract()
    expected = 2 if enabled else None

    assert hparams["model"]["head"]["multilevel_suppression"].get("version") == expected
    assert hparams["losses"]["multilevel_suppression"].get("version") == expected
    assert metadata.get("multilevel_suppression_version") == expected
    assert (
        metadata["model"]["transformer"]["multilevel_suppression"].get(
            "version"
        )
        == expected
    )
    assert contract["model"].get("multilevel_suppression_version") == expected


def test_multilevel_suppression_diagnostics_are_validated_and_reported(tmp_path):
    trainer = _suppression_trainer(tmp_path)
    trainer._current_multilevel_suppression_progress = 0.5
    diagnostic_values = {
        "effective_ratio": 0.075,
        "coarse_erased_fraction": 0.08,
        "fine_erased_fraction": 0.09,
        "global_cam_active_fraction": 0.90,
        "coarse_cam_active_fraction": 0.80,
    }
    features = {
        "_multilevel_suppression_diagnostics": {key: torch.tensor(value) for key, value in diagnostic_values.items()}
    }

    actual = trainer._multilevel_suppression_diagnostics(features)
    torch.testing.assert_close(
        actual,
        torch.tensor(tuple(diagnostic_values.values())),
    )

    malformed = {
        "_multilevel_suppression_diagnostics": {
            **features["_multilevel_suppression_diagnostics"],
            "coarse_cam_active_fraction": torch.tensor(float("nan")),
        }
    }
    with pytest.raises(RuntimeError, match="must be finite"):
        trainer._multilevel_suppression_diagnostics(malformed)

    metrics = TrainMetrics(
        epoch=35,
        loss=1.0,
        id_loss=0.5,
        triplet_loss=0.5,
        center_loss=0.0,
        lr=7e-4,
        elapsed_s=1.0,
        multilevel_suppression_loss=1.25,
        multilevel_suppression_weight=0.10,
        multilevel_suppression_effective_ratio=0.075,
        multilevel_suppression_coarse_erased_fraction=0.08,
        multilevel_suppression_fine_erased_fraction=0.09,
        multilevel_suppression_global_cam_active_fraction=0.90,
        multilevel_suppression_coarse_cam_active_fraction=0.80,
    )
    trainer._save_metrics(tmp_path, [metrics], [], 0, 0.0, 0.0)
    saved = json.loads((tmp_path / "metrics.json").read_text())["train"][0]
    for key, value in diagnostic_values.items():
        assert saved[f"multilevel_suppression_{key}"] == pytest.approx(value)


def test_multilevel_suppression_diagnostics_accept_fp16_schedule_rounding(
    tmp_path,
):
    trainer = _suppression_trainer(tmp_path)
    trainer._current_multilevel_suppression_progress = 0.5
    values = (0.075, 0.08, 0.09, 0.90, 0.80)
    features = {
        "_multilevel_suppression_diagnostics": {
            key: torch.tensor(value, dtype=torch.float16)
            for key, value in zip(
                trainer._MULTILEVEL_SUPPRESSION_DIAGNOSTIC_KEYS,
                values,
            )
        }
    }

    actual = trainer._multilevel_suppression_diagnostics(features)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual,
        torch.tensor(values, dtype=torch.float16).float(),
    )
