"""Regression checks for the training-only Jigsaw Patch Module."""

from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import torch
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.reid.backbones.families.csl_tinyvit import (
    JigsawPatchAuxiliary,
    csl_tinyvit_7m,
)
from boxmot.reid.training.ablation import resolve_csl_tinyvit_ablation
from boxmot.reid.training.config import ReIDTrainConfig, trainer_kwargs_from_args
from boxmot.reid.training.losses import TripletLoss
from boxmot.reid.training.resume import contract_differences
from boxmot.reid.training.trainer import ReIDTrainer


def _model(*, jpm: bool):
    return csl_tinyvit_7m(
        num_classes=8,
        loss="triplet",
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        feat_dim=384,
        neck_dim=384,
        head_pool="gelu_gem",
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=False,
        attention_window_layout="rect",
        attention_bias="absolute",
        interpolate_pretrained_attention_bias=True,
        attention_mask=True,
        jpm=jpm,
    )


def _trainer(tmp_path, **kwargs) -> ReIDTrainer:
    values = {
        "model_name": "csl_tinyvit_7m",
        "dataset_name": "market1501",
        "data_dir": str(tmp_path),
        "device": "cpu",
        "epochs": 200,
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
        "jpm": True,
    }
    values.update(kwargs)
    return ReIDTrainer(**values)


def test_jpm_shift_shuffle_matches_transreid_order():
    tokens = torch.arange(12).reshape(1, 12, 1)

    shuffled = JigsawPatchAuxiliary.rearrange_patches(
        tokens,
        num_groups=4,
        shift=5,
    )

    assert shuffled.flatten().tolist() == [
        5,
        8,
        11,
        2,
        6,
        9,
        0,
        3,
        7,
        10,
        1,
        4,
    ]


def test_jpm_outputs_four_local_losses_with_live_gradients(tmp_path):
    module = JigsawPatchAuxiliary(32, 4, token_dim=16, num_heads=4)
    module.train()
    spatial = torch.randn(8, 32, 6, 4, requires_grad=True)
    global_map = torch.randn(8, 32, 6, 4, requires_grad=True)
    logits, features = module(spatial, global_map)
    labels = torch.arange(4).repeat_interleave(2)

    trainer = _trainer(tmp_path)
    id_loss, metric_loss = trainer._jpm_auxiliary_losses(
        torch.nn.CrossEntropyLoss(),
        TripletLoss(margin=0.3),
        {"_jpm_logits": logits, "_jpm_features": features},
        labels,
    )
    (id_loss + metric_loss).backward()

    assert len(logits) == len(features) == 4
    assert all(feature.shape == (8, 16) for feature in features)
    assert torch.isfinite(id_loss) and torch.isfinite(metric_loss)
    assert module.input_projection.weight.grad is not None
    assert spatial.grad is not None and spatial.grad.abs().sum() > 0
    assert global_map.grad is not None and global_map.grad.abs().sum() > 0


def test_jpm_preserves_same_seed_model_and_1152d_inference():
    torch.manual_seed(11)
    control = _model(jpm=False)
    torch.manual_seed(11)
    treatment = _model(jpm=True)

    treatment_state = treatment.state_dict()
    assert all(
        torch.equal(value, treatment_state[key])
        for key, value in control.state_dict().items()
    )
    assert (
        sum(parameter.numel() for parameter in treatment.parameters())
        - sum(parameter.numel() for parameter in control.parameters())
        == 153_504
    )

    calls = []
    hook = treatment.head.jpm.register_forward_hook(
        lambda *_args: calls.append(True)
    )
    control.eval()
    treatment.eval()
    image = torch.randn(1, 3, 384, 128)
    with torch.inference_mode():
        control_descriptor = control(image)
        treatment_descriptor = treatment(image)
    hook.remove()

    assert calls == []
    assert control_descriptor.shape == treatment_descriptor.shape == (1, 1152)
    assert torch.equal(control_descriptor, treatment_descriptor)


def test_freeze_schedule_preserves_rgb_and_jpm_frozen_bnneck_biases(tmp_path):
    for enabled in (False, True):
        model = _model(jpm=enabled)
        trainer = _trainer(
            tmp_path,
            jpm=enabled,
            backbone_freeze_epochs=10,
        )
        parameter_groups = trainer._build_vit_param_groups(model)
        optimized_parameters = {
            id(parameter)
            for group in parameter_groups
            for parameter in group["params"]
        }
        frozen_biases = {
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }

        trainer._set_backbone_freeze_trainability(model, True)
        trainer._set_backbone_freeze_trainability(model, False)
        trainer._set_head_warmup_trainability(model, True)
        trainer._set_head_warmup_trainability(model, False)
        trainer._set_gradual_unfreeze_trainability(model, "head")
        trainer._set_gradual_unfreeze_trainability(model, "stage")
        trainer._set_gradual_unfreeze_trainability(model, "full")

        assert frozen_biases
        assert all(
            not dict(model.named_parameters())[name].requires_grad
            for name in frozen_biases
        )
        assert not [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in optimized_parameters
        ]


def test_jpm_is_logged_and_resume_contract_is_strict(tmp_path):
    trainer = _trainer(tmp_path)
    plan = resolve_csl_tinyvit_ablation(trainer)
    contract = trainer._resume_contract()

    assert "architecture.jpm" in plan.active_names
    assert contract["model"]["jpm"] is True
    assert contract["loss"]["jpm_id_loss_weight"] == 1.0

    changed = copy.deepcopy(contract)
    changed["model"]["jpm_shift"] = 4
    assert contract_differences(contract, changed) == [
        "model.jpm_shift: saved=5, requested=4"
    ]

    disabled = _trainer(tmp_path, jpm=False)._resume_contract()
    historical = copy.deepcopy(disabled)
    for key in tuple(historical["model"]):
        if key == "jpm" or key.startswith("jpm_"):
            del historical["model"][key]
    for key in tuple(historical["loss"]):
        if key.startswith("jpm_"):
            del historical["loss"][key]
    assert contract_differences(historical, disabled) == []


def test_cli_propagates_jpm_options(monkeypatch):
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
            "--jpm",
            "--jpm-num-groups",
            "4",
            "--jpm-shift",
            "5",
            "--jpm-token-dim",
            "96",
            "--jpm-num-heads",
            "4",
            "--jpm-id-loss-weight",
            "1",
            "--jpm-metric-loss-weight",
            "1",
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
    assert captured["args"].jpm is True
    assert captured["args"].jpm_num_groups == 4
    assert captured["args"].jpm_shift == 5
    assert captured["args"].jpm_token_dim == 96
    kwargs = trainer_kwargs_from_args(captured["args"], {})
    config = ReIDTrainConfig.from_flat_kwargs(**kwargs)
    trainer = ReIDTrainer.from_config(config)
    assert trainer.jpm is True
    assert trainer.jpm_num_groups == 4
    assert trainer.jpm_shift == 5
    assert trainer.jpm_token_dim == 96
    model = trainer._build_model(num_classes=8)
    assert isinstance(model.head.jpm, JigsawPatchAuxiliary)
