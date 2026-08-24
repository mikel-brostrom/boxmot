"""Checks for the V20 recipe transfer to MobileNetV4 Medium backbones."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from click.testing import CliRunner

from boxmot.engine.config import load_training_recipe
from boxmot.engine.cli import boxmot
from boxmot.reid.backbones.families.csl_tinyvit.deployment import (
    FoldedBNNeck,
    optimize_csl_tinyvit_for_inference,
)
from boxmot.reid.backbones.heads.bnneck import BNNeck3
from boxmot.reid.backbones.mobilenetv4 import (
    mobilenetv4_conv_medium,
    mobilenetv4_conv_medium_v20,
    mobilenetv4_hybrid_medium,
    mobilenetv4_hybrid_medium_v20,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.exporters.base_exporter import as_inference_export_model
from boxmot.reid.training.trainer import ReIDTrainer


class _FeatureInfo:
    def __init__(self, channels: tuple[int, ...]) -> None:
        self._channels = channels

    def channels(self) -> tuple[int, ...]:
        return self._channels


class _FakeMobileNetV4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (32, 48, 80, 160, 960)
        self.feature_info = _FeatureInfo(channels)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(1, 1, 1, bias=False) for _ in range(3)]
            + [nn.Conv2d(1, 1, 1, stride=2, bias=False)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(channels[-1], 1280, 1, bias=False)
        self.norm_head = nn.BatchNorm2d(1280)
        self.act2 = nn.ReLU(inplace=True)

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        base = x.mean(dim=1, keepdim=True)
        outputs = []
        for index, channels in enumerate(
            self.feature_info.channels(),
            start=1,
        ):
            height = max(x.shape[-2] // (2**index), 1)
            width = max(x.shape[-1] // (2**index), 1)
            pooled = F.adaptive_avg_pool2d(base, (height, width))
            outputs.append(pooled.repeat(1, channels, 1, 1))
        return outputs

    def forward_intermediates(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outputs = self._features(x)
        return outputs[-1], outputs

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self._features(x)


def _install_fake_timm(monkeypatch, captured: dict) -> None:
    available = (
        "mobilenetv4_conv_medium.e500_r256_in1k",
        "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
        "mobilenetv4_hybrid_medium.ix_e550_r256_in1k",
    )

    def create_model(name: str, **kwargs):
        captured[name] = kwargs
        return _FakeMobileNetV4()

    monkeypatch.setitem(
        sys.modules,
        "timm",
        SimpleNamespace(
            list_models=lambda pattern, pretrained=False: list(available),
            create_model=create_model,
        ),
    )


def _build_fake_mobile_mcpt_pose_model():
    """Build a small shared-head treatment while retaining production topology."""
    return mobilenetv4_conv_medium(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        timm_model_name="mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
        timm_head_mode="spatial_linear",
        mobilenetv4_last_stride=1,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        pyramid_resize_mode="pool_bilinear",
        spatial_conv_mode="depthwise_separable",
        feat_dim=32,
        neck_dim=32,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
        mcpt_mode="shared_multiscale",
        mcpt_hidden_dim=8,
    )


@pytest.mark.parametrize(
    (
        "recipe_name",
        "model_name",
        "expected_lr",
        "expected_timm_model",
        "expected_timm_head",
        "expected_last_stride",
    ),
    (
        (
            "mobilenetv4_conv_medium_v20",
            "mobilenetv4_conv_medium",
            5e-4,
            "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
            "spatial_linear",
            1,
        ),
        (
            "mobilenetv4_hybrid_medium_v20",
            "mobilenetv4_hybrid_medium",
            3.5e-4,
            "",
            "pooled",
            2,
        ),
    ),
)
def test_medium_v20_recipes_preserve_policy_with_mobile_optimization(
    recipe_name,
    model_name,
    expected_lr,
    expected_timm_model,
    expected_timm_head,
    expected_last_stride,
):
    recipe = load_training_recipe(recipe_name)

    assert recipe["model"] == model_name
    assert recipe["feature_fusion"] == ("global_final_parts_stage0_semantic_fine")
    assert recipe["pyramid_resize_mode"] == "pool_bilinear"
    assert recipe["spatial_conv_mode"] == "depthwise_separable"
    assert recipe.get("timm_model_name", "") == expected_timm_model
    assert recipe["timm_head_mode"] == expected_timm_head
    assert recipe["mobilenetv4_last_stride"] == expected_last_stride
    assert recipe["mobilenetv4_neck_mode"] == "cnn"
    assert recipe["feat_dim"] == recipe["neck_dim"] == 384
    assert recipe["head_parts"] == [1, 2, 4]
    assert recipe["scale_balanced_branches"] is True
    assert recipe["anatomical_auxiliary"] is True
    assert recipe["anatomical_token_dim"] == 96
    assert recipe["anatomical_multiscale"] is True
    assert recipe["anatomical_target_type"] == "learned_pose_concat_ema"
    assert recipe["anatomical_min_effective_coverage"] == pytest.approx(0.8)
    assert recipe["mcpt_mode"] == "shared_multiscale"
    assert recipe["mcpt_hidden_dim"] == 64
    assert recipe["mcpt_max_displacement"] == pytest.approx(0.15)
    assert recipe["mcpt_start_epoch"] == 10
    assert recipe["mcpt_ramp_end_epoch"] == 25
    assert recipe["mcpt_smoothness_weight"] == pytest.approx(0.01)
    assert recipe["mcpt_identity_weight"] == pytest.approx(0.02)
    assert recipe["mcpt_identity_decay_epoch"] == 35
    assert recipe["mcpt_lr_multiplier"] == pytest.approx(2.0)
    assert recipe["mcpt_disabled_eval"] is False
    assert recipe["p_ids"] == 12
    assert recipe["k_instances"] == 8
    assert recipe["epochs"] == 100
    assert recipe["warmup_epochs"] == 20
    assert recipe["lr"] == expected_lr
    assert recipe["backbone_lr_mult"] == 1.0
    assert recipe["weight_decay"] == 1e-4
    assert recipe["ema_decay"] == 0.999
    assert recipe["anatomical_student_ramp_end_epoch"] == 25
    assert recipe["anatomical_decay_start_epoch"] == 60
    assert recipe["anatomical_decay_end_epoch"] == 85
    assert recipe["drop_path_rate"] == 0.1
    assert recipe["flip_tta"] is False


@pytest.mark.parametrize(
    (
        "recipe_name",
        "expected_model",
        "expected_timm_model",
        "expected_timm_head",
        "expected_last_stride",
    ),
    (
        (
            "mobilenetv4_conv_medium_v20",
            "mobilenetv4_conv_medium",
            "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
            "spatial_linear",
            1,
        ),
        (
            "mobilenetv4_hybrid_medium_v20",
            "mobilenetv4_hybrid_medium",
            "",
            "pooled",
            2,
        ),
    ),
)
def test_medium_v20_cli_recipe_resolves_training_contract(
    monkeypatch,
    recipe_name,
    expected_model,
    expected_timm_model,
    expected_timm_head,
    expected_last_stride,
):
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
        ["train", "--recipe", recipe_name, "--data-dir", "."],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == expected_model
    assert args.feat_dim == args.neck_dim == 384
    assert args.head_parts == (1, 2, 4)
    assert args.scale_balanced_branches is True
    assert args.timm_model_name == expected_timm_model
    assert args.timm_head_mode == expected_timm_head
    assert args.mobilenetv4_last_stride == expected_last_stride
    assert args.mobilenetv4_neck_mode == "cnn"
    assert args.anatomical_auxiliary is True
    assert args.anatomical_token_dim == 96
    assert args.anatomical_multiscale is True
    assert args.anatomical_min_effective_coverage == pytest.approx(0.8)
    assert args.mcpt_mode == "shared_multiscale"
    assert args.mcpt_hidden_dim == 64
    assert args.mcpt_max_displacement == pytest.approx(0.15)
    assert args.mcpt_start_epoch == 10
    assert args.mcpt_ramp_end_epoch == 25
    assert args.mcpt_identity_decay_epoch == 35
    assert args.mcpt_lr_multiplier == pytest.approx(2.0)
    assert args.p_ids == 12
    assert args.k_instances == 8
    assert args.epochs == 100
    assert args.backbone_lr_mult == 1.0
    assert args.anatomical_student_ramp_end_epoch == 25
    assert args.anatomical_decay_start_epoch == 60
    assert args.anatomical_decay_end_epoch == 85


def test_conv_medium_v20_cli_accepts_spatial_timm_head_ablation(monkeypatch):
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
            "mobilenetv4_conv_medium_v20",
            "--data-dir",
            ".",
            "--timm-head-mode",
            "spatial",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].timm_head_mode == "spatial"


@pytest.mark.parametrize(
    ("model_name", "builder"),
    (
        ("mobilenetv4_conv_medium_v20", mobilenetv4_conv_medium_v20),
        ("mobilenetv4_hybrid_medium_v20", mobilenetv4_hybrid_medium_v20),
    ),
)
def test_promoted_mobile_model_alias_selects_v20_recipe(
    monkeypatch,
    model_name,
    builder,
):
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
        ["train", "--model", model_name, "--data-dir", "."],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.model == model_name
    assert args.head_type == "standard"
    assert args.head_parts == (1, 2, 4)
    assert args.scale_balanced_branches is True
    assert args.mcpt_mode == "shared_multiscale"
    assert args.anatomical_auxiliary is True
    assert args.epochs == 100
    assert callable(builder)


def test_conv_medium_v20_cli_accepts_followup_ablation_axes(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.reid.trainer",
        SimpleNamespace(main=fake_main),
    )
    timm_model_name = "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k"
    result = CliRunner().invoke(
        boxmot,
        [
            "train",
            "--recipe",
            "mobilenetv4_conv_medium_v20",
            "--data-dir",
            ".",
            "--timm-head-mode",
            "spatial_linear",
            "--mobilenetv4-last-stride",
            "1",
            "--mobilenetv4-neck-mode",
            "spatial_ln",
            "--backbone-lr-mult",
            "0.25",
            "--timm-model-name",
            timm_model_name,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].timm_head_mode == "spatial_linear"
    assert captured["args"].mobilenetv4_last_stride == 1
    assert captured["args"].mobilenetv4_neck_mode == "spatial_ln"
    assert captured["args"].backbone_lr_mult == pytest.approx(0.25)
    assert captured["args"].timm_model_name == timm_model_name


def test_conv_medium_accepts_exact_resolution_matched_timm_checkpoint(monkeypatch):
    captured = {}
    _install_fake_timm(monkeypatch, captured)
    timm_model_name = "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k"

    model = mobilenetv4_conv_medium(
        num_classes=4,
        loss="triplet",
        pretrained=True,
        timm_model_name=timm_model_name,
    )

    assert model.timm_model_name == timm_model_name
    assert captured[timm_model_name]["pretrained"] is True
    assert model.pretrained_url == (
        f"https://huggingface.co/timm/{timm_model_name}"
    )
    assert len(model.pretrained_sha256) == 64


@pytest.mark.parametrize("last_stride", (2, 1))
def test_cross_scale_fusion_arms_preserve_shared_initialization(monkeypatch, last_stride):
    captured = {}
    _install_fake_timm(monkeypatch, captured)
    modes = {
        "anchor": "global_final_parts_stage0_semantic_fine",
        "panet_lite": "global_final_parts_stage0_panet_lite",
        "bifpn_lite": "global_final_parts_stage0_bifpn_lite",
    }
    models = {}
    shared_states = {}
    inputs = torch.randn(1, 3, 64, 32)

    for label, feature_fusion in modes.items():
        torch.manual_seed(0)
        model = mobilenetv4_conv_medium(
            num_classes=4,
            loss="triplet",
            pretrained=False,
            timm_model_name="mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
            timm_head_mode="spatial_linear",
            mobilenetv4_last_stride=last_stride,
            mobilenetv4_neck_mode="cnn",
            feature_fusion=feature_fusion,
            pyramid_resize_mode="pool_bilinear",
            spatial_conv_mode="depthwise_separable",
            feat_dim=32,
            neck_dim=32,
            metric_feature="raw_concat",
            inference_feature="norm_concat_bn",
            head_pool="gelu_gem",
            head_parts=(1, 2, 4),
            scale_balanced_branches=True,
            anatomical_auxiliary=True,
            anatomical_token_dim=16,
            anatomical_multiscale=True,
            anatomical_target_type="learned_pose_concat_ema",
        ).eval()
        models[label] = model
        shared_states[label] = {
            key: value for key, value in model.state_dict().items() if not key.startswith("feature_fusion_module.")
        }

    anchor_state = shared_states["anchor"]
    for label in ("panet_lite", "bifpn_lite"):
        candidate_state = shared_states[label]
        assert candidate_state.keys() == anchor_state.keys()
        for key, anchor_value in anchor_state.items():
            torch.testing.assert_close(candidate_state[key], anchor_value)

    with torch.inference_mode():
        descriptors = {label: model(inputs) for label, model in models.items()}
    torch.testing.assert_close(descriptors["panet_lite"], descriptors["anchor"])
    torch.testing.assert_close(descriptors["bifpn_lite"], descriptors["anchor"])


@pytest.mark.parametrize(
    ("builder", "expected_timm_name"),
    (
        (
            mobilenetv4_conv_medium,
            "mobilenetv4_conv_medium.e500_r256_in1k",
        ),
        (
            mobilenetv4_hybrid_medium,
            "mobilenetv4_hybrid_medium.ix_e550_r256_in1k",
        ),
    ),
)
def test_medium_backbones_support_v20_multiscale_pose_teacher(
    monkeypatch,
    builder,
    expected_timm_name,
):
    captured = {}
    _install_fake_timm(monkeypatch, captured)
    model = builder(
        num_classes=4,
        loss="triplet",
        pretrained=True,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        pyramid_resize_mode="pool_bilinear",
        spatial_conv_mode="depthwise_separable",
        feat_dim=384,
        neck_dim=384,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_pool="gelu_gem",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=96,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
        mcpt_mode="shared_multiscale",
        mcpt_hidden_dim=64,
        mcpt_max_displacement=0.15,
        mcpt_start_epoch=10,
        mcpt_ramp_end_epoch=25,
        drop_path_rate=0.1,
    )

    assert model.timm_model_name == expected_timm_name
    assert captured[expected_timm_name]["drop_path_rate"] == 0.1
    assert model.feature_fusion_module.stage_indices == (1, 2, 0)
    assert model.head.anatomical_auxiliary_pool.token_dim == 96
    assert model.head.mcpt is not None

    inputs = torch.randn(2, 3, 64, 32)
    model.eval()
    with torch.inference_mode():
        descriptor = model(inputs)
    assert descriptor.shape == (2, 1152)

    model.train()
    logits, features = model(
        inputs,
        anatomical_pose=torch.zeros(2, 17, 3),
    )
    assert len(logits) == 7
    assert features["raw_concat"].shape == (2, 1152)
    assert features["_anatomical_student_tokens"].shape == (
        2,
        6,
        96,
    )
    assert "_mcpt_smoothness" in features
    assert "_mcpt_identity" in features


@pytest.mark.parametrize(
    "model_name",
    ("mobilenetv4_conv_medium", "mobilenetv4_hybrid_medium"),
)
def test_medium_trainers_accept_v20_anatomical_contract(
    tmp_path,
    model_name,
):
    trainer = ReIDTrainer(
        model_name=model_name,
        dataset_name="market1501",
        data_dir=str(tmp_path),
        epochs=100,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        feat_dim=384,
        neck_dim=384,
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        anatomical_auxiliary=True,
        anatomical_metadata_dir="pose-metadata",
        anatomical_token_dim=96,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_pose_teacher_weight=0.03,
        anatomical_student_ramp_end_epoch=25,
        anatomical_decay_start_epoch=60,
        anatomical_decay_end_epoch=85,
        anatomical_pose_only_reliability=0.0,
        mcpt_mode="shared_multiscale",
        mcpt_hidden_dim=64,
        mcpt_max_displacement=0.15,
        mcpt_smoothness_weight=0.01,
        mcpt_identity_weight=0.02,
        mcpt_identity_decay_epoch=35,
        mcpt_lr_multiplier=2.0,
        mcpt_start_epoch=10,
        mcpt_ramp_end_epoch=25,
    )

    trainer._validate_config()


def test_mobile_mcpt_optimizer_uses_multiplier_and_owner_aware_decay(
    monkeypatch,
    tmp_path,
):
    captured = {}
    _install_fake_timm(monkeypatch, captured)
    trainer = ReIDTrainer(
        model_name="mobilenetv4_conv_medium",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        device="cpu",
        epochs=80,
        lr=5e-4,
        weight_decay=1e-4,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        feat_dim=384,
        neck_dim=384,
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        mcpt_mode="shared_multiscale",
    )
    model = mobilenetv4_conv_medium(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        feat_dim=384,
        neck_dim=384,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        mcpt_mode="shared_multiscale",
    )

    groups = trainer._build_mobilenetv4_param_groups(model)
    by_parameter = {
        id(parameter): group
        for group in groups
        for parameter in group["params"]
    }

    predictor = by_parameter[id(model.head.mcpt.row_output.weight)]
    gate = by_parameter[id(model.head.mcpt.local_gate_delta)]
    reduction = by_parameter[id(model.head.bn_global.reduction.weight)]
    classifier = by_parameter[id(model.head.bn_global.classifier.weight)]
    batch_norm = by_parameter[id(model.head.bn_global.bn.weight)]

    assert predictor["lr"] == pytest.approx(
        trainer.lr * trainer.mcpt_lr_multiplier
    )
    assert predictor["weight_decay"] == pytest.approx(trainer.weight_decay)
    assert gate["lr"] == pytest.approx(
        trainer.lr * trainer.mcpt_lr_multiplier
    )
    assert gate["weight_decay"] == 0
    assert reduction["weight_decay"] == pytest.approx(trainer.weight_decay)
    assert classifier["weight_decay"] == pytest.approx(trainer.weight_decay)
    assert batch_norm["weight_decay"] == 0


def test_mobile_mcpt_checkpoint_preserves_exact_constructor_contract(
    monkeypatch,
    tmp_path,
):
    captured = {}
    _install_fake_timm(monkeypatch, captured)
    timm_model_name = "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k"
    trainer = ReIDTrainer(
        model_name="mobilenetv4_conv_medium",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        device="cpu",
        epochs=80,
        timm_model_name=timm_model_name,
        timm_head_mode="spatial_linear",
        mobilenetv4_last_stride=1,
        mobilenetv4_neck_mode="cnn",
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        feat_dim=384,
        neck_dim=384,
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        mcpt_mode="shared_multiscale",
        mcpt_hidden_dim=64,
        mcpt_max_displacement=0.15,
        mcpt_start_epoch=10,
        mcpt_ramp_end_epoch=40,
    )
    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)
    checkpoint = tmp_path / "mobile-mcpt-metadata.pt"
    torch.save(metadata, checkpoint)

    stored = ReIDModelRegistry.get_checkpoint_model_kwargs(checkpoint)

    assert stored["timm_model_name"] == timm_model_name
    assert stored["timm_head_mode"] == "spatial_linear"
    assert stored["mobilenetv4_last_stride"] == 1
    assert stored["mobilenetv4_neck_mode"] == "cnn"
    assert stored["mcpt_mode"] == "shared_multiscale"
    assert stored["mcpt_hidden_dim"] == 64
    assert stored["mcpt_max_displacement"] == pytest.approx(0.15)
    assert stored["mcpt_start_epoch"] == 10
    assert stored["mcpt_ramp_end_epoch"] == 40


def test_mobile_deployment_prunes_teacher_and_classifiers_but_keeps_mcpt(
    monkeypatch,
):
    captured = {}
    _install_fake_timm(monkeypatch, captured)
    model = _build_fake_mobile_mcpt_pose_model().eval()
    mcpt = model.head.mcpt
    with torch.no_grad():
        mcpt.row_output.weight.normal_(0, 0.1)
        mcpt.row_output.bias.fill_(0.2)
    inputs = torch.randn(2, 3, 64, 32)
    with torch.inference_mode():
        expected = model(inputs)
    before_parameters = sum(parameter.numel() for parameter in model.parameters())

    optimize_csl_tinyvit_for_inference(model)

    with torch.inference_mode():
        actual = model(inputs)
    assert model.head.anatomical_auxiliary_pool is None
    assert model.head.mcpt is mcpt
    assert sum(isinstance(module, FoldedBNNeck) for module in model.modules()) == 7
    assert not any(isinstance(module, BNNeck3) for module in model.modules())
    assert not any("classifier" in name for name, _ in model.named_parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) < before_parameters
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_mobile_mcpt_torchscript_prunes_teacher_and_preserves_descriptor(
    monkeypatch,
):
    captured = {}
    _install_fake_timm(monkeypatch, captured)
    model = _build_fake_mobile_mcpt_pose_model().eval()
    with torch.no_grad():
        model.head.mcpt.row_output.weight.normal_(0, 0.1)
        model.head.mcpt.row_output.bias.fill_(0.2)
    inputs = torch.randn(2, 3, 64, 32)
    # Populate the old BNNeck cache under inference_mode to cover export after
    # an ordinary eager inference call.
    with torch.inference_mode():
        expected = model(inputs)

    export_model = as_inference_export_model(model)
    traced = torch.jit.trace(export_model, inputs, strict=False)
    with torch.inference_mode():
        actual = traced(inputs)
    parameter_names = tuple(name for name, _ in traced.named_parameters())

    assert model.head.anatomical_auxiliary_pool is None
    assert model.head.mcpt is not None
    assert not any("anatomical_auxiliary_pool" in name for name in parameter_names)
    assert not any("classifier" in name for name in parameter_names)
    assert any(".mcpt." in name for name in parameter_names)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
