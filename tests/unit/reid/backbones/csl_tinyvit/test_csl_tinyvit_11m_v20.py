"""Checks for the canonical promoted CSL-TinyViT 11M V20 preset."""

from __future__ import annotations

import pytest
import torch

from boxmot.reid.backbones import BACKBONE_REGISTRY, get_backbone_spec
from boxmot.reid.backbones.families.csl_tinyvit import (
    csl_tinyvit_11m,
    csl_tinyvit_11m_v20,
)
from boxmot.reid.backbones.families.csl_tinyvit import pretrained as csl_tinyvit_pretrained
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.training.trainer import ReIDTrainer


def _v20_trainer(tmp_path, **overrides) -> ReIDTrainer:
    params = {
        "model_name": "csl_tinyvit_11m_v20",
        "dataset_name": "market1501",
        "data_dir": str(tmp_path),
        "pretrained": False,
        "epochs": 200,
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "spatial_conv_mode": "depthwise_separable",
        "head_type": "standard",
        "head_pool": "gelu_gem",
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "metric_feature": "raw_concat",
        "inference_feature": "norm_concat_bn",
        "scale_balanced_branches": True,
        "feat_dim": 512,
        "neck_dim": 512,
        "attention_window_layout": "rect",
        "attention_bias": "absolute",
        "interpolate_pretrained_attention_bias": True,
        "attention_mask": True,
        "loss_type": "triplet",
        "classifier_loss": "ce",
    }
    params.update(overrides)
    return ReIDTrainer(**params)


def test_11m_v20_direct_preset_has_exact_promoted_topology_and_descriptor():
    model = csl_tinyvit_11m_v20(
        num_classes=751,
        loss="triplet",
        pretrained=False,
        use_gpu=False,
    ).eval()

    assert [layer.dim for layer in model.layers] == [64, 128, 256, 448]
    assert [len(layer.blocks) for layer in model.layers] == [2, 2, 6, 2]
    assert model.feature_fusion == "global_final_parts_stage0_semantic_fine"
    assert model.feature_fusion_module.stage_indices == (1, 2, 0)
    assert model.feature_fusion_module.local_channels == 512
    assert model.feature_fusion_module.fine_output_channels == 512
    assert model.pyramid_resize_mode == "bilinear"
    assert model.spatial_conv_mode == "depthwise_separable"
    assert model.layers[1].blocks[0].window_size == (12, 4)
    assert model.layers[2].blocks[0].window_size == (12, 8)
    assert model.layers[3].blocks[0].window_size == (12, 8)
    assert all(
        block.attn.bias_mode == "absolute" and block.attention_mask
        for layer in model.layers[1:]
        for block in layer.blocks
    )
    assert model.interpolate_pretrained_attention_bias is True
    assert model.head.head_pool == "gelu_gem"
    assert model.head.head_parts == (1, 2, 4)
    assert model.head.part_pooling == "stripes"
    assert model.head.metric_feature == "raw_concat"
    assert model.head.inference_feature == "norm_concat_bn"
    assert model.head.scale_balanced_branches is True
    assert model.head.anatomical_auxiliary_pool is None
    assert model.head.metric_dim == 1536
    assert model.head.classifier_dim == 1536
    assert model.head.center_dim == 1536
    assert sum(parameter.numel() for parameter in model.parameters()) == 13_514_597

    with torch.inference_mode():
        descriptor = model(torch.randn(1, 3, 384, 128))
    assert descriptor.shape == (1, 1536)
    torch.testing.assert_close(
        descriptor.norm(dim=1),
        torch.ones(1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_11m_v20_pretrained_uses_official_11m_spec_with_full_coverage(monkeypatch):
    official_layout = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        use_gpu=False,
    )
    official_backbone = {
        key: value.clone()
        for key, value in official_layout.state_dict().items()
        if key.startswith(("patch_embed.", "layers."))
    }
    captured = {}

    def fake_load(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return official_backbone

    monkeypatch.setattr(csl_tinyvit_pretrained, "load_hub_checkpoint", fake_load)

    model = csl_tinyvit_11m_v20(
        num_classes=4,
        pretrained=True,
        use_gpu=False,
    )

    assert captured["url"].endswith("tiny_vit_11m_22kto1k_distill.pth")
    assert captured["sha256"] == (
        "98d4dde231bb9b8d98df178393e725ae8258115e939a6fb50210970f5f0d3192"
    )
    assert captured["weights_only"] is True
    assert model.pretrained_backbone_tensor_coverage == 1.0
    assert model.pretrained_backbone_numel_coverage == 1.0
    assert model.pretrained_backbone_required_tensor_count == 292
    assert model.pretrained_backbone_matched_tensor_count == 292
    assert model.pretrained_missing_backbone_keys == ()
    assert len(model.pretrained_interpolated_attention_biases) == 10


def test_11m_v20_registry_checkpoint_name_reconstructs_strictly(tmp_path):
    source = csl_tinyvit_11m_v20(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        use_gpu=False,
    )
    weights = tmp_path / "best.pt"
    checkpoint = {
        "model_name": "csl_tinyvit_11m_v20",
        "num_classes": 4,
        "model_kwargs_schema_version": 1,
        "model_kwargs": {},
        "state_dict": source.state_dict(),
    }
    torch.save(checkpoint, weights)

    model_name = ReIDModelRegistry.get_model_name(weights)
    model_kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)
    reconstructed = ReIDModelRegistry.build_model(
        model_name,
        weights=weights,
        num_classes=ReIDModelRegistry.get_nr_classes(weights),
        loss="triplet",
        pretrained=False,
        use_gpu=False,
        **model_kwargs,
    )
    report = ReIDModelRegistry.load_deployment_weights(reconstructed, weights)

    assert model_name == "csl_tinyvit_11m_v20"
    assert model_kwargs == {}
    assert reconstructed.feature_fusion == source.feature_fusion
    assert reconstructed.head.metric_dim == source.head.metric_dim == 1536
    assert reconstructed.state_dict().keys() == source.state_dict().keys()
    assert report.tensor_coverage == 1.0
    assert report.numel_coverage == 1.0


def test_11m_v20_is_publicly_registered_with_transformer_capabilities():
    assert "csl_tinyvit_11m_v20" in BACKBONE_REGISTRY

    spec = get_backbone_spec("csl_tinyvit_11m_v20")
    assert spec.family == "transformer"
    assert spec.default_img_size == (384, 128)
    assert spec.supports_layer_decay is True
    assert spec.supports_drop_path is True
    assert spec.pretrained_source == "TinyViT model zoo"


@pytest.mark.parametrize(
    "treatment",
    (
        {"hierarchical_branch_attention": True},
        {"branch_set_attention": True},
        {"multiscale_query_decoder": True},
        {
            "hierarchical_late_interaction": True,
            "late_interaction_loss_weight": 0.1,
        },
        {"csmm_loss_weight": 0.1},
        {"treeboost_loss_weight": 0.1},
        {"head_type": "multiscale_channel2"},
        {"compact_deployment_head": True},
    ),
    ids=(
        "hierarchical-branch-attention",
        "branch-set-attention",
        "query-decoder",
        "late-interaction",
        "csmm",
        "treeboost",
        "specialist-head",
        "compact-head",
    ),
)
def test_11m_v20_accepts_11m_family_treatments(tmp_path, treatment):
    trainer = _v20_trainer(tmp_path, **treatment)

    assert trainer.model_name == "csl_tinyvit_11m_v20"


def test_11m_v20_mcpt_keeps_512_channel_contract(tmp_path):
    trainer = _v20_trainer(tmp_path, mcpt_mode="shared_multiscale")

    assert trainer.feat_dim == 512
    assert trainer.neck_dim == 512

    with pytest.raises(ValueError, match="requires feat_dim=neck_dim=512"):
        _v20_trainer(
            tmp_path,
            mcpt_mode="shared_multiscale",
            feat_dim=384,
            neck_dim=384,
        )
