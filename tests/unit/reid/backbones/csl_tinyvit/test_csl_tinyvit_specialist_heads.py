import pytest
import torch
import torch.nn.functional as F

from boxmot.reid.backbones.families.csl_tinyvit.heads import MultiBranchHead
from boxmot.reid.backbones.families.csl_tinyvit.pooling import SpatialTopSuppression


@pytest.mark.parametrize(
    ("mode", "classifier_count", "descriptor_dim"),
    [
        ("stage2_channel2", 9, 1792),
        ("stage2_pg", 8, 1664),
        ("stage2_gpc_lite", 10, 1920),
        ("stage2_gpc_lite_gate", 10, 1920),
        ("stage2_pg_gate", 8, 1664),
        ("suppressed_global", 8, 1664),
    ],
)
def test_specialist_heads_preserve_main_metric_and_extend_retrieval(
    mode: str,
    classifier_count: int,
    descriptor_dim: int,
):
    head = MultiBranchHead(
        in_ch=512,
        feat_dim=512,
        num_classes=5,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_pool="gelu_gem",
        scale_balanced_branches=True,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        hierarchical_scales=True,
        specialist_mode=mode,
    )
    maps = (
        torch.randn(3, 512, 24, 8),
        torch.randn(3, 512, 24, 8),
        torch.randn(3, 512, 48, 16),
    )

    head.train()
    logits, metric = head(maps)
    assert len(logits) == classifier_count
    assert all(score.shape == (3, 5) for score in logits)
    assert metric.shape == (3, 1536)

    head.eval()
    with torch.no_grad():
        descriptor = head(maps)
    assert descriptor.shape == (3, descriptor_dim)
    torch.testing.assert_close(descriptor.norm(dim=1), torch.ones(3), atol=1e-5, rtol=1e-5)


def test_gpc_lite_specialist_overhead_stays_below_point_three_million_parameters():
    baseline = MultiBranchHead(
        in_ch=512,
        feat_dim=512,
        num_classes=751,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_pool="gelu_gem",
        scale_balanced_branches=True,
        head_parts=(1, 2, 4),
        hierarchical_scales=True,
    )
    gpc = MultiBranchHead(
        in_ch=512,
        feat_dim=512,
        num_classes=751,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_pool="gelu_gem",
        scale_balanced_branches=True,
        head_parts=(1, 2, 4),
        hierarchical_scales=True,
        specialist_mode="stage2_gpc_lite_gate",
    )

    overhead = sum(parameter.numel() for parameter in gpc.parameters()) - sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert overhead < 300_000
    assert gpc.bn_stage2_channel_shared is not None


def test_gpc_lite_retrieval_uses_conservative_fixed_specialist_weights():
    head = MultiBranchHead(
        in_ch=8,
        feat_dim=8,
        num_classes=3,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_pool="gelu_gem",
        scale_balanced_branches=True,
        head_parts=(1, 2, 4),
        hierarchical_scales=True,
        specialist_mode="stage2_gpc_lite",
    ).eval()
    maps = (
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 12, 4),
    )

    with torch.no_grad():
        descriptor = head(maps)
    chunks = torch.split(descriptor, (8, 4, 4, 2, 2, 2, 2, 128, 128, 128), dim=1)
    expected_pre_normalization = torch.tensor(
        [1.0, 2**-0.5, 2**-0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.20, 0.20]
    )
    expected = expected_pre_normalization / expected_pre_normalization.square().sum().sqrt()
    observed = torch.stack([chunk.norm(dim=1) for chunk in chunks], dim=1)
    torch.testing.assert_close(observed, expected.expand(2, -1), atol=1e-5, rtol=1e-5)


def test_specialist_gate_is_initialized_to_requested_weight_and_trained_by_ce():
    head = MultiBranchHead(
        in_ch=8,
        feat_dim=8,
        num_classes=3,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_pool="gelu_gem",
        scale_balanced_branches=True,
        head_parts=(1, 2, 4),
        hierarchical_scales=True,
        specialist_mode="stage2_gpc_lite_gate",
    )
    maps = (
        torch.randn(4, 8, 6, 2),
        torch.randn(4, 8, 6, 2),
        torch.randn(4, 8, 12, 4),
    )
    gate_input = torch.cat(
        (maps[0].mean(dim=(2, 3)), maps[1].mean(dim=(2, 3))),
        dim=1,
    )
    torch.testing.assert_close(
        head.specialist_gate(gate_input).sigmoid(),
        torch.full((4, 1), head.SPECIALIST_GATE_INIT),
    )

    head.train()
    logits, _ = head(maps)
    labels = torch.tensor([0, 1, 2, 0])
    specialist_loss = sum(F.cross_entropy(score, labels) for score in logits[-3:])
    specialist_loss.backward()
    assert head.specialist_gate.weight.grad is not None
    assert torch.count_nonzero(head.specialist_gate.weight.grad) > 0


def test_spatial_top_suppression_is_active_during_evaluation():
    suppression = SpatialTopSuppression(h_ratio=0.25).eval()
    feature = torch.ones(1, 2, 4, 2)
    feature[:, :, 2, :] = 10

    suppressed = suppression(feature)

    assert torch.count_nonzero(suppressed[:, :, 2, :]) == 0
    assert torch.equal(suppressed[:, :, :2, :], feature[:, :, :2, :])
