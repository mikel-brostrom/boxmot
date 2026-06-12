import json
from types import SimpleNamespace

import torch
import torch.nn as nn

from boxmot.engine.reid import trainer as engine_trainer
from boxmot.reid.backbones.csl_tinyvit import (
    MultiBranchHead,
    csl_tinyvit_7m,
    csl_tinyvit_11m,
    csl_tinyvit_23m,
    csl_tinyvit_large,
    csl_tinyvit_normal,
    csl_tinyvit_small,
)
from boxmot.reid.datasets import build_combined_dataset, build_dataset
from boxmot.reid.training.trainer import ReIDTrainer


def _trainer(tmp_path, **kwargs):
    params = {
        "model_name": "csl_tinyvit_7m",
        "dataset_name": "market1501",
        "data_dir": str(tmp_path),
        "lr": 3.5e-4,
        "weight_decay": 5e-4,
        "center_loss_weight": 5e-4,
    }
    params.update(kwargs)
    return ReIDTrainer(**params)


def _write_market_style_dataset(root, name):
    ds_root = root / name
    for split_dir in ("bounding_box_train", "query", "bounding_box_test"):
        (ds_root / split_dir).mkdir(parents=True)
    (ds_root / "bounding_box_train" / "0001_c1s1_000001_00.jpg").write_bytes(b"")
    (ds_root / "query" / "0001_c1s1_000002_00.jpg").write_bytes(b"")
    (ds_root / "bounding_box_test" / "0002_c2s1_000003_00.jpg").write_bytes(b"")
    return ds_root


def test_mot17_1501_market_style_dataset_alias(tmp_path):
    _write_market_style_dataset(tmp_path, "MOT17-1501")
    fixed_root = _write_market_style_dataset(tmp_path, "MOT17-1501-fixed")
    _write_market_style_dataset(tmp_path, "Market-1501-v15.09.15")

    mot17 = build_dataset("mot17_1501", str(tmp_path))
    combined = build_combined_dataset(["mot17_1501", "market1501"], str(tmp_path))

    assert mot17.name == "mot17_1501"
    assert mot17.root == fixed_root
    assert mot17.train.num_imgs == 1
    assert combined.name == "mot17_1501+market1501"
    assert combined.train.num_imgs == 2


def test_vit_defaults_apply_to_implicit_training_values(tmp_path):
    trainer = _trainer(tmp_path)

    trainer._apply_vit_training_defaults()

    assert trainer.lr == 7e-4
    assert trainer.weight_decay == 0.1
    assert trainer.warmup_epochs == 20
    assert trainer.center_loss_weight == 5e-3


def test_vit_defaults_respect_explicit_training_values(tmp_path):
    trainer = _trainer(
        tmp_path,
        center_loss_weight=0.0,
        explicit_hparams={"lr", "weight_decay", "center_loss_weight"},
    )

    trainer._apply_vit_training_defaults()

    assert trainer.lr == 3.5e-4
    assert trainer.weight_decay == 5e-4
    assert trainer.center_loss_weight == 0.0


def test_resume_hparams_do_not_override_explicit_cli_values(monkeypatch, tmp_path):
    run_dir = tmp_path / "exp"
    run_dir.mkdir()
    (run_dir / "hparams.json").write_text(
        json.dumps(
            {
                "model_name": "csl_tinyvit_7m",
                "dataset": "market1501",
                "data_dir": str(tmp_path),
                "loss_type": "triplet",
                "lr": 7e-4,
                "center_loss_weight": 5e-3,
                "head_pool": "gem",
                "head_parts": [1, 2, 4],
                "branch_aware_metric": True,
                "branch_metric_part_weight": 0.25,
                "head_warmup_epochs": 10,
                "head_warmup_lr_mult": 3.0,
            }
        )
    )
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return SimpleNamespace(weights_path=run_dir / "best.pt", best_mAP=0.0, best_rank1=0.0)

    monkeypatch.setattr(engine_trainer, "ReIDTrainer", FakeTrainer)
    args = SimpleNamespace(
        model="csl_tinyvit_7m",
        dataset="market1501",
        data_dir=str(tmp_path),
        loss="triplet",
        imgsz=(384, 128),
        lr=3.5e-4,
        center_loss_weight=0.0,
        resume=str(run_dir),
        train_explicit_keys=("lr", "center_loss_weight"),
    )

    engine_trainer.main(args)

    assert captured["lr"] == 3.5e-4
    assert captured["center_loss_weight"] == 0.0
    assert captured["head_pool"] == "gem"
    assert captured["head_parts"] == [1, 2, 4]
    assert captured["branch_aware_metric"] is True
    assert captured["branch_metric_part_weight"] == 0.25
    assert captured["head_warmup_epochs"] == 10
    assert captured["head_warmup_lr_mult"] == 3.0
    assert captured["explicit_hparams"] == {"lr", "center_loss_weight"}


def test_csl_tinyvit_metric_feature_mode_follows_loss():
    triplet_model = csl_tinyvit_7m(num_classes=4, loss="triplet", pretrained=False)
    ms_model = csl_tinyvit_7m(num_classes=4, loss="ms", pretrained=False)

    assert triplet_model.head.metric_feature == "raw_mean"
    assert ms_model.head.metric_feature == "concat_bn"


def test_csl_tinyvit_family_uses_standard_widths_and_512_neck():
    small = csl_tinyvit_7m(num_classes=4, pretrained=False)
    normal = csl_tinyvit_11m(num_classes=4, pretrained=False)
    large = csl_tinyvit_23m(num_classes=4, pretrained=False)

    assert [layer.dim for layer in small.layers] == [64, 128, 160, 320]
    assert [layer.dim for layer in normal.layers] == [64, 128, 256, 448]
    assert [layer.dim for layer in large.layers] == [96, 192, 384, 576]
    assert [layer.depth for layer in small.layers] == [2, 2, 6, 2]
    assert [layer.depth for layer in normal.layers] == [2, 2, 6, 2]
    assert [layer.depth for layer in large.layers] == [2, 2, 6, 2]
    assert small.neck[0].out_channels == 512
    assert normal.neck[0].out_channels == 512
    assert large.neck[0].out_channels == 512


def test_csl_tinyvit_size_aliases_build_expected_variants():
    small = csl_tinyvit_small(num_classes=4, pretrained=False)
    normal = csl_tinyvit_normal(num_classes=4, pretrained=False)
    large = csl_tinyvit_large(num_classes=4, pretrained=False)

    assert [layer.dim for layer in small.layers] == [64, 128, 160, 320]
    assert [layer.dim for layer in normal.layers] == [64, 128, 256, 448]
    assert [layer.dim for layer in large.layers] == [96, 192, 384, 576]


def test_multibranch_head_metric_feature_shapes():
    x = torch.randn(4, 8, 2, 1)

    raw_head = MultiBranchHead(8, feat_dim=4, num_classes=3, metric_feature="raw_mean")
    raw_head.train()
    raw_logits, raw_features = raw_head(x)

    concat_head = MultiBranchHead(8, feat_dim=4, num_classes=3, metric_feature="concat_bn")
    concat_head.train()
    concat_logits, concat_features = concat_head(x)
    concat_head.eval()
    with torch.no_grad():
        eval_features = concat_head(x)

    assert len(raw_logits) == 3
    assert len(concat_logits) == 3
    assert raw_features.shape == (4, 4)
    assert concat_features.shape == (4, 12)
    assert eval_features.shape == (4, 12)

    gem_head = MultiBranchHead(8, feat_dim=4, num_classes=3, head_pool="gem")
    gem_head.train()
    _, gem_features = gem_head(x)
    assert gem_head.head_pool == "gem"
    assert gem_features.shape == (4, 4)

    branch_head = MultiBranchHead(8, feat_dim=4, num_classes=3, branch_metric=True)
    branch_head.train()
    _, branch_features = branch_head(x)
    assert set(branch_features) == {"global", "part0", "part1", "raw_mean", "concat_bn"}
    assert branch_features["global"].shape == (4, 4)
    assert branch_features["concat_bn"].shape == (4, 12)


def test_multibranch_head_inference_feature_modes():
    x = torch.randn(2, 8, 2, 1)

    concat_head = MultiBranchHead(8, feat_dim=4, num_classes=3, inference_feature="concat_bn")
    concat_head.eval()
    with torch.no_grad():
        assert concat_head(x).shape == (2, 12)

    global_head = MultiBranchHead(8, feat_dim=4, num_classes=3, inference_feature="global")
    global_head.eval()
    with torch.no_grad():
        assert global_head(x).shape == (2, 4)

    raw_mean_head = MultiBranchHead(8, feat_dim=4, num_classes=3, inference_feature="raw_mean")
    raw_mean_head.eval()
    with torch.no_grad():
        assert raw_mean_head(x).shape == (2, 4)


def test_multibranch_head_feat_dim_1024_projects_each_branch():
    x = torch.randn(2, 8, 2, 1)
    head = MultiBranchHead(8, feat_dim=1024, num_classes=3, metric_feature="raw_mean")

    head.train()
    _, train_features = head(x)
    assert train_features.shape == (2, 1024)

    head.eval()
    with torch.no_grad():
        eval_features = head(x)
    assert eval_features.shape == (2, 3072)


def test_multibranch_head_supports_multi_granularity_parts():
    x = torch.randn(2, 8, 8, 2)
    head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        metric_feature="concat_bn",
        inference_feature="concat_bn",
        head_parts=(1, 2, 4),
    )

    head.train()
    logits, train_features = head(x)
    assert len(logits) == 7
    assert train_features.shape == (2, 28)

    branch_head = MultiBranchHead(8, feat_dim=4, num_classes=3, branch_metric=True, head_parts=(1, 2, 4))
    branch_head.train()
    _, branch_features = branch_head(x)
    assert {"global", "part0", "part1", "part2", "part3", "part4", "part5"} <= set(branch_features)
    assert branch_features["raw_mean"].shape == (2, 4)
    assert branch_features["concat_bn"].shape == (2, 28)

    head.eval()
    with torch.no_grad():
        eval_features = head(x)
    assert eval_features.shape == (2, 28)


def test_trainer_effective_metric_feature_modes(tmp_path):
    assert _trainer(tmp_path, loss_type="triplet")._effective_metric_feature() == "raw_mean"
    assert _trainer(tmp_path, loss_type="ms")._effective_metric_feature() == "concat_bn"
    assert _trainer(tmp_path, loss_type="ms", metric_feature="raw_mean")._effective_metric_feature() == "raw_mean"


def test_trainer_normalizes_head_parts_from_string(tmp_path):
    assert _trainer(tmp_path, head_parts="1,2,4").head_parts == (1, 2, 4)


def test_trainer_branch_aware_metric_loss_uses_branch_dict(tmp_path):
    trainer = _trainer(tmp_path, branch_aware_metric=True, branch_metric_part_weight=0.5)
    pids = torch.tensor([0, 0, 1, 1])
    features = {
        "global": torch.randn(4, 8),
        "part0": torch.randn(4, 8),
        "part1": torch.randn(4, 8),
        "raw_mean": torch.randn(4, 8),
        "concat_bn": torch.randn(4, 24),
    }
    calls = []

    def criterion(inputs, targets):
        calls.append(inputs.shape)
        assert targets is pids
        return inputs.square().mean()

    loss = trainer._metric_loss_for_features(criterion, features, pids)

    assert loss.ndim == 0
    assert calls == [torch.Size([4, 8]), torch.Size([4, 8]), torch.Size([4, 8])]


def test_trainer_branch_aware_metric_loss_uses_dynamic_part_keys(tmp_path):
    trainer = _trainer(tmp_path, branch_aware_metric=True, branch_metric_part_weight=0.5)
    pids = torch.tensor([0, 0, 1, 1])
    features = {
        "global": torch.randn(4, 8),
        "part0": torch.randn(4, 8),
        "part1": torch.randn(4, 8),
        "part2": torch.randn(4, 8),
        "part3": torch.randn(4, 8),
        "part4": torch.randn(4, 8),
        "part5": torch.randn(4, 8),
        "raw_mean": torch.randn(4, 8),
        "concat_bn": torch.randn(4, 56),
    }
    calls = []

    def criterion(inputs, targets):
        calls.append(inputs.shape)
        assert targets is pids
        return inputs.square().mean()

    loss = trainer._metric_loss_for_features(criterion, features, pids)

    assert loss.ndim == 0
    assert calls == [torch.Size([4, 8])] * 7


def test_trainer_head_warmup_toggles_backbone_trainability(tmp_path):
    trainer = _trainer(tmp_path)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(1, 1)
            self.layers = nn.Linear(1, 1)
            self.neck = nn.Linear(1, 1)
            self.head = nn.Linear(1, 1)

    model = TinyModel()

    trainer._set_head_warmup_trainability(model, True)
    assert not model.patch_embed.weight.requires_grad
    assert not model.layers.weight.requires_grad
    assert model.neck.weight.requires_grad
    assert model.head.weight.requires_grad

    trainer._set_head_warmup_trainability(model, False)
    assert all(param.requires_grad for param in model.parameters())
