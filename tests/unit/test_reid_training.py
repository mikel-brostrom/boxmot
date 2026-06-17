import json
from types import SimpleNamespace

import torch
import torch.nn as nn

from boxmot.engine.reid import trainer as engine_trainer
import boxmot.reid.backbones.lmbn.lmbn_ain_n as lmbn_ain_n_module
import boxmot.reid.backbones.lmbn.lmbn_n as lmbn_n_module
from boxmot.reid.backbones.lmbn.bnneck import BNNeck3
from boxmot.reid.backbones.csl_tinyvit import (
    CSLTinyViTFeatureFusion,
    LMBNStyleMultiBranchHead,
    MultiBranchHead,
    csl_tinyvit_7m,
    csl_tinyvit_7m_lmbn,
    csl_tinyvit_11m,
    csl_tinyvit_11m_lmbn,
    csl_tinyvit_23m,
    csl_tinyvit_23m_lmbn,
    csl_tinyvit_large,
    csl_tinyvit_lmbn,
    csl_tinyvit_normal,
    csl_tinyvit_small,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.datasets import build_combined_dataset, build_dataset
from boxmot.reid.training.losses import ArcFaceLoss, CenterLoss, CircleLoss, CosFaceLoss
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


def _fake_osnet_backbone(*, with_ain_pools: bool = False):
    backbone = SimpleNamespace(
        conv1=nn.Identity(),
        maxpool=nn.Identity(),
        conv2=nn.Identity(),
        conv3=nn.Sequential(nn.Identity(), nn.Identity()),
        conv4=nn.Identity(),
        conv5=nn.Identity(),
    )
    if with_ain_pools:
        backbone.pool2 = nn.Identity()
        backbone.pool3 = nn.Identity()
    return backbone


def test_lmbn_n_uses_requested_osnet_imagenet_pretraining(monkeypatch):
    called = {}

    def fake_osnet_x1_0(*, pretrained=False):
        called["pretrained"] = pretrained
        return _fake_osnet_backbone()

    monkeypatch.setattr(lmbn_n_module, "osnet_x1_0", fake_osnet_x1_0)

    lmbn_n_module.LMBN_n(num_classes=4, loss="ms", pretrained=True, use_gpu=False)

    assert called["pretrained"] is True


def test_lmbn_ain_n_uses_requested_osnet_imagenet_pretraining(monkeypatch):
    called = {}

    def fake_osnet_ain_x1_0(*, pretrained=False):
        called["pretrained"] = pretrained
        return _fake_osnet_backbone(with_ain_pools=True)

    monkeypatch.setattr(lmbn_ain_n_module, "osnet_ain_x1_0", fake_osnet_ain_x1_0)

    lmbn_ain_n_module.LMBN_ain_n(
        args=None,
        test_only=True,
        num_classes=4,
        loss="ms",
        pretrained=True,
        use_gpu=False,
    )

    assert called["pretrained"] is True


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


def test_lmbn_augment_flags_are_config_driven(tmp_path):
    trainer = ReIDTrainer(
        model_name="lmbn_n",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        color_jitter=False,
        gaussian_blur=False,
        random_grayscale=0.0,
        random_erasing=0.5,
        random_patch=False,
        color_augmentation=False,
        flip_tta=True,
    )

    assert trainer.color_jitter is False
    assert trainer.gaussian_blur is False
    assert trainer.random_grayscale == 0.0
    assert trainer.random_erasing == 0.5
    assert trainer.random_patch is False
    assert trainer.color_augmentation is False
    assert trainer.flip_tta is True


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
                "feature_fusion": "last2",
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
    assert captured["feature_fusion"] == "last2"
    assert captured["branch_aware_metric"] is True
    assert captured["branch_metric_part_weight"] == 0.25
    assert captured["head_warmup_epochs"] == 10
    assert captured["head_warmup_lr_mult"] == 3.0
    assert captured["explicit_hparams"] == {"lr", "center_loss_weight"}


def test_resume_hparams_nested_layout_applies_defaults(monkeypatch, tmp_path):
    run_dir = tmp_path / "exp_nested"
    run_dir.mkdir()
    (run_dir / "hparams.json").write_text(
        json.dumps(
            {
                "run": {
                    "model_name": "csl_tinyvit_7m",
                },
                "data": {
                    "dataset": "market1501",
                    "data_dir": str(tmp_path),
                    "img_size": [384, 128],
                    "sampler": {"p": 16, "k": 4},
                },
                "model": {
                    "feature_fusion": "last2",
                    "head": {"pool": "gem", "parts": [1, 2, 4]},
                    "branch": {"aware_metric": True, "metric_part_weight": 0.25},
                },
                "optimization": {
                    "epochs": 250,
                    "scheduler": {"warmup_epochs": 20},
                },
                "losses": {
                    "loss_type": "triplet",
                    "weights": {"center_loss_weight": 0.005},
                },
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
        resume=str(run_dir),
        train_explicit_keys=("lr",),
    )

    engine_trainer.main(args)

    assert captured["lr"] == 3.5e-4
    assert captured["feature_fusion"] == "last2"
    assert captured["head_pool"] == "gem"
    assert captured["head_parts"] == [1, 2, 4]
    assert captured["branch_aware_metric"] is True
    assert captured["branch_metric_part_weight"] == 0.25
    assert captured["center_loss_weight"] == 0.005
    assert captured["p"] == 16
    assert captured["k"] == 4
    assert captured["warmup_epochs"] == 20


def test_reid_checkpoint_saves_center_loss_state(tmp_path):
    trainer = _trainer(tmp_path)
    model = nn.Linear(3, 2)
    criterion_center = CenterLoss(num_classes=2, feat_dim=3)
    expected_centers = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with torch.no_grad():
        criterion_center.centers.copy_(expected_centers)

    ckpt_path = tmp_path / "last.pt"
    trainer._save_checkpoint(
        model,
        ckpt_path,
        epoch=3,
        val=None,
        criterion_center=criterion_center,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert "center_loss_state_dict" in ckpt
    assert torch.allclose(ckpt["center_loss_state_dict"]["centers"], expected_centers)


def test_reid_resume_restores_center_loss_state(tmp_path):
    trainer = _trainer(tmp_path)
    criterion_center = CenterLoss(num_classes=2, feat_dim=3)
    expected_centers = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ckpt = {"center_loss_state_dict": {"centers": expected_centers}}

    trainer._restore_center_loss_state(
        ckpt,
        criterion_center,
        model=nn.Identity(),
        train_loader=[],
        resume_path=tmp_path / "last.pt",
    )

    assert torch.allclose(criterion_center.centers, expected_centers)


def test_reid_resume_initializes_missing_center_loss_state_from_features(tmp_path):
    trainer = _trainer(tmp_path)
    criterion_center = CenterLoss(num_classes=3, feat_dim=2)
    with torch.no_grad():
        criterion_center.centers.zero_()

    class FeatureModel(nn.Module):
        def forward(self, inputs):
            logits = torch.zeros(inputs.shape[0], 3, device=inputs.device)
            return logits, inputs

    train_loader = [
        (
            torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 0.0]]),
            torch.tensor([0, 0, 1]),
            None,
        )
    ]

    trainer._restore_center_loss_state(
        {},
        criterion_center,
        model=FeatureModel(),
        train_loader=train_loader,
        resume_path=tmp_path / "old_last.pt",
    )

    assert torch.allclose(criterion_center.centers[0], torch.tensor([2.0, 4.0]))
    assert torch.allclose(criterion_center.centers[1], torch.tensor([10.0, 0.0]))
    assert torch.allclose(criterion_center.centers[2], torch.zeros(2))


def test_csl_tinyvit_metric_feature_mode_follows_loss():
    triplet_model = csl_tinyvit_7m(num_classes=4, loss="triplet", pretrained=False)
    ms_model = csl_tinyvit_7m(num_classes=4, loss="ms", pretrained=False)

    assert triplet_model.head.metric_feature == "raw_mean"
    assert ms_model.head.metric_feature == "concat_bn"


def test_reid_registry_reads_custom_checkpoint_metadata(tmp_path):
    weights = tmp_path / "best.pt"
    torch.save({"model_name": "csl_tinyvit_23m", "num_classes": 751}, weights)

    assert ReIDModelRegistry.get_model_name(weights) == "csl_tinyvit_23m"
    assert ReIDModelRegistry.get_nr_classes(weights) == 751


def test_circle_loss_accepts_pk_batch():
    loss_fn = CircleLoss()
    features = torch.randn(4, 8)
    pids = torch.tensor([0, 0, 1, 1])

    loss = loss_fn(features, pids)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_margin_classifier_losses_accept_embeddings():
    features = torch.randn(4, 8)
    pids = torch.tensor([0, 0, 1, 1])

    arc_loss = ArcFaceLoss(feat_dim=8, num_classes=2)(features, pids)
    cos_loss = CosFaceLoss(feat_dim=8, num_classes=2)(features, pids)

    assert arc_loss.ndim == 0
    assert cos_loss.ndim == 0
    assert torch.isfinite(arc_loss)
    assert torch.isfinite(cos_loss)


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


def test_csl_tinyvit_lmbn_variant_builds_lmbn_style_head():
    model = csl_tinyvit_lmbn(num_classes=4, pretrained=False, loss="ms")

    assert isinstance(model.head, LMBNStyleMultiBranchHead)
    assert model.head.metric_feature == "raw_mean"

    model.train()
    logits, train_features = model(torch.randn(2, 3, 384, 128))
    assert len(logits) == 7
    assert isinstance(train_features, list)
    assert len(train_features) == 3
    assert all(feature.shape == (2, 512) for feature in train_features)

    model.eval()
    with torch.no_grad():
        eval_features = model(torch.randn(2, 3, 384, 128))
    assert eval_features.shape == (2, 3584)


def test_csl_tinyvit_lmbn_triplet_returns_lmbn_metric_feature_list():
    model = csl_tinyvit_lmbn(num_classes=4, pretrained=False, loss="triplet")
    model.train()

    logits, train_features = model(torch.randn(2, 3, 384, 128))

    assert len(logits) == 7
    assert isinstance(train_features, list)
    assert len(train_features) == 3
    assert all(feature.shape == (2, 512) for feature in train_features)


def test_csl_tinyvit_lmbn_variants_cover_7m_11m_23m_widths():
    small = csl_tinyvit_7m_lmbn(num_classes=4, pretrained=False, loss="ms")
    normal = csl_tinyvit_11m_lmbn(num_classes=4, pretrained=False, loss="ms")
    large = csl_tinyvit_23m_lmbn(num_classes=4, pretrained=False, loss="ms")

    assert isinstance(small.head, LMBNStyleMultiBranchHead)
    assert isinstance(normal.head, LMBNStyleMultiBranchHead)
    assert isinstance(large.head, LMBNStyleMultiBranchHead)
    assert [layer.dim for layer in small.layers] == [64, 128, 160, 320]
    assert [layer.dim for layer in normal.layers] == [64, 128, 256, 448]
    assert [layer.dim for layer in large.layers] == [96, 192, 384, 576]


def test_csl_tinyvit_feature_fusion_preserves_output_shape():
    model = csl_tinyvit_7m(num_classes=4, pretrained=False, feature_fusion="last2")
    model.eval()

    with torch.no_grad():
        features = model(torch.randn(1, 3, 384, 128))

    assert model.feature_fusion == "last2"
    assert isinstance(model.feature_fusion_module, CSLTinyViTFeatureFusion)
    assert model.feature_fusion_module.projections["2"][0].in_channels == 320
    assert model.fusion_scales["2"].item() == 0.0
    assert features.shape == (1, 1536)


def test_csl_tinyvit_weighted_feature_fusion_preserves_output_shape():
    model = csl_tinyvit_7m(num_classes=4, pretrained=False, feature_fusion="weighted_last3")
    model.eval()

    with torch.no_grad():
        features = model(torch.randn(1, 3, 384, 128))

    weights = model._normalized_fusion_weights()
    assert model.feature_fusion == "weighted_last3"
    assert model._fusion_stage_indices == (1, 2)
    assert isinstance(model.feature_fusion_module, CSLTinyViTFeatureFusion)
    assert model.fusion_weights.shape == (3,)
    assert weights[0] > 0.99
    assert torch.all(weights[1:] > 0)
    assert features.shape == (1, 1536)


def test_csl_tinyvit_feature_fusion_module_handles_variable_paths():
    module = CSLTinyViTFeatureFusion(
        fusion_type="weighted",
        stage_indices=(0, 1, 2),
        path_channels={0: 4, 1: 8, 2: 16},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 4, 2)
    path_features = {
        0: torch.randn(2, 4, 4, 2),
        1: torch.randn(2, 8, 8, 4),
        2: torch.randn(2, 16, 4, 2),
    }

    output = module(final_feature, path_features)

    assert module.stage_indices == (0, 1, 2)
    assert module.projections["0"][0].in_channels == 4
    assert module.projections["1"][0].in_channels == 8
    assert module.projections["2"][0].in_channels == 16
    assert module.fusion_weights.shape == (4,)
    assert output.shape == final_feature.shape


def test_csl_tinyvit_feature_fusion_uses_23m_path_channels():
    model = csl_tinyvit_23m(num_classes=4, pretrained=False, feature_fusion="weighted_last3")

    assert model.feature_fusion_module.projections["1"][0].in_channels == 384
    assert model.feature_fusion_module.projections["2"][0].in_channels == 576


def test_csl_tinyvit_feature_fusion_uses_11m_last3_path_channels():
    model = csl_tinyvit_11m(num_classes=4, pretrained=False, feature_fusion="weighted_last3")

    assert model._fusion_stage_indices == (1, 2)
    assert model.neck[0].in_channels == 448
    assert model.feature_fusion_module.projections["1"][0].in_channels == 256
    assert model.feature_fusion_module.projections["2"][0].in_channels == 448


def test_csl_tinyvit_feature_fusion_loads_legacy_direct_keys():
    model = csl_tinyvit_7m(num_classes=4, pretrained=False, feature_fusion="last2")
    legacy_state = {}
    for key, value in model.state_dict().items():
        legacy_key = key.replace("feature_fusion_module.projections.", "fusion_projections.")
        legacy_key = legacy_key.replace("feature_fusion_module.residual_scales.", "fusion_scales.")
        legacy_state[legacy_key] = value.clone()

    fresh = csl_tinyvit_7m(num_classes=4, pretrained=False, feature_fusion="last2")

    fresh.load_state_dict(legacy_state, strict=True)


def test_registry_loads_legacy_csl_tinyvit_feature_fusion_keys(tmp_path):
    source = csl_tinyvit_7m(num_classes=4, pretrained=False, feature_fusion="last2")
    source.feature_fusion_module.projections["2"][0].weight.data.fill_(0.123)
    source.feature_fusion_module.residual_scales["2"].data.fill_(0.456)

    legacy_state = {}
    for key, value in source.state_dict().items():
        legacy_key = key.replace("feature_fusion_module.projections.", "fusion_projections.")
        legacy_key = legacy_key.replace("feature_fusion_module.residual_scales.", "fusion_scales.")
        legacy_state[legacy_key] = value.clone()

    weights = tmp_path / "csl_tinyvit_7m_legacy.pt"
    torch.save({"state_dict": legacy_state}, weights)

    loaded = csl_tinyvit_7m(num_classes=4, pretrained=False, feature_fusion="last2")
    ReIDModelRegistry.load_pretrained_weights(loaded, weights)

    torch.testing.assert_close(
        loaded.feature_fusion_module.projections["2"][0].weight,
        source.feature_fusion_module.projections["2"][0].weight,
    )
    torch.testing.assert_close(
        loaded.feature_fusion_module.residual_scales["2"],
        source.feature_fusion_module.residual_scales["2"],
    )


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


def test_bnneck3_skips_classifier_in_eval_return_features():
    neck = BNNeck3(input_dim=8, class_num=3, feat_dim=4, return_f=True)
    neck.eval()

    feature, score, raw = neck(torch.randn(2, 8, 1, 1))

    assert feature.shape == (2, 4)
    assert score is None
    assert raw.shape == (2, 4)


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


def test_trainer_uses_embedding_model_contract_for_margin_classifier(tmp_path):
    trainer = _trainer(tmp_path, loss_type="softmax", classifier_loss="arcface")

    assert trainer._model_loss_type() == "triplet"


def test_trainer_triplet_soft_margin_can_be_forced(tmp_path):
    assert _trainer(tmp_path, triplet_soft_margin=False)._use_soft_margin_triplet(is_vit=True) is False
    assert _trainer(tmp_path, triplet_soft_margin=True)._use_soft_margin_triplet(is_vit=False) is True


def test_trainer_margin_classifier_uses_effective_metric_feature(tmp_path):
    trainer = _trainer(tmp_path, metric_feature="concat_bn")
    features = {
        "global": torch.randn(4, 8),
        "raw_mean": torch.randn(4, 8),
        "concat_bn": torch.randn(4, 24),
    }

    assert trainer._classification_features(features) is features["concat_bn"]


def test_trainer_builds_margin_classifier_with_selected_feature_dim(tmp_path):
    trainer = _trainer(tmp_path, classifier_loss="arcface", metric_feature="concat_bn")

    criterion = trainer._build_classifier_loss(num_classes=3, feat_dim=24, label_smooth=0.0)

    assert criterion.weight.shape == (3, 24)


def test_resume_scheduler_extension_does_not_raise_checkpoint_lr(tmp_path):
    trainer = _trainer(tmp_path, epochs=300, warmup_epochs=20, eta_min=0.0)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "hparams.json").write_text(json.dumps({"epochs": 200}))
    param = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([param], lr=0.1)
    optimizer.param_groups[0]["lr"] = 0.01
    optimizer.param_groups[0]["initial_lr"] = 0.1
    optimizer.param_groups[0]["_base_lr"] = 0.1

    scheduler = trainer._build_resume_scheduler(
        optimizer,
        resumed_epoch=170,
        resume_path=run_dir / "last.pt",
        ckpt={"optimizer": {}},
    )

    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["initial_lr"] == 0.01
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 0.01


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
