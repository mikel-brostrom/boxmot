import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import boxmot.reid.backbones.families.csl_tinyvit.pretrained as csl_tinyvit_pretrained
import boxmot.reid.backbones.lmbn_ain_n as lmbn_ain_n_module
import boxmot.reid.backbones.lmbn_n as lmbn_n_module
from boxmot.engine.reid import trainer as workflow_trainer
from boxmot.reid.backbones.families.csl_tinyvit import (
    Attention,
    BranchSetAttention,
    CSLTinyViTFeatureFusion,
    DSELitePool,
    GeM,
    GPCLiteMultiBranchHead,
    HierarchicalBranchAttention,
    HierarchicalLateInteractionMatcher,
    LMBNStyleMultiBranchHead,
    MultiBranchHead,
    NormPreservingWidthMerge,
    PostFusionLocalMixer,
    ReIDResidualAdapter,
    ResidualMultiScaleQueryDecoder,
    TinyViTBlock,
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
from boxmot.reid.backbones.heads.bnneck import BNNeck3
from boxmot.reid.backbones.mobilenetv4 import TimmMobileNetV4ReID, mobilenetv4_conv_small
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.datasets import build_combined_dataset, build_dataset
from boxmot.reid.training.base import BaseTrainer
from boxmot.reid.training.config import ReIDTrainConfig
from boxmot.reid.training.losses import (
    METRIC_LOSS_REGISTRY,
    ArcFaceLoss,
    CenterLoss,
    CircleLoss,
    CosFaceLoss,
    CrossScaleMajorityMarginLoss,
    TreeBoostAPLoss,
    TripletLoss,
    WeightedRegularizedTripletLoss,
)
from boxmot.reid.training.provenance import model_pretrained_provenance
from boxmot.reid.training.trainer import (
    DatasetBundle,
    LoaderBundle,
    LossBundle,
    ModelBundle,
    OptimizationBundle,
    ReIDTrainer,
    ValMetrics,
    _TrainingTimeEstimator,
)


def _vit_tiny_module():
    return pytest.importorskip(
        "boxmot.reid.backbones.vit_tiny",
        reason="vit_tiny backbones are not present in this checkout",
    )


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


def test_reid_trainer_uses_base_trainer_contract():
    assert issubclass(ReIDTrainer, BaseTrainer)


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


class _ClassifierToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.classifier(self.backbone(x))


class _TinyViTLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(2, 2)
        self.blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.head = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.head(x))


class _FakeTimmFeatureInfo:
    def __init__(self, channels):
        self._channels = tuple(channels)

    def channels(self):
        return self._channels


class _FakeTimmMobileNetV4(nn.Module):
    def __init__(self, channels=(16, 24, 40, 80, 160)):
        super().__init__()
        self.feature_info = _FakeTimmFeatureInfo(channels)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(1, 1, kernel_size=1, bias=False) for _ in range(3)]
            + [nn.Conv2d(1, 1, kernel_size=1, stride=2, bias=False)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(channels[-1], 192, kernel_size=1, bias=False)
        self.norm_head = nn.BatchNorm2d(192)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Identity()

    def _features(self, x):
        base = x.mean(dim=1, keepdim=True)
        outputs = []
        for index, channels in enumerate(self.feature_info.channels(), start=1):
            divisor = 2**index
            if index == len(self.feature_info.channels()) and self.blocks[-1].stride == (1, 1):
                divisor //= 2
            height = max(x.shape[-2] // divisor, 1)
            width = max(x.shape[-1] // divisor, 1)
            pooled = F.adaptive_avg_pool2d(base, (height, width))
            outputs.append(pooled.repeat(1, channels, 1, 1))
        return outputs

    def forward_intermediates(self, x):
        outputs = self._features(x)
        return outputs[-1], outputs

    def forward(self, x):
        return self._features(x)


def _install_fake_timm(monkeypatch, *, available=None, captured=None):
    captured = captured if captured is not None else {}
    available = available or (
        "mobilenetv4_conv_small.e2400_r224_in1k",
        "mobilenetv4_conv_medium.e500_r256_in1k",
        "mobilenetv4_conv_large.e600_r384_in1k",
        "mobilenetv4_hybrid_medium.ix_e550_r256_in1k",
        "mobilenetv4_hybrid_large.e600_r384_in1k",
    )

    def create_model(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return _FakeTimmMobileNetV4()

    fake_timm = SimpleNamespace(
        list_models=lambda pattern, pretrained=False: list(available),
        create_model=create_model,
    )
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    return captured


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


@pytest.mark.parametrize(
    ("module", "model_cls", "builder_name", "with_ain_pools"),
    [
        (lmbn_n_module, lmbn_n_module.LMBN_n, "osnet_x1_0", False),
        (lmbn_ain_n_module, lmbn_ain_n_module.LMBN_ain_n, "osnet_ain_x1_0", True),
    ],
)
def test_lmbn_backbones_expose_feature_contract(monkeypatch, module, model_cls, builder_name, with_ain_pools):
    def fake_osnet_builder(*, pretrained=False):
        return _fake_osnet_backbone(with_ain_pools=with_ain_pools)

    monkeypatch.setattr(module, builder_name, fake_osnet_builder)

    model = model_cls(num_classes=4, loss="ms", pretrained=False, use_gpu=False)
    inputs = torch.randn(2, 512, 8, 4)

    model.eval()
    feature_maps = model.forward_features(inputs)
    embeddings = model.forward_head(feature_maps)

    assert model.feature_dim == 3584
    assert model.featuremaps(inputs).shape == (2, 512, 8, 4)
    assert embeddings.shape == (2, model.feature_dim)
    assert model(inputs).shape == (2, model.feature_dim)

    model.train()
    logits, metric_features = model(inputs)

    assert len(logits) == 7
    assert len(metric_features) == 3


def test_lmbn_state_dict_uses_backbone_prefix(monkeypatch):
    def fake_osnet_builder(*, pretrained=False):
        return SimpleNamespace(
            conv1=nn.Conv2d(3, 4, kernel_size=1, bias=False),
            maxpool=nn.Identity(),
            conv2=nn.Conv2d(4, 8, kernel_size=1, bias=False),
            conv3=nn.Sequential(nn.Conv2d(8, 512, kernel_size=1, bias=False), nn.Identity()),
            conv4=nn.Identity(),
            conv5=nn.Identity(),
        )

    monkeypatch.setattr(lmbn_n_module, "osnet_x1_0", fake_osnet_builder)

    source = lmbn_n_module.LMBN_n(num_classes=4, loss="ms", pretrained=False, use_gpu=False)
    state_dict = {key: value.clone() for key, value in source.state_dict().items()}
    state_keys = tuple(state_dict)

    assert not hasattr(source, "backone")
    assert any(key.startswith("backbone.") for key in state_keys)
    assert not any(key.startswith("backone.") for key in state_keys)


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


def test_trainer_resolves_training_recipe_from_model_family(monkeypatch, tmp_path):
    _install_fake_timm(monkeypatch)
    transformer_trainer = _trainer(tmp_path, model_name="csl_tinyvit_7m")
    cnn_trainer = _trainer(tmp_path, model_name="osnet_x0_25")
    hybrid_trainer = _trainer(tmp_path, model_name="mobilenetv4_conv_small")

    transformer_recipe = transformer_trainer._resolve_training_recipe(_TinyViTLikeModel())
    cnn_recipe = cnn_trainer._resolve_training_recipe(_ClassifierToyModel())
    hybrid_recipe = hybrid_trainer._resolve_training_recipe(mobilenetv4_conv_small(num_classes=4, pretrained=False))

    assert transformer_recipe.family == "transformer"
    assert transformer_recipe.name == "transformer_reid"
    assert transformer_recipe.optimizer_name == "AdamW"
    assert transformer_recipe.default_flip_tta is True
    assert cnn_recipe.family == "cnn"
    assert cnn_recipe.name == "cnn_reid"
    assert cnn_recipe.optimizer_name == "Adam"
    assert cnn_recipe.default_flip_tta is False
    assert hybrid_recipe.family == "hybrid"
    assert hybrid_recipe.name == "hybrid_reid"
    assert hybrid_recipe.optimizer_name == "AdamW"
    assert hybrid_recipe.grad_clip == 1.0
    assert hybrid_recipe.default_triplet_soft_margin is True


def test_training_recipe_drives_optimizer_and_grad_clip(tmp_path):
    trainer = _trainer(tmp_path)
    center_loss = CenterLoss(num_classes=2, feat_dim=2)
    losses = LossBundle(
        criterion_id=nn.Identity(),
        criterion_metric=None,
        criterion_center=center_loss,
        label_smooth=0.0,
        soft_margin=False,
        metric_dim=2,
        classifier_dim=2,
    )

    vit_bundle = ModelBundle(
        model=_TinyViTLikeModel(),
        ema_model=None,
        val_model=_TinyViTLikeModel(),
        is_transformer=True,
        training_family="transformer",
        recipe=trainer._training_recipe_for_family("transformer"),
    )
    cnn_bundle = ModelBundle(
        model=_ClassifierToyModel(),
        ema_model=None,
        val_model=_ClassifierToyModel(),
        is_transformer=False,
        training_family="cnn",
        recipe=trainer._training_recipe_for_family("cnn"),
    )

    vit_optimization = trainer._build_optimization_bundle(vit_bundle, losses)
    cnn_optimization = trainer._build_optimization_bundle(cnn_bundle, losses)

    assert isinstance(vit_optimization.optimizer, torch.optim.AdamW)
    assert vit_optimization.grad_clip == 1.0
    assert isinstance(cnn_optimization.optimizer, torch.optim.Adam)
    assert cnn_optimization.grad_clip == 0.0


def test_mobilenetv4_training_recipe_uses_adamw_no_decay_groups(monkeypatch, tmp_path):
    _install_fake_timm(monkeypatch)
    trainer = _trainer(
        tmp_path,
        model_name="mobilenetv4_conv_small",
        weight_decay=1e-4,
    )
    model = mobilenetv4_conv_small(num_classes=4, loss="triplet", pretrained=False)
    losses = LossBundle(
        criterion_id=nn.Identity(),
        criterion_metric=None,
        criterion_center=CenterLoss(num_classes=4, feat_dim=384),
        label_smooth=0.0,
        soft_margin=True,
        metric_dim=384,
        classifier_dim=384,
    )
    bundle = ModelBundle(
        model=model,
        ema_model=None,
        val_model=model,
        is_transformer=False,
        training_family="hybrid",
        recipe=trainer._resolve_training_recipe(model),
    )

    optimization = trainer._build_optimization_bundle(bundle, losses)

    assert isinstance(optimization.optimizer, torch.optim.AdamW)
    assert optimization.grad_clip == 1.0
    assert {group["weight_decay"] for group in optimization.optimizer.param_groups} == {0.0, 1e-4}
    assert any(group.get("no_weight_decay") for group in optimization.optimizer.param_groups)
    assert any(group.get("is_head") for group in optimization.optimizer.param_groups)


def test_mobilenetv4_prebuild_defaults_are_mobile_safe(monkeypatch, tmp_path):
    _install_fake_timm(monkeypatch)
    trainer = _trainer(
        tmp_path,
        model_name="mobilenetv4_conv_small",
        img_size=(384, 128),
        batch_size=64,
        feature_fusion="last2",
        head_pool="gelu_gem",
        head_parts=(1, 2),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        weight_decay=5e-4,
        epochs=200,
        eta_min=1e-6,
        random_erasing=0.5,
        random_patch=True,
        color_jitter=True,
        gaussian_blur=True,
        random_grayscale=0.1,
    )

    bundle = trainer._build_model_bundle(num_classes=4)

    assert bundle.recipe.name == "hybrid_reid"
    assert trainer.img_size == (384, 128)
    assert trainer.batch_size == 64
    assert trainer.feature_fusion == "final"
    assert trainer.head_pool == "avg"
    assert trainer.head_parts == (1,)
    assert trainer.metric_feature == "auto"
    assert trainer.inference_feature == "concat_bn"
    assert trainer.weight_decay == 1e-4
    assert trainer.epochs == 120
    assert trainer.eta_min == 1e-7
    assert trainer.triplet_soft_margin is True
    assert trainer.ema_decay == 0.999
    assert trainer.random_erasing == 0.35
    assert trainer.random_patch is False
    assert trainer.color_jitter is False
    assert trainer.gaussian_blur is False
    assert trainer.random_grayscale == 0.0
    assert trainer.backbone_freeze_epochs == 10
    assert trainer.gradual_unfreeze is False
    assert trainer.head_warmup_epochs == 0
    assert trainer.head_warmup_lr_mult == 2.0
    assert getattr(bundle.model, "feature_fusion") == "final"
    assert bundle.model.head.head_parts == (1,)


def test_mobilenetv4_prebuild_defaults_preserve_explicit_imgsz_alias(monkeypatch, tmp_path):
    _install_fake_timm(monkeypatch)
    trainer = _trainer(
        tmp_path,
        model_name="mobilenetv4_conv_small",
        img_size=(384, 128),
        explicit_hparams=("imgsz",),
    )

    bundle = trainer._build_model_bundle(num_classes=4)

    assert trainer.img_size == (384, 128)
    assert bundle.model.img_size == (384, 128)


def test_hparams_and_checkpoint_metadata_record_training_family(tmp_path):
    trainer = _trainer(tmp_path, model_name="osnet_x0_25")
    model = _ClassifierToyModel()
    data = DatasetBundle(dataset=None, num_classes=2, default_eval_name="market1501")
    losses = LossBundle(
        criterion_id=nn.Identity(),
        criterion_metric=None,
        criterion_center=CenterLoss(num_classes=2, feat_dim=2),
        label_smooth=0.0,
        soft_margin=False,
        metric_dim=2,
        classifier_dim=2,
    )
    bundle = ModelBundle(
        model=model,
        ema_model=None,
        val_model=model,
        is_transformer=False,
        training_family="cnn",
        recipe=trainer._training_recipe_for_family("cnn"),
    )

    trainer._write_hparams(tmp_path, data, bundle, losses)
    hparams = json.loads((tmp_path / "hparams.json").read_text())
    metadata = trainer._checkpoint_metadata(model)

    assert hparams["model"]["training_family"] == "cnn"
    assert hparams["model"]["training_recipe"] == "cnn_reid"
    assert hparams["model"]["is_transformer"] is False
    assert hparams["model"]["cnn"]["feature_fusion"] == trainer.feature_fusion
    assert hparams["optimization"]["optimizer"] == "Adam"
    assert hparams["optimization"]["grad_clip"] == 0.0
    assert hparams["optimization"]["recipe"] == "cnn_reid"
    assert metadata["training_family"] == "cnn"
    assert metadata["training_recipe"] == "cnn_reid"
    assert metadata["is_transformer"] is False
    assert metadata["optimizer_name"] == "Adam"
    assert metadata["model"]["family"] == "cnn"
    assert metadata["model"]["is_transformer"] is False
    assert metadata["model"]["cnn"]["feature_fusion"] == trainer.feature_fusion
    assert metadata["optimization"]["recipe"] == "cnn_reid"
    assert metadata["optimization"]["optimizer"] == "Adam"


def test_global_ap_hparams_record_identity_only_pair_contract(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="osnet_x0_25",
        epochs=200,
        inference_feature="norm_concat_bn",
        global_ap_loss_weight=0.1,
    )
    model = _ClassifierToyModel()
    data = DatasetBundle(dataset=None, num_classes=2, default_eval_name="market1501")
    losses = LossBundle(
        criterion_id=nn.Identity(),
        criterion_metric=None,
        criterion_center=CenterLoss(num_classes=2, feat_dim=2),
        label_smooth=0.0,
        soft_margin=False,
        metric_dim=2,
        classifier_dim=2,
    )
    bundle = ModelBundle(
        model=model,
        ema_model=None,
        val_model=model,
        is_transformer=False,
        training_family="cnn",
        recipe=trainer._training_recipe_for_family("cnn"),
    )

    trainer._write_hparams(tmp_path, data, bundle, losses)

    global_ap = json.loads((tmp_path / "hparams.json").read_text())["losses"]["global_ap"]
    assert global_ap["label_source"] == "person_identity"
    assert global_ap["positive_policy"] == "same_identity_nonself"
    assert global_ap["negative_policy"] == "different_identity"
    assert global_ap["loss_inputs"] == ["norm_concat_bn", "sample_indices", "identity_labels"]
    assert "camera" not in json.dumps(global_ap).lower()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loss_type": "unknown"}, "Unsupported loss_type"),
        ({"classifier_loss": "unknown"}, "classifier_loss"),
        ({"epochs": 10, "warmup_epochs": 10}, "warmup_epochs"),
        ({"p": 0}, "p and k"),
        ({"k": 0}, "p and k"),
        ({"batch_size": 0}, "evaluation batch size"),
        ({"center_loss_weight": -1}, "center_loss_weight"),
        ({"early_id_loss_weight": -1}, "early_id_loss_weight"),
        ({"epochs": 20, "early_id_loss_epochs": 21}, "early_id_loss_epochs"),
        (
            {"center_loss_ramp_start_epoch": 10, "center_loss_ramp_end_epoch": 10},
            "center_loss_ramp_end_epoch",
        ),
        ({"random_erasing": 1.1}, "random_erasing"),
        ({"random_crop_scale": 0.99}, "random_crop_scale"),
        ({"eta_min": 1.0}, "eta_min"),
        ({"drop_global_aux_ratio": 0.0}, "drop_global_aux_ratio"),
        ({"drop_global_aux": True, "classifier_loss": "arcface"}, "drop_global_aux requires classifier_loss"),
        ({"drop_global_aux": True, "head_type": "gpc_lite"}, "drop_global_aux requires head_type"),
    ],
)
def test_trainer_rejects_invalid_config_early(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _trainer(tmp_path, **kwargs)


def test_trainer_applies_epoch_loss_schedules(tmp_path):
    trainer = _trainer(
        tmp_path,
        center_loss_weight=5e-3,
        early_id_loss_weight=1.25,
        early_id_loss_epochs=40,
        center_loss_ramp_start_epoch=10,
        center_loss_ramp_end_epoch=20,
    )

    assert trainer._effective_id_loss_weight(1) == 1.25
    assert trainer._effective_id_loss_weight(40) == 1.25
    assert trainer._effective_id_loss_weight(41) == 1.0
    assert trainer._effective_center_loss_weight(10) == 0.0
    assert trainer._effective_center_loss_weight(15) == pytest.approx(2.5e-3)
    assert trainer._effective_center_loss_weight(20) == pytest.approx(5e-3)
    assert trainer._effective_center_loss_weight(21) == pytest.approx(5e-3)


def test_trainer_applies_delayed_csmm_weight_ramp(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        epochs=50,
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        csmm_loss_weight=0.10,
        csmm_start_epoch=20,
        csmm_ramp_end_epoch=40,
    )

    assert trainer._effective_csmm_loss_weight(19) == 0.0
    assert trainer._effective_csmm_loss_weight(20) == 0.0
    assert trainer._effective_csmm_loss_weight(30) == pytest.approx(0.05)
    assert trainer._effective_csmm_loss_weight(40) == pytest.approx(0.10)


def test_trainer_applies_delayed_treeboost_weight_ramp(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        epochs=80,
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        treeboost_loss_weight=0.15,
        treeboost_start_epoch=30,
        treeboost_ramp_end_epoch=60,
    )

    assert trainer._effective_treeboost_loss_weight(30) == 0.0
    assert trainer._effective_treeboost_loss_weight(45) == pytest.approx(0.075)
    assert trainer._effective_treeboost_loss_weight(60) == pytest.approx(0.15)


def test_trainer_exposes_distinct_train_and_eval_batch_sizes(tmp_path):
    trainer = _trainer(tmp_path, batch_size=96, p=12, k=4)

    assert trainer.train_batch_size == 48
    assert trainer.eval_batch_size == 96
    assert trainer.batch_size == 96


def test_typed_training_config_preserves_legacy_constructor_values(tmp_path):
    data_specs = [{"name": "market1501", "root": str(tmp_path / "market")}]
    config = ReIDTrainConfig.from_flat_kwargs(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        data_specs=data_specs,
        batch_size=96,
        p=12,
        k=4,
        seed=7,
        deterministic=True,
        center_loss_weight=0.0,
        layer_decay=0.83,
    )

    trainer = ReIDTrainer.from_config(config)

    assert trainer.model_name == "csl_tinyvit_7m"
    assert trainer.train_batch_size == 48
    assert trainer.eval_batch_size == 96
    assert trainer.seed == 7
    assert trainer.center_loss_weight == 0.0
    assert config.optimization.layer_decay == pytest.approx(0.83)
    assert trainer.layer_decay == pytest.approx(0.83)
    assert trainer.data_specs == ({"name": "market1501", "root": str((tmp_path / "market").resolve())},)


def test_run_is_thin_orchestration_over_typed_bundles(monkeypatch, tmp_path):
    trainer = _trainer(tmp_path)
    calls = []
    data = SimpleNamespace(num_classes=4)
    models = object()
    loaders = object()
    losses = object()
    optimization = object()
    state = object()
    expected = SimpleNamespace(best_mAP=0.5)

    monkeypatch.setattr(trainer, "_prepare_runtime", lambda: calls.append("runtime"))
    monkeypatch.setattr(trainer, "_build_dataset_bundle", lambda: calls.append("data") or data)
    monkeypatch.setattr(
        trainer,
        "_build_model_bundle",
        lambda num_classes: calls.append(("model", num_classes)) or models,
    )
    monkeypatch.setattr(
        trainer,
        "_build_loader_bundle",
        lambda bundle: calls.append(("loaders", bundle)) or loaders,
    )
    monkeypatch.setattr(
        trainer,
        "_build_loss_bundle",
        lambda model_bundle, num_classes: calls.append(("losses", num_classes)) or losses,
    )
    monkeypatch.setattr(
        trainer,
        "_build_optimization_bundle",
        lambda model_bundle, loss_bundle: calls.append("optimization") or optimization,
    )
    monkeypatch.setattr(
        trainer,
        "_restore_if_needed",
        lambda *args: calls.append("restore") or state,
    )
    monkeypatch.setattr(trainer, "_make_save_dir", lambda: tmp_path / "run")
    monkeypatch.setattr(trainer, "_write_hparams", lambda *args: calls.append("hparams"))
    monkeypatch.setattr(
        trainer,
        "_fit",
        lambda **kwargs: calls.append("fit") or expected,
    )

    result = trainer.run()

    assert result is expected
    assert calls == [
        "runtime",
        "data",
        ("model", 4),
        ("loaders", data),
        ("losses", 4),
        "optimization",
        "restore",
        "hparams",
        "fit",
    ]


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
        random_crop_scale=1.05,
        color_augmentation=False,
        flip_tta=True,
    )

    assert trainer.color_jitter is False
    assert trainer.gaussian_blur is False
    assert trainer.random_grayscale == 0.0
    assert trainer.random_erasing == 0.5
    assert trainer.random_patch is False
    assert trainer.random_crop_scale == 1.05
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
                "seed": 91,
                "deterministic": False,
                "lr": 7e-4,
                "center_loss_weight": 5e-3,
                "early_id_loss_weight": 1.25,
                "early_id_loss_epochs": 40,
                "center_loss_ramp_start_epoch": 10,
                "center_loss_ramp_end_epoch": 20,
                "head_pool": "gem",
                "head_parts": [1, 2, 4],
                "part_pooling": "tokens",
                "num_part_tokens": 4,
                "decouple_patterns": True,
                "pattern_adapter_dim": 128,
                "feature_fusion": "last2",
                "reid_adapter_stages": [2, 3],
                "reid_adapter_reduction": 8,
                "branch_aware_metric": True,
                "branch_metric_part_weight": 0.25,
                "head_warmup_epochs": 10,
                "head_warmup_lr_mult": 3.0,
                "vit_lr_profile": "reid_lrd",
                "backbone_freeze_epochs": 20,
                "gradual_unfreeze": True,
                "gradual_unfreeze_head_epochs": 5,
                "gradual_unfreeze_stage_epochs": 10,
                "gradual_unfreeze_backbone_lr_mult": 0.1,
                "gradual_unfreeze_backbone_lr_epochs": 5,
            }
        )
    )
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        @classmethod
        def from_config(cls, config):
            return cls(**config.to_trainer_kwargs())

        def run(self):
            return SimpleNamespace(weights_path=run_dir / "best.pt", best_mAP=0.0, best_rank1=0.0)

    monkeypatch.setattr(workflow_trainer, "ReIDTrainer", FakeTrainer)
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

    workflow_trainer.main(args)

    assert captured["lr"] == 3.5e-4
    assert captured["center_loss_weight"] == 0.0
    assert captured["early_id_loss_weight"] == 1.25
    assert captured["early_id_loss_epochs"] == 40
    assert captured["center_loss_ramp_start_epoch"] == 10
    assert captured["center_loss_ramp_end_epoch"] == 20
    assert captured["seed"] == 91
    assert captured["deterministic"] is False
    assert captured["head_pool"] == "gem"
    assert captured["head_parts"] == [1, 2, 4]
    assert captured["part_pooling"] == "tokens"
    assert captured["num_part_tokens"] == 4
    assert captured["decouple_patterns"] is True
    assert captured["pattern_adapter_dim"] == 128
    assert captured["feature_fusion"] == "last2"
    assert captured["reid_adapter_stages"] == [2, 3]
    assert captured["reid_adapter_reduction"] == 8
    assert captured["branch_aware_metric"] is True
    assert captured["branch_metric_part_weight"] == 0.25
    assert captured["head_warmup_epochs"] == 10
    assert captured["head_warmup_lr_mult"] == 3.0
    assert captured["vit_lr_profile"] == "reid_lrd"
    assert captured["backbone_freeze_epochs"] == 20
    assert captured["gradual_unfreeze"] is True
    assert captured["gradual_unfreeze_head_epochs"] == 5
    assert captured["gradual_unfreeze_stage_epochs"] == 10
    assert captured["gradual_unfreeze_backbone_lr_mult"] == 0.1
    assert captured["gradual_unfreeze_backbone_lr_epochs"] == 5
    assert captured["explicit_hparams"] == {"lr", "center_loss_weight"}


def test_resume_hparams_nested_layout_applies_defaults(monkeypatch, tmp_path):
    run_dir = tmp_path / "exp_nested"
    run_dir.mkdir()
    (run_dir / "hparams.json").write_text(
        json.dumps(
            {
                "run": {
                    "model_name": "csl_tinyvit_7m",
                    "seed": 73,
                    "deterministic": False,
                },
                "data": {
                    "dataset": "market1501",
                    "data_dir": str(tmp_path),
                    "img_size": [384, 128],
                    "sampler": {"p": 16, "k": 4},
                },
                "model": {
                    "feature_fusion": "last2",
                    "attention": {
                        "window_layout": "rect",
                        "bias": "absolute",
                        "interpolate_pretrained_bias": True,
                    },
                    "reid_adapters": {"stages": [3], "reduction": 4},
                    "head": {
                        "pool": "gem",
                        "parts": [1, 2, 4],
                        "part_pooling": "tokens",
                        "num_part_tokens": 4,
                        "decouple_patterns": True,
                        "pattern_adapter_dim": 128,
                    },
                    "branch": {"aware_metric": True, "metric_part_weight": 0.25},
                },
                "optimization": {
                    "epochs": 250,
                    "vit_lr_profile": "reid_lrd",
                    "backbone_freeze_epochs": 40,
                    "gradual_unfreeze": {
                        "enabled": True,
                        "head_epochs": 5,
                        "stage_epochs": 10,
                        "backbone_lr_mult": 0.1,
                        "backbone_lr_epochs": 5,
                    },
                    "scheduler": {"warmup_epochs": 20},
                },
                "losses": {
                    "loss_type": "triplet",
                    "weights": {"center_loss_weight": 0.005},
                    "schedules": {
                        "early_id_loss": {"weight": 1.25, "epochs": 40},
                        "center_loss_ramp": {"start_epoch": 10, "end_epoch": 20},
                    },
                },
            }
        )
    )
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        @classmethod
        def from_config(cls, config):
            return cls(**config.to_trainer_kwargs())

        def run(self):
            return SimpleNamespace(weights_path=run_dir / "best.pt", best_mAP=0.0, best_rank1=0.0)

    monkeypatch.setattr(workflow_trainer, "ReIDTrainer", FakeTrainer)
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

    workflow_trainer.main(args)

    assert captured["lr"] == 3.5e-4
    assert captured["seed"] == 73
    assert captured["deterministic"] is False
    assert captured["feature_fusion"] == "last2"
    assert captured["attention_window_layout"] == "rect"
    assert captured["interpolate_pretrained_attention_bias"] is True
    assert captured["head_pool"] == "gem"
    assert captured["head_parts"] == [1, 2, 4]
    assert captured["part_pooling"] == "tokens"
    assert captured["num_part_tokens"] == 4
    assert captured["decouple_patterns"] is True
    assert captured["pattern_adapter_dim"] == 128
    assert captured["reid_adapter_stages"] == [3]
    assert captured["reid_adapter_reduction"] == 4
    assert captured["branch_aware_metric"] is True
    assert captured["branch_metric_part_weight"] == 0.25
    assert captured["center_loss_weight"] == 0.005
    assert captured["early_id_loss_weight"] == 1.25
    assert captured["early_id_loss_epochs"] == 40
    assert captured["center_loss_ramp_start_epoch"] == 10
    assert captured["center_loss_ramp_end_epoch"] == 20
    assert captured["p"] == 16
    assert captured["k"] == 4
    assert captured["warmup_epochs"] == 20
    assert captured["vit_lr_profile"] == "reid_lrd"
    assert captured["backbone_freeze_epochs"] == 40
    assert captured["gradual_unfreeze"] is True
    assert captured["gradual_unfreeze_head_epochs"] == 5
    assert captured["gradual_unfreeze_stage_epochs"] == 10
    assert captured["gradual_unfreeze_backbone_lr_mult"] == 0.1
    assert captured["gradual_unfreeze_backbone_lr_epochs"] == 5


def test_reid_checkpoint_saves_center_loss_state(tmp_path):
    trainer = _trainer(tmp_path)
    model = nn.Linear(3, 2)
    criterion_center = CenterLoss(num_classes=2, feat_dim=3)
    expected_centers = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with torch.no_grad():
        criterion_center.centers.copy_(expected_centers)

    ckpt_path = tmp_path / "last.pt"
    trainer.checkpoint_manager.save_last(
        ckpt_path,
        model=model,
        epoch=3,
        val=None,
        optimizer=None,
        optimizer_center=None,
        criterion_center=criterion_center,
        criterion_classifier=None,
        ema_model=None,
        best_mAP=0.0,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert "center_loss_state_dict" in ckpt
    assert ckpt["checkpoint_precision"] == "native"
    assert ckpt["center_loss_state_dict"]["centers"].dtype == torch.float32
    torch.testing.assert_close(ckpt["center_loss_state_dict"]["centers"], expected_centers)
    assert ckpt["seed"] == trainer.seed
    assert ckpt["deterministic"] is trainer.deterministic
    assert {"python", "numpy", "torch"} <= set(ckpt["rng_state"])


def test_last_checkpoint_keeps_live_and_ema_weights_separate(tmp_path):
    trainer = _trainer(tmp_path)
    live_model = nn.Linear(2, 2)
    ema_model = nn.Linear(2, 2)
    with torch.no_grad():
        live_model.weight.fill_(1.0)
        ema_model.weight.fill_(2.0)
    optimizer = torch.optim.AdamW(live_model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = live_model(torch.ones(2, 2)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    criterion_center = CenterLoss(num_classes=2, feat_dim=2)
    path = tmp_path / "last.pt"

    trainer.checkpoint_manager.save_last(
        path,
        model=live_model,
        epoch=4,
        val=None,
        optimizer=optimizer,
        optimizer_center=None,
        criterion_center=criterion_center,
        criterion_classifier=nn.Identity(),
        ema_model=ema_model,
        best_mAP=0.7,
        scheduler=scheduler,
        grad_scaler=grad_scaler,
        best_epoch=3,
        best_rank1=0.8,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_precision"] == "native"
    torch.testing.assert_close(checkpoint["state_dict"]["weight"], live_model.weight)
    torch.testing.assert_close(checkpoint["ema_state_dict"]["weight"], ema_model.weight)
    assert checkpoint["state_dict"]["weight"].dtype == torch.float32
    assert checkpoint["ema_state_dict"]["weight"].dtype == torch.float32
    optimizer_tensors = [
        tensor
        for state in checkpoint["optimizer"]["state"].values()
        for tensor in state.values()
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor)
    ]
    assert optimizer_tensors
    assert {tensor.dtype for tensor in optimizer_tensors} == {torch.float32}
    assert checkpoint["checkpoint_type"] == "last"
    assert checkpoint["resumable"] is True
    assert checkpoint["best_mAP"] == 0.7
    assert checkpoint["best_epoch"] == 3
    assert checkpoint["best_rank1"] == 0.8
    assert checkpoint["scheduler"] == scheduler.state_dict()
    assert checkpoint["grad_scaler"] == grad_scaler.state_dict()


def test_native_last_checkpoint_replays_next_adamw_step_exactly(tmp_path):
    trainer = _trainer(tmp_path)
    torch.manual_seed(37)
    live_model = nn.Linear(3, 2)
    live_optimizer = torch.optim.AdamW(live_model.parameters(), lr=3e-4)
    first_inputs = torch.randn(5, 3)
    first_targets = torch.randn(5, 2)
    second_inputs = torch.randn(5, 3)
    second_targets = torch.randn(5, 2)

    first_loss = nn.functional.mse_loss(live_model(first_inputs), first_targets)
    first_loss.backward()
    live_optimizer.step()
    live_optimizer.zero_grad(set_to_none=True)

    path = tmp_path / "last.pt"
    trainer.checkpoint_manager.save_last(
        path,
        model=live_model,
        epoch=1,
        val=None,
        optimizer=live_optimizer,
        optimizer_center=None,
        criterion_center=None,
        criterion_classifier=None,
        ema_model=None,
        best_mAP=0.0,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    resumed_model = nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=3e-4)
    resumed_model.load_state_dict(checkpoint["state_dict"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])

    for model, optimizer in (
        (live_model, live_optimizer),
        (resumed_model, resumed_optimizer),
    ):
        loss = nn.functional.mse_loss(model(second_inputs), second_targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for live_parameter, resumed_parameter in zip(
        live_model.parameters(),
        resumed_model.parameters(),
    ):
        assert torch.equal(live_parameter, resumed_parameter)


def test_last_checkpoint_keeps_train_only_classifier_weights_for_resume(tmp_path):
    trainer = _trainer(tmp_path)
    model = _ClassifierToyModel()
    path = tmp_path / "last.pt"

    trainer.checkpoint_manager.save_last(
        path,
        model=model,
        epoch=4,
        val=None,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        optimizer_center=None,
        criterion_center=None,
        criterion_classifier=None,
        ema_model=None,
        best_mAP=0.7,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["state_dict"]["backbone.weight"].dtype == torch.float32
    assert checkpoint["state_dict"]["classifier.weight"].dtype == torch.float32


def test_best_checkpoint_records_metric_and_is_weights_only(tmp_path):
    trainer = _trainer(tmp_path, classifier_loss="arcface")
    model = _ClassifierToyModel()
    model.pretrained_url = "https://example.invalid/tinyvit.pth"
    model.pretrained_sha256 = "a" * 64
    model.pretrained_backbone_required_tensor_count = 292
    model.pretrained_backbone_matched_tensor_count = 292
    model.pretrained_backbone_tensor_coverage = 1.0
    model.pretrained_backbone_required_numel = 5_078_939
    model.pretrained_backbone_matched_numel = 5_078_939
    model.pretrained_backbone_numel_coverage = 1.0
    validation = ValMetrics(epoch=3, mAP=0.81, rank1=0.92, rank5=0.0, rank10=0.0)
    criterion_center = CenterLoss(num_classes=2, feat_dim=2)
    criterion_classifier = ArcFaceLoss(num_classes=2, feat_dim=2)
    path = tmp_path / "best.pt"

    trainer.checkpoint_manager.save_best(
        path,
        model=model,
        epoch=3,
        val=validation,
        criterion_center=criterion_center,
        criterion_classifier=criterion_classifier,
        best_mAP=validation.mAP,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_precision"] == "float16"
    assert checkpoint["state_dict"]["backbone.weight"].dtype == torch.float16
    assert not any(key.startswith("classifier.") or ".classifier." in key for key in checkpoint["state_dict"])
    assert checkpoint["checkpoint_type"] == "best"
    assert checkpoint["resumable"] is False
    assert checkpoint["model_kwargs_schema_version"] == 1
    assert checkpoint["model_kwargs"]["img_size"] == trainer.img_size
    assert checkpoint["img_size"] == list(trainer.img_size)
    assert checkpoint["pretrained"] == {
        "url": "https://example.invalid/tinyvit.pth",
        "sha256": "a" * 64,
        "required_tensor_count": 292,
        "matched_tensor_count": 292,
        "tensor_coverage": 1.0,
        "required_numel": 5_078_939,
        "matched_numel": 5_078_939,
        "numel_coverage": 1.0,
    }
    assert checkpoint["model"]["pretrained"] == checkpoint["pretrained"]
    assert checkpoint["best_mAP"] == validation.mAP
    assert checkpoint["mAP"] == validation.mAP
    assert "optimizer" not in checkpoint
    assert "optimizer_center" not in checkpoint
    assert "center_loss_state_dict" not in checkpoint
    assert "classifier_loss_state_dict" not in checkpoint
    assert "ema_state_dict" not in checkpoint
    assert "rng_state" not in checkpoint


def test_weights_only_checkpoint_is_rejected_for_resume(tmp_path):
    checkpoint_path = tmp_path / "best.pt"
    source_model = _ClassifierToyModel()
    with torch.no_grad():
        source_model.backbone.weight.fill_(3.0)
        source_model.classifier.weight.fill_(5.0)
    state_dict = {
        key: value.half() for key, value in source_model.state_dict().items() if not key.startswith("classifier.")
    }
    torch.save(
        {
            "state_dict": state_dict,
            "epoch": 4,
            "epochs": 120,
            "best_mAP": 0.0,
            "mAP": 0.77,
            "rank1": 0.88,
            "resumable": False,
        },
        checkpoint_path,
    )
    trainer = _trainer(
        tmp_path,
        resume=str(checkpoint_path),
        center_loss_weight=0.0,
    )
    live_model = _ClassifierToyModel()
    ema_model = _ClassifierToyModel()
    model_bundle = ModelBundle(
        model=live_model,
        ema_model=ema_model,
        val_model=ema_model,
        is_transformer=False,
    )
    loaders = LoaderBundle(train=[], query=[], gallery=[], cross_domain={})
    losses = LossBundle(
        criterion_id=nn.Identity(),
        criterion_metric=None,
        criterion_center=CenterLoss(2, 2),
        label_smooth=0.0,
        soft_margin=False,
        metric_dim=2,
        classifier_dim=2,
    )
    optimizer = torch.optim.SGD(live_model.parameters(), lr=0.1)
    optimization = OptimizationBundle(
        optimizer=optimizer,
        optimizer_center=torch.optim.SGD(losses.criterion_center.parameters(), lr=0.5),
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
        grad_clip=0.0,
    )

    with pytest.raises(ValueError, match="does not contain a resumable training state"):
        trainer._restore_if_needed(model_bundle, loaders, losses, optimization)


def test_resume_round_trips_pretrained_provenance_to_live_and_ema_models(tmp_path):
    source_trainer = _trainer(
        tmp_path,
        center_loss_weight=0.0,
        ema_decay=0.9,
    )
    source_model = _ClassifierToyModel()
    source_model.pretrained_url = "https://example.invalid/tinyvit.pth"
    source_model.pretrained_sha256 = "a" * 64
    source_model.pretrained_backbone_required_tensor_count = 292
    source_model.pretrained_backbone_matched_tensor_count = 292
    source_model.pretrained_backbone_tensor_coverage = 1.0
    source_model.pretrained_backbone_required_numel = 5_078_939
    source_model.pretrained_backbone_matched_numel = 5_078_939
    source_model.pretrained_backbone_numel_coverage = 1.0
    source_ema = _ClassifierToyModel()
    source_ema.load_state_dict(source_model.state_dict())
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1)
    source_center = CenterLoss(2, 2)
    source_optimizer_center = torch.optim.SGD(source_center.parameters(), lr=0.5)
    source_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        source_optimizer,
        T_max=10,
    )
    resume_path = tmp_path / "last.pt"
    source_trainer.checkpoint_manager.save_last(
        resume_path,
        model=source_model,
        epoch=0,
        val=None,
        optimizer=source_optimizer,
        optimizer_center=source_optimizer_center,
        criterion_center=source_center,
        criterion_classifier=nn.Identity(),
        ema_model=source_ema,
        best_mAP=0.0,
        scheduler=source_scheduler,
    )

    resumed_trainer = _trainer(
        tmp_path,
        center_loss_weight=0.0,
        ema_decay=0.9,
        resume=str(resume_path),
    )
    live_model = _ClassifierToyModel()
    ema_model = _ClassifierToyModel()
    model_bundle = ModelBundle(
        model=live_model,
        ema_model=ema_model,
        val_model=ema_model,
        is_transformer=False,
    )
    loaders = LoaderBundle(train=[], query=[], gallery=[], cross_domain={})
    center = CenterLoss(2, 2)
    losses = LossBundle(
        criterion_id=nn.Identity(),
        criterion_metric=None,
        criterion_center=center,
        label_smooth=0.0,
        soft_margin=False,
        metric_dim=2,
        classifier_dim=2,
    )
    optimizer = torch.optim.SGD(live_model.parameters(), lr=0.1)
    optimizer_center = torch.optim.SGD(center.parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    optimization = OptimizationBundle(
        optimizer=optimizer,
        optimizer_center=optimizer_center,
        scheduler=scheduler,
        grad_clip=0.0,
    )

    resumed_trainer._restore_if_needed(
        model_bundle,
        loaders,
        losses,
        optimization,
    )

    expected = model_pretrained_provenance(source_model)
    assert model_pretrained_provenance(live_model) == expected
    assert model_pretrained_provenance(ema_model) == expected

    round_trip_path = tmp_path / "round-trip-last.pt"
    resumed_trainer.checkpoint_manager.save_last(
        round_trip_path,
        model=live_model,
        epoch=0,
        val=None,
        optimizer=optimizer,
        optimizer_center=optimizer_center,
        criterion_center=center,
        criterion_classifier=nn.Identity(),
        ema_model=ema_model,
        best_mAP=0.0,
        scheduler=optimization.scheduler,
    )
    round_trip = torch.load(round_trip_path, map_location="cpu", weights_only=False)
    assert round_trip["pretrained"] == expected
    assert round_trip["model"]["pretrained"] == expected


def test_checkpoint_fp16_compaction_clamps_non_finite_weights(tmp_path):
    trainer = _trainer(tmp_path)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1e10, -1e10], [float("nan"), 42.0]]))
    validation = ValMetrics(epoch=1, mAP=0.5, rank1=0.6, rank5=0.0, rank10=0.0)
    path = tmp_path / "best.pt"

    trainer.checkpoint_manager.save_best(
        path,
        model=model,
        epoch=1,
        val=validation,
        criterion_center=None,
        criterion_classifier=None,
        best_mAP=validation.mAP,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint["state_dict"]["weight"]

    assert weights.dtype == torch.float16
    assert torch.isfinite(weights).all()
    assert weights[0, 0].item() == torch.finfo(torch.float16).max
    assert weights[0, 1].item() == -torch.finfo(torch.float16).max
    assert weights[1, 0].item() == 0.0
    assert weights[1, 1].item() == 42.0


def test_center_loss_matches_full_distance_matrix_value_and_gradients():
    torch.manual_seed(7)
    inputs = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    centers = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 1, 1, 2, 3, 3, 3, 4])

    full_distances = (
        inputs.square().sum(dim=1, keepdim=True) + centers.square().sum(dim=1).unsqueeze(0) - 2 * inputs @ centers.t()
    ).clamp_min(1e-12)
    reference = full_distances.gather(1, targets[:, None]).mean()
    reference.backward()
    expected_input_grad = inputs.grad.clone()
    expected_center_grad = centers.grad.clone()

    optimized_inputs = inputs.detach().clone().requires_grad_(True)
    criterion = CenterLoss(num_classes=5, feat_dim=16).double()
    with torch.no_grad():
        criterion.centers.copy_(centers.detach())
    actual = criterion(optimized_inputs, targets)
    actual.backward()

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(optimized_inputs.grad, expected_input_grad)
    torch.testing.assert_close(criterion.centers.grad, expected_center_grad)


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


def test_feature_dim_fallback_restores_bn_buffers_modes_caches_and_rng(tmp_path):
    trainer = _trainer(
        tmp_path,
        img_size=(16, 8),
        pretrained=False,
    )

    class LegacyFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.neck = BNNeck3(3, 2, 4, return_f=True)

        def forward(self, inputs):
            pooled = inputs.mean(dim=(2, 3), keepdim=True)
            bn_features, logits, raw_features = self.neck(pooled)
            return logits, raw_features

    model = LegacyFeatureModel().eval()
    model.neck.prepare_for_inference()
    model.neck.bn.train()
    modes_before = [module.training for module in model.modules()]
    buffers_before = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state().clone()

    assert trainer._probe_feat_dim(model) == 4

    assert torch.equal(torch.get_rng_state(), rng_before)
    assert [module.training for module in model.modules()] == modes_before
    buffers_after = dict(model.named_buffers())
    assert buffers_after.keys() == buffers_before.keys()
    assert all(
        torch.equal(buffers_after[name], value)
        for name, value in buffers_before.items()
    )


def test_explicit_feature_dimensions_bypass_training_forward(tmp_path):
    trainer = _trainer(tmp_path, classifier_loss="arcface")

    class DeclaredHead(nn.Module):
        center_dim = 1152
        metric_dim = 1152
        classifier_dim = 1152

    class DeclaredModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = DeclaredHead()

        def forward(self, inputs):
            raise AssertionError("explicit dimensions must bypass shape probing")

    model = DeclaredModel()

    assert trainer._probe_feat_dim(model) == 1152
    assert trainer._probe_classifier_feat_dim(model) == 1152


def test_vit_tiny_dpt_fpn_reid_uses_intermediate_block_maps():
    vit_tiny = _vit_tiny_module()
    model = vit_tiny.vit_tiny_dpt_fpn_reid(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        img_size=(64, 32),
        feat_dim=64,
        neck_dim=128,
        drop_path_rate=0.0,
        head_parts=(1, 2),
    )

    assert isinstance(model, vit_tiny.ViTTinyDPTFPNReID)
    assert model.out_indices == (2, 5, 8, 11)
    assert model.patch_embed.grid_size == (4, 2)

    model.eval()
    inputs = torch.randn(2, 3, 64, 32)
    with torch.no_grad():
        fused_map = model.forward_features(inputs)
        embeddings = model(inputs)

    assert fused_map.shape == (2, 128, 4, 2)
    assert embeddings.shape == (2, 64 * 3)


def test_registry_passes_reid_head_kwargs_to_vit_tiny_dpt_fpn_reid(tmp_path):
    vit_tiny = _vit_tiny_module()
    model = ReIDModelRegistry.build_model(
        name="vit_tiny_dpt_fpn_reid",
        weights=tmp_path / "vit_tiny_dpt_fpn_reid_market1501.pt",
        num_classes=4,
        loss="triplet",
        pretrained=False,
        use_gpu=False,
        img_size=(64, 32),
        feat_dim=64,
        neck_dim=128,
        head_pool="gelu_gem",
        head_parts=(1, 4),
        stripe_visibility=True,
        inference_feature="norm_concat_bn",
    )

    assert isinstance(model, vit_tiny.ViTTinyDPTFPNReID)
    assert model.head.head_pool == "gelu_gem"
    assert model.head.head_parts == (1, 4)
    assert model.head.stripe_visibility is True
    assert model.head.visibility_granularity == 4


def test_vit_tiny_dpt_fpn_layers_are_reid_adaptation_params():
    assert ReIDTrainer._is_head_or_neck_param("fpn_projections.0.0.weight")
    assert ReIDTrainer._is_head_or_neck_param("output_norms.2.weight")
    assert ReIDTrainer._is_head_or_neck_param("fusion_logits")
    assert ReIDTrainer._is_reid_adaptation_param("fpn_projections.0.0.weight")
    assert ReIDTrainer._is_reid_adaptation_param("output_norms.2.weight")
    assert ReIDTrainer._is_reid_adaptation_param("fusion_logits")


def test_vit_tiny_dpt_gradual_unfreeze_handles_flat_blocks(tmp_path):
    vit_tiny = _vit_tiny_module()
    trainer = _trainer(
        tmp_path,
        model_name="vit_tiny_dpt_fpn_reid",
        gradual_unfreeze=True,
    )
    model = vit_tiny.vit_tiny_dpt_fpn_reid(
        num_classes=4,
        pretrained=False,
        img_size=(64, 32),
        feat_dim=64,
        neck_dim=128,
        drop_path_rate=0.1,
    )

    model.train()
    trainer._set_gradual_unfreeze_trainability(model, "head")

    assert not model.blocks.training
    assert all(not block.training for block in model.blocks)
    assert model.fpn_projections.training
    assert model.head.training
    assert model.fusion_logits.requires_grad
    assert not model.blocks[-1].attn.qkv.weight.requires_grad

    model.train()
    trainer._set_gradual_unfreeze_trainability(model, "stage")

    assert not model.blocks[0].training
    assert model.blocks[-1].training
    assert model.fpn_projections[0][0].weight.requires_grad
    assert not model.blocks[-1].attn.qkv.weight.requires_grad
    assert model.blocks[-1].mlp.fc1.weight.requires_grad


def test_mobilenetv4_uses_timm_pretrained_head_backbone(monkeypatch):
    captured = _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=True,
        feature_fusion="last3",
        inference_feature="norm_concat_bn",
    )

    assert isinstance(model, TimmMobileNetV4ReID)
    assert model.timm_model_name == "mobilenetv4_conv_small.e2400_r224_in1k"
    assert model.pretrained_source == "huggingface/pytorch-image-models (timm)"
    assert captured["name"] == "mobilenetv4_conv_small.e2400_r224_in1k"
    assert captured["kwargs"]["pretrained"] is True
    assert captured["kwargs"]["num_classes"] == 0
    assert "features_only" not in captured["kwargs"]
    assert model.use_timm_head is True
    assert model.timm_head_mode == "pooled"
    assert model.timm_head_channels == 192
    assert model.feature_fusion_module.projections["1"][0].in_channels == 40
    assert model.feature_fusion_module.projections["2"][0].in_channels == 80

    model.eval()
    with torch.no_grad():
        eval_features = model(torch.randn(2, 3, 64, 32))

    assert eval_features.shape == (2, 512)

    model.train()
    logits, train_features = model(torch.randn(2, 3, 64, 32))

    assert len(logits) == 1
    assert train_features.shape == (2, 512)


def test_mobilenetv4_final_fusion_uses_timm_head_global_map(monkeypatch):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="final",
    )

    model.eval()
    with torch.no_grad():
        final_map = model.forward_features(torch.randn(2, 3, 64, 32))

    assert model.use_timm_head is True
    assert model.timm_head_mode == "pooled"
    assert final_map.shape == (2, 512, 1, 1)


def test_mobilenetv4_spatial_timm_head_preserves_c5_map(monkeypatch):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="final",
        timm_head_mode="spatial",
    )

    model.eval()
    with torch.no_grad():
        final_map = model.forward_features(torch.randn(2, 3, 64, 32))

    assert model.use_timm_head is True
    assert model.timm_head_mode == "spatial"
    assert final_map.shape == (2, 512, 2, 1)


@pytest.mark.parametrize("timm_head_mode", ("spatial_adapt_norm", "spatial_linear"))
def test_mobilenetv4_followup_timm_heads_preserve_c5_map(
    monkeypatch,
    timm_head_mode,
):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="final",
        timm_head_mode=timm_head_mode,
    )

    model.eval()
    with torch.no_grad():
        final_map = model.forward_features(torch.randn(2, 3, 64, 32))

    assert model.timm_head_mode == timm_head_mode
    assert final_map.shape == (2, 512, 2, 1)


def test_mobilenetv4_last_stride_one_retains_stride16_c5(monkeypatch):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="final",
        timm_head_mode="spatial_linear",
        mobilenetv4_last_stride=1,
    )

    model.eval()
    with torch.no_grad():
        final_map = model.forward_features(torch.randn(2, 3, 64, 32))

    assert model.mobilenetv4_last_stride == 1
    assert model.backbone.blocks[-1].stride == (1, 1)
    assert final_map.shape == (2, 512, 4, 2)


def test_mobilenetv4_spatial_ln_neck_matches_tinyvit_projection(monkeypatch):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="final",
        timm_head_mode="spatial_linear",
        spatial_conv_mode="depthwise_separable",
        mobilenetv4_neck_mode="spatial_ln",
    )

    model.eval()
    with torch.no_grad():
        final_map = model.forward_features(torch.randn(2, 3, 64, 32))

    assert model.mobilenetv4_neck_mode == "spatial_ln"
    assert isinstance(model.neck[0], nn.Conv2d)
    assert model.neck[1].__class__.__name__ == "LayerNorm2d"
    assert isinstance(model.neck[2], nn.Sequential)
    assert model.neck[3].__class__.__name__ == "LayerNorm2d"
    assert final_map.shape == (2, 512, 2, 1)


def test_mobilenetv4_spatial_linear_skips_pooled_domain_norm(monkeypatch):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="final",
        timm_head_mode="spatial_linear",
    )

    class FailIfCalled(nn.Module):
        def forward(self, inputs):
            raise AssertionError("spatial_linear must bypass timm norm_head")

    model.backbone.norm_head = FailIfCalled()
    model.eval()
    with torch.no_grad():
        final_map = model.forward_features(torch.randn(2, 3, 64, 32))

    assert final_map.shape == (2, 512, 2, 1)


def test_mobilenetv4_off_timm_head_preserves_raw_c5_map(monkeypatch):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="final",
        timm_head_mode="off",
    )

    model.eval()
    with torch.no_grad():
        final_map = model.forward_features(torch.randn(2, 3, 64, 32))

    assert model.use_timm_head is False
    assert model.timm_head_mode == "off"
    assert final_map.shape == (2, 512, 2, 1)


def test_mobilenetv4_global_final_parts_stage2_uses_final_global_and_stage2_parts(monkeypatch):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="global_final_parts_stage2",
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_pool="gelu_gem",
        head_parts=(1, 2),
        part_pooling="stripes",
        post_fusion_mixer="none",
    )

    model.eval()
    with torch.no_grad():
        global_map, local_map = model.forward_features(torch.randn(2, 3, 64, 32))
        eval_features = model(torch.randn(2, 3, 64, 32))

    assert model.feature_fusion == "global_final_parts_stage2"
    assert model._fusion_stage_indices == (1, 2)
    assert model.feature_fusion_module.split_global_local is True
    assert model.feature_fusion_module.target_stage_index == 2
    assert model.feature_fusion_module.projections["1"][0].in_channels == 40
    assert model.feature_fusion_module.projections["2"][0].in_channels == 80
    assert global_map.shape == (2, 512, 1, 1)
    assert local_map.shape == (2, 512, 4, 2)
    assert eval_features.shape == (2, 1536)

    model.train()
    logits, train_features = model(torch.randn(2, 3, 64, 32))

    assert model.head.metric_feature == "raw_concat"
    assert len(logits) == 3
    assert train_features.shape == (2, 1536)


def test_mobilenetv4_stage0_semantic_fine_uses_c3_c4_and_scale_balanced_head(monkeypatch):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine_reference",
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        feat_dim=384,
        neck_dim=384,
        head_pool="gelu_gem",
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
    )

    model.eval()
    with torch.no_grad():
        global_map, coarse_map, fine_map = model.forward_features(torch.randn(2, 3, 64, 32))
        eval_features = model(torch.randn(2, 3, 64, 32))

    assert model._fusion_source_indices == {0: -3, 1: -3, 2: -2}
    assert model.feature_fusion_module.stage0_fine_projection[0].in_channels == 40
    assert model.feature_fusion_module.projections["1"][0].in_channels == 40
    assert model.feature_fusion_module.projections["2"][0].in_channels == 80
    assert global_map.shape == (2, 384, 1, 1)
    assert coarse_map.shape == (2, 384, 4, 2)
    assert fine_map.shape == (2, 384, 8, 4)
    assert model.head.hierarchical_scales is True
    assert model.head.scale_balanced_branches is True
    assert [
        getattr(model.head, model.head._bn_attr(key)).reduction.out_channels
        for key, _, _ in model.head.branch_specs
    ] == [384, 192, 192, 96, 96, 96, 96]
    assert eval_features.shape == (2, 1152)

    model.train()
    logits, train_features = model(torch.randn(2, 3, 64, 32))

    assert len(logits) == 7
    assert train_features.shape == (2, 1152)


def test_mobilenetv4_executes_only_the_selected_final_projection(monkeypatch):
    _install_fake_timm(monkeypatch)

    global_model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        feat_dim=32,
        neck_dim=32,
        head_parts=(1, 2, 4),
    ).eval()
    global_calls = {"neck": 0, "spatial_neck": 0}
    global_model.neck.register_forward_hook(
        lambda *_: global_calls.__setitem__("neck", global_calls["neck"] + 1)
    )
    global_model.spatial_neck.register_forward_hook(
        lambda *_: global_calls.__setitem__("spatial_neck", global_calls["spatial_neck"] + 1)
    )

    spatial_model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        feature_fusion="last3",
        feat_dim=32,
        neck_dim=32,
    ).eval()
    spatial_calls = {"neck": 0, "spatial_neck": 0}
    spatial_model.neck.register_forward_hook(
        lambda *_: spatial_calls.__setitem__("neck", spatial_calls["neck"] + 1)
    )
    spatial_model.spatial_neck.register_forward_hook(
        lambda *_: spatial_calls.__setitem__("spatial_neck", spatial_calls["spatial_neck"] + 1)
    )

    with torch.no_grad():
        global_model.forward_features(torch.randn(2, 3, 64, 32))
        spatial_model.forward_features(torch.randn(2, 3, 64, 32))

    assert global_calls == {"neck": 1, "spatial_neck": 0}
    assert spatial_calls == {"neck": 0, "spatial_neck": 1}


def test_mobilenetv4_gradual_unfreeze_handles_timm_backbone_blocks(monkeypatch, tmp_path):
    _install_fake_timm(monkeypatch)
    trainer = _trainer(
        tmp_path,
        model_name="mobilenetv4_conv_small",
        gradual_unfreeze=True,
        gradual_unfreeze_head_epochs=5,
        gradual_unfreeze_stage_epochs=20,
    )
    model = mobilenetv4_conv_small(num_classes=4, loss="triplet", pretrained=False)

    parameter_groups = trainer._build_cnn_param_groups(model)
    assert [group["is_backbone"] for group in parameter_groups] == [True, False]
    assert [group["is_head"] for group in parameter_groups] == [False, True]

    model.train()
    trainer._set_gradual_unfreeze_trainability(model, "head")

    assert model.backbone.training is False
    assert not model.backbone.blocks[0].weight.requires_grad
    assert not model.backbone.blocks[-1].weight.requires_grad
    assert model.neck[0].weight.requires_grad
    assert next(model.head.parameters()).requires_grad

    model.train()
    trainer._set_gradual_unfreeze_trainability(model, "stage")

    assert model.backbone.training is False
    assert model.backbone.blocks[0].training is False
    assert model.backbone.blocks[-1].training is True
    assert not model.backbone.blocks[0].weight.requires_grad
    assert model.backbone.blocks[-1].weight.requires_grad
    assert model.neck[0].weight.requires_grad


def test_mobilenetv4_supports_postfusion_mixer_and_drop_global_aux(monkeypatch):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        post_fusion_mixer="dwconv",
        post_fusion_mixer_reduction=4,
        post_fusion_mixer_kernel=(5, 3),
        post_fusion_mixer_gamma_init=1e-4,
        drop_global_aux=True,
        drop_global_aux_ratio=0.25,
    )

    assert isinstance(model.post_fusion_mixer_module, PostFusionLocalMixer)
    assert model.post_fusion_mixer_module.gamma.item() == pytest.approx(1e-4)
    assert model.head.drop_global_aux_enabled is True
    assert model.head.drop_global_aux_ratio == 0.25

    model.train()
    logits, train_features = model(torch.randn(2, 3, 64, 32))

    assert len(logits) == 2
    assert train_features.shape == (2, 512)


def test_mobilenetv4_honors_metric_feature_directly(monkeypatch):
    _install_fake_timm(monkeypatch)

    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        metric_feature="raw_concat",
    )

    model.train()
    logits, train_features = model(torch.randn(2, 3, 64, 32))

    assert model.head.metric_feature == "raw_concat"
    assert len(logits) == 1
    assert train_features.shape == (2, 512)


def test_registry_passes_mobilenetv4_checkpoint_model_kwargs(monkeypatch, tmp_path):
    _install_fake_timm(monkeypatch)
    weights = tmp_path / "mobilenetv4_conv_small_market1501.pt"

    model = ReIDModelRegistry.build_model(
        "mobilenetv4_conv_small",
        weights=weights,
        num_classes=4,
        loss="triplet",
        pretrained=False,
        use_gpu=False,
        timm_model_name="mobilenetv4_conv_small.e2400_r224_in1k",
        feature_fusion="last2",
        metric_feature="raw_concat",
        post_fusion_mixer="dwconv",
        post_fusion_mixer_gamma_init=1e-4,
    )

    assert isinstance(model, TimmMobileNetV4ReID)
    assert model.feature_fusion == "last2"
    assert model.head.metric_feature == "raw_concat"
    assert model._fusion_stage_indices == (2,)
    assert isinstance(model.post_fusion_mixer_module, PostFusionLocalMixer)
    assert model.post_fusion_mixer_module.gamma.item() == pytest.approx(1e-4)
    assert ReIDModelRegistry.get_model_name(weights) == "mobilenetv4_conv_small"


def test_registry_reads_mobilenetv4_checkpoint_kwargs(tmp_path):
    weights = tmp_path / "mobilenetv4.pt"
    torch.save(
        {
            "model_name": "mobilenetv4_conv_small",
            "timm_model_name": "mobilenetv4_conv_small.e2400_r224_in1k",
            "timm_head_mode": "spatial",
            "feature_fusion": "last2",
            "post_fusion_mixer": "dwconv",
            "post_fusion_mixer_reduction": 4,
            "post_fusion_mixer_kernel": [5, 3],
            "post_fusion_mixer_gamma_init": 1e-4,
        },
        weights,
    )

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)

    assert kwargs["timm_model_name"] == "mobilenetv4_conv_small.e2400_r224_in1k"
    assert kwargs["timm_head_mode"] == "spatial"
    assert kwargs["feature_fusion"] == "last2"
    assert kwargs["post_fusion_mixer"] == "dwconv"
    assert kwargs["post_fusion_mixer_reduction"] == 4
    assert kwargs["post_fusion_mixer_kernel"] == (5, 3)
    assert kwargs["post_fusion_mixer_gamma_init"] == 1e-4


def test_registry_reads_pattern_head_checkpoint_kwargs(tmp_path):
    weights = tmp_path / "pattern_head.pt"
    torch.save(
        {
            "head_type": "gpc_lite",
            "part_pooling": "tokens",
            "num_part_tokens": 4,
            "decouple_patterns": True,
            "pattern_adapter_dim": 128,
            "stripe_visibility": True,
            "drop_global_aux": True,
            "drop_global_aux_ratio": 0.25,
            "scale_balanced_branches": True,
            "pyramid_resize_mode": "pool_nearest",
            "spatial_conv_mode": "depthwise_separable",
            "interpolate_pretrained_attention_bias": True,
            "post_fusion_mixer": "dwconv",
            "post_fusion_mixer_reduction": 4,
            "post_fusion_mixer_kernel": [5, 3],
            "post_fusion_mixer_gamma_init": 1e-4,
            "reid_adapter_stages": [2, 3],
            "reid_adapter_reduction": 8,
        },
        weights,
    )

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)

    assert kwargs["head_type"] == "gpc_lite"
    assert kwargs["part_pooling"] == "tokens"
    assert kwargs["num_part_tokens"] == 4
    assert kwargs["decouple_patterns"] is True
    assert kwargs["pattern_adapter_dim"] == 128
    assert kwargs["stripe_visibility"] is True
    assert kwargs["drop_global_aux"] is True
    assert kwargs["drop_global_aux_ratio"] == 0.25
    assert kwargs["scale_balanced_branches"] is True
    assert kwargs["pyramid_resize_mode"] == "pool_nearest"
    assert kwargs["spatial_conv_mode"] == "depthwise_separable"
    assert kwargs["interpolate_pretrained_attention_bias"] is True
    assert kwargs["post_fusion_mixer"] == "dwconv"
    assert kwargs["post_fusion_mixer_reduction"] == 4
    assert kwargs["post_fusion_mixer_kernel"] == (5, 3)
    assert kwargs["post_fusion_mixer_gamma_init"] == 1e-4
    assert kwargs["reid_adapter_stages"] == (2, 3)
    assert kwargs["reid_adapter_reduction"] == 8


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


def test_weighted_regularized_triplet_matches_soft_pair_weighting():
    features = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [10.0, 0.0], [11.0, 0.0], [13.0, 0.0]],
        requires_grad=True,
    )
    pids = torch.tensor([0, 0, 0, 1, 1, 1])

    loss = WeightedRegularizedTripletLoss()(features, pids)

    distances = torch.cdist(features, features)
    expected_losses = []
    for anchor in range(features.shape[0]):
        positive_mask = pids.eq(pids[anchor])
        positive_mask[anchor] = False
        positive_distances = distances[anchor][positive_mask]
        negative_distances = distances[anchor][~pids.eq(pids[anchor])]
        weighted_positive = (F.softmax(positive_distances, dim=0) * positive_distances).sum()
        weighted_negative = (F.softmax(-negative_distances, dim=0) * negative_distances).sum()
        expected_losses.append(F.softplus(weighted_positive - weighted_negative))
    expected = torch.stack(expected_losses).mean()

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert METRIC_LOSS_REGISTRY["wrt"] is WeightedRegularizedTripletLoss


def test_weighted_regularized_triplet_ignores_anchors_without_positive_pairs():
    features = torch.randn(3, 8, requires_grad=True)
    pids = torch.tensor([0, 1, 1])

    loss = WeightedRegularizedTripletLoss()(features, pids)

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(features.grad).all()


def test_cross_scale_majority_margin_requires_two_correct_scales():
    correct = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
        requires_grad=True,
    )
    undecided = torch.tensor(
        [[1.0, 0.2], [1.0, -0.2], [1.0, 0.1], [1.0, -0.1]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    criterion = CrossScaleMajorityMarginLoss(
        margin=0.10,
        temperature=0.05,
        topk_negatives=1,
    )

    majority_loss = criterion((correct, correct, undecided), labels)
    minority_loss = criterion((correct, undecided, undecided), labels)

    assert majority_loss < minority_loss
    minority_loss.backward()
    assert torch.isfinite(correct.grad).all()
    assert torch.isfinite(undecided.grad).all()


def _treeboost_features(batch_size=6):
    return (
        torch.randn(batch_size, 8, requires_grad=True),
        tuple(torch.randn(batch_size, 4, requires_grad=True) for _ in range(2)),
        tuple(torch.randn(batch_size, 2, requires_grad=True) for _ in range(4)),
    )


def test_treeboost_ap_routes_coarse_gradients_without_changing_global_gradient():
    torch.manual_seed(0)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    camera_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    baseline_features = _treeboost_features()
    coarse_features = (
        baseline_features[0].detach().clone().requires_grad_(),
        tuple(feature.detach().clone().requires_grad_() for feature in baseline_features[1]),
        tuple(feature.detach().clone().requires_grad_() for feature in baseline_features[2]),
    )

    global_only = TreeBoostAPLoss(
        coarse_coefficient=0.0,
        fine_coefficient=0.0,
        node_coefficient=0.0,
        regression_coefficient=0.0,
    )(baseline_features, labels, camera_ids)
    global_only.backward()

    with_coarse = TreeBoostAPLoss(
        coarse_coefficient=1.0,
        fine_coefficient=0.0,
        node_coefficient=0.0,
        regression_coefficient=0.0,
    )(coarse_features, labels, camera_ids)
    with_coarse.backward()

    torch.testing.assert_close(
        coarse_features[0].grad,
        baseline_features[0].grad,
        atol=1e-6,
        rtol=1e-5,
    )
    assert all(feature.grad.abs().sum() > 0 for feature in coarse_features[1])
    assert all(feature.grad is None or feature.grad.count_nonzero() == 0 for feature in coarse_features[2])


def test_treeboost_ap_skips_batches_without_cross_camera_positives():
    features = _treeboost_features(batch_size=4)
    labels = torch.tensor([0, 0, 1, 1])
    camera_ids = torch.zeros(4, dtype=torch.long)

    loss = TreeBoostAPLoss()(features, labels, camera_ids)

    assert loss == 0
    loss.backward()
    assert all(feature.grad is not None for feature in (features[0], *features[1], *features[2]))


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


def test_csl_tinyvit_builds_pattern_decoupled_part_token_head():
    model = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        part_pooling="tokens",
        num_part_tokens=4,
        decouple_patterns=True,
        pattern_adapter_dim=64,
    )

    assert model.head.part_pooling == "tokens"
    assert model.head.num_part_tokens == 4
    assert model.head.decouple_patterns is True
    assert model.head.pattern_adapter_dim == 64


def test_trainer_builds_pattern_decoupled_part_token_head(tmp_path):
    trainer = _trainer(
        tmp_path,
        pretrained=False,
        part_pooling="tokens",
        num_part_tokens=4,
        decouple_patterns=True,
        pattern_adapter_dim=64,
    )

    model = trainer._build_model(num_classes=4)

    assert model.head.part_pooling == "tokens"
    assert model.head.num_part_tokens == 4
    assert model.head.decouple_patterns is True
    assert model.head.pattern_adapter_dim == 64


def test_trainer_builds_postfusion_mixer_and_drop_global_aux(tmp_path):
    trainer = _trainer(
        tmp_path,
        pretrained=False,
        post_fusion_mixer="dwconv",
        post_fusion_mixer_reduction=4,
        post_fusion_mixer_kernel=(5, 3),
        post_fusion_mixer_gamma_init=1e-4,
        drop_global_aux=True,
        drop_global_aux_ratio=0.25,
    )

    model = trainer._build_model(num_classes=4)

    assert isinstance(model.post_fusion_mixer_module, PostFusionLocalMixer)
    assert model.post_fusion_mixer_module.gamma.item() == pytest.approx(1e-4)
    assert model.head.drop_global_aux_enabled is True
    assert model.head.drop_global_aux_ratio == 0.25


def test_csl_tinyvit_drop_path_rate_is_configurable():
    model = csl_tinyvit_23m(num_classes=4, pretrained=False, drop_path_rate=0.1)

    max_drop = max(
        block.drop_path.drop_prob
        for layer in model.layers
        for block in layer.blocks
        if hasattr(block.drop_path, "drop_prob")
    )

    assert abs(max_drop - 0.1) < 1e-6


def test_csl_tinyvit_23m_default_drop_path_rate_is_point_two():
    model = csl_tinyvit_23m(num_classes=4, pretrained=False)

    max_drop = max(
        block.drop_path.drop_prob
        for layer in model.layers
        for block in layer.blocks
        if hasattr(block.drop_path, "drop_prob")
    )

    assert abs(max_drop - 0.2) < 1e-6


@pytest.mark.parametrize("factory", [csl_tinyvit_7m, csl_tinyvit_11m, csl_tinyvit_23m])
def test_csl_tinyvit_blocks_alias_is_not_serialized_for_supported_scales(factory):
    model = factory(num_classes=4, pretrained=False)
    state_keys = tuple(model.state_dict())

    assert model.blocks is model.layers
    assert "blocks" not in model._modules
    assert any(key.startswith("layers.") for key in state_keys)
    assert not any(key.startswith("blocks.") for key in state_keys)


def test_csl_tinyvit_rectangular_shifted_attention_config():
    model = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        attention_window_layout="rect",
        attention_bias="signed_factorized",
        attention_mask=True,
        attention_shift=True,
        stage3_global=True,
    )

    assert model.layers[1].blocks[0].window_size == (12, 4)
    assert model.layers[1].blocks[1].shift_size == (6, 2)
    assert model.layers[2].blocks[1].window_size == (12, 8)
    assert model.layers[2].blocks[1].shift_size == (6, 4)
    assert model.layers[3].blocks[-1].window_size == (24, 8)
    assert model.layers[1].blocks[0].attn.bias_mode == "signed_factorized"
    assert model.layers[1].blocks[0].attention_mask is True


def test_csl_tinyvit_interpolates_pretrained_absolute_attention_biases(monkeypatch):
    source = csl_tinyvit_7m(num_classes=4, pretrained=False)
    source_backbone = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if key.startswith(("patch_embed.", "layers."))
    }
    source_biases = {
        key: torch.arange(value.numel(), dtype=value.dtype).reshape_as(value)
        for key, value in source_backbone.items()
        if key.endswith("attention_biases")
    }
    source_backbone.update(source_biases)
    target = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        attention_window_layout="rect",
        interpolate_pretrained_attention_bias=True,
    )
    monkeypatch.setattr(
        csl_tinyvit_pretrained,
        "load_hub_checkpoint",
        lambda *args, **kwargs: source_backbone,
    )

    csl_tinyvit_pretrained.load_pretrained_tinyvit(target, "test://tinyvit")

    assert len(source_biases) == 10
    assert target.pretrained_match_count == len(source_backbone)
    assert target.pretrained_backbone_tensor_coverage == 1.0
    assert target.pretrained_backbone_numel_coverage == 1.0
    assert set(target.pretrained_interpolated_attention_biases) == set(source_biases)
    target_state = target.state_dict()
    target_resolutions = {
        f"{name}.attention_biases": module.resolution
        for name, module in target.named_modules()
        if isinstance(module, Attention) and module.bias_mode == "absolute"
    }
    for key, source_bias in source_biases.items():
        expected = csl_tinyvit_pretrained._resize_absolute_attention_bias(
            source_bias,
            target_resolutions[key],
        )
        torch.testing.assert_close(target_state[key], expected)


def test_csl_tinyvit_pretrained_requires_complete_backbone_and_safe_load(monkeypatch):
    source = csl_tinyvit_7m(num_classes=4, pretrained=False)
    source_backbone = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if key.startswith(("patch_embed.", "layers."))
    }
    source_backbone.pop("patch_embed.seq.0.c.weight")
    captured = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        return source_backbone

    monkeypatch.setattr(csl_tinyvit_pretrained, "load_hub_checkpoint", fake_load)
    target = csl_tinyvit_7m(num_classes=4, pretrained=False)

    with pytest.raises(RuntimeError, match="Incomplete TinyViT pretrained backbone load"):
        csl_tinyvit_pretrained.load_pretrained_tinyvit(
            target,
            "test://tinyvit",
            sha256="0" * 64,
        )

    assert captured["weights_only"] is True
    assert captured["sha256"] == "0" * 64


def test_csl_tinyvit_pretrained_requires_native_backbone_but_not_reid_adapters(monkeypatch):
    source = csl_tinyvit_7m(num_classes=4, pretrained=False)
    source_backbone = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if key.startswith(("patch_embed.", "layers."))
    }
    target = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        reid_adapter_stages=(1, 2),
    )
    adapter_keys = {
        key for key in target.state_dict() if ".reid_adapters." in key
    }
    assert adapter_keys
    monkeypatch.setattr(
        csl_tinyvit_pretrained,
        "load_hub_checkpoint",
        lambda *args, **kwargs: source_backbone,
    )

    csl_tinyvit_pretrained.load_pretrained_tinyvit(target, "test://tinyvit")

    required = csl_tinyvit_pretrained._required_pretrained_keys(target.state_dict())
    assert adapter_keys.isdisjoint(required)
    assert target.pretrained_backbone_tensor_coverage == 1.0
    assert target.pretrained_backbone_numel_coverage == 1.0


def test_reid_residual_adapter_is_identity_at_initialization():
    adapter = ReIDResidualAdapter(dim=8, reduction_ratio=4)
    x = torch.randn(2, 6, 8)

    y = adapter(x, (3, 2))

    torch.testing.assert_close(y, x)
    assert adapter.gamma.item() == 0.0


def test_post_fusion_local_mixer_is_identity_at_zero_gamma():
    mixer = PostFusionLocalMixer(channels=8, reduction=4, kernel_size=(5, 3), gamma_init=0.0)
    x = torch.randn(2, 8, 6, 4)

    y = mixer(x)

    torch.testing.assert_close(y, x)
    assert mixer.gamma.item() == 0.0


def test_csl_tinyvit_inserts_post_fusion_local_mixer():
    model = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        post_fusion_mixer="dwconv",
        post_fusion_mixer_reduction=4,
        post_fusion_mixer_kernel=(5, 3),
        post_fusion_mixer_gamma_init=1e-4,
    )

    assert isinstance(model.post_fusion_mixer_module, PostFusionLocalMixer)
    assert model.post_fusion_mixer == "dwconv"
    assert model.post_fusion_mixer_kernel == (5, 3)
    assert model.post_fusion_mixer_module.gamma.item() == pytest.approx(1e-4)


def test_csl_tinyvit_inserts_zero_gated_reid_adapters_in_requested_stages():
    model = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        reid_adapter_stages=(3,),
        reid_adapter_reduction=8,
    )

    assert len(model.layers[2].reid_adapters) == 0
    assert len(model.layers[3].reid_adapters) == len(model.layers[3].blocks)
    assert all(adapter.gamma.item() == 0.0 for adapter in model.layers[3].reid_adapters)


def test_signed_factorized_attention_bias_keeps_direction():
    attention = Attention(dim=8, key_dim=4, num_heads=2, resolution=(3, 2), bias_mode="signed_factorized")

    top_to_bottom = attention.attention_bias_h_idxs[0, 2].item()
    bottom_to_top = attention.attention_bias_h_idxs[2, 0].item()

    assert top_to_bottom != bottom_to_top
    assert attention.attention_bias_h.shape == (2, 5)
    assert attention.attention_bias_w.shape == (2, 3)


def test_tinyvit_block_masks_padded_tokens():
    block = TinyViTBlock(dim=8, input_resolution=(3, 3), num_heads=2, window_size=(2, 2), attention_mask=True)
    captured = {}

    def fake_attention(x, attn_mask=None):
        captured["attn_mask"] = attn_mask
        return torch.zeros_like(x)

    block.attn.forward = fake_attention
    block(torch.randn(1, 9, 8), (3, 3))

    mask = captured["attn_mask"]
    assert mask.shape == (4, 4, 4)
    assert not mask.all()


def test_tinyvit_block_shift_builds_attention_mask_without_padding():
    block = TinyViTBlock(dim=8, input_resolution=(4, 4), num_heads=2, window_size=(2, 2), shift_size=(1, 1))
    captured = {}

    def fake_attention(x, attn_mask=None):
        captured["attn_mask"] = attn_mask
        return torch.zeros_like(x)

    block.attn.forward = fake_attention
    block(torch.randn(1, 16, 8), (4, 4))

    mask = captured["attn_mask"]
    assert mask.shape == (4, 4, 4)
    assert not mask.all()


def test_gem_uses_safe_exponent_parameterization():
    gem = GeM((1, 1), p=3.0)

    assert torch.allclose(gem.effective_p(), torch.tensor([3.0]))
    gem.raw_p.data.fill_(20.0)
    assert gem.effective_p().item() == 8.0


def test_gem_loads_legacy_p_parameter():
    gem = GeM((1, 1), p=3.0)

    gem.load_state_dict({"p": torch.tensor([4.0])}, strict=True)

    assert torch.allclose(gem.effective_p(), torch.tensor([4.0]))


def test_registry_loads_legacy_gem_p_parameter(tmp_path):
    source = csl_tinyvit_7m(num_classes=4, pretrained=False, head_pool="gem")
    legacy_state = {}
    for key, value in source.state_dict().items():
        if key.endswith(".raw_p"):
            legacy_state[f"{key[:-6]}.p"] = torch.tensor([4.0])
        else:
            legacy_state[key] = value.clone()
    weights = tmp_path / "csl_tinyvit_7m_legacy_gem.pt"
    torch.save({"state_dict": legacy_state}, weights)

    loaded = csl_tinyvit_7m(num_classes=4, pretrained=False, head_pool="gem")
    ReIDModelRegistry.load_pretrained_weights(loaded, weights)

    assert torch.allclose(loaded.head.global_pool.effective_p(), torch.tensor([4.0]))
    assert torch.allclose(loaded.head.partial_pool.effective_p(), torch.tensor([4.0]))


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


def test_csl_tinyvit_last3_stage2_target_fuses_at_stage2_resolution():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last3_stage2_target",
        path_channels={1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {
        1: torch.randn(2, 4, 12, 4),
        2: torch.randn(2, 4, 6, 3),
    }

    output = module(final_feature, path_features)
    expected = F.interpolate(final_feature, size=(6, 3), mode="bilinear", align_corners=False)

    assert module.mode == "last3_stage2_target"
    assert module.fusion_type == "residual"
    assert module.stage_indices == (1, 2)
    assert module.target_stage_index == 2
    assert output.shape[-2:] == path_features[2].shape[-2:]
    torch.testing.assert_close(output, expected)


def test_csl_tinyvit_last3_stage1_concat_fuses_at_24x8_and_compresses_channels():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last3_stage1_concat",
        path_channels={1: 6, 2: 8},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {
        1: torch.randn(2, 6, 6, 4),
        2: torch.randn(2, 8, 3, 2),
    }

    output = module(final_feature, path_features)

    assert module.mode == "last3_stage1_concat"
    assert module.fusion_type == "concat_compress"
    assert module.stage_indices == (1, 2)
    assert module.target_stage_index == 1
    assert module.concat_projection[0].in_channels == 12
    assert module.concat_projection[0].out_channels == 4
    assert output.shape == (2, 4, 6, 4)


def test_csl_tinyvit_last3_stage1_concat_model_outputs_24x8_feature_map():
    model = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        feature_fusion="last3_stage1_concat",
        neck_dim=32,
        feat_dim=16,
    )
    model.eval()

    with torch.no_grad():
        feature_map = model.forward_features(torch.randn(1, 3, 384, 128))

    assert model._fusion_stage_indices == (1, 2)
    assert feature_map.shape == (1, 32, 24, 8)


def test_csl_tinyvit_last3_fpn_stage1_add_uses_recursive_top_down_addition():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last3_fpn_stage1_add",
        path_channels={1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {1: torch.randn(2, 4, 6, 4), 2: torch.randn(2, 4, 3, 2)}
    output = module(final_feature, path_features)
    assert module.mode == "last3_fpn_stage1_add"
    assert module.fusion_type == "fpn_topdown"
    assert module.stage_indices == (2, 1)
    assert module.target_stage_index == 1
    assert output.shape == (2, 4, 6, 4)


def test_csl_tinyvit_last3_fpn_stage1_split_routes_p2_global_and_p1_parts():
    module = CSLTinyViTFeatureFusion.from_mode("last3_fpn_stage1_split", path_channels={1: 4, 2: 4}, out_channels=4)
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {1: torch.randn(2, 4, 6, 4), 2: torch.randn(2, 4, 3, 2)}
    global_feature, local_feature = module(final_feature, path_features)
    assert global_feature.shape == (2, 4, 3, 2)
    assert local_feature.shape == (2, 4, 6, 4)


def test_csl_tinyvit_pool_nearest_uses_pooling_down_and_nearest_up():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last3_fpn_stage1_split",
        path_channels={1: 4, 2: 4},
        out_channels=4,
        resize_mode="pool_nearest",
    )
    high_resolution = torch.arange(48, dtype=torch.float32).reshape(1, 2, 6, 4)
    low_resolution = high_resolution[:, :, :3, :2]

    downsampled = module._resize_feature(high_resolution, (3, 2))
    upsampled = module._resize_feature(low_resolution, (6, 4))

    torch.testing.assert_close(downsampled, F.adaptive_avg_pool2d(high_resolution, (3, 2)))
    torch.testing.assert_close(upsampled, F.interpolate(low_resolution, size=(6, 4), mode="nearest"))


def test_csl_tinyvit_pool_bilinear_uses_pooling_down_and_bilinear_up():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last3_fpn_stage1_split",
        path_channels={1: 4, 2: 4},
        out_channels=4,
        resize_mode="pool_bilinear",
    )
    high_resolution = torch.arange(48, dtype=torch.float32).reshape(1, 2, 6, 4)
    low_resolution = high_resolution[:, :, :3, :2]

    downsampled = module._resize_feature(high_resolution, (1, 1))
    upsampled = module._resize_feature(low_resolution, (6, 4))
    mixed = module._resize_feature(high_resolution, (3, 8))

    torch.testing.assert_close(downsampled, high_resolution.mean(dim=(-2, -1), keepdim=True))
    torch.testing.assert_close(
        upsampled,
        F.interpolate(low_resolution, size=(6, 4), mode="bilinear", align_corners=False),
    )
    expected_mixed = F.interpolate(
        F.adaptive_avg_pool2d(high_resolution, (3, 4)),
        size=(3, 8),
        mode="bilinear",
        align_corners=False,
    )
    torch.testing.assert_close(mixed, expected_mixed)


def test_csl_tinyvit_depthwise_separable_neck_and_fpn_preserve_branch_shapes():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last3_fpn_stage1_split",
        path_channels={1: 4, 2: 4},
        out_channels=4,
        spatial_conv_mode="depthwise_separable",
    )
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {1: torch.randn(2, 4, 6, 4), 2: torch.randn(2, 4, 3, 2)}

    global_feature, local_feature = module(final_feature, path_features)

    assert isinstance(module.fpn_output[0], nn.Sequential)
    assert module.fpn_output[0][0].groups == 4
    assert module.fpn_output[0][1].kernel_size == (1, 1)
    assert global_feature.shape == (2, 4, 3, 2)
    assert local_feature.shape == (2, 4, 6, 4)


def test_trainer_builds_efficient_csl_tinyvit_fpn(tmp_path):
    trainer = _trainer(
        tmp_path,
        pretrained=False,
        feature_fusion="last3_fpn_stage1_split",
        pyramid_resize_mode="pool_nearest",
        spatial_conv_mode="depthwise_separable",
        neck_dim=32,
        feat_dim=16,
    )

    model = trainer._build_model(num_classes=4)

    assert model.pyramid_resize_mode == "pool_nearest"
    assert model.spatial_conv_mode == "depthwise_separable"
    assert isinstance(model.neck[2], nn.Sequential)
    assert model.neck[2][0].groups == 32
    assert model.feature_fusion_module.resize_mode == "pool_nearest"
    assert model.feature_fusion_module.spatial_conv_mode == "depthwise_separable"
    metadata = trainer._checkpoint_metadata(model)
    assert metadata["pyramid_resize_mode"] == "pool_nearest"
    assert metadata["spatial_conv_mode"] == "depthwise_separable"
    assert metadata["model"]["pyramid_resize_mode"] == "pool_nearest"
    assert metadata["model"]["spatial_conv_mode"] == "depthwise_separable"


def test_trainer_records_pretrained_attention_bias_interpolation(tmp_path):
    trainer = _trainer(
        tmp_path,
        pretrained=False,
        attention_window_layout="rect",
        interpolate_pretrained_attention_bias=True,
        neck_dim=32,
        feat_dim=16,
    )

    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)

    assert model.interpolate_pretrained_attention_bias is True
    assert metadata["interpolate_pretrained_attention_bias"] is True
    assert metadata["model"]["transformer"]["attention"]["interpolate_pretrained_bias"] is True


def test_csl_tinyvit_last3_panet_stage1_split_returns_low_res_global_and_high_res_parts():
    module = CSLTinyViTFeatureFusion.from_mode("last3_panet_stage1_split", path_channels={1: 4, 2: 4}, out_channels=4)
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {1: torch.randn(2, 4, 6, 4), 2: torch.randn(2, 4, 3, 2)}
    global_map, local_map = module(final_feature, path_features)
    assert module.fusion_type == "panet"
    assert module.panet_downsample[0].stride == (2, 2)
    assert global_map.shape == (2, 4, 3, 2)
    assert local_map.shape == (2, 4, 6, 4)


def test_csl_tinyvit_last3_panet_stage1_shared_returns_final_panet_map():
    module = CSLTinyViTFeatureFusion.from_mode("last3_panet_stage1_shared", path_channels={1: 4, 2: 4}, out_channels=4)
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {1: torch.randn(2, 4, 6, 4), 2: torch.randn(2, 4, 3, 2)}
    shared_map = module(final_feature, path_features)
    assert isinstance(shared_map, torch.Tensor)
    assert shared_map.shape == (2, 4, 3, 2)


def test_csl_tinyvit_panet_constructs_coarser_semantic_level_from_equal_resolution_stages():
    module = CSLTinyViTFeatureFusion.from_mode("last3_panet_stage1_split", path_channels={1: 4, 2: 4}, out_channels=4)
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {1: torch.randn(2, 4, 3, 2), 2: torch.randn(2, 4, 3, 2)}
    global_map, local_map = module(final_feature, path_features)
    assert global_map.shape == (2, 4, 1, 1)
    assert local_map.shape == (2, 4, 3, 2)


def test_csl_tinyvit_last3_bifpn_stage1_split_learns_normalized_bidirectional_fusion():
    module = CSLTinyViTFeatureFusion.from_mode("last3_bifpn_stage1_split", path_channels={1: 4, 2: 4}, out_channels=4)
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {1: torch.randn(2, 4, 6, 4), 2: torch.randn(2, 4, 3, 2)}
    global_map, local_map = module(final_feature, path_features)
    assert module.fusion_type == "bifpn"
    assert module.bifpn_blocks["top_high"][0].groups == 4
    assert global_map.shape == (2, 4, 3, 2)
    assert local_map.shape == (2, 4, 6, 4)


def test_csl_tinyvit_best_global_layer0_fpn_parts_uses_high_resolution_local_map():
    module = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_fpn_layer0", path_channels={0: 5, 1: 6, 2: 8}, out_channels=4
    )
    final_feature = torch.randn(2, 4, 3, 1)
    path_features = {
        0: torch.randn(2, 5, 12, 4), 1: torch.randn(2, 6, 6, 2), 2: torch.randn(2, 8, 3, 1)
    }
    global_feature, local_feature = module(final_feature, path_features)
    assert set(module.residual_scales) == {"1", "2"}
    assert global_feature.shape == (2, 4, 3, 1)
    assert local_feature.shape == (2, 4, 12, 4)


def test_csl_tinyvit_panet_scale_aware_gates_semantic_and_detailed_stripes():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last3_panet_stage1_scale_aware", path_channels={1: 4, 2: 4}, out_channels=4
    )
    global_map, local_map = module(
        torch.randn(2, 4, 3, 2),
        {1: torch.randn(2, 4, 6, 4), 2: torch.randn(2, 4, 3, 2)},
    )
    torch.testing.assert_close(torch.sigmoid(module.panet_scale_gate.bias), torch.full((4,), 0.7))
    assert global_map.shape == (2, 4, 3, 2)
    assert local_map.shape == (2, 4, 6, 4)


def test_csl_tinyvit_branch_aware_bifpn_uses_bidirectional_and_branch_nodes():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last3_bifpn_stage1_branch_aware", path_channels={1: 4, 2: 4}, out_channels=4
    )
    global_map, local_map = module(
        torch.randn(2, 4, 3, 2),
        {1: torch.randn(2, 4, 6, 4), 2: torch.randn(2, 4, 3, 2)},
    )
    assert tuple(module.bifpn_branch_weights) == ("global", "local")
    assert module.bifpn_branch_weights["global"] is not module.bifpn_branch_weights["local"]
    assert global_map.shape == (2, 4, 3, 2)
    assert local_map.shape == (2, 4, 6, 4)

    (global_map.square().mean() + local_map.square().mean()).backward()

    assert all(parameter.grad is not None for parameter in module.bifpn_weights.values())
    assert all(parameter.grad is not None for parameter in module.bifpn_blocks.parameters())
    assert all(parameter.grad is not None for parameter in module.bifpn_branch_weights.values())
    assert all(parameter.grad is not None for parameter in module.bifpn_branch_blocks.parameters())


def test_hierarchical_control_matches_stage2_aggregation_and_head_contract():
    module = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage2_hierarchical_control",
        path_channels={1: 6, 2: 8},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 6, 2)
    path_features = {
        1: torch.randn(2, 6, 6, 2),
        2: torch.randn(2, 8, 6, 2),
    }

    global_map, local_map, fine_map = module(final_feature, path_features)

    assert set(module.residual_scales) == {"1", "2"}
    assert global_map.shape == local_map.shape == (2, 4, 6, 2)
    assert fine_map.shape == (2, 4, 12, 4)
    torch.testing.assert_close(fine_map, module._resize_feature(local_map, (12, 4)))


def test_stage2_semantic_residual_preserves_baseline_at_zero_gate_and_refines_only_parts():
    torch.manual_seed(0)
    module = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage2_semantic_residual",
        path_channels={1: 6, 2: 8},
        out_channels=4,
    )
    control = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage2",
        path_channels={1: 6, 2: 8},
        out_channels=4,
    )
    control.projections.load_state_dict(module.projections.state_dict())
    control.residual_scales.load_state_dict(module.residual_scales.state_dict())
    final_feature = torch.randn(2, 4, 6, 2)
    path_features = {
        1: torch.randn(2, 6, 6, 2),
        2: torch.randn(2, 8, 6, 2),
    }

    global_map, local_map = module(final_feature, path_features)
    control_global, control_local = control(final_feature, path_features)

    assert module.local_semantic_residual is True
    assert module.local_semantic_adapter[0][0].groups == 4
    torch.testing.assert_close(module.local_semantic_gate, torch.zeros(4))
    torch.testing.assert_close(global_map, control_global)
    torch.testing.assert_close(local_map, control_local)

    with torch.no_grad():
        module.local_semantic_gate.fill_(1.0)
    refined_global, refined_local = module(final_feature, path_features)

    torch.testing.assert_close(refined_global, control_global)
    assert not torch.allclose(refined_local, control_local)


def test_hierarchical_fpn_routes_global_coarse_and_fine_maps_without_dead_scale():
    module = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_hierarchical_fpn",
        path_channels={0: 5, 1: 6, 2: 8},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 6, 2)
    path_features = {
        0: torch.randn(2, 5, 12, 4),
        1: torch.randn(2, 6, 6, 2),
        2: torch.randn(2, 8, 6, 2),
    }

    global_map, coarse_map, fine_map = module(final_feature, path_features)

    assert set(module.residual_scales) == {"1", "2"}
    assert global_map.shape == (2, 4, 3, 1)
    assert coarse_map.shape == (2, 4, 6, 2)
    assert fine_map.shape == (2, 4, 12, 4)


def test_stage0_semantic_fine_preserves_baseline_branches_and_zero_gates_fine_detail():
    torch.manual_seed(0)
    module = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage0_semantic_fine",
        path_channels={0: 5, 1: 6, 2: 8},
        out_channels=8,
    )
    control = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage2_hierarchical_control",
        path_channels={1: 6, 2: 8},
        out_channels=8,
    )
    control.projections.load_state_dict({key: value for key, value in module.projections.state_dict().items()})
    control.residual_scales.load_state_dict(module.residual_scales.state_dict())
    final_feature = torch.randn(2, 8, 6, 2)
    path_features = {
        0: torch.randn(2, 5, 12, 4),
        1: torch.randn(2, 6, 6, 2),
        2: torch.randn(2, 8, 6, 2),
    }

    global_map, local_map, fine_map = module(final_feature, path_features)
    control_global, control_local, control_fine = control(
        final_feature,
        {index: path_features[index] for index in (1, 2)},
    )

    assert set(module.projections) == {"1", "2"}
    assert set(module.residual_scales) == {"1", "2"}
    torch.testing.assert_close(module.stage0_fine_gate, torch.zeros(8))
    torch.testing.assert_close(global_map, control_global)
    torch.testing.assert_close(local_map, control_local)
    torch.testing.assert_close(fine_map, control_fine)
    assert global_map.shape == local_map.shape == (2, 8, 6, 2)
    assert fine_map.shape == (2, 8, 12, 4)


def test_stage0_semantic_fine_optimized_execution_matches_reference():
    torch.manual_seed(0)
    optimized = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage0_semantic_fine",
        path_channels={0: 5, 1: 6, 2: 8},
        out_channels=8,
    )
    reference = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage0_semantic_fine_reference",
        path_channels={0: 5, 1: 6, 2: 8},
        out_channels=8,
    )
    with torch.no_grad():
        optimized.stage0_fine_gate.fill_(0.2)
    reference.load_state_dict(optimized.state_dict())
    final_feature = torch.randn(2, 8, 6, 2)
    paths = {
        0: torch.randn(2, 5, 12, 4),
        1: torch.randn(2, 6, 6, 2),
        2: torch.randn(2, 8, 6, 2),
    }
    projection_calls = {"optimized": 0, "reference": 0}
    semantic_input_shapes = {}

    def count_projection(_module, _args, *, execution):
        projection_calls[execution] += 1

    def record_semantic_shape(_module, args, *, execution):
        semantic_input_shapes[execution] = args[0].shape[-2:]

    handles = [
        optimized.projections["2"].register_forward_pre_hook(
            lambda module, args: count_projection(module, args, execution="optimized")
        ),
        reference.projections["2"].register_forward_pre_hook(
            lambda module, args: count_projection(module, args, execution="reference")
        ),
        optimized.stage0_semantic_projection[0].register_forward_pre_hook(
            lambda module, args: record_semantic_shape(module, args, execution="optimized")
        ),
        reference.stage0_semantic_projection[0].register_forward_pre_hook(
            lambda module, args: record_semantic_shape(module, args, execution="reference")
        ),
    ]
    try:
        optimized_maps = optimized(final_feature, paths)
        reference_maps = reference(final_feature, paths)
    finally:
        for handle in handles:
            handle.remove()

    assert projection_calls == {"optimized": 1, "reference": 2}
    assert semantic_input_shapes == {"optimized": (6, 2), "reference": (12, 4)}
    for optimized_map, reference_map in zip(optimized_maps, reference_maps, strict=True):
        # Projection and bilinear interpolation commute mathematically, but the
        # optimized order accumulates floating-point products differently.
        torch.testing.assert_close(optimized_map, reference_map, rtol=2e-4, atol=1e-5)


def test_csl_tinyvit_stage0_semantic_fine_builds_compact_hierarchical_head():
    model = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        head_parts=(1, 2, 4),
        neck_dim=32,
        feat_dim=16,
        scale_balanced_branches=True,
    )

    assert model.feature_fusion_module.stage_indices == (1, 2, 0)
    assert model.feature_fusion_module.stage0_fine_projection[0].out_channels == 8
    assert model.head.hierarchical_scales is True
    assert model.head.scale_balanced_branches is True


def test_csl_tinyvit_fusion_arms_preserve_shared_initialization():
    modes = (
        "global_final_parts_stage0_semantic_fine",
        "global_final_parts_stage0_panet_lite",
        "global_final_parts_stage0_bifpn_lite",
    )
    shared_states = {}

    for mode in modes:
        torch.manual_seed(0)
        model = csl_tinyvit_7m(
            num_classes=4,
            pretrained=False,
            feature_fusion=mode,
            head_parts=(1, 2, 4),
            neck_dim=32,
            feat_dim=16,
            scale_balanced_branches=True,
        )
        shared_states[mode] = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("feature_fusion_module.")
        }

    anchor_state = shared_states[modes[0]]
    for mode in modes[1:]:
        candidate_state = shared_states[mode]
        assert candidate_state.keys() == anchor_state.keys()
        for key, anchor_value in anchor_state.items():
            torch.testing.assert_close(candidate_state[key], anchor_value)


def test_csl_tinyvit_multiscale_geometry_anatomy_runs_full_rgb_model():
    model = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        head_parts=(1, 2, 4),
        neck_dim=32,
        feat_dim=16,
        scale_balanced_branches=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
    ).train()

    _, features = model(torch.randn(2, 3, 384, 128))

    assert features["_anatomical_feature_map"].shape == (2, 2, 24, 8)
    assert features["_anatomical_fine_feature_map"].shape == (2, 2, 48, 16)
    assert features["_anatomical_student_tokens"].shape == (2, 6, 16)
    assert features["_anatomical_fine_student_tokens"].shape == (2, 6, 16)
    assert "_anatomical_teacher_feature_map" not in features
    assert "_anatomical_online_teacher_feature_map" not in features

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, 4),
        torch.tensor([-0.5, 0.5]),
        indexing="ij",
    )
    canonical_grid = torch.stack((grid_x, grid_y), dim=-1)
    targets = {
        "masks": torch.ones(2, 6, 96, 32),
        "canonical_grid": canonical_grid[None, None].repeat(
            2,
            6,
            1,
            1,
            1,
        ),
        "canonical_grid_valid": torch.ones(
            2,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "visibility": torch.ones(2, 6),
        "reliability": torch.ones(2, 6),
        "valid": torch.ones(2, dtype=torch.bool),
    }
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_multiscale = True
    trainer.anatomical_distill_weight = 0.20
    trainer.anatomical_attention_weight = 0.10
    trainer.anatomical_visibility_weight = 0.05
    trainer.anatomical_contrastive_weight = 0.10
    trainer.anatomical_descriptor_distill_weight = 0.0
    trainer.anatomical_pose_teacher_weight = 0.10
    trainer.anatomical_local_scale_weight = 0.60
    trainer.anatomical_fine_scale_weight = 0.40
    trainer.anatomical_cross_scale_weight = 0.05
    trainer.anatomical_pose_only_reliability = 0.35
    trainer.anatomical_temperature = 0.07

    loss, components = trainer._anatomical_auxiliary_loss(
        features,
        targets,
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        return_components=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["local_scale"].item() > 0
    assert components["fine_scale"].item() > 0
    pool = model.head.anatomical_auxiliary_pool
    assert pool.feature_projection.weight.grad is not None
    assert pool.feature_projection.weight.grad.abs().sum().item() > 0
    assert pool.fine_feature_projection.weight.grad is not None
    assert pool.fine_feature_projection.weight.grad.abs().sum().item() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in model.patch_embed.parameters()
    )


def test_csl_tinyvit_global_only_stage3_downsample_preserves_local_resolution():
    model = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        stage3_downsample=True,
    ).eval()

    with torch.inference_mode():
        global_map, local_map, fine_map = model.forward_features(torch.randn(1, 3, 384, 128))

    assert global_map.shape == (1, 512, 12, 4)
    assert local_map.shape == (1, 512, 24, 8)
    assert fine_map.shape == (1, 512, 48, 16)
    assert model.layers[2].downsample.stride == 2
    assert {block.window_size for block in model.layers[3].blocks} == {(12, 4)}


def test_norm_preserving_width_merge_uses_activation_norm_weights():
    merge = NormPreservingWidthMerge(eps=1e-12)
    tokens = torch.tensor(
        [[[[3.0, 0.0], [0.0, 4.0], [1.0, 0.0], [2.0, 0.0]]]],
    ).reshape(1, 4, 2)

    merged, size = merge(tokens, (1, 4))
    merged = merged.view(1, 1, 2, 2)

    assert size == (1, 2)
    torch.testing.assert_close(torch.linalg.vector_norm(merged, dim=-1), torch.tensor([[[4.0, 2.0]]]))
    expected_direction = torch.tensor([3.0 * 3.0 / 7.0, 4.0 * 4.0 / 7.0])
    torch.testing.assert_close(
        F.normalize(merged[0, 0, 0], dim=0),
        F.normalize(expected_direction, dim=0),
    )
    zero, _ = merge(torch.zeros(1, 4, 2), (1, 4))
    assert torch.isfinite(zero).all()
    assert torch.count_nonzero(zero) == 0


def test_csl_tinyvit_late_width_merge_preserves_fine_and_coarse_taps():
    model = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        stage2_width_merge_after=2,
        neck_dim=32,
        feat_dim=16,
    ).eval()

    with torch.inference_mode():
        global_map, local_map, fine_map = model.forward_features(torch.randn(1, 3, 384, 128))

    assert global_map.shape == (1, 32, 24, 4)
    assert local_map.shape == (1, 32, 24, 8)
    assert fine_map.shape == (1, 32, 48, 16)
    assert model.layers[2].width_merge_after_blocks == 2
    assert [block.window_size for block in model.layers[2].blocks] == [
        (12, 8),
        (12, 8),
        (12, 4),
        (12, 4),
        (12, 4),
        (12, 4),
    ]
    assert {block.window_size for block in model.layers[3].blocks} == {(12, 4)}
    assert model.feature_fusion_module.projections["2"][0].in_channels == model.layers[2].dim


def test_csl_tinyvit_native_branch_widths_preserve_seven_branch_descriptor():
    model = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        stage3_downsample=True,
        native_branch_widths=True,
        neck_dim=32,
        feat_dim=16,
    ).eval()

    with torch.inference_mode():
        maps = model.forward_features(torch.randn(1, 3, 384, 128))
        descriptor = model.forward_head(maps)

    assert [feature.shape for feature in maps] == [
        (1, 32, 12, 4),
        (1, 16, 24, 8),
        (1, 8, 48, 16),
    ]
    assert descriptor.shape == (1, 48)  # 16 + 2x8 + 4x4
    assert model.head.branch_input_channels == (32, 16, 8)


def test_multibranch_norm_concat_fast_path_matches_full_descriptor_formula():
    torch.manual_seed(0)
    head = MultiBranchHead(
        8,
        feat_dim=8,
        num_classes=4,
        inference_feature="norm_concat_bn",
        head_pool="avg",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        hierarchical_scales=True,
    ).eval()
    sources = (torch.randn(2, 8, 6, 2), torch.randn(2, 8, 6, 2), torch.randn(2, 8, 12, 4))

    with torch.inference_mode():
        actual = head(sources)
        pooled = {
            1: head.global_pool(sources[0]),
            2: head.partial_pool(sources[1]),
            4: head.part_pool_4(sources[2]),
        }
        branches = []
        for key, granularity, stripe_index in head.branch_specs:
            branch = pooled[granularity]
            if granularity > 1:
                branch = branch[:, :, stripe_index : stripe_index + 1, :]
            bn_feature = getattr(head, head._bn_attr(key))(branch)[0]
            branches.append(F.normalize(bn_feature, p=2, dim=1) * head._descriptor_scale(granularity))
        expected = F.normalize(torch.cat(branches, dim=1), p=2, dim=1)

    torch.testing.assert_close(actual, expected)


def test_compact_deployment_head_trains_with_teacher_and_skips_it_at_inference():
    head = MultiBranchHead(
        8,
        feat_dim=8,
        num_classes=4,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_pool="avg",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        hierarchical_scales=True,
        compact_deployment_head=True,
    )
    sources = (torch.randn(4, 8, 6, 2), torch.randn(4, 8, 6, 2), torch.randn(4, 8, 12, 4))

    head.train()
    logits, features = head(sources)
    assert len(logits) == 7
    assert features["_compact_logits"].shape == (4, 4)
    assert features["raw_concat"].shape == (4, 24)
    assert features["_compact_student"].shape == (4, 8)
    assert features["_compact_decoded"].shape == (4, 24)

    teacher_calls = []
    hooks = [
        getattr(head, head._bn_attr(key)).register_forward_hook(
            lambda *_args, key=key: teacher_calls.append(key)
        )
        for key, _, _ in head.branch_specs
    ]
    try:
        head.eval()
        with torch.inference_mode():
            descriptor = head(sources)
    finally:
        for hook in hooks:
            hook.remove()

    assert descriptor.shape == (4, 8)
    torch.testing.assert_close(torch.linalg.vector_norm(descriptor, dim=1), torch.ones(4))
    assert teacher_calls == []


def test_compact_student_losses_distill_direction_and_pairwise_geometry(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        compact_deployment_head=True,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
    )
    teacher = torch.randn(4, 24, requires_grad=True)
    features = {
        "_compact_logits": torch.randn(4, 2, requires_grad=True),
        "_compact_student": torch.randn(4, 8, requires_grad=True),
        "_compact_student_bn": torch.randn(4, 8, requires_grad=True),
        "_compact_teacher": teacher,
        "_compact_decoded": torch.randn(4, 24, requires_grad=True),
    }
    pids = torch.tensor([0, 0, 1, 1])
    compact_id = trainer._compact_student_id_loss(nn.CrossEntropyLoss(), features, pids)
    metric, cosine, pairwise = trainer._compact_student_losses(
        TripletLoss(margin=0.3, soft_margin=True),
        features,
        pids,
    )

    total = compact_id + metric + cosine + pairwise
    assert torch.isfinite(total)
    total.backward()
    assert features["_compact_student"].grad is not None
    assert features["_compact_student_bn"].grad is not None
    assert features["_compact_decoded"].grad is not None
    assert features["_compact_logits"].grad is not None
    assert teacher.grad is None


def test_csl_tinyvit_stage3_capacity_and_bottleneck_neck_options():
    model = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        stage3_downsample=True,
        native_branch_widths=True,
        stage3_mlp_ratio=3.0,
        stage3_depth=1,
        spatial_conv_mode="bottleneck_depthwise",
    )

    assert len(model.layers[3].blocks) == 1
    assert model.layers[3].blocks[0].mlp.fc1.out_features == 448 * 3
    spatial_neck = model.neck[2]
    sample = torch.randn(2, 512, 5, 3)
    torch.testing.assert_close(spatial_neck(sample), sample)


def test_csl_tinyvit_fine_map_width_preserves_seven_branch_descriptor():
    model = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        neck_dim=32,
        feat_dim=16,
        fine_map_dim=16,
        inference_feature="norm_concat_bn",
    ).eval()

    with torch.inference_mode():
        maps = model.forward_features(torch.randn(1, 3, 384, 128))
        descriptor = model.forward_head(maps)

    assert [feature.shape for feature in maps] == [
        (1, 32, 24, 8),
        (1, 32, 24, 8),
        (1, 16, 48, 16),
    ]
    assert model.head.branch_input_channels == (32, 32, 16)
    assert descriptor.shape == (1, 48)
    torch.testing.assert_close(descriptor.norm(dim=1), torch.ones(1))


def test_csl_tinyvit_reallocates_one_stage3_block_to_stage2():
    model = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        stage2_mlp_ratio=3.0,
        stage3_mlp_ratio=3.0,
        stage2_depth=7,
        stage3_depth=1,
    )

    assert [len(model.layers[index].blocks) for index in (1, 2, 3)] == [2, 7, 1]
    assert model.layers[2].blocks[-1].mlp.fc1.out_features == 256 * 3
    assert model.layers[3].blocks[0].mlp.fc1.out_features == 448 * 3
    assert sum(len(layer.blocks) for layer in model.layers) == 12


def test_trainer_records_scale_balancing_for_checkpoint_retrieval(tmp_path):
    trainer = _trainer(
        tmp_path,
        pretrained=False,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        head_parts=(1, 2, 4),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        neck_dim=32,
        feat_dim=16,
    )

    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)

    assert model.head.metric_feature == "raw_concat"
    assert model.head.inference_feature == "norm_concat_bn"
    assert model.head.scale_balanced_branches is True
    assert metadata["scale_balanced_branches"] is True


def test_checkpoint_reconstructs_width_merge_and_compact_deployment_head(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        pretrained=False,
        img_size=(384, 128),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        attention_mask=True,
        head_parts=(1, 2, 4),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        stage2_width_merge_after=2,
        stage2_mlp_ratio=3.0,
        stage2_depth=7,
        fine_map_dim=0,
        compact_deployment_head=True,
        neck_dim=32,
        feat_dim=16,
    )
    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)
    weights = tmp_path / "compact_widthmerge.pt"
    torch.save({**metadata, "state_dict": model.state_dict()}, weights)

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)
    assert metadata["stage2_width_merge_after"] == 2
    assert metadata["stage2_mlp_ratio"] == 3.0
    assert metadata["stage2_depth"] == 7
    assert metadata["fine_map_dim"] == 0
    assert metadata["compact_deployment_head"] is True
    assert metadata["model"]["transformer"]["speed"]["stage2_width_merge_after"] == 2
    assert metadata["model"]["transformer"]["deployment"] == {
        "compact_head": True,
        "descriptor_dim": 16,
    }
    assert kwargs["stage2_width_merge_after"] == 2
    assert kwargs["stage2_mlp_ratio"] == 3.0
    assert kwargs["stage2_depth"] == 7
    assert kwargs["fine_map_dim"] == 0
    assert kwargs["compact_deployment_head"] is True


def test_checkpoint_reconstructs_deployed_anatomical_descriptor(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        pretrained=False,
        img_size=(384, 128),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        attention_mask=True,
        head_parts=(1, 2, 4),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=True,
        anatomical_metadata_dir=str(tmp_path),
        anatomical_token_dim=16,
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_multiscale=True,
        anatomical_deployment=True,
        anatomical_deployment_dim=8,
        anatomical_deployment_alpha=0.25,
        anatomical_descriptor_distill_weight=0.0,
        anatomical_branch_distill_weight=0.0,
        neck_dim=32,
        feat_dim=16,
    )
    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)
    weights = tmp_path / "deployed_anatomy.pt"
    torch.save({**metadata, "state_dict": model.state_dict()}, weights)

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)
    assert kwargs["anatomical_auxiliary"] is True
    assert kwargs["anatomical_token_dim"] == 16
    assert kwargs["anatomical_target_type"] == "learned_pose_concat_ema"
    assert kwargs["anatomical_multiscale"] is True
    assert kwargs["anatomical_deployment"] is True
    assert kwargs["anatomical_deployment_dim"] == 8
    assert kwargs["anatomical_deployment_alpha"] == 0.25

    reconstructed = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        **kwargs,
    )
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    reconstructed.eval()
    descriptor = reconstructed(torch.randn(2, 3, 384, 128))
    assert descriptor.shape == (2, 96)
    torch.testing.assert_close(
        descriptor.norm(dim=1),
        torch.ones(2),
    )


def test_checkpoint_reconstructs_privileged_mask_pose_attention(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        pretrained=False,
        img_size=(384, 128),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        attention_mask=True,
        head_parts=(1, 2, 4),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=True,
        anatomical_metadata_dir=str(tmp_path),
        anatomical_person_mask_dir=str(tmp_path),
        anatomical_token_dim=16,
        anatomical_target_type="privileged_mask_pose_attention",
        anatomical_multiscale=True,
        anatomical_deployment=False,
        anatomical_descriptor_distill_weight=0.0,
        anatomical_branch_distill_weight=0.0,
        neck_dim=32,
        feat_dim=16,
    )
    model = trainer._build_model(num_classes=4)
    model.head.anatomical_attention_adapter.foreground_gate_logit.data.fill_(
        0.4
    )
    model.head.anatomical_attention_adapter.part_gate_logit.data.fill_(-0.2)
    model.head.anatomical_fine_attention_adapter.foreground_gate_logit.data.fill_(
        -0.25
    )
    model.head.anatomical_fine_attention_adapter.part_gate_logit.data.fill_(
        0.3
    )
    metadata = trainer._checkpoint_metadata(model)
    weights = tmp_path / "privileged_mask_pose_attention.pt"
    torch.save({**metadata, "state_dict": model.state_dict()}, weights)

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)
    assert kwargs["anatomical_auxiliary"] is True
    assert kwargs["anatomical_target_type"] == (
        "privileged_mask_pose_attention"
    )
    assert kwargs["anatomical_multiscale"] is True
    assert kwargs["anatomical_deployment"] is False

    reconstructed = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        **kwargs,
    )
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    reconstructed.eval()
    images = torch.randn(2, 3, 384, 128)
    with torch.no_grad():
        expected = model(images)
        actual = reconstructed(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 48)
    torch.testing.assert_close(
        actual.norm(dim=1),
        torch.ones(2),
    )


def test_checkpoint_reconstructs_pose_semantic_teacher(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        pretrained=False,
        img_size=(384, 128),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        attention_mask=True,
        head_parts=(1, 2, 4),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=True,
        anatomical_metadata_dir=str(tmp_path),
        anatomical_person_mask_dir=str(tmp_path),
        anatomical_token_dim=16,
        anatomical_target_type="learned_pose_semantic_fused_ema",
        anatomical_multiscale=True,
        anatomical_foreground_weight=0.03,
        anatomical_semantic_part_weight=0.05,
        anatomical_deployment=False,
        anatomical_descriptor_distill_weight=0.0,
        anatomical_branch_distill_weight=0.025,
        neck_dim=32,
        feat_dim=16,
    )
    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)
    weights = tmp_path / "pose_semantic_teacher.pt"
    torch.save({**metadata, "state_dict": model.state_dict()}, weights)

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)
    assert kwargs["anatomical_target_type"] == (
        "learned_pose_semantic_fused_ema"
    )
    assert kwargs["anatomical_multiscale"] is True
    reconstructed = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        **kwargs,
    )
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    reconstructed.eval()
    images = torch.randn(2, 3, 384, 128)
    with torch.no_grad():
        expected = model(images)
        actual = reconstructed(images)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 48)


def test_checkpoint_reconstructs_multiscale_channel_pose_teacher(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        pretrained=False,
        img_size=(384, 128),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        attention_mask=True,
        head_parts=(1, 2, 4),
        head_type="multiscale_channel2",
        multiscale_channel_alpha=0.5,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=True,
        anatomical_metadata_dir=str(tmp_path),
        anatomical_token_dim=16,
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_multiscale=True,
        anatomical_deployment=False,
        anatomical_descriptor_distill_weight=0.0,
        anatomical_branch_distill_weight=0.0,
        neck_dim=32,
        feat_dim=16,
    )
    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)
    weights = tmp_path / "multiscale_channel_pose_teacher.pt"
    torch.save({**metadata, "state_dict": model.state_dict()}, weights)

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)
    assert kwargs["head_type"] == "multiscale_channel2"
    assert kwargs["multiscale_channel_alpha"] == 0.5
    assert kwargs["anatomical_target_type"] == "learned_pose_concat_ema"
    reconstructed = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        **kwargs,
    )
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    reconstructed.eval()
    images = torch.randn(2, 3, 384, 128)
    with torch.no_grad():
        expected = model(images)
        actual = reconstructed(images)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 48 + 6 * 128)


def test_checkpoint_reconstructs_hierarchical_branch_attention(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        pretrained=False,
        img_size=(384, 128),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        attention_window_layout="rect",
        attention_mask=True,
        head_parts=(1, 2, 4),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        hierarchical_branch_attention=True,
        branch_attention_token_dim=12,
        branch_attention_num_heads=3,
        branch_attention_num_layers=1,
        branch_attention_mlp_ratio=2.0,
        branch_attention_dropout=0.0,
        neck_dim=32,
        feat_dim=32,
    )
    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)
    weights = tmp_path / "hierarchical_attention.pt"
    torch.save({**metadata, "state_dict": model.state_dict()}, weights)

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)

    assert metadata["model"]["transformer"]["hierarchical_attention"] == {
        "enabled": True,
        "token_dim": 12,
        "num_heads": 3,
        "num_layers": 1,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "mask": "1_to_2_to_4_tree",
        "output_init": "zero",
    }
    assert kwargs["hierarchical_branch_attention"] is True
    assert kwargs["branch_attention_token_dim"] == 12
    assert kwargs["branch_attention_num_heads"] == 3
    assert kwargs["branch_attention_num_layers"] == 1
    assert kwargs["branch_attention_mlp_ratio"] == 2.0
    assert kwargs["branch_attention_dropout"] == 0.0

    reconstructed = csl_tinyvit_11m(num_classes=4, pretrained=False, **kwargs)
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    assert isinstance(reconstructed.head.branch_attention, HierarchicalBranchAttention)


def test_checkpoint_reconstructs_branch_set_attention(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        pretrained=False,
        img_size=(384, 128),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        attention_window_layout="rect",
        attention_mask=True,
        head_parts=(1, 2, 4),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        branch_set_attention=True,
        branch_set_attention_token_dim=12,
        branch_set_attention_num_heads=3,
        branch_set_attention_num_layers=1,
        branch_set_attention_mlp_ratio=2.0,
        branch_set_attention_dropout=0.0,
        neck_dim=32,
        feat_dim=32,
    )
    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)
    weights = tmp_path / "branch_set_attention.pt"
    torch.save({**metadata, "state_dict": model.state_dict()}, weights)

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)

    assert metadata["model"]["transformer"]["branch_set_attention"] == {
        "enabled": True,
        "token_dim": 12,
        "num_heads": 3,
        "num_layers": 1,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "mask": "none",
        "output_init": "zero",
        "input_location": "post_pool_pre_reduction",
    }
    assert kwargs["branch_set_attention"] is True
    assert kwargs["branch_set_attention_token_dim"] == 12
    assert kwargs["branch_set_attention_num_heads"] == 3
    assert kwargs["branch_set_attention_num_layers"] == 1
    assert kwargs["branch_set_attention_mlp_ratio"] == 2.0
    assert kwargs["branch_set_attention_dropout"] == 0.0

    reconstructed = csl_tinyvit_11m(num_classes=4, pretrained=False, **kwargs)
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    assert isinstance(reconstructed.head.branch_set_attention, BranchSetAttention)


def test_checkpoint_reconstructs_multiscale_query_decoder(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        pretrained=False,
        img_size=(384, 128),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        attention_window_layout="rect",
        attention_mask=True,
        head_parts=(1, 2, 4),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        multiscale_query_decoder=True,
        query_decoder_dim=12,
        query_decoder_num_heads=3,
        query_decoder_num_layers=1,
        query_decoder_mlp_ratio=2.0,
        query_decoder_dropout=0.0,
        neck_dim=32,
        feat_dim=32,
    )
    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)
    weights = tmp_path / "multiscale_query_decoder.pt"
    torch.save({**metadata, "state_dict": model.state_dict()}, weights)

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)

    assert metadata["model"]["transformer"]["multiscale_query_decoder"] == {
        "enabled": True,
        "token_dim": 12,
        "num_heads": 3,
        "num_layers": 1,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "query_seeds": "existing_7_pooled_outputs",
        "memory": "final_stage2_stage0_maps",
        "position_encoding": "2d_sine_cosine",
        "attention_masks": "none",
        "memory_projection": "shared",
        "output_init": "zero",
    }
    assert kwargs["multiscale_query_decoder"] is True
    assert kwargs["query_decoder_dim"] == 12
    assert kwargs["query_decoder_num_heads"] == 3
    assert kwargs["query_decoder_num_layers"] == 1
    assert kwargs["query_decoder_mlp_ratio"] == 2.0
    assert kwargs["query_decoder_dropout"] == 0.0

    reconstructed = csl_tinyvit_11m(num_classes=4, pretrained=False, **kwargs)
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    assert isinstance(
        reconstructed.head.multiscale_query_decoder,
        ResidualMultiScaleQueryDecoder,
    )


def test_checkpoint_reconstructs_hierarchical_late_interaction(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        pretrained=False,
        epochs=60,
        img_size=(384, 128),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        attention_window_layout="rect",
        attention_mask=True,
        head_parts=(1, 2, 4),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        hierarchical_late_interaction=True,
        late_interaction_dim=12,
        late_interaction_num_heads=3,
        late_interaction_sinkhorn_iters=5,
        late_interaction_start_epoch=20,
        late_interaction_ramp_end_epoch=50,
        neck_dim=32,
        feat_dim=32,
    )
    model = trainer._build_model(num_classes=4)
    metadata = trainer._checkpoint_metadata(model)
    weights = tmp_path / "hierarchical_late_interaction.pt"
    torch.save({**metadata, "state_dict": model.state_dict()}, weights)

    kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights)

    assert kwargs["hierarchical_late_interaction"] is True
    assert kwargs["late_interaction_dim"] == 12
    assert kwargs["late_interaction_num_heads"] == 3
    assert kwargs["late_interaction_num_layers"] == 1
    assert kwargs["late_interaction_sinkhorn_iters"] == 5
    assert kwargs["late_interaction_null_tokens"] == 1
    assert kwargs["late_interaction_base_score_init"] == 0.9

    reconstructed = csl_tinyvit_11m(num_classes=4, pretrained=False, **kwargs)
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    assert isinstance(
        reconstructed.head.late_interaction_matcher,
        HierarchicalLateInteractionMatcher,
    )


def test_stage0_panet_lite_matches_lite_fpn_at_zero_bottom_up_gate():
    torch.manual_seed(0)
    panet = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage0_panet_lite",
        path_channels={0: 5, 1: 6, 2: 8},
        out_channels=8,
    )
    fpn = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage0_semantic_fine",
        path_channels={0: 5, 1: 6, 2: 8},
        out_channels=8,
    )
    fpn.projections.load_state_dict(panet.projections.state_dict())
    fpn.residual_scales.load_state_dict(panet.residual_scales.state_dict())
    fpn.stage0_fine_projection.load_state_dict(panet.stage0_fine_projection.state_dict())
    fpn.stage0_semantic_projection.load_state_dict(panet.stage0_semantic_projection.state_dict())
    fpn.stage0_fine_mixer.load_state_dict(panet.stage0_fine_mixer.state_dict())
    fpn.stage0_fine_gate.data.copy_(panet.stage0_fine_gate.data)
    final_feature = torch.randn(2, 8, 6, 2)
    paths = {
        0: torch.randn(2, 5, 12, 4),
        1: torch.randn(2, 6, 6, 2),
        2: torch.randn(2, 8, 6, 2),
    }

    panet_maps = panet(final_feature, paths)
    fpn_maps = fpn(final_feature, paths)

    torch.testing.assert_close(panet.stage0_panet_gate, torch.zeros(8))
    for panet_map, fpn_map in zip(panet_maps, fpn_maps, strict=True):
        torch.testing.assert_close(panet_map, fpn_map)


def test_stage0_bifpn_lite_starts_from_matched_hierarchical_control():
    torch.manual_seed(0)
    bifpn = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage0_bifpn_lite",
        path_channels={0: 5, 1: 6, 2: 8},
        out_channels=8,
    )
    control = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage2_hierarchical_control",
        path_channels={1: 6, 2: 8},
        out_channels=8,
    )
    control.projections.load_state_dict(bifpn.projections.state_dict())
    control.residual_scales.load_state_dict(bifpn.residual_scales.state_dict())
    final_feature = torch.randn(2, 8, 6, 2)
    paths = {
        0: torch.randn(2, 5, 12, 4),
        1: torch.randn(2, 6, 6, 2),
        2: torch.randn(2, 8, 6, 2),
    }

    bifpn_maps = bifpn(final_feature, paths)
    control_maps = control(final_feature, {index: paths[index] for index in (1, 2)})

    assert set(bifpn.stage0_bifpn_weights) == {"top_down", "bottom_up"}
    assert all(torch.count_nonzero(gate) == 0 for gate in bifpn.stage0_bifpn_gates.values())
    for bifpn_map, control_map in zip(bifpn_maps, control_maps, strict=True):
        torch.testing.assert_close(bifpn_map, control_map)


def test_stage0_native_pyramid_only_projects_the_fine_map_before_head_pooling():
    module = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage0_native_pyramid",
        path_channels={0: 5, 1: 6, 2: 8},
        out_channels=8,
    )
    final_feature = torch.randn(2, 8, 6, 2)
    paths = {
        0: torch.randn(2, 5, 12, 4),
        1: torch.randn(2, 6, 6, 2),
        2: torch.randn(2, 8, 6, 2),
    }

    global_map, local_map, fine_map = module(final_feature, paths)
    expected_fine = module.projections["0"](paths[0])

    assert global_map.shape == local_map.shape == (2, 8, 6, 2)
    assert fine_map.shape == (2, 8, 12, 4)
    torch.testing.assert_close(fine_map, expected_fine)


def test_stage0_fine_lite_skips_semantic_projection_and_spatial_mixer():
    module = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage0_fine_lite",
        path_channels={0: 5, 1: 6, 2: 8},
        out_channels=8,
        fine_map_dim=2,
    )
    final_feature = torch.randn(2, 8, 6, 2)
    paths = {
        0: torch.randn(2, 5, 12, 4),
        1: torch.randn(2, 6, 6, 2),
        2: torch.randn(2, 8, 6, 2),
    }

    _, local_map, fine_map = module(final_feature, paths)
    expected_control = module._resize_feature(module.local_to_fine(local_map), (12, 4))

    assert isinstance(module.stage0_semantic_projection, nn.Identity)
    assert len(module.stage0_fine_mixer) == 0
    assert fine_map.shape == (2, 2, 12, 4)
    torch.testing.assert_close(fine_map, expected_control)


def test_stage0_pool_first_passes_native_maps_to_the_head_without_spatial_projection():
    module = CSLTinyViTFeatureFusion.from_mode(
        "global_final_parts_stage0_pool_first",
        path_channels={0: 5, 2: 8},
        out_channels=16,
    )
    final_feature = torch.randn(2, 16, 6, 2)
    paths = {
        0: torch.randn(2, 5, 12, 4),
        2: torch.randn(2, 8, 6, 2),
    }

    global_map, local_map, fine_map = module(final_feature, paths)

    assert module.stage_indices == (2, 0)
    assert module.local_channels == 8
    assert module.fine_output_channels == 5
    assert len(module.projections) == 0
    assert sum(parameter.numel() for parameter in module.parameters()) == 0
    assert global_map is final_feature
    assert local_map is paths[2]
    assert fine_map is paths[0]


def test_csl_tinyvit_pool_first_projects_only_after_native_stripe_pooling():
    model = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage0_pool_first",
        attention_window_layout="rect",
        head_parts=(1, 2, 4),
        scale_balanced_branches=True,
        neck_dim=32,
        feat_dim=16,
        inference_feature="norm_concat_bn",
    ).eval()

    with torch.inference_mode():
        maps = model.forward_features(torch.randn(1, 3, 384, 128))
        descriptor = model.forward_head(maps)

    assert [feature.shape for feature in maps] == [
        (1, 32, 24, 8),
        (1, 448, 24, 8),
        (1, 128, 48, 16),
    ]
    assert model.head.branch_input_channels == (32, 448, 128)
    assert descriptor.shape == (1, 48)


@pytest.mark.parametrize(
    "mode",
    (
        "global_final_parts_stage0_semantic_fine_reference",
        "global_final_parts_stage0_fine_lite",
        "global_final_parts_stage0_panet_lite",
        "global_final_parts_stage0_bifpn_lite",
        "global_final_parts_stage0_native_pyramid",
    ),
)
def test_csl_tinyvit_compact_stage0_modes_build_hierarchical_head(mode):
    model = csl_tinyvit_11m(
        num_classes=4,
        pretrained=False,
        feature_fusion=mode,
        head_parts=(1, 2, 4),
        neck_dim=32,
        feat_dim=16,
    )

    assert model.feature_fusion_module.stage_indices == (1, 2, 0)
    assert model.head.hierarchical_scales is True


@pytest.mark.parametrize(
    "mode",
    (
        "global_final_parts_stage0_semantic_fine_reference",
        "global_final_parts_stage0_semantic_fine",
        "global_final_parts_stage0_fine_lite",
        "global_final_parts_stage0_panet_lite",
        "global_final_parts_stage0_bifpn_lite",
        "global_final_parts_stage0_native_pyramid",
    ),
)
def test_compact_stage0_fusions_stay_below_parameter_budget(mode):
    stage_indices = CSLTinyViTFeatureFusion.stage_indices_for_mode(mode)
    module = CSLTinyViTFeatureFusion.from_mode(
        mode,
        path_channels={0: 128, 1: 256, 2: 448},
        out_channels=512,
    )

    assert stage_indices == (1, 2, 0)
    assert sum(parameter.numel() for parameter in module.parameters()) < 750_000


@pytest.mark.parametrize(
    "mode",
    (
        "last3_stage1_concat",
        "last3_fpn_stage1_split",
        "global_final_parts_stage1_concat",
        "global_final_parts_fpn_layer0",
        "last3_panet_stage1_scale_aware",
        "last3_bifpn_stage1_branch_aware",
        "global_final_parts_stage2_semantic_residual",
        "global_final_parts_stage2_hierarchical_control",
        "global_final_parts_stage0_semantic_fine_reference",
        "global_final_parts_stage0_semantic_fine",
        "global_final_parts_stage0_fine_lite",
        "global_final_parts_stage0_panet_lite",
        "global_final_parts_stage0_bifpn_lite",
        "global_final_parts_stage0_native_pyramid",
        "global_final_parts_stage0_pool_first",
        "global_final_parts_hierarchical_fpn",
    ),
)
def test_feature_aggregation_modes_have_no_trainable_parameters_outside_the_forward_graph(mode):
    stage_indices = CSLTinyViTFeatureFusion.stage_indices_for_mode(mode)
    module = CSLTinyViTFeatureFusion.from_mode(
        mode,
        path_channels={index: 8 for index in stage_indices},
        out_channels=8,
    )
    final_feature = torch.randn(2, 8, 6, 2, requires_grad=True)
    all_paths = {
        0: torch.randn(2, 8, 12, 4),
        1: torch.randn(2, 8, 6, 2),
        2: torch.randn(2, 8, 6, 2),
    }

    output = module(final_feature, {index: all_paths[index] for index in stage_indices})
    outputs = output if isinstance(output, tuple) else (output,)
    sum(feature.square().mean() for feature in outputs).backward()

    missing_gradients = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing_gradients == []


def test_hierarchical_head_routes_scales_and_keeps_compact_descriptor():
    head = MultiBranchHead(
        8, feat_dim=32, num_classes=3, inference_feature="raw_concat",
        head_parts=(1, 2, 4), hierarchical_scales=True,
    )
    head.eval()
    with torch.no_grad():
        descriptor = head((
            torch.randn(2, 8, 3, 1),
            torch.randn(2, 8, 6, 2),
            torch.randn(2, 8, 12, 4),
        ))
    assert descriptor.shape == (2, 96)


def test_scale_balanced_hierarchical_head_uses_all_branches_with_equal_scale_energy():
    torch.manual_seed(0)
    head = MultiBranchHead(
        8,
        feat_dim=32,
        num_classes=3,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
        hierarchical_scales=True,
        scale_balanced_branches=True,
    )
    features = (
        torch.randn(2, 8, 3, 1),
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 12, 4),
    )
    chunk_sizes = (32, 16, 16, 8, 8, 8, 8)

    head.train()
    logits, metric_descriptor = head(features)
    metric_chunks = torch.split(metric_descriptor, chunk_sizes, dim=1)

    assert len(logits) == 7
    torch.testing.assert_close(
        torch.stack([chunk.norm(dim=1) for chunk in metric_chunks], dim=1),
        torch.tensor([[1.0, 2**-0.5, 2**-0.5, 0.5, 0.5, 0.5, 0.5]]).expand(2, -1),
        atol=1e-5,
        rtol=1e-5,
    )

    head.eval()
    with torch.no_grad():
        retrieval_descriptor = head(features)
    retrieval_chunks = torch.split(retrieval_descriptor, chunk_sizes, dim=1)
    expected_scale_norms = torch.tensor(
        [[3**-0.5, 6**-0.5, 6**-0.5, 12**-0.5, 12**-0.5, 12**-0.5, 12**-0.5]]
    ).expand(2, -1)

    torch.testing.assert_close(retrieval_descriptor.norm(dim=1), torch.ones(2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        torch.stack([chunk.norm(dim=1) for chunk in retrieval_chunks], dim=1),
        expected_scale_norms,
        atol=1e-5,
        rtol=1e-5,
    )


def test_hierarchical_head_exposes_equal_width_csmm_groups_only_during_training():
    head = MultiBranchHead(
        8,
        feat_dim=32,
        num_classes=3,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
        hierarchical_scales=True,
        scale_balanced_branches=True,
        return_cross_scale_features=True,
    )
    features = (
        torch.randn(2, 8, 3, 1),
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 12, 4),
    )

    head.train()
    logits, training_features = head(features)
    scale_groups = training_features["_cross_scale_features"]
    assert len(logits) == 7
    assert training_features["raw_concat"].shape == (2, 96)
    assert [group.shape for group in scale_groups] == [(2, 32), (2, 32), (2, 32)]
    for group in scale_groups:
        torch.testing.assert_close(group.norm(dim=1), torch.ones(2), atol=1e-5, rtol=1e-5)

    head.eval()
    with torch.no_grad():
        descriptor = head(features)
    assert isinstance(descriptor, torch.Tensor)
    assert descriptor.shape == (2, 96)


def test_hierarchical_head_exposes_post_bn_treeboost_branches_only_during_training():
    head = MultiBranchHead(
        8,
        feat_dim=32,
        num_classes=3,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
        hierarchical_scales=True,
        scale_balanced_branches=True,
        return_treeboost_features=True,
    )
    features = (
        torch.randn(2, 8, 3, 1),
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 12, 4),
    )

    head.train()
    logits, training_features = head(features)
    global_feature, coarse_features, fine_features = training_features["_treeboost_features"]
    assert len(logits) == 7
    assert global_feature.shape == (2, 32)
    assert [feature.shape for feature in coarse_features] == [(2, 16), (2, 16)]
    assert [feature.shape for feature in fine_features] == [(2, 8)] * 4
    for feature in (global_feature, *coarse_features, *fine_features):
        torch.testing.assert_close(feature.norm(dim=1), torch.ones(2), atol=1e-5, rtol=1e-5)

    head.eval()
    with torch.no_grad():
        descriptor = head(features)
    assert descriptor.shape == (2, 96)


def test_scale_balanced_hierarchical_head_can_train_global_metric_and_retrieve_all_branches():
    head = MultiBranchHead(
        8,
        feat_dim=32,
        num_classes=3,
        metric_feature="global",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
        hierarchical_scales=True,
        scale_balanced_branches=True,
    )
    features = (
        torch.randn(2, 8, 3, 1),
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 12, 4),
    )

    head.train()
    logits, metric_descriptor = head(features)
    assert len(logits) == 7
    assert metric_descriptor.shape == (2, 32)

    head.eval()
    with torch.no_grad():
        retrieval_descriptor = head(features)
    assert retrieval_descriptor.shape == (2, 96)


def test_scale_balanced_hierarchical_head_can_train_coarse_metric_and_retrieve_all_branches():
    head = MultiBranchHead(
        8,
        feat_dim=32,
        num_classes=3,
        metric_feature="coarse_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
        hierarchical_scales=True,
        scale_balanced_branches=True,
    )
    features = (
        torch.randn(2, 8, 3, 1),
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 12, 4),
    )

    head.train()
    logits, metric_descriptor = head(features)
    metric_chunks = torch.split(metric_descriptor, (32, 16, 16), dim=1)
    assert len(logits) == 7
    assert metric_descriptor.shape == (2, 64)
    torch.testing.assert_close(
        torch.stack([chunk.norm(dim=1) for chunk in metric_chunks], dim=1),
        torch.tensor([[1.0, 2**-0.5, 2**-0.5]]).expand(2, -1),
        atol=1e-5,
        rtol=1e-5,
    )

    head.eval()
    with torch.no_grad():
        retrieval_descriptor = head(features)
    assert retrieval_descriptor.shape == (2, 96)


def test_csl_tinyvit_last4_layer0_target_fuses_at_layer0_resolution():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last4_layer0_target",
        path_channels={0: 4, 1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {
        0: torch.randn(2, 4, 12, 4),
        1: torch.randn(2, 4, 6, 3),
        2: torch.randn(2, 4, 3, 2),
    }

    output = module(final_feature, path_features)
    expected = F.interpolate(final_feature, size=(12, 4), mode="bilinear", align_corners=False)

    assert module.mode == "last4_layer0_target"
    assert module.fusion_type == "residual"
    assert module.stage_indices == (0, 1, 2)
    assert module.target_stage_index == 0
    assert output.shape[-2:] == path_features[0].shape[-2:]
    torch.testing.assert_close(output, expected)


def test_csl_tinyvit_last4_layer0_target_captures_layer0_feature_map():
    model = csl_tinyvit_7m(num_classes=4, pretrained=False, feature_fusion="last4_layer0_target")
    model.eval()

    with torch.no_grad():
        features = model.forward_features(torch.randn(1, 3, 384, 128))

    assert model.feature_fusion == "last4_layer0_target"
    assert model._fusion_stage_indices == (0, 1, 2)
    assert model.feature_fusion_module.projections["0"][0].in_channels == 128
    assert features.shape == (1, 512, 48, 16)


def test_csl_tinyvit_last3_fpn_stage2_averages_paths_at_stage2_resolution():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last3_fpn_stage2",
        path_channels={1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {
        1: torch.randn(2, 4, 12, 4),
        2: torch.randn(2, 4, 6, 3),
    }

    projected = module._ordered_features(final_feature, path_features)
    output = module(final_feature, path_features)
    expected = torch.stack(projected, dim=0).mean(dim=0)

    assert module.mode == "last3_fpn_stage2"
    assert module.fusion_type == "fpn"
    assert module.stage_indices == (2, 1)
    assert module.target_stage_index == 2
    assert output.shape[-2:] == path_features[2].shape[-2:]
    torch.testing.assert_close(output, expected)


def test_csl_tinyvit_last3_pafpn_stage2_uses_top_down_bottom_up_path():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last3_pafpn_stage2",
        path_channels={1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {
        1: torch.randn(2, 4, 12, 4),
        2: torch.randn(2, 4, 6, 3),
    }

    output = module(final_feature, path_features)

    assert module.mode == "last3_pafpn_stage2"
    assert module.fusion_type == "pafpn"
    assert module.stage_indices == (2, 1)
    assert module.target_stage_index == 2
    assert module.pafpn_top_down["2"][0].in_channels == 8
    assert module.pafpn_bottom_up["2"][0].in_channels == 8
    assert output.shape == (2, 4, 6, 3)


def test_csl_tinyvit_last3_pafpn_stage2_forward_preserves_embedding_shape():
    model = csl_tinyvit_7m(num_classes=4, pretrained=False, feature_fusion="last3_pafpn_stage2")
    model.eval()

    with torch.no_grad():
        features = model(torch.randn(2, 3, 384, 128))

    assert model.feature_fusion == "last3_pafpn_stage2"
    assert model._fusion_stage_indices == (2, 1)
    assert features.shape == (2, 1536)


def test_csl_tinyvit_last4_fpn_layer0_target_averages_paths_at_layer0_resolution():
    module = CSLTinyViTFeatureFusion.from_mode(
        "last4_fpn_layer0_target",
        path_channels={0: 4, 1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {
        0: torch.randn(2, 4, 12, 4),
        1: torch.randn(2, 4, 6, 3),
        2: torch.randn(2, 4, 3, 2),
    }

    projected = module._ordered_features(final_feature, path_features)
    output = module(final_feature, path_features)
    expected = torch.stack(projected, dim=0).mean(dim=0)

    assert module.mode == "last4_fpn_layer0_target"
    assert module.fusion_type == "fpn"
    assert module.stage_indices == (2, 1, 0)
    assert module.target_stage_index == 0
    assert output.shape[-2:] == path_features[0].shape[-2:]
    torch.testing.assert_close(output, expected)


def test_csl_tinyvit_last4_fpn_layer0_target_captures_layer0_feature_map():
    model = csl_tinyvit_7m(num_classes=4, pretrained=False, feature_fusion="last4_fpn_layer0_target")
    model.eval()

    with torch.no_grad():
        features = model.forward_features(torch.randn(1, 3, 384, 128))

    assert model.feature_fusion == "last4_fpn_layer0_target"
    assert model._fusion_stage_indices == (2, 1, 0)
    assert model.feature_fusion_module.projections["0"][0].in_channels == 128
    assert features.shape == (1, 512, 48, 16)


def test_csl_tinyvit_split_global_local_modes_return_1536d_embeddings():
    for fusion in (
        "global_final_parts_stage2",
        "global_final_parts_stage2_semantic_residual",
        "late_concat_stage2",
    ):
        model = csl_tinyvit_7m(
            num_classes=4,
            pretrained=False,
            feature_fusion=fusion,
            inference_feature="norm_concat_bn",
        )
        model.eval()

        with torch.no_grad():
            features = model(torch.randn(1, 3, 384, 128))

        assert model.feature_fusion == fusion
        assert features.shape == (1, 1536)

        model.train()
        logits, train_features = model(torch.randn(2, 3, 384, 128))

        assert len(logits) == 3
        assert train_features.shape == (2, 512)


def test_norm_preserved_feature_fusion_preserves_max_path_norm():
    module = CSLTinyViTFeatureFusion.from_mode(
        "normpres_last3",
        path_channels={1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 3, 2)
    path_features = {
        1: torch.randn(2, 4, 6, 4),
        2: torch.randn(2, 4, 3, 2),
    }

    projected = module._ordered_features(final_feature, path_features)
    output = module(final_feature, path_features)
    max_norm = torch.stack([feature.norm(p=2, dim=1) for feature in projected], dim=0).max(dim=0).values

    assert module.fusion_type == "norm_preserved"
    assert module.stage_indices == (1, 2)
    torch.testing.assert_close(output.norm(p=2, dim=1), max_norm, atol=1e-5, rtol=1e-5)


def test_csl_tinyvit_dynamic_feature_fusion_preserves_output_shape():
    model = csl_tinyvit_7m(num_classes=4, pretrained=False, feature_fusion="dynamic_last3")
    model.eval()

    with torch.no_grad():
        features = model(torch.randn(2, 3, 384, 128))

    assert model.feature_fusion == "dynamic_last3"
    assert model._fusion_stage_indices == (2, 1)
    assert model.feature_fusion_module.dynamic_gate[-1].out_features == 3
    assert features.shape == (2, 1536)


def test_dynamic_feature_fusion_uses_per_image_softmax_weights():
    module = CSLTinyViTFeatureFusion.from_mode(
        "dynamic_last3",
        path_channels={1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 4, 2)
    path_features = {
        1: torch.randn(2, 4, 8, 4),
        2: torch.randn(2, 4, 4, 2),
    }

    weights = module.dynamic_weights(final_feature, path_features)

    assert weights.shape == (2, 3)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(
        weights.mean(dim=0),
        torch.tensor([0.8, 0.1, 0.1]),
        atol=0.01,
        rtol=0.0,
    )


def test_dynamic_image_gate_depends_only_on_final_feature():
    module = CSLTinyViTFeatureFusion.from_mode(
        "dynamic_last3",
        path_channels={1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(1, 4, 4, 2).repeat(2, 1, 1, 1)
    path_features = {
        1: torch.stack([torch.zeros(4, 8, 4), torch.randn(4, 8, 4)]),
        2: torch.stack([torch.randn(4, 4, 2), torch.zeros(4, 4, 2)]),
    }

    weights = module.dynamic_weights(final_feature, path_features)

    torch.testing.assert_close(weights[0], weights[1])


def test_dynamic_scale_token_responds_to_multiscale_path_content():
    module = CSLTinyViTFeatureFusion.from_mode(
        "dynamic_last3_scale_token",
        path_channels={1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(1, 4, 4, 2).repeat(2, 1, 1, 1)
    ascending = torch.arange(4, dtype=torch.float32)[:, None, None]
    descending = ascending.flip(0)
    path_features = {
        1: torch.stack([ascending.expand(4, 8, 4), descending.expand(4, 8, 4)]),
        2: torch.stack([descending.expand(4, 4, 2), ascending.expand(4, 4, 2)]),
    }

    weights = module.dynamic_weights(final_feature, path_features)

    assert module.stage_indices == (2, 1)
    assert module.scale_token_projection is not None
    assert module.scale_tokens.shape[0] == 3
    assert weights.shape == (2, 3)
    assert not torch.allclose(weights[0], weights[1], atol=1e-8, rtol=0.0)


def test_dynamic_fusion_initialization_keeps_side_path_gradients_active():
    module = CSLTinyViTFeatureFusion.from_mode(
        "dynamic_last3_scale_token",
        path_channels={1: 4, 2: 4},
        out_channels=4,
    )
    final_feature = torch.randn(2, 4, 4, 2, requires_grad=True)
    path_features = {
        1: torch.randn(2, 4, 8, 4, requires_grad=True),
        2: torch.randn(2, 4, 4, 2, requires_grad=True),
    }

    module(final_feature, path_features).square().mean().backward()

    assert module.dynamic_gate[-1].weight.grad.abs().sum() > 0
    assert module.scale_tokens.grad.abs().sum() > 0
    assert path_features[1].grad.abs().sum() > 0
    assert path_features[2].grad.abs().sum() > 0


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
    layer0_target = csl_tinyvit_23m(num_classes=4, pretrained=False, feature_fusion="last4_layer0_target")

    assert model.feature_fusion_module.projections["1"][0].in_channels == 384
    assert model.feature_fusion_module.projections["2"][0].in_channels == 576
    assert layer0_target.feature_fusion_module.projections["0"][0].in_channels == 192


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


def test_csl_tinyvit_loads_legacy_top_level_blocks_alias_keys():
    source = csl_tinyvit_7m(num_classes=4, pretrained=False)
    legacy_state = {}
    for key, value in source.state_dict().items():
        legacy_key = key.replace("layers.", "blocks.", 1) if key.startswith("layers.") else key
        legacy_state[legacy_key] = value.clone()

    fresh = csl_tinyvit_7m(num_classes=4, pretrained=False)

    fresh.load_state_dict(legacy_state, strict=True)

    assert not any(key.startswith("blocks.") for key in fresh.state_dict())
    torch.testing.assert_close(
        fresh.layers[3].blocks[0].mlp.fc1.weight,
        source.layers[3].blocks[0].mlp.fc1.weight,
    )


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
    assert set(branch_features) == {
        "global",
        "part0",
        "part1",
        "raw_mean",
        "raw_concat",
        "coarse_concat",
        "concat_bn",
        "norm_concat_bn",
    }
    assert branch_features["global"].shape == (4, 4)
    assert branch_features["raw_concat"].shape == (4, 12)
    assert branch_features["concat_bn"].shape == (4, 12)


def test_multibranch_head_drop_global_aux_is_ce_only():
    x = torch.randn(4, 8, 4, 2)
    head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        drop_global_aux=True,
        drop_global_aux_ratio=0.25,
    )

    head.train()
    logits, features = head(x)

    assert len(logits) == 4
    assert features.shape == (4, 12)

    head.eval()
    with torch.no_grad():
        eval_features = head(x)

    assert eval_features.shape == (4, 12)


def test_multibranch_head_overlap_stripes_keeps_branch_contract():
    x = torch.randn(4, 8, 8, 2)
    head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        part_pooling="overlap_stripes",
        head_parts=(1, 2),
    )

    bounds = head._overlap_window_bounds(height=8, granularity=2)
    head.train()
    logits, features = head(x)

    assert head.part_pooling == "overlap_stripes"
    assert bounds == [(0, 6), (2, 8)]
    assert len(logits) == 3
    assert features.shape == (4, 12)

    head.eval()
    with torch.no_grad():
        eval_features = head(x)

    assert eval_features.shape == (4, 12)


def test_bnneck3_skips_classifier_in_eval_return_features():
    neck = BNNeck3(input_dim=8, class_num=3, feat_dim=4, return_f=True)
    neck.eval()

    feature, score, raw = neck(torch.randn(2, 8, 1, 1))

    assert feature.shape == (2, 4)
    assert score is None
    assert raw.shape == (2, 4)


def test_hierarchical_branch_attention_is_tree_masked_identity_at_initialization():
    attention = HierarchicalBranchAttention(
        global_dim=32,
        coarse_dim=16,
        fine_dim=8,
        token_dim=12,
        num_heads=3,
        num_layers=1,
        mlp_ratio=2.0,
        dropout=0.0,
    ).eval()
    global_feature = torch.randn(2, 32)
    coarse_features = (torch.randn(2, 16), torch.randn(2, 16))
    fine_features = tuple(torch.randn(2, 8) for _ in range(4))

    with torch.no_grad():
        refined_global, refined_coarse, refined_fine = attention(
            global_feature,
            coarse_features,
            fine_features,
        )

    expected_allowed = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(~attention.attention_mask, expected_allowed)
    torch.testing.assert_close(refined_global, global_feature, rtol=0, atol=0)
    for refined, original in zip(refined_coarse, coarse_features, strict=True):
        torch.testing.assert_close(refined, original, rtol=0, atol=0)
    for refined, original in zip(refined_fine, fine_features, strict=True):
        torch.testing.assert_close(refined, original, rtol=0, atol=0)


def test_hierarchical_branch_attention_zero_outputs_can_learn():
    attention = HierarchicalBranchAttention(
        global_dim=32,
        coarse_dim=16,
        fine_dim=8,
        token_dim=12,
        num_heads=3,
        dropout=0.0,
    )
    outputs = attention(
        torch.randn(4, 32),
        (torch.randn(4, 16), torch.randn(4, 16)),
        tuple(torch.randn(4, 8) for _ in range(4)),
    )
    global_feature, coarse_features, fine_features = outputs
    (global_feature.sum() + sum(x.sum() for x in coarse_features + fine_features)).backward()

    for projection in (attention.global_out, attention.coarse_out, attention.fine_out):
        assert projection.weight.grad is not None
        assert projection.weight.grad.abs().sum() > 0


def test_hierarchical_branch_attention_preserves_initial_retrieval_descriptor():
    kwargs = {
        "in_ch": 8,
        "feat_dim": 32,
        "num_classes": 3,
        "inference_feature": "norm_concat_bn",
        "head_parts": (1, 2, 4),
        "hierarchical_scales": True,
        "scale_balanced_branches": True,
    }
    torch.manual_seed(13)
    control = MultiBranchHead(**kwargs).eval()
    torch.manual_seed(13)
    treatment = MultiBranchHead(
        **kwargs,
        hierarchical_branch_attention=True,
        branch_attention_token_dim=12,
        branch_attention_num_heads=3,
    ).eval()
    features = (
        torch.randn(2, 8, 3, 1),
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 12, 4),
    )

    with torch.no_grad():
        control_descriptor = control(features)
        treatment_descriptor = treatment(features)

    assert treatment_descriptor.shape == (2, 96)
    torch.testing.assert_close(treatment_descriptor, control_descriptor, rtol=0, atol=0)


def test_branch_set_attention_is_identity_trainable_and_compact():
    attention = BranchSetAttention(
        input_dim=512,
        token_dim=128,
        num_heads=4,
        num_layers=1,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    branches = torch.randn(3, 7, 512)

    output = attention(branches)

    torch.testing.assert_close(output, branches, rtol=0, atol=0)
    output.sum().backward()
    assert attention.output_proj.weight.grad is not None
    assert attention.output_proj.weight.grad.abs().sum() > 0
    assert sum(parameter.numel() for parameter in attention.parameters()) < 280_000


def test_branch_set_attention_refines_pooled_512_equivalents_before_branch_reductions():
    kwargs = {
        "in_ch": 8,
        "feat_dim": 32,
        "num_classes": 3,
        "inference_feature": "norm_concat_bn",
        "head_parts": (1, 2, 4),
        "hierarchical_scales": True,
        "scale_balanced_branches": True,
    }
    torch.manual_seed(19)
    control = MultiBranchHead(**kwargs).eval()
    torch.manual_seed(19)
    treatment = MultiBranchHead(
        **kwargs,
        branch_set_attention=True,
        branch_set_attention_token_dim=12,
        branch_set_attention_num_heads=3,
    ).eval()
    observed_shapes = []
    hook = treatment.branch_set_attention.register_forward_pre_hook(
        lambda _module, inputs: observed_shapes.append(tuple(inputs[0].shape))
    )
    features = (
        torch.randn(2, 8, 3, 1),
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 12, 4),
    )

    with torch.no_grad():
        control_descriptor = control(features)
        treatment_descriptor = treatment(features)
    hook.remove()

    assert observed_shapes == [(2, 7, 8)]
    assert treatment.bn_global.reduction.out_channels == 32
    assert treatment.bn_part0.reduction.out_channels == 16
    assert treatment.bn_part2.reduction.out_channels == 8
    assert treatment_descriptor.shape == (2, 96)
    torch.testing.assert_close(treatment_descriptor, control_descriptor, rtol=0, atol=0)


def test_multiscale_query_decoder_is_identity_trainable_and_compact():
    decoder = ResidualMultiScaleQueryDecoder(
        input_dim=512,
        token_dim=128,
        num_heads=4,
        num_layers=1,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    branches = torch.randn(2, 7, 512)
    maps = (
        torch.randn(2, 512, 3, 1),
        torch.randn(2, 512, 6, 2),
        torch.randn(2, 512, 12, 4),
    )

    output = decoder(branches, maps)

    torch.testing.assert_close(output, branches, rtol=0, atol=0)
    output.sum().backward()
    assert decoder.output_projection.weight.grad is not None
    assert decoder.output_projection.weight.grad.abs().sum() > 0
    parameter_count = sum(parameter.numel() for parameter in decoder.parameters())
    assert 300_000 < parameter_count < 500_000


def test_multiscale_query_decoder_reads_all_maps_before_branch_reductions():
    kwargs = {
        "in_ch": 8,
        "feat_dim": 32,
        "num_classes": 3,
        "inference_feature": "norm_concat_bn",
        "head_parts": (1, 2, 4),
        "hierarchical_scales": True,
        "scale_balanced_branches": True,
    }
    torch.manual_seed(23)
    control = MultiBranchHead(**kwargs).eval()
    torch.manual_seed(23)
    treatment = MultiBranchHead(
        **kwargs,
        multiscale_query_decoder=True,
        query_decoder_dim=12,
        query_decoder_num_heads=3,
    ).eval()
    observed_shapes = []

    def observe_decoder_inputs(_module, inputs):
        observed_shapes.append(
            (
                tuple(inputs[0].shape),
                tuple(tuple(feature_map.shape) for feature_map in inputs[1]),
            )
        )

    hook = treatment.multiscale_query_decoder.register_forward_pre_hook(observe_decoder_inputs)
    features = (
        torch.randn(2, 8, 3, 1),
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 12, 4),
    )

    with torch.no_grad():
        control_descriptor = control(features)
        treatment_descriptor = treatment(features)
    hook.remove()

    assert observed_shapes == [
        (
            (2, 7, 8),
            ((2, 8, 3, 1), (2, 8, 6, 2), (2, 8, 12, 4)),
        )
    ]
    assert treatment.bn_global.reduction.out_channels == 32
    assert treatment.bn_part0.reduction.out_channels == 16
    assert treatment.bn_part2.reduction.out_channels == 8
    assert treatment_descriptor.shape == (2, 96)
    torch.testing.assert_close(treatment_descriptor, control_descriptor, rtol=0, atol=0)


def test_hierarchical_late_interaction_is_symmetric_trainable_and_compact():
    matcher = HierarchicalLateInteractionMatcher(
        global_dim=32,
        coarse_dim=16,
        fine_dim=8,
        token_dim=12,
        num_heads=3,
        num_layers=1,
        sinkhorn_iters=5,
        base_score_init=0.9,
    )

    def hierarchy(batch_size: int):
        return (
            torch.randn(batch_size, 32),
            (torch.randn(batch_size, 16), torch.randn(batch_size, 16)),
            tuple(torch.randn(batch_size, 8) for _ in range(4)),
        )

    query = hierarchy(4)
    gallery = hierarchy(4)
    query_base = torch.randn(4, 96)
    gallery_base = torch.randn(4, 96)
    forward_scores = matcher.score_pairs(query, gallery, query_base, gallery_base)
    reverse_scores = matcher.score_pairs(gallery, query, gallery_base, query_base)

    torch.testing.assert_close(forward_scores, reverse_scores, atol=1e-6, rtol=1e-6)
    forward_scores.sum().backward()
    assert matcher.query_projections[0].weight.grad is not None
    assert matcher.query_projections[0].weight.grad.abs().sum() > 0
    assert matcher.null_token.grad is not None
    assert matcher.base_score_logit.grad is not None

    full_matcher = HierarchicalLateInteractionMatcher(512, 256, 128)
    assert sum(parameter.numel() for parameter in full_matcher.parameters()) < 250_000


def test_hierarchical_late_interaction_preserves_default_descriptor_and_emits_packet():
    kwargs = {
        "in_ch": 8,
        "feat_dim": 32,
        "num_classes": 3,
        "metric_feature": "raw_concat",
        "inference_feature": "norm_concat_bn",
        "head_parts": (1, 2, 4),
        "hierarchical_scales": True,
        "scale_balanced_branches": True,
    }
    torch.manual_seed(17)
    control = MultiBranchHead(**kwargs).eval()
    torch.manual_seed(17)
    treatment = MultiBranchHead(
        **kwargs,
        hierarchical_late_interaction=True,
        late_interaction_dim=12,
        late_interaction_num_heads=3,
    ).eval()
    features = (
        torch.randn(2, 8, 3, 1),
        torch.randn(2, 8, 6, 2),
        torch.randn(2, 8, 12, 4),
    )

    with torch.no_grad():
        control_descriptor = control(features)
        treatment_descriptor = treatment(features)
        treatment.emit_late_interaction_packet = True
        packet = treatment(features)

    torch.testing.assert_close(treatment_descriptor, control_descriptor, rtol=0, atol=0)
    assert treatment_descriptor.shape == (2, 96)
    assert packet.shape == (2, 192)
    torch.testing.assert_close(packet[:, :96], treatment_descriptor, rtol=0, atol=0)


def test_hierarchical_late_interaction_listwise_and_distillation_losses_backpropagate(tmp_path):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        epochs=60,
        feature_fusion="global_final_parts_stage0_semantic_fine",
        spatial_conv_mode="depthwise_separable",
        head_parts=(1, 2, 4),
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        hierarchical_late_interaction=True,
        late_interaction_dim=12,
        late_interaction_num_heads=3,
        late_interaction_negative_identities=3,
        late_interaction_start_epoch=0,
        late_interaction_ramp_end_epoch=1,
    )
    head = MultiBranchHead(
        8,
        feat_dim=32,
        num_classes=4,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
        hierarchical_scales=True,
        scale_balanced_branches=True,
        hierarchical_late_interaction=True,
        late_interaction_dim=12,
        late_interaction_num_heads=3,
    )
    sources = (
        torch.randn(8, 8, 3, 1),
        torch.randn(8, 8, 6, 2),
        torch.randn(8, 8, 12, 4),
    )
    pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    camera_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    head.train()
    _, training_features = head(sources)

    matcher_loss, distillation_loss = trainer._hierarchical_late_interaction_losses(
        SimpleNamespace(head=head),
        training_features,
        pids,
        camera_ids,
    )
    (matcher_loss + distillation_loss).backward()

    assert matcher_loss.isfinite() and matcher_loss > 0
    assert distillation_loss.isfinite() and distillation_loss >= 0
    assert head.late_interaction_matcher.query_projections[0].weight.grad is not None
    assert head.bn_global.reduction.weight.grad is not None


def test_csl_tinyvit_restores_bnneck_classifier_init_after_global_init():
    model = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage2",
        head_parts=(1, 2),
        part_pooling="stripes",
    )
    classifier = model.head.bn_global.classifier

    assert classifier.weight.std().item() == pytest.approx(0.001, abs=2.5e-4)
    assert model.head.bn_global.bn.bias.requires_grad is False


def test_csl_tinyvit_restores_semantic_visibility_priors_after_global_init():
    model = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        feature_fusion="global_final_parts_stage2",
        head_parts=(1, 4),
        part_pooling="semantic_parts",
    )
    pool = model.head.semantic_part_pool

    torch.testing.assert_close(pool.visibility_predictor.weight, torch.zeros_like(pool.visibility_predictor.weight))
    torch.testing.assert_close(
        pool.visibility_predictor.bias,
        torch.full_like(pool.visibility_predictor.bias, math.log(9.0)),
    )
    torch.testing.assert_close(pool.rarity_predictor.weight, torch.zeros_like(pool.rarity_predictor.weight))
    torch.testing.assert_close(pool.rarity_predictor.bias, torch.zeros_like(pool.rarity_predictor.bias))
    torch.testing.assert_close(pool.null_predictor.weight, torch.zeros_like(pool.null_predictor.weight))
    torch.testing.assert_close(
        pool.null_predictor.bias,
        torch.full_like(pool.null_predictor.bias, math.log(1.0 / 9.0)),
    )
    torch.testing.assert_close(
        torch.sigmoid(pool.visibility_predictor.bias),
        torch.full_like(pool.visibility_predictor.bias, 0.9),
    )
    torch.testing.assert_close(torch.sigmoid(pool.null_predictor.bias), torch.full_like(pool.null_predictor.bias, 0.1))


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

    norm_concat_head = MultiBranchHead(8, feat_dim=4, num_classes=3, inference_feature="norm_concat_bn")
    norm_concat_head.eval()
    with torch.no_grad():
        features = norm_concat_head(x)
    assert features.shape == (2, 12)
    assert torch.allclose(features.norm(dim=1), torch.ones(2), atol=1e-5)

    raw_concat_head = MultiBranchHead(8, feat_dim=4, num_classes=3, metric_feature="raw_concat")
    raw_concat_head.train()
    _, train_features = raw_concat_head(x)
    assert train_features.shape == (2, 12)


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


def test_multibranch_head_supports_learned_part_tokens():
    x = torch.randn(2, 8, 6, 2)
    head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        metric_feature="concat_bn",
        inference_feature="concat_bn",
        part_pooling="tokens",
        num_part_tokens=4,
    )

    head.train()
    logits, train_features = head(x)
    assert len(logits) == 5
    assert train_features.shape == (2, 20)

    head.eval()
    with torch.no_grad():
        eval_features = head(x)
    assert eval_features.shape == (2, 20)


def test_multibranch_head_supports_semantic_visibility_parts():
    x = (torch.randn(2, 8, 4, 2), torch.randn(2, 8, 8, 2))
    head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        metric_feature="global",
        inference_feature="visibility_weighted_parts",
        head_parts=(1, 4),
        part_pooling="semantic_parts",
        branch_metric=True,
    )

    head.train()
    logits, train_features = head(x)
    assert len(logits) == 5
    assert train_features["_visibility"].shape == (2, 4)
    assert {"global", "part0", "part1", "part2", "part3"} <= set(train_features)
    loss = sum(logit.mean() for logit in logits)
    loss = loss + train_features["_visibility"].mean()
    loss.backward()
    assert head.semantic_part_pool.pool.queries.grad is not None
    assert head.semantic_part_pool.pool.queries.grad.abs().sum() > 0

    head.eval()
    with torch.no_grad():
        eval_features = head(x)
    assert eval_features.shape == (2, 24)


def test_multibranch_head_supports_evidence_sinkhorn_packet():
    x = (torch.randn(2, 8, 4, 2), torch.randn(2, 8, 8, 2))
    head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        metric_feature="global",
        inference_feature="evidence_sinkhorn",
        head_parts=(1, 3),
        part_pooling="semantic_parts",
        branch_metric=True,
        evidence_num_roles=5,
    )

    head.train()
    logits, train_features = head(x)

    assert len(logits) == 4
    assert train_features["_visibility"].shape == (2, 3)
    assert train_features["_rarity"].shape == (2, 3)
    assert train_features["_role_logits"].shape == (2, 3, 5)
    assert train_features["_nullness"].shape == (2, 3)
    assert {"global", "part0", "part1", "part2"} <= set(train_features)

    loss = (
        sum(logit.mean() for logit in logits)
        + train_features["_visibility"].mean()
        + train_features["_rarity"].mean()
        + train_features["_role_logits"].mean()
        + train_features["_nullness"].mean()
    )
    loss.backward()

    assert head.semantic_part_pool.role_predictor.weight.grad is not None
    assert head.semantic_part_pool.role_predictor.weight.grad.abs().sum() > 0
    assert head.semantic_part_pool.null_predictor.weight.grad is not None
    assert head.semantic_part_pool.null_predictor.weight.grad.abs().sum() > 0

    head.eval()
    with torch.no_grad():
        eval_features = head(x)

    # [global(D), parts(K*D), visibility(K), rarity(K), roles(K*R), nullness(K)]
    assert eval_features.shape == (2, 40)


def test_dse_lite_pool_weights_tokens_without_changing_shape():
    pool = DSELitePool((3, 1))
    x = torch.zeros(1, 4, 6, 2)
    x[:, :, 3, :] = 2.0

    pooled = pool(x)

    assert pooled.shape == (1, 4, 3, 1)
    assert pooled[:, :, 1].mean() > pooled[:, :, 0].mean()


def test_multibranch_head_supports_dse_pool_and_mix_descriptor():
    x = torch.randn(2, 8, 6, 2)
    dse_head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_pool="dse",
    )
    dse_head.train()
    logits, features = dse_head(x)
    assert len(logits) == 3
    assert features.shape == (2, 12)

    mix_head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        metric_feature="dse_mix",
        inference_feature="dse_mix",
    )
    mix_head.train()
    _, train_features = mix_head(x)
    assert train_features.shape == (2, 24)

    mix_head.eval()
    with torch.no_grad():
        eval_features = mix_head(x)
    assert eval_features.shape == (2, 24)


def test_learned_part_tokens_receive_gradients():
    head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        part_pooling="tokens",
        num_part_tokens=4,
    )
    head.train()

    logits, features = head(torch.randn(2, 8, 6, 2))
    loss = features.square().mean() + sum(score.square().mean() for score in logits)
    loss.backward()

    assert head.part_token_pool.queries.grad is not None
    assert head.part_token_pool.queries.grad.abs().sum() > 0


def test_pattern_adapters_are_identity_at_initialization():
    head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        decouple_patterns=True,
        pattern_adapter_dim=4,
    )
    x = torch.randn(2, 8, 6, 2)

    torch.testing.assert_close(head.global_adapter(x), x)
    torch.testing.assert_close(head.local_adapter(x), x)


def test_multibranch_head_combines_part_tokens_and_pattern_adapters():
    head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        inference_feature="concat_bn",
        part_pooling="tokens",
        num_part_tokens=4,
        decouple_patterns=True,
        pattern_adapter_dim=4,
    )
    head.eval()

    with torch.no_grad():
        features = head(torch.randn(2, 8, 6, 2))

    assert features.shape == (2, 20)
    assert head.decouple_patterns is True
    assert head.part_pooling == "tokens"


def test_gpc_lite_head_has_global_three_part_and_two_channel_branches():
    head = GPCLiteMultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        metric_feature="raw_mean",
        inference_feature="norm_concat_bn",
        head_parts=(1, 3),
        branch_metric=True,
    )
    x = torch.randn(2, 8, 6, 2)

    head.train()
    logits, features = head(x)

    assert len(logits) == 6
    assert {"global", "part0", "part1", "part2", "ch0", "ch1"} <= set(features)
    torch.testing.assert_close(features["raw_mean"], features["global"])

    head.eval()
    with torch.no_grad():
        embedding = head(x)
    assert embedding.shape == (2, 24)
    torch.testing.assert_close(embedding.norm(dim=1), torch.ones(2))


def test_stripe_visibility_weights_local_descriptors_and_receives_gradients():
    head = MultiBranchHead(
        8,
        feat_dim=4,
        num_classes=3,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 3),
        stripe_visibility=True,
    )
    x = torch.randn(2, 8, 6, 2)

    pooled = head.part_pool_3(x)
    confidence = head.visibility_gate(pooled)
    torch.testing.assert_close(
        confidence,
        torch.full_like(confidence, 0.9),
        atol=1e-6,
        rtol=0.0,
    )

    head.train()
    logits, features = head(x)
    loss = features.square().mean() + sum(score.square().mean() for score in logits)
    loss.backward()

    assert len(logits) == 4
    assert features.shape == (2, 16)
    assert head.visibility_gate.predictor.weight.grad is not None
    assert head.visibility_gate.predictor.weight.grad.abs().sum() > 0

    head.eval()
    with torch.no_grad():
        head.visibility_gate.predictor.bias.fill_(torch.logit(torch.tensor(0.1)))
        low_visibility = head(x)
        head.visibility_gate.predictor.bias.fill_(torch.logit(torch.tensor(0.9)))
        high_visibility = head(x)
    assert not torch.allclose(low_visibility, high_visibility)


def test_trainer_builds_gpc_lite_and_visibility_heads(tmp_path):
    gpc_trainer = _trainer(
        tmp_path,
        pretrained=False,
        head_type="gpc_lite",
        head_parts=(1, 3),
        part_pooling="stripes",
        decouple_patterns=False,
        stripe_visibility=False,
        metric_feature="raw_mean",
    )
    gpc_model = gpc_trainer._build_model(num_classes=4)
    assert isinstance(gpc_model.head, GPCLiteMultiBranchHead)

    visibility_trainer = _trainer(
        tmp_path,
        pretrained=False,
        head_type="standard",
        head_parts=(1, 3),
        part_pooling="stripes",
        stripe_visibility=True,
    )
    visibility_model = visibility_trainer._build_model(num_classes=4)
    assert visibility_model.head.stripe_visibility is True


def test_trainer_effective_metric_feature_modes(tmp_path):
    assert _trainer(tmp_path, loss_type="triplet")._effective_metric_feature() == "raw_mean"
    assert _trainer(tmp_path, loss_type="ms")._effective_metric_feature() == "concat_bn"
    assert _trainer(tmp_path, loss_type="ms", metric_feature="raw_mean")._effective_metric_feature() == "raw_mean"
    assert _trainer(tmp_path, metric_feature="global")._effective_metric_feature() == "global"


def test_trainer_aux_ce_weight_preserves_default_and_can_drop(tmp_path):
    trainer = _trainer(tmp_path, aux_ce_weight=0.1, aux_ce_drop_epoch=2)
    logits = [
        torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
        torch.tensor([[0.0, 5.0], [5.0, 0.0]]),
    ]
    pids = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()

    before_drop = trainer._classification_loss_for_logits(criterion, logits, pids, epoch=2)
    after_drop = trainer._classification_loss_for_logits(criterion, logits, pids, epoch=3)

    assert before_drop > after_drop
    torch.testing.assert_close(after_drop, criterion(logits[0], pids))


def test_trainer_scale_balanced_ce_averages_within_each_scale_and_uses_every_output(tmp_path):
    trainer = _trainer(
        tmp_path,
        scale_balanced_branches=True,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
    )
    logits = [
        torch.full((2, 1), value, requires_grad=True)
        for value in (3.0, 1.0, 3.0, 2.0, 4.0, 6.0, 8.0)
    ]

    def criterion(logit, pids):
        del pids
        return logit.mean()

    loss = trainer._classification_loss_for_logits(
        criterion,
        logits,
        torch.tensor([0, 1]),
        epoch=1,
    )
    loss.backward()

    # (global=3 + mean(two-stripe)=2 + mean(four-stripe)=5) / 3
    torch.testing.assert_close(loss.detach(), torch.tensor(10.0 / 3.0))
    assert all(logit.grad is not None and torch.count_nonzero(logit.grad) > 0 for logit in logits)


def test_multiscale_channel_ce_allocates_power_inside_each_scale(
    tmp_path,
):
    trainer = _trainer(
        tmp_path,
        model_name="csl_tinyvit_11m",
        feature_fusion="global_final_parts_stage0_semantic_fine",
        head_type="multiscale_channel2",
        multiscale_channel_alpha=0.5,
        scale_balanced_branches=True,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
    )
    logits = [
        torch.full((2, 1), float(value), requires_grad=True)
        for value in range(1, 14)
    ]

    def criterion(logit, pids):
        del pids
        return logit.mean()

    loss = trainer._classification_loss_for_logits(
        criterion,
        logits,
        torch.tensor([0, 1]),
        epoch=1,
    )
    loss.backward()

    spatial_means = torch.tensor(
        [
            1.0,
            (2.0 + 3.0) / 2.0,
            (4.0 + 5.0 + 6.0 + 7.0) / 4.0,
        ]
    )
    channel_means = torch.tensor(
        [
            (8.0 + 9.0) / 2.0,
            (10.0 + 11.0) / 2.0,
            (12.0 + 13.0) / 2.0,
        ]
    )
    expected = (0.75 * spatial_means + 0.25 * channel_means).mean()
    torch.testing.assert_close(loss.detach(), expected)
    assert all(
        logit.grad is not None and torch.count_nonzero(logit.grad) > 0
        for logit in logits
    )


def test_trainer_scale_balanced_center_loss_uses_the_full_retrieval_structure(tmp_path):
    trainer = _trainer(
        tmp_path,
        scale_balanced_branches=True,
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
    )
    features = {
        "global": torch.randn(2, 32),
        "raw_mean": torch.randn(2, 32),
        "raw_concat": torch.randn(2, 96),
    }

    center_features = trainer._center_features(features)

    assert center_features is features["raw_concat"]
    assert center_features.shape == (2, 96)


def test_trainer_scale_balanced_global_metric_routes_center_loss_to_global(tmp_path):
    trainer = _trainer(
        tmp_path,
        scale_balanced_branches=True,
        metric_feature="global",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
    )
    features = {
        "global": torch.randn(2, 32),
        "raw_mean": torch.randn(2, 32),
        "raw_concat": torch.randn(2, 96),
    }

    center_features = trainer._center_features(features)

    assert center_features is features["global"]
    assert center_features.shape == (2, 32)


def test_trainer_scale_balanced_coarse_metric_routes_center_loss_to_coarse_descriptor(tmp_path):
    trainer = _trainer(
        tmp_path,
        scale_balanced_branches=True,
        metric_feature="coarse_concat",
        inference_feature="norm_concat_bn",
        head_parts=(1, 2, 4),
    )
    features = {
        "global": torch.randn(2, 32),
        "coarse_concat": torch.randn(2, 64),
        "raw_mean": torch.randn(2, 32),
        "raw_concat": torch.randn(2, 96),
    }

    center_features = trainer._center_features(features)

    assert center_features is features["coarse_concat"]
    assert center_features.shape == (2, 64)


def test_trainer_aux_ce_ignores_null_evidence_token_without_metadata_grad(tmp_path):
    trainer = _trainer(tmp_path, aux_ce_weight=1.0)
    logits = [
        torch.full((2, 2), 1.0, requires_grad=True),
        torch.full((2, 2), 10.0, requires_grad=True),
        torch.full((2, 2), 100.0, requires_grad=True),
    ]
    features = {
        "_visibility": torch.ones(2, 2, requires_grad=True),
        "_nullness": torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True),
    }

    def criterion(logit, pids):
        del pids
        return logit.mean()

    loss = trainer._classification_loss_for_logits(
        criterion,
        logits,
        torch.tensor([0, 1]),
        epoch=1,
        features=features,
    )
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(5.5))
    assert features["_visibility"].grad is None
    assert features["_nullness"].grad is None
    assert logits[2].grad.abs().sum() == 0


def test_trainer_branch_metric_includes_raw_concat_when_selected(tmp_path):
    trainer = _trainer(tmp_path, metric_feature="raw_concat", branch_aware_metric=True, branch_metric_part_weight=0.5)
    features = {
        "global": torch.randn(4, 8),
        "part0": torch.randn(4, 8),
        "part1": torch.randn(4, 8),
        "raw_mean": torch.randn(4, 8),
        "raw_concat": torch.randn(4, 24),
    }
    called_shapes = []

    def criterion(feature, pids):
        called_shapes.append(tuple(feature.shape))
        return feature.sum() * 0

    trainer._metric_loss_for_features(criterion, features, torch.tensor([0, 0, 1, 1]))

    assert called_shapes == [(4, 8), (4, 24), (4, 8), (4, 8)]


def test_trainer_uses_embedding_model_contract_for_margin_classifier(tmp_path):
    trainer = _trainer(tmp_path, loss_type="softmax", classifier_loss="arcface")

    assert trainer._model_loss_type() == "triplet"


def test_trainer_uses_embedding_model_contract_for_wrt(tmp_path):
    trainer = _trainer(tmp_path, loss_type="wrt")

    assert trainer._model_loss_type() == "triplet"
    assert trainer._effective_metric_feature() == "raw_mean"


def test_trainer_triplet_soft_margin_can_be_forced(tmp_path):
    assert _trainer(tmp_path, triplet_soft_margin=False)._use_soft_margin_triplet(default_soft_margin=True) is False
    assert _trainer(tmp_path, triplet_soft_margin=True)._use_soft_margin_triplet(default_soft_margin=False) is True


def test_trainer_eta_min_accepts_scientific_notation_string(tmp_path):
    trainer = _trainer(tmp_path, eta_min="1e-07")

    assert trainer.eta_min == 1e-07


def test_triplet_soft_margin_respects_margin_value():
    inputs = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    low_margin_loss = TripletLoss(margin=0.0, soft_margin=True)(inputs, targets)
    high_margin_loss = TripletLoss(margin=0.5, soft_margin=True)(inputs, targets)

    assert high_margin_loss > low_margin_loss


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


def test_resume_rejects_pre_v2_optimizer_contract(tmp_path):
    trainer = _trainer(tmp_path)
    saved_contract = trainer._resume_contract()
    del saved_contract["optimization"]["optimizer_contract_version"]
    checkpoint = {
        "resume_contract": saved_contract,
        "epoch": 1,
        "epochs": trainer.epochs,
    }

    with pytest.raises(ValueError, match="optimizer_contract_version"):
        trainer._assert_resume_compatible(checkpoint, tmp_path / "last.pt")


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


def test_resume_scheduler_inside_warmup_keeps_linear_warmup_lr(tmp_path):
    trainer = _trainer(tmp_path, epochs=200, warmup_epochs=20, eta_min=0.0, lr=7e-4)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    param = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [param],
                "lr": 3.5e-4,
                "initial_lr": 7e-4,
                "_base_lr": 7e-4,
            }
        ],
        lr=7e-4,
    )

    scheduler = trainer._build_resume_scheduler(
        optimizer,
        resumed_epoch=10,
        resume_path=run_dir / "last.pt",
        ckpt={"epochs": 200, "optimizer": {}},
    )

    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(7e-4)
    assert optimizer.param_groups[0]["_base_lr"] == pytest.approx(7e-4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.5e-4)


def test_resume_scheduler_restores_exact_saved_state(tmp_path):
    trainer = _trainer(tmp_path, epochs=200, warmup_epochs=20, eta_min=1e-7)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=7e-4)
    original = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=180,
        eta_min=1e-7,
    )
    for _ in range(7):
        optimizer.step()
        original.step()
    saved_state = original.state_dict()

    restored = trainer._build_resume_scheduler(
        optimizer,
        resumed_epoch=27,
        resume_path=run_dir / "last.pt",
        ckpt={"epochs": 200, "optimizer": {}, "scheduler": saved_state},
    )

    assert restored.state_dict() == saved_state
    assert optimizer.param_groups[0]["lr"] == pytest.approx(original.get_last_lr()[0])


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


def test_trainer_evidence_auxiliary_loss_is_finite_and_differentiable(tmp_path):
    trainer = _trainer(
        tmp_path,
        evidence_num_roles=5,
        evidence_alignment_loss_weight=0.2,
        evidence_null_loss_weight=0.1,
        evidence_diversity_loss_weight=0.05,
        evidence_sinkhorn_iters=3,
        evidence_sinkhorn_temperature=0.2,
    )
    pids = torch.tensor([0, 0, 1, 1])
    features = {
        "global": torch.randn(4, 8, requires_grad=True),
        "part0": torch.randn(4, 8, requires_grad=True),
        "part1": torch.randn(4, 8, requires_grad=True),
        "part2": torch.randn(4, 8, requires_grad=True),
        "_visibility": torch.full((4, 3), 0.9, requires_grad=True),
        "_rarity": torch.full((4, 3), 0.5, requires_grad=True),
        "_role_logits": torch.randn(4, 3, 5, requires_grad=True),
        "_nullness": torch.full((4, 3), 0.1, requires_grad=True),
    }

    loss = trainer._evidence_auxiliary_loss(features, pids)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert features["part0"].grad is not None
    assert features["_role_logits"].grad is not None
    assert features["_nullness"].grad is not None


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


def test_trainer_head_warmup_runs_after_backbone_freeze(tmp_path):
    trainer = _trainer(tmp_path, backbone_freeze_epochs=10, head_warmup_epochs=5)

    assert trainer._backbone_freeze_active(1) is True
    assert trainer._head_warmup_active(1) is False
    assert trainer._backbone_freeze_active(10) is True
    assert trainer._head_warmup_active(10) is False
    assert trainer._backbone_freeze_active(11) is False
    assert trainer._head_warmup_active(11) is True
    assert trainer._head_warmup_active(15) is True
    assert trainer._head_warmup_active(16) is False
    assert trainer._head_warmup_start_epoch() == 11


def test_epoch_warmup_sets_current_epoch_lr_without_one_epoch_lag(tmp_path):
    trainer = _trainer(tmp_path, warmup_epochs=4)
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([{"params": [parameter], "lr": 0.2}])
    optimizer.param_groups[0]["_base_lr"] = 0.2

    assert trainer._apply_epoch_warmup_lrs(optimizer, 1) is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert trainer._apply_epoch_warmup_lrs(optimizer, 2) is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert trainer._apply_epoch_warmup_lrs(optimizer, 4) is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
    assert trainer._apply_epoch_warmup_lrs(optimizer, 5) is False
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)


def test_train_epoch_restores_head_warmup_lrs_before_scheduler(tmp_path):
    trainer = _trainer(
        tmp_path,
        epochs=3,
        warmup_epochs=0,
        head_warmup_epochs=1,
        head_warmup_lr_mult=2.0,
        center_loss_weight=0.0,
    )

    class TinyWarmupModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            features = self.backbone(x.flatten(1))
            return [self.head(features)], features

    model = TinyWarmupModel()
    criterion_center = CenterLoss(num_classes=2, feat_dim=4)
    optimizer = torch.optim.SGD(
        [
            {"params": model.backbone.parameters(), "lr": 0.1, "is_backbone": True},
            {"params": model.head.parameters(), "lr": 0.2, "is_head": True},
        ],
        lr=0.1,
    )
    optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loader = [(torch.randn(2, 1, 2, 2), torch.tensor([0, 1]), torch.zeros(2, dtype=torch.long))]

    metrics = trainer._train_epoch(
        1,
        model,
        loader,
        nn.CrossEntropyLoss(),
        None,
        criterion_center,
        optimizer,
        optimizer_center,
        scheduler,
    )

    assert metrics.backbone_lr == 0.0
    assert metrics.head_lr == pytest.approx(0.4)
    assert metrics.forward_elapsed_s > 0.0
    assert metrics.forward_elapsed_s <= metrics.elapsed_s
    assert optimizer.param_groups[0]["lr"] > 0.0
    assert optimizer.param_groups[1]["lr"] > 0.0


def test_train_epoch_rgb_control_does_not_run_pose_teacher_lifecycle(tmp_path):
    trainer = _trainer(
        tmp_path,
        epochs=3,
        warmup_epochs=0,
        center_loss_weight=0.0,
        anatomical_auxiliary=False,
        anatomical_multiscale=False,
        anatomical_target_type="learned_pose_concat_ema",
    )

    class TinyRGBModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 4)
            self.classifier = nn.Linear(4, 2)

        def forward(self, x):
            features = self.encoder(x.flatten(1))
            return self.classifier(features), features

    model = TinyRGBModel()
    criterion_center = CenterLoss(num_classes=2, feat_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    loader = [
        (
            torch.randn(2, 1, 2, 2),
            torch.tensor([0, 1]),
            torch.zeros(2, dtype=torch.long),
        )
    ]

    metrics = trainer._train_epoch(
        1,
        model,
        loader,
        nn.CrossEntropyLoss(),
        None,
        criterion_center,
        optimizer,
        optimizer_center,
        scheduler,
    )

    assert metrics.loss > 0


def test_center_optimizer_update_is_invariant_to_model_loss_scale(tmp_path):
    """Branch aggregation may scale model gradients, not the center-table LR."""

    class TinyCenterModel(nn.Module):
        def __init__(self, center_scale: float):
            super().__init__()
            self.encoder = nn.Linear(4, 4)
            self.classifier = nn.Linear(4, 2)
            self.center_scale = center_scale

        def forward(self, x):
            features = self.encoder(x.flatten(1))
            return self.classifier(features), {
                "global": features,
                "_center_features": (features,),
                "_center_loss_scale": self.center_scale,
            }

    images = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[3.0, 2.0], [1.0, 0.0]]],
        ]
    )
    labels = torch.tensor([0, 1])
    cameras = torch.zeros(2, dtype=torch.long)

    def center_after_one_step(center_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(23)
        trainer = _trainer(
            tmp_path,
            epochs=3,
            warmup_epochs=0,
            center_loss_weight=5e-4,
        )
        model = TinyCenterModel(center_scale)
        criterion_center = CenterLoss(num_classes=2, feat_dim=4)
        initial_centers = criterion_center.centers.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

        trainer._train_epoch(
            1,
            model,
            [(images, labels, cameras)],
            nn.CrossEntropyLoss(),
            None,
            criterion_center,
            optimizer,
            optimizer_center,
            scheduler,
        )
        return initial_centers, criterion_center.centers.detach().clone()

    initial, scale_one = center_after_one_step(1.0)
    _, scale_eleven = center_after_one_step(11.0)

    assert not torch.allclose(scale_one, initial)
    assert torch.allclose(scale_one, scale_eleven, atol=1e-7, rtol=1e-6)


def test_train_epoch_applies_pav_clean_view_id_and_consistency(tmp_path):
    trainer = _trainer(
        tmp_path,
        epochs=3,
        warmup_epochs=0,
        center_loss_weight=0.0,
        pav_mosaic=True,
        pav_metadata_dir=str(tmp_path / "pose"),
        pav_mosaic_decay_start_epoch=0,
        pav_mosaic_min_unaltered=0.5,
        pav_consistency_weight=0.2,
    )

    class TinyPAVModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 4)
            self.classifier = nn.Linear(4, 2)
            self.forward_calls = 0

        def forward(self, x):
            self.forward_calls += 1
            features = self.encoder(x.flatten(1))
            return self.classifier(features), {
                "raw_concat": features,
                "norm_concat_bn": features,
            }

    model = TinyPAVModel()
    criterion_center = CenterLoss(num_classes=2, feat_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    mosaic = torch.randn(4, 1, 2, 2)
    clean = mosaic + 0.5
    loader = [
        (
            mosaic,
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 1, 1, 0]),
            clean,
            torch.tensor([True, True, True, False]),
        )
    ]

    metrics = trainer._train_epoch(
        1,
        model,
        loader,
        nn.CrossEntropyLoss(),
        None,
        criterion_center,
        optimizer,
        optimizer_center,
        scheduler,
    )

    assert model.forward_calls == 2
    assert metrics.pav_consistency_loss > 0
    assert metrics.loss > metrics.id_loss


def test_pav_clean_view_is_skipped_when_batch_overflow_is_negligible(
    tmp_path,
):
    trainer = _trainer(
        tmp_path,
        epochs=3,
        warmup_epochs=0,
        pav_mosaic=True,
        pav_metadata_dir=str(tmp_path / "pose"),
        pav_mosaic_decay_start_epoch=0,
        pav_mosaic_probability=0.25,
        pav_mosaic_min_unaltered=0.5,
        pav_consistency_weight=0.0,
    )

    assert trainer._pav_requires_clean_view(96) is False


def test_pav_clean_view_is_kept_for_consistency_or_likely_reversion(
    tmp_path,
):
    trainer = _trainer(
        tmp_path,
        epochs=3,
        warmup_epochs=0,
        pav_mosaic=True,
        pav_metadata_dir=str(tmp_path / "pose"),
        pav_mosaic_decay_start_epoch=0,
        pav_mosaic_probability=0.25,
        pav_mosaic_min_unaltered=0.5,
        pav_consistency_weight=0.2,
    )
    assert trainer._pav_requires_clean_view(96) is True

    trainer.pav_consistency_weight = 0.0
    trainer.pav_mosaic_probability = 0.5
    assert trainer._pav_requires_clean_view(96) is True

    trainer.pav_mosaic_probability = 0.25
    trainer.background_mosaic = True
    trainer.background_mosaic_probability = 0.3
    assert trainer._pav_requires_clean_view(96) is True


def test_train_epoch_reverts_single_pav_pair_before_bn_clean_forward(tmp_path):
    trainer = _trainer(
        tmp_path,
        epochs=3,
        warmup_epochs=0,
        center_loss_weight=0.0,
        pav_mosaic=True,
        pav_metadata_dir=str(tmp_path / "pose"),
        pav_mosaic_decay_start_epoch=0,
        pav_mosaic_min_unaltered=0.5,
        pav_consistency_weight=0.2,
    )

    class TinyBNPAVModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 4)
            self.bn = nn.BatchNorm1d(4)
            self.classifier = nn.Linear(4, 2)
            self.forward_calls = 0
            self.last_input = None

        def forward(self, x):
            self.forward_calls += 1
            self.last_input = x.detach().clone()
            features = self.bn(self.encoder(x.flatten(1)))
            return self.classifier(features), {
                "raw_concat": features,
                "norm_concat_bn": features,
            }

    model = TinyBNPAVModel()
    criterion_center = CenterLoss(num_classes=2, feat_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    mosaic = torch.randn(4, 1, 2, 2)
    clean = mosaic.clone()
    mosaic[0] += 0.5
    loader = [
        (
            mosaic,
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 1, 1, 0]),
            clean,
            torch.tensor([True, False, False, False]),
        )
    ]

    metrics = trainer._train_epoch(
        1,
        model,
        loader,
        nn.CrossEntropyLoss(),
        None,
        criterion_center,
        optimizer,
        optimizer_center,
        scheduler,
    )

    assert model.forward_calls == 1
    assert torch.equal(model.last_input[0], clean[0])
    assert metrics.pav_consistency_loss == 0


def test_save_metrics_records_average_forward_time_beside_epoch_time(tmp_path):
    trainer = _trainer(tmp_path, epochs=30)

    trainer._save_metrics(
        tmp_path,
        history=[],
        val_history=[],
        best_epoch=0,
        best_mAP=0.0,
        best_rank1=0.0,
        average_epoch_time_s=12.34567,
        average_forward_time_s=4.56789,
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    keys = list(metrics)
    assert metrics["average_epoch_time_s"] == 12.3457
    assert metrics["average_forward_time_s"] == 4.5679
    assert keys.index("average_forward_time_s") == keys.index("average_epoch_time_s") + 1


def test_training_time_estimator_separates_epochs_evaluations_and_phases():
    estimator = _TrainingTimeEstimator(total_epochs=200, eval_interval=10)

    for duration in (30.0, 31.0, 32.0):
        estimator.add_epoch(duration, phase="backbone_frozen")
    assert estimator.epoch_duration_s == 31.0

    # A phase transition discards the faster frozen-backbone samples.
    for duration in (45.0, 46.0, 47.0):
        estimator.add_epoch(duration, phase="full_model")
    for duration in (70.0, 72.0, 300.0):
        estimator.add_evaluation(duration)

    assert estimator.epoch_duration_s == 46.0
    assert estimator.evaluation_duration_s == 72.0
    assert estimator.remaining_evaluations(52) == 15
    assert estimator.estimate_remaining_s(52) == pytest.approx(148 * 46.0 + 15 * 72.0)
    assert ReIDTrainer._format_eta(estimator.estimate_remaining_s(52)) == "2h 11m"


def test_training_time_estimator_counts_final_unscheduled_evaluation_and_resume_fallbacks():
    estimator = _TrainingTimeEstimator(
        total_epochs=25,
        eval_interval=10,
        fallback_epoch_s=43.37,
        fallback_eval_s=72.59,
    )

    assert estimator.remaining_evaluations(0) == 3
    assert estimator.remaining_evaluations(10) == 2
    assert estimator.remaining_evaluations(20) == 1
    assert estimator.remaining_evaluations(25) == 0
    assert estimator.estimate_remaining_s(20) == pytest.approx(5 * 43.37 + 72.59)


def test_trainer_restores_timing_fallbacks_for_resumed_eta(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.resume = tmp_path / "last.pt"
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "average_epoch_time_s": 43.3714,
                "average_eval_time_s": 72.592,
            }
        )
    )

    assert trainer._restore_timing_averages(tmp_path) == (43.3714, 72.592)


def test_ema_update_uses_names_and_ignores_dynamic_nonpersistent_buffers(tmp_path):
    trainer = _trainer(tmp_path)

    class DynamicBufferModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cache_owner = nn.Module()
            self.cache_owner.register_buffer("indexes", torch.arange(2), persistent=False)
            self.linear = nn.Linear(3, 3, bias=False)
            self.bn = nn.BatchNorm1d(128)

    live_model = DynamicBufferModel()
    ema_model = DynamicBufferModel()
    ema_model.load_state_dict(live_model.state_dict())
    live_model.cache_owner.register_buffer("ab", torch.ones(2, 48, 48), persistent=False)

    with torch.no_grad():
        live_model.linear.weight.fill_(4.0)
        ema_model.linear.weight.zero_()
        live_model.bn.running_mean.fill_(2.0)
        ema_model.bn.running_mean.zero_()
        live_model.bn.num_batches_tracked.fill_(7)
        ema_model.bn.num_batches_tracked.zero_()

    assert any(
        live_buffer.shape != ema_buffer.shape
        for live_buffer, ema_buffer in zip(live_model.buffers(), ema_model.buffers())
    )

    trainer._update_ema_model(ema_model, live_model, decay=0.5)

    torch.testing.assert_close(ema_model.linear.weight, torch.full_like(ema_model.linear.weight, 2.0))
    torch.testing.assert_close(ema_model.bn.running_mean, torch.ones_like(ema_model.bn.running_mean))
    assert ema_model.bn.num_batches_tracked.item() == 7
    assert not hasattr(ema_model.cache_owner, "ab")


def test_ema_decay_ramp_forgets_random_initialization_before_first_validation(tmp_path):
    trainer = _trainer(tmp_path)
    target_decay = 0.999

    assert trainer._ema_decay_for_update(target_decay, 1) == pytest.approx(2 / 11)
    assert trainer._ema_decay_for_update(target_decay, 620) == pytest.approx(621 / 630)
    assert trainer._ema_decay_for_update(target_decay, 10_000) == target_decay

    ema_value = 0.0
    for update in range(1, 621):
        decay = trainer._ema_decay_for_update(target_decay, update)
        ema_value = decay * ema_value + (1.0 - decay)

    assert ema_value > 0.999999


def test_ema_decay_ramp_rejects_nonpositive_updates(tmp_path):
    trainer = _trainer(tmp_path)

    with pytest.raises(ValueError, match="EMA update must be positive"):
        trainer._ema_decay_for_update(0.999, 0)


def test_trainer_reid_lrd_uses_requested_stage_lr_scales(tmp_path):
    trainer = _trainer(tmp_path, vit_lr_profile="reid_lrd")

    assert trainer._vit_lr_scale_for_param("head.bn_global.weight", depth=4) == 1.0
    assert trainer._vit_lr_scale_for_param("feature_fusion_module.projections.2.weight", depth=4) == 1.0
    assert trainer._vit_lr_scale_for_param("layers.3.blocks.0.attn.qkv.weight", depth=4) == 0.5
    assert trainer._vit_lr_scale_for_param("layers.2.blocks.0.attn.qkv.weight", depth=4) == 0.25
    assert trainer._vit_lr_scale_for_param("layers.1.blocks.0.attn.qkv.weight", depth=4) == 0.1
    assert trainer._vit_lr_scale_for_param("layers.0.blocks.0.conv1.c.weight", depth=4) == 0.05
    assert trainer._vit_lr_scale_for_param("patch_embed.seq.0.c.weight", depth=4) == 0.05
    assert trainer._vit_lr_scale_for_param("layers.2.reid_adapters.0.gamma", depth=4) == 1.0


def test_trainer_layer_decay_is_configurable_and_stage_based(tmp_path):
    trainer = _trainer(
        tmp_path,
        vit_lr_profile="layer_decay",
        layer_decay=0.8,
    )

    assert trainer._vit_lr_scale_for_param("patch_embed.seq.0.c.weight", depth=4) == pytest.approx(0.8**5)
    assert trainer._vit_lr_scale_for_param("layers.0.blocks.0.conv1.c.weight", depth=4) == pytest.approx(0.8**4)
    assert trainer._vit_lr_scale_for_param("layers.3.blocks.1.attn.qkv.weight", depth=4) == pytest.approx(0.8)
    assert trainer._vit_lr_scale_for_param("head.bn_global.weight", depth=4) == 1.0


def test_trainer_vit_no_wd_is_shape_and_module_aware(tmp_path):
    trainer = _trainer(tmp_path, weight_decay=0.1)

    class CollisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(4, 4, bias=False)
            self.blocks = nn.ModuleList()
            self.head = nn.Module()
            self.head.bn_global = nn.Module()
            self.head.bn_global.reduction = nn.Linear(4, 4, bias=False)
            self.head.bn_global.classifier = nn.Linear(4, 3, bias=False)
            self.head.norm = nn.LayerNorm(4)
            self.head.raw_p = nn.Parameter(torch.zeros(1))
            self.head.residual_scale = nn.Parameter(torch.zeros(()))
            self.attention_biases = nn.Parameter(torch.zeros(2, 3))

    model = CollisionModel()
    groups_by_parameter = {
        id(parameter): group
        for group in trainer._build_vit_param_groups(model)
        for parameter in group["params"]
    }
    named = dict(model.named_parameters())

    for name in (
        "head.bn_global.reduction.weight",
        "head.bn_global.classifier.weight",
    ):
        assert groups_by_parameter[id(named[name])]["weight_decay"] == pytest.approx(0.1)
    for name in (
        "head.norm.weight",
        "head.norm.bias",
        "head.raw_p",
        "head.residual_scale",
        "attention_biases",
    ):
        assert groups_by_parameter[id(named[name])]["weight_decay"] == 0.0


def test_trainer_exempts_reid_adapter_gammas_from_weight_decay(tmp_path):
    trainer = _trainer(tmp_path)
    model = csl_tinyvit_7m(
        num_classes=4,
        pretrained=False,
        reid_adapter_stages=(3,),
        reid_adapter_reduction=4,
        reid_adapter_suppression_tau=0.7,
    )

    parameter_groups = {
        id(parameter): group
        for group in trainer._build_vit_param_groups(model)
        for parameter in group["params"]
    }
    adapters = model.layers[3].reid_adapters
    assert adapters
    assert all(
        parameter_groups[id(adapter.gamma)]["weight_decay"] == 0.0
        for adapter in adapters
    )
    assert parameter_groups[id(adapters[0].adapter[0].weight)][
        "weight_decay"
    ] == pytest.approx(trainer.weight_decay)


def test_trainer_preserves_vit_param_grouping_unless_gradual_unfreeze(tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList()
            self.other = nn.Linear(1, 1, bias=False)
            self.head = nn.Linear(1, 1, bias=False)

    model = TinyModel()
    default_trainer = _trainer(tmp_path)
    gradual_trainer = _trainer(tmp_path, gradual_unfreeze=True)

    default_groups = default_trainer._build_vit_param_groups(model)
    gradual_groups = gradual_trainer._build_vit_param_groups(model)

    mixed_default_groups = [
        group
        for group in default_groups
        if group.get("lr_scale") == 1.0 and group.get("weight_decay") == default_trainer.weight_decay
    ]
    assert len(mixed_default_groups) == 1
    assert mixed_default_groups[0]["is_head"] is True
    assert mixed_default_groups[0]["is_backbone"] is True

    split_gradual_groups = [
        group
        for group in gradual_groups
        if group.get("lr_scale") == 1.0 and group.get("weight_decay") == gradual_trainer.weight_decay
    ]
    assert len(split_gradual_groups) == 2
    assert sorted(group["is_backbone"] for group in split_gradual_groups) == [False, True]


def test_trainer_backbone_freeze_keeps_reid_modules_trainable(tmp_path):
    trainer = _trainer(tmp_path)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(1, 1)
            self.layers = nn.ModuleList(
                [
                    nn.Linear(1, 1),
                    nn.Linear(1, 1),
                    nn.ModuleDict({"reid_adapters": nn.ModuleList([nn.Linear(1, 1)])}),
                ]
            )
            self.feature_fusion_module = nn.Linear(1, 1)
            self.neck = nn.Linear(1, 1)
            self.head = nn.Linear(1, 1)

    model = TinyModel()

    trainer._set_backbone_freeze_trainability(model, True)
    assert not model.patch_embed.weight.requires_grad
    assert not model.layers[0].weight.requires_grad
    assert not model.layers[1].weight.requires_grad
    assert model.patch_embed.training is False
    assert model.layers.training is False
    assert model.layers[2]["reid_adapters"][0].weight.requires_grad
    assert model.feature_fusion_module.weight.requires_grad
    assert model.neck.weight.requires_grad
    assert model.head.weight.requires_grad
    assert model.feature_fusion_module.training is True

    trainer._set_backbone_freeze_trainability(model, False)
    assert all(param.requires_grad for param in model.parameters())


def test_mobilenetv4_spatial_adapt_norm_updates_only_head_norm_during_freeze(
    monkeypatch,
    tmp_path,
):
    _install_fake_timm(monkeypatch)
    trainer = _trainer(tmp_path, model_name="mobilenetv4_conv_small")
    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
        timm_head_mode="spatial_adapt_norm",
    )

    model.train()
    trainer._set_backbone_freeze_trainability(model, True)

    assert model.backbone.training is False
    assert model.backbone.norm_head.training is True
    assert all(param.requires_grad for param in model.backbone.norm_head.parameters())
    assert not model.backbone.conv_head.weight.requires_grad
    assert not model.backbone.blocks[0].weight.requires_grad
    assert model.neck[0].weight.requires_grad
    assert next(model.head.parameters()).requires_grad


def test_mobilenetv4_backbone_lr_multiplier_only_scales_pretrained_params(
    monkeypatch,
    tmp_path,
):
    _install_fake_timm(monkeypatch)
    trainer = _trainer(
        tmp_path,
        model_name="mobilenetv4_conv_small",
        lr=4e-4,
        backbone_lr_mult=0.25,
    )
    model = mobilenetv4_conv_small(
        num_classes=4,
        loss="triplet",
        pretrained=False,
    )

    groups = trainer._build_mobilenetv4_param_groups(model)
    backbone_groups = [group for group in groups if group["is_backbone"]]
    reid_groups = [group for group in groups if group["is_head"]]

    assert backbone_groups
    assert reid_groups
    assert all(group["lr"] == pytest.approx(1e-4) for group in backbone_groups)
    assert all(group["lr"] == pytest.approx(4e-4) for group in reid_groups)


def test_trainer_gradual_unfreeze_stages_trainability_and_backbone_lr(tmp_path):
    trainer = _trainer(
        tmp_path,
        gradual_unfreeze=True,
        gradual_unfreeze_head_epochs=5,
        gradual_unfreeze_stage_epochs=10,
        gradual_unfreeze_backbone_lr_mult=0.1,
        gradual_unfreeze_backbone_lr_epochs=5,
    )

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(1, 1)
            self.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(4)])
            self.layers[3].attn = nn.Module()
            self.layers[3].attn.attention_biases = nn.Parameter(torch.ones(1))
            self.feature_fusion_module = nn.Linear(1, 1)
            self.neck = nn.Linear(1, 1)
            self.head = nn.Linear(1, 1)

    model = TinyModel()

    trainer._set_gradual_unfreeze_trainability(model, "head")
    assert not model.patch_embed.weight.requires_grad
    assert not model.layers[3].weight.requires_grad
    assert model.feature_fusion_module.weight.requires_grad
    assert model.neck.weight.requires_grad
    assert model.head.weight.requires_grad
    assert model.patch_embed.training is False
    assert model.layers[0].training is False
    assert model.layers[3].training is False

    trainer._set_gradual_unfreeze_trainability(model, "stage")
    assert not model.patch_embed.weight.requires_grad
    assert not model.layers[2].weight.requires_grad
    assert model.layers[3].weight.requires_grad
    assert not model.layers[3].attn.attention_biases.requires_grad
    assert model.feature_fusion_module.weight.requires_grad
    assert model.neck.weight.requires_grad
    assert model.head.weight.requires_grad
    assert model.patch_embed.training is False
    assert model.layers[2].training is False
    assert model.layers[3].training is True

    optimizer = torch.optim.SGD(
        [
            {"params": [model.layers[0].weight], "lr": 1.0, "is_backbone": True},
            {"params": [model.layers[3].weight], "lr": 1.0, "is_backbone": True},
            {"params": [model.head.weight], "lr": 2.0, "is_head": True, "is_backbone": False},
        ],
        lr=1.0,
    )
    original_lrs = trainer._apply_gradual_backbone_lrs(optimizer, epoch=6)

    assert original_lrs == [1.0, 1.0, 2.0]
    assert [group["lr"] for group in optimizer.param_groups] == [0.0, 0.1, 2.0]
    assert trainer._optimizer_lr_summary(optimizer) == (2.0, 0.1, 2.0)
    for group, lr in zip(optimizer.param_groups, original_lrs):
        group["lr"] = lr

    trainer._set_gradual_unfreeze_trainability(model, "full")
    assert all(param.requires_grad for param in model.parameters())
    assert model.layers[3].attn.attention_biases.requires_grad
    original_lrs = trainer._apply_gradual_backbone_lrs(optimizer, epoch=11)
    assert original_lrs == [1.0, 1.0, 2.0]
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.1, 2.0]
    assert trainer._gradual_unfreeze_phase(5) == "head"
    assert trainer._gradual_unfreeze_phase(6) == "stage"
    assert trainer._gradual_unfreeze_phase(11) == "full"
    assert trainer._gradual_backbone_lr_active(5) is False
    assert trainer._gradual_backbone_lr_active(6) is True
    assert trainer._gradual_backbone_lr_active(11) is True
    assert trainer._gradual_backbone_lr_active(15) is True
    assert trainer._gradual_backbone_lr_active(16) is False
