import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import boxmot.reid.backbones.lmbn_ain_n as lmbn_ain_n_module
import boxmot.reid.backbones.lmbn_n as lmbn_n_module
from boxmot.engine.reid import trainer as workflow_trainer
from boxmot.reid.backbones.families.csl_tinyvit import (
    Attention,
    CSLTinyViTFeatureFusion,
    DSELitePool,
    GeM,
    GPCLiteMultiBranchHead,
    LMBNStyleMultiBranchHead,
    MultiBranchHead,
    PostFusionLocalMixer,
    ReIDResidualAdapter,
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
from boxmot.reid.training.losses import ArcFaceLoss, CenterLoss, CircleLoss, CosFaceLoss, TripletLoss
from boxmot.reid.training.trainer import (
    DatasetBundle,
    LoaderBundle,
    LossBundle,
    ModelBundle,
    OptimizationBundle,
    ReIDTrainer,
    ValMetrics,
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
        self.blocks = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=1, bias=False) for _ in range(4)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(channels[-1], 192, kernel_size=1, bias=False)
        self.norm_head = nn.BatchNorm2d(192)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Identity()

    def _features(self, x):
        base = x.mean(dim=1, keepdim=True)
        outputs = []
        for index, channels in enumerate(self.feature_info.channels(), start=1):
            height = max(x.shape[-2] // (2**index), 1)
            width = max(x.shape[-1] // (2**index), 1)
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
    )

    trainer = ReIDTrainer.from_config(config)

    assert trainer.model_name == "csl_tinyvit_7m"
    assert trainer.train_batch_size == 48
    assert trainer.eval_batch_size == 96
    assert trainer.seed == 7
    assert trainer.center_loss_weight == 0.0
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
    assert ckpt["checkpoint_precision"] == "float16"
    assert ckpt["center_loss_state_dict"]["centers"].dtype == torch.float16
    torch.testing.assert_close(ckpt["center_loss_state_dict"]["centers"], expected_centers.half())
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
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_precision"] == "float16"
    torch.testing.assert_close(checkpoint["state_dict"]["weight"], live_model.weight.half())
    torch.testing.assert_close(checkpoint["ema_state_dict"]["weight"], ema_model.weight.half())
    assert checkpoint["state_dict"]["weight"].dtype == torch.float16
    assert checkpoint["ema_state_dict"]["weight"].dtype == torch.float16
    optimizer_tensors = [
        tensor
        for state in checkpoint["optimizer"]["state"].values()
        for tensor in state.values()
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor)
    ]
    assert optimizer_tensors
    assert {tensor.dtype for tensor in optimizer_tensors} == {torch.float16}
    assert checkpoint["checkpoint_type"] == "last"
    assert checkpoint["resumable"] is True
    assert checkpoint["best_mAP"] == 0.7


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

    assert checkpoint["state_dict"]["backbone.weight"].dtype == torch.float16
    assert checkpoint["state_dict"]["classifier.weight"].dtype == torch.float16


def test_best_checkpoint_records_metric_and_is_weights_only(tmp_path):
    trainer = _trainer(tmp_path, classifier_loss="arcface")
    model = _ClassifierToyModel()
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
    assert checkpoint["best_mAP"] == validation.mAP
    assert checkpoint["mAP"] == validation.mAP
    assert "optimizer" not in checkpoint
    assert "optimizer_center" not in checkpoint
    assert "center_loss_state_dict" not in checkpoint
    assert "classifier_loss_state_dict" not in checkpoint
    assert "ema_state_dict" not in checkpoint
    assert "rng_state" not in checkpoint


def test_weights_only_resume_syncs_ema_and_falls_back_to_map(tmp_path):
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
    live_classifier_before = live_model.classifier.weight.detach().clone()
    ema_classifier_before = ema_model.classifier.weight.detach().clone()
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

    state = trainer._restore_if_needed(model_bundle, loaders, losses, optimization)

    torch.testing.assert_close(live_model.backbone.weight, source_model.backbone.weight)
    torch.testing.assert_close(ema_model.backbone.weight, source_model.backbone.weight)
    torch.testing.assert_close(live_model.classifier.weight, live_classifier_before)
    torch.testing.assert_close(ema_model.classifier.weight, ema_classifier_before)
    assert live_model.backbone.weight.dtype == torch.float32
    assert ema_model.backbone.weight.dtype == torch.float32
    assert state.start_epoch == 5
    assert state.best_mAP == 0.77
    assert state.best_rank1 == 0.88


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
    assert final_map.shape == (2, 512, 1, 1)


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
    for fusion in ("global_final_parts_stage2", "late_concat_stage2"):
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
    assert set(branch_features) == {"global", "part0", "part1", "raw_mean", "raw_concat", "concat_bn", "norm_concat_bn"}
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
    assert optimizer.param_groups[0]["lr"] > 0.0
    assert optimizer.param_groups[1]["lr"] > 0.0


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
