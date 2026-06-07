import json
from types import SimpleNamespace

import torch

from boxmot.engine.reid import trainer as engine_trainer
from boxmot.reid.backbones.csl_tinyvit import MultiBranchHead, csl_tinyvit_5m
from boxmot.reid.training.trainer import ReIDTrainer


def _trainer(tmp_path, **kwargs):
    params = {
        "model_name": "csl_tinyvit_5m",
        "dataset_name": "market1501",
        "data_dir": str(tmp_path),
        "lr": 3.5e-4,
        "weight_decay": 5e-4,
        "center_loss_weight": 5e-4,
    }
    params.update(kwargs)
    return ReIDTrainer(**params)


def test_vit_defaults_apply_to_implicit_training_values(tmp_path):
    trainer = _trainer(tmp_path)

    trainer._apply_vit_training_defaults()

    assert trainer.lr == 7e-4
    assert trainer.weight_decay == 0.05
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
                "model_name": "csl_tinyvit_5m",
                "dataset": "market1501",
                "data_dir": str(tmp_path),
                "loss_type": "triplet",
                "lr": 7e-4,
                "center_loss_weight": 5e-3,
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
        model="csl_tinyvit_5m",
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
    assert captured["explicit_hparams"] == {"lr", "center_loss_weight"}


def test_csl_tinyvit_metric_feature_mode_follows_loss():
    triplet_model = csl_tinyvit_5m(num_classes=4, loss="triplet", pretrained=False)
    ms_model = csl_tinyvit_5m(num_classes=4, loss="ms", pretrained=False)

    assert triplet_model.head.metric_feature == "raw_mean"
    assert ms_model.head.metric_feature == "concat_bn"


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


def test_trainer_effective_metric_feature_modes(tmp_path):
    assert _trainer(tmp_path, loss_type="triplet")._effective_metric_feature() == "raw_mean"
    assert _trainer(tmp_path, loss_type="ms")._effective_metric_feature() == "concat_bn"
    assert _trainer(tmp_path, loss_type="ms", metric_feature="raw_mean")._effective_metric_feature() == "raw_mean"
