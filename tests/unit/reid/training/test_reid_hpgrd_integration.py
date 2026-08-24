from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from boxmot.engine.config import build_mode_namespace
from boxmot.reid.datasets.base import ReIDSample
from boxmot.reid.datasets.torch_dataset import ReIDImageDataset
from boxmot.reid.training.checkpoint import CheckpointManager, _is_train_only_model_key
from boxmot.reid.training.config import ReIDTrainConfig, trainer_kwargs_from_args
from boxmot.reid.training.losses import CenterLoss
from boxmot.reid.training.trainer import ReIDTrainer
from boxmot.reid.training.trainer_components.global_ap import IdentityGlobalAP
from boxmot.reid.training.trainer_components.hpgrd_integration import (
    _HumanPrivilegedRetrievalMixin,
)
from boxmot.reid.training.trainer_components.privileged_graph import (
    PrivilegedGraphLoss,
    PrivilegedGraphTeacherCache,
)


class _IntegrationHarness(_HumanPrivilegedRetrievalMixin):
    def __init__(self):
        self.global_ap_start_epoch = 20
        self.global_ap_ramp_end_epoch = 50
        self.global_ap_decay_start_epoch = 130
        self.global_ap_decay_end_epoch = 170
        self.hpgrd_global_weight = 0.0
        self.hpgrd_part_weight = 0.0
        self.hpgrd_background_weight = 0.0
        self.hpgrd_part_drop_weight = 0.05
        self.hpgrd_part_drop_probability = 1.0
        self.hpgrd_min_confidence = 0.05


@pytest.mark.parametrize(
    ("epoch", "expected"),
    ((20, 0.0), (35, 0.5), (50, 1.0), (130, 1.0), (150, 0.5), (170, 0.0)),
)
def test_privileged_retrieval_schedule_warms_holds_and_retires(epoch, expected):
    harness = _IntegrationHarness()
    assert harness._retrieval_auxiliary_schedule_scale(epoch) == pytest.approx(expected)


def test_semantic_drop_keeps_repeated_pid_positive_groups_without_camera_metadata():
    torch.manual_seed(4)
    harness = _IntegrationHarness()
    images = torch.ones(4, 3, 8, 4)
    pids = torch.tensor([0, 0, 1, 1])
    masks = torch.zeros(4, 2, 8, 4)
    masks[:, 0, :4] = 1
    masks[:, 1, 4:] = 1
    targets = {
        "masks": masks,
        "reliability": torch.ones(4, 2),
    }

    dropped, base_indices, parts, confidence = harness._build_hpgrd_semantic_drop_view(
        images,
        pids,
        targets,
    )

    assert dropped is not None
    assert dropped.shape == images.shape
    assert base_indices.tolist() == [0, 1, 2, 3]
    assert parts.unique().numel() == 1
    assert confidence.tolist() == [1.0] * 4
    assert bool((dropped == 0).any())
    assert set(pids[base_indices].tolist()) == {0, 1}
    assert all((pids[base_indices] == pid).sum().item() == 2 for pid in (0, 1))


def test_semantic_drop_rejects_identity_sets_without_a_repeated_pid():
    harness = _IntegrationHarness()
    images = torch.ones(4, 3, 8, 4)
    pids = torch.arange(4)
    targets = {
        "masks": torch.ones(4, 1, 8, 4),
        "reliability": torch.ones(4, 1),
    }

    result = harness._build_hpgrd_semantic_drop_view(images, pids, targets)

    assert result == (None, None, None, None)


def test_intervention_forward_preserves_batchnorm_state_and_gradients():
    harness = _IntegrationHarness()
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
    )
    model.train()
    before_mean = model[-1].running_mean.clone()
    before_var = model[-1].running_var.clone()
    before_batches = model[-1].num_batches_tracked.clone()

    output = harness._hpgrd_intervention_forward(
        model,
        torch.randn(3, 1, 2, 2),
        detached=False,
    )

    assert output.requires_grad
    assert model[-1].training is True
    torch.testing.assert_close(model[-1].running_mean, before_mean)
    torch.testing.assert_close(model[-1].running_var, before_var)
    torch.testing.assert_close(model[-1].num_batches_tracked, before_batches)


def test_fixed_part_pooling_has_no_adapter_and_backpropagates_to_shared_map():
    harness = _IntegrationHarness()
    harness.hpgrd_part_weight = 0.15
    feature_map = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
        requires_grad=True,
    )
    masks = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]])

    packet = harness._hpgrd_student_packet(
        {
            "norm_concat_bn": torch.ones(1, 3, requires_grad=True),
            "_hpgrd_feature_map": feature_map,
        },
        anatomical_targets={"masks": masks},
    )

    expected = torch.tensor([[[1.5, 5.5], [3.5, 7.5]]])
    torch.testing.assert_close(packet["_anatomical_student_tokens"], expected)
    torch.testing.assert_close(
        packet["_hpgrd_student_part_reliability"],
        torch.ones(1, 2),
    )
    packet["_anatomical_student_tokens"].sum().backward()
    assert feature_map.grad is not None
    assert bool(feature_map.grad.ne(0).all())


def test_background_control_is_a_detached_target():
    harness = _IntegrationHarness()
    harness.hpgrd_part_weight = 0.0
    primary = torch.randn(3, 4, requires_grad=True)
    clean = torch.randn(3, 4, requires_grad=True)

    packet = harness._hpgrd_student_packet(
        {"norm_concat_bn": primary},
        background_features={"norm_concat_bn": clean},
        background_indices=torch.tensor([0, 2]),
    )
    packet["norm_concat_bn"].sum().backward()

    assert packet["norm_concat_bn"].requires_grad
    assert not packet["_privileged_graph_background_descriptors"].requires_grad
    torch.testing.assert_close(
        packet["_privileged_graph_background_descriptors"],
        clean.detach()[[0, 2]],
    )
    assert primary.grad is not None
    assert clean.grad is None


def test_full_batch_semantic_intervention_selects_only_declared_rows():
    harness = _IntegrationHarness()
    harness.hpgrd_part_weight = 0.0
    primary = torch.randn(4, 3, requires_grad=True)
    semantic = torch.randn(4, 3, requires_grad=True)

    packet = harness._hpgrd_student_packet(
        {"norm_concat_bn": primary},
        semantic_drop_features={"norm_concat_bn": semantic},
        semantic_drop_indices=torch.tensor([1, 3]),
        semantic_drop_parts=torch.tensor([0, 0]),
    )

    selected = packet["_privileged_graph_semantic_drop_descriptors"]
    torch.testing.assert_close(selected, semantic[[1, 3]])
    selected.sum().backward()
    assert semantic.grad is not None
    assert bool(semantic.grad[[1, 3]].ne(0).all())
    assert bool(semantic.grad[[0, 2]].eq(0).all())


def test_hpgrd_recipe_round_trips_to_typed_trainer_configuration():
    namespace = build_mode_namespace(
        "train",
        {"recipe": "csl_tinyvit_7m_hpgrd"},
    )
    kwargs = trainer_kwargs_from_args(namespace)
    config = ReIDTrainConfig.from_flat_kwargs(**kwargs)
    resolved = config.to_trainer_kwargs()

    assert resolved["model_name"] == "csl_tinyvit_7m_v20"
    assert resolved["p"] == 24
    assert resolved["k"] == 4
    assert resolved["camera_aware_sampler"] is True
    assert resolved["inference_feature"] == "norm_concat_bn"
    assert resolved["global_ap_loss_weight"] == pytest.approx(0.15)
    assert resolved["hpgrd_global_weight"] == pytest.approx(0.30)
    assert resolved["hpgrd_part_weight"] == pytest.approx(0.15)
    assert resolved["pretrained"] is True
    assert resolved["pretrained_weights"] is None
    assert resolved["background_mosaic_mask_dir"] == "Market-1501-mosaic-highconf-masks"
    assert resolved["anatomical_person_mask_dir"] == "Market-1501-mosaic-highconf-person-masks"
    assert resolved["eval_interval"] == 200
    # Camera diversity is a sampler concern only in the canonical path. Every
    # optional objective that still has legacy camera-conditioned mining is
    # explicitly inactive, and HP-GRD owns mask generation/fixed pooling.
    assert resolved["anatomical_contrastive_weight"] == 0.0
    assert resolved["anatomical_query_relational_distill_weight"] == 0.0
    assert resolved["anatomical_part_triplet_weight"] == 0.0
    assert resolved["hierarchical_late_interaction"] is False
    assert resolved["treeboost_loss_weight"] == 0.0
    assert ReIDTrainer.from_config(config)._hpgrd_owns_anatomical_runtime() is True


def test_hpgrd_recipe_preserves_v20_deployment_model_and_descriptor_contract():
    models = []
    for recipe_name in ("csl_tinyvit_7m_v20", "csl_tinyvit_7m_hpgrd"):
        namespace = build_mode_namespace("train", {"recipe": recipe_name})
        config = ReIDTrainConfig.from_flat_kwargs(**trainer_kwargs_from_args(namespace))
        trainer = ReIDTrainer.from_config(config)
        trainer.pretrained = False
        models.append(trainer._build_model(num_classes=751))

    v20_model, hpgrd_model = models

    def deployed_schema(model):
        return {
            key: tuple(value.shape) for key, value in model.state_dict().items() if not _is_train_only_model_key(key)
        }

    assert deployed_schema(v20_model) == deployed_schema(hpgrd_model)
    assert sum(parameter.numel() for parameter in v20_model.parameters()) == 7_165_011
    assert sum(parameter.numel() for parameter in hpgrd_model.parameters()) == 7_165_011
    for model in models:
        assert model.head.inference_feature == "norm_concat_bn"
        assert model.head._declared_feature_dim("norm_concat_bn") == 1152

    hpgrd_model.train()
    hpgrd_model.head.set_anatomical_auxiliary_active(False)
    hpgrd_model.head.set_retrieval_packet_active(True)
    hpgrd_model.head.set_hpgrd_part_packet_active(True)
    with torch.no_grad():
        _, training_features = hpgrd_model(torch.randn(2, 3, 384, 128))
    assert training_features["_hpgrd_feature_map"].ndim == 4
    assert training_features["norm_concat_bn"].shape == (2, 1152)

    hpgrd_model.eval()
    with torch.no_grad():
        inference_descriptor = hpgrd_model(torch.randn(1, 3, 384, 128))
    assert inference_descriptor.shape == (1, 1152)


def test_global_ap_training_state_is_strictly_resumable():
    harness = _IntegrationHarness()
    harness._global_ap = SimpleNamespace(state_dict=lambda: {"memory_step": torch.tensor(7)})
    state = harness._training_auxiliary_state()
    assert state["global_ap_state_dict"]["memory_step"].item() == 7

    loaded = {}
    harness._global_ap = SimpleNamespace(
        load_state_dict=lambda value, strict: loaded.update(value=value, strict=strict)
    )
    harness._restore_training_auxiliary_state(state)
    assert loaded["strict"] is True
    assert loaded["value"]["memory_step"].item() == 7
    with pytest.raises(ValueError, match="missing its memory state"):
        harness._restore_training_auxiliary_state(None)


def test_resume_contract_pins_hpgrd_manifest_but_ignores_disabled_family(tmp_path):
    kwargs = {
        "model_name": "csl_tinyvit_7m",
        "dataset_name": "market1501",
        "data_dir": str(tmp_path),
        "hpgrd_cache_dir": str(tmp_path / "cache.pt"),
        "hpgrd_global_weight": 0.1,
        "inference_feature": "norm_concat_bn",
        "epochs": 200,
    }
    first = ReIDTrainer(**kwargs)
    second = ReIDTrainer(**kwargs)
    first._hpgrd_manifest_sha256 = "a" * 64
    second._hpgrd_manifest_sha256 = "b" * 64

    first_contract = first._resume_contract()
    second_contract = second._resume_contract()

    assert first_contract["loss"]["hpgrd_manifest_sha256"] == "a" * 64
    assert first_contract != second_contract
    disabled = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
    )._resume_contract()
    assert not any(key.startswith("hpgrd_") for key in disabled["loss"])


def test_stable_dataset_index_is_appended_without_changing_default_contract(tmp_path):
    image_path = tmp_path / "person.jpg"
    Image.new("RGB", (4, 8)).save(image_path)
    sample = ReIDSample(str(image_path), pid=7, camid=2)

    default_item = ReIDImageDataset([sample])[0]
    indexed_item = ReIDImageDataset([sample], return_sample_index=True)[0]

    assert len(default_item) == 3
    assert indexed_item[1:] == (7, 2, 0)


def test_training_only_memory_is_saved_only_in_resumable_checkpoint(tmp_path):
    manager = CheckpointManager(
        metadata_factory=lambda model: {},
        rng_state_factory=lambda: {"torch": torch.get_rng_state()},
        classifier_loss="ce",
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    last_path = tmp_path / "last.pt"
    manager.save_last(
        last_path,
        model=model,
        epoch=1,
        val=None,
        optimizer=optimizer,
        optimizer_center=None,
        criterion_center=None,
        criterion_classifier=None,
        ema_model=None,
        best_mAP=0.0,
        training_state={"global_ap_state_dict": {"memory_step": torch.tensor(3)}},
    )
    last = torch.load(last_path, weights_only=False)
    assert last["training_state"]["global_ap_state_dict"]["memory_step"].item() == 3

    best_path = tmp_path / "best.pt"
    manager.save_best(
        best_path,
        model=model,
        epoch=1,
        val=SimpleNamespace(mAP=0.1, rank1=0.2),
        criterion_center=None,
        criterion_classifier=None,
        best_mAP=0.1,
    )
    best = torch.load(best_path, weights_only=False)
    assert "training_state" not in best


def test_train_epoch_consumes_indices_and_updates_global_ap_memory(tmp_path):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        epochs=3,
        warmup_epochs=0,
        center_loss_weight=0.0,
        inference_feature="norm_concat_bn",
        global_ap_loss_weight=0.1,
        global_ap_topk=4,
        global_ap_memory_size=4,
        global_ap_start_epoch=0,
        global_ap_ramp_end_epoch=1,
        global_ap_decay_start_epoch=2,
        global_ap_decay_end_epoch=3,
    )

    class TinyIndexedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.classifier = torch.nn.Linear(4, 2)

        def forward(self, images):
            raw = self.encoder(images.flatten(1))
            deployed = torch.nn.functional.normalize(raw, dim=1)
            return self.classifier(raw), {
                "global": raw,
                "raw_concat": raw,
                "norm_concat_bn": deployed,
            }

    model = TinyIndexedModel()
    trainer._global_ap = IdentityGlobalAP(
        memory_size=4,
        feature_dim=4,
        top_k=4,
        max_age=None,
    )
    trainer._privileged_graph_cache = None
    trainer._privileged_graph_loss = None
    center = CenterLoss(num_classes=2, feat_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    center_optimizer = torch.optim.SGD(center.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    loader = [
        (
            torch.randn(2, 1, 2, 2),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([0, 2]),
        ),
        (
            torch.randn(2, 1, 2, 2),
            torch.tensor([0, 1]),
            torch.tensor([1, 1]),
            torch.tensor([1, 3]),
        ),
    ]

    metrics = trainer._train_epoch(
        1,
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        None,
        center,
        optimizer,
        center_optimizer,
        scheduler,
    )

    assert trainer._global_ap.memory_valid.sum().item() == 4
    assert trainer._global_ap.memory_step.item() == 2
    assert metrics.global_ap_loss > 0


def test_train_epoch_applies_cached_privileged_graph_with_gradient_budget(tmp_path):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        epochs=3,
        warmup_epochs=0,
        center_loss_weight=0.0,
        inference_feature="norm_concat_bn",
        hpgrd_cache_dir=str(tmp_path / "teacher.pt"),
        hpgrd_global_weight=0.3,
        hpgrd_gradient_fraction=0.3,
        global_ap_start_epoch=0,
        global_ap_ramp_end_epoch=1,
        global_ap_decay_start_epoch=2,
        global_ap_decay_end_epoch=3,
    )

    class TinyPrivilegedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.classifier = torch.nn.Linear(4, 2)

        def forward(self, images):
            raw = self.encoder(images.flatten(1))
            return self.classifier(raw), {
                "global": raw,
                "raw_concat": raw,
                "norm_concat_bn": torch.nn.functional.normalize(raw, dim=1),
            }

    teacher_global = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
    trainer._global_ap = None
    trainer._privileged_graph_cache = PrivilegedGraphTeacherCache(
        part_names=("whole",),
        sample_indices=torch.arange(4),
        global_descriptors=teacher_global,
        part_descriptors=teacher_global[:, None, :],
        part_visibility=torch.ones(4, 1),
        part_confidence=torch.ones(4, 1),
    )
    trainer._privileged_graph_loss = PrivilegedGraphLoss(
        global_weight=0.3,
        part_weight=0.0,
        background_weight=0.0,
        semantic_drop_weight=0.0,
    )

    def reject_disabled_anatomical_deployment(*args, **kwargs):
        pytest.fail("disabled anatomical deployment must not receive camera metadata")

    trainer._anatomical_deployment_losses = reject_disabled_anatomical_deployment
    model = TinyPrivilegedModel()
    center = CenterLoss(num_classes=2, feat_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    center_optimizer = torch.optim.SGD(center.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    loader = [
        (
            torch.tensor(
                [
                    [[[1.0, 0.0], [0.0, 0.0]]],
                    [[[0.8, 0.2], [0.0, 0.0]]],
                    [[[0.0, 1.0], [0.0, 0.0]]],
                    [[[0.2, 0.8], [0.0, 0.0]]],
                ]
            ),
            torch.tensor([0, 0, 1, 1]),
            # Camera labels remain part of the ordinary training batch, but
            # every sample deliberately shares one camera. HP-GRD must still
            # form both repeated-PID positive groups.
            torch.zeros(4, dtype=torch.long),
            torch.arange(4),
        )
    ]

    metrics = trainer._train_epoch(
        1,
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        None,
        center,
        optimizer,
        center_optimizer,
        scheduler,
    )

    assert metrics.hpgrd_global_loss > 0
    assert metrics.hpgrd_loss > 0
    assert 0 < metrics.hpgrd_gradient_scale <= 1
