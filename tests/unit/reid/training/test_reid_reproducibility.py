import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import boxmot.reid.training.trainer as trainer_module
from boxmot.reid.datasets.base import ReIDSample
from boxmot.reid.datasets.sampler import PKSampler, SourceBalancedPKSampler, parse_source_balance
from boxmot.reid.datasets.transforms import (
    EpochAwareCompose,
    Random2DTranslation,
    RandomPatch,
    ResizePad,
    build_train_transforms,
)
from boxmot.reid.training.resume import contract_differences
from boxmot.reid.training.trainer import ReIDTrainer, _seed_data_worker


def _samples(num_pids: int = 16, instances: int = 4) -> list[ReIDSample]:
    return [
        ReIDSample(img_path=f"{pid}_{index}.jpg", pid=pid, camid=index % 2)
        for pid in range(num_pids)
        for index in range(instances)
    ]


def _source_samples(instances: int = 4) -> list[ReIDSample]:
    return [
        ReIDSample(
            img_path=f"{source}_{pid}_{index}.jpg",
            pid=pid,
            camid=index % 2,
            source=source,
        )
        for source, pid_offset in (("market1501", 0), ("mot17_1501", 100))
        for pid in range(pid_offset, pid_offset + 4)
        for index in range(instances)
    ]


def test_pk_sampler_is_deterministic_per_seed_and_epoch():
    first = PKSampler(_samples(), p=4, k=4, seed=42)
    second = PKSampler(_samples(), p=4, k=4, seed=42)

    first.set_epoch(3)
    second.set_epoch(3)

    assert list(first) == list(second)

    second.set_epoch(4)
    assert list(first) != list(second)


def test_fixed_camera_aware_pk_sampler_has_stable_length_and_diverse_groups():
    samples = [
        ReIDSample(
            img_path=f"{pid}_{camera}_{instance}.jpg",
            pid=pid,
            camid=camera,
        )
        for pid in range(20)
        for camera in range(6)
        for instance in range(2)
    ]
    sampler = PKSampler(
        samples,
        p=16,
        k=6,
        seed=42,
        steps_per_epoch=62,
        camera_aware=True,
    )

    indices = list(sampler)

    assert len(sampler) == 62 * 16 * 6
    assert len(indices) == len(sampler)
    for batch_start in range(0, len(indices), 16 * 6):
        batch = indices[batch_start : batch_start + 16 * 6]
        batch_pids = []
        for group_start in range(0, len(batch), 6):
            group = batch[group_start : group_start + 6]
            group_samples = [samples[index] for index in group]
            assert len({sample.pid for sample in group_samples}) == 1
            assert len({sample.camid for sample in group_samples}) == 6
            assert len(set(group)) == 6
            batch_pids.append(group_samples[0].pid)
        assert len(set(batch_pids)) == 16


def test_fixed_camera_aware_pk_sampler_is_deterministic_per_epoch():
    first = PKSampler(
        _samples(num_pids=20, instances=8),
        p=16,
        k=6,
        seed=42,
        steps_per_epoch=3,
        camera_aware=True,
    )
    second = PKSampler(
        _samples(num_pids=20, instances=8),
        p=16,
        k=6,
        seed=42,
        steps_per_epoch=3,
        camera_aware=True,
    )

    first.set_epoch(3)
    second.set_epoch(3)
    assert list(first) == list(second)

    second.set_epoch(4)
    assert list(first) != list(second)


def test_trainer_builds_fixed_camera_aware_pk_sampler(tmp_path):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        p=16,
        k=6,
        pk_steps_per_epoch=62,
        camera_aware_sampler=True,
    )
    split = SimpleNamespace(samples=_samples(num_pids=20, instances=8))
    dataset = SimpleNamespace(train=split, query=split, gallery=split)

    loader = trainer._build_train_loader(dataset)

    assert isinstance(loader.sampler, PKSampler)
    assert loader.batch_size == 96
    assert loader.sampler.steps_per_epoch == 62
    assert loader.sampler.camera_aware is True
    assert len(loader.sampler) == 62 * 96


def test_source_balance_spec_parses_normalized_groups():
    groups = parse_source_balance("market1501+dukemtmc-reid:8,4;mot17_1501:8,4")

    assert groups[0].sources == ("market1501", "dukemtmcreid")
    assert groups[0].batch_size == 32
    assert groups[1].sources == ("mot171501",)
    assert groups[1].batch_size == 32


def test_source_balanced_sampler_is_deterministic_and_mixes_sources():
    samples = _source_samples()
    first = SourceBalancedPKSampler(samples, "market1501:2,2;mot17_1501:2,2", seed=42)
    second = SourceBalancedPKSampler(samples, "market1501:2,2;mot17_1501:2,2", seed=42)

    first.set_epoch(3)
    second.set_epoch(3)
    first_indices = list(first)
    second_indices = list(second)

    assert first_indices == second_indices
    assert len(first) == 16
    batch = first_indices[: first.batch_size]
    sources = [samples[index].source for index in batch]
    assert sources.count("market1501") == 4
    assert sources.count("mot17_1501") == 4

    second.set_epoch(4)
    assert first_indices != list(second)


def test_seed_everything_controls_python_numpy_and_torch():
    ReIDTrainer._seed_everything(17)
    expected = (random.random(), np.random.random(), torch.rand(1))

    ReIDTrainer._seed_everything(17)
    actual = (random.random(), np.random.random(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_trainer_defaults_to_seed_zero_and_deterministic_mode(tmp_path):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
    )

    assert trainer.seed == 0
    assert trainer.deterministic is True


def test_deterministic_mode_is_configurable_independently_of_seed(monkeypatch):
    original = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    try:
        ReIDTrainer._configure_determinism(True)
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

        ReIDTrainer._seed_everything(17)
        assert torch.are_deterministic_algorithms_enabled()

        ReIDTrainer._configure_determinism(False)
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        ReIDTrainer._configure_determinism(original)


def test_seed_data_worker_controls_python_numpy_and_torch(monkeypatch):
    monkeypatch.setattr(torch, "initial_seed", lambda: 123)

    _seed_data_worker(0)
    expected = (random.random(), np.random.random(), torch.rand(1))
    _seed_data_worker(0)
    actual = (random.random(), np.random.random(), torch.rand(1))

    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


def test_epoch_seed_controls_sampler_augmentations_and_torch(tmp_path):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        seed=42,
    )
    sampler = PKSampler(_samples(), p=4, k=4, seed=trainer.seed)
    loader = SimpleNamespace(sampler=sampler)

    trainer._seed_training_epoch(5, loader)
    expected = (
        list(sampler),
        random.random(),
        np.random.random(),
        torch.rand(1),
        torch.rand(1, generator=trainer._train_generator),
    )
    trainer._seed_training_epoch(5, loader)
    actual = (
        list(sampler),
        random.random(),
        np.random.random(),
        torch.rand(1),
        torch.rand(1, generator=trainer._train_generator),
    )

    assert actual[:3] == expected[:3]
    torch.testing.assert_close(actual[3], expected[3])
    torch.testing.assert_close(actual[4], expected[4])


def test_epoch_seed_resets_stateful_augmentation_for_exact_resume(tmp_path):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        seed=42,
    )
    patch = RandomPatch(min_sample_size=1)
    patch.patchpool.append(object())
    dataset = SimpleNamespace(transform=EpochAwareCompose([patch]))
    dataset.set_epoch = lambda epoch: dataset.transform.set_epoch(epoch)
    sampler = PKSampler(_samples(), p=4, k=4, seed=trainer.seed)
    loader = SimpleNamespace(dataset=dataset, sampler=sampler)

    trainer._seed_training_epoch(5, loader)

    assert list(patch.patchpool) == []
    assert sampler.epoch == 5


def test_random_patch_epoch_stream_matches_fresh_resumed_transform():
    uninterrupted = RandomPatch(
        prob_happen=1.0,
        min_sample_size=1,
        prob_rotate=1.0,
        prob_flip_leftright=1.0,
    )
    resumed = RandomPatch(
        prob_happen=1.0,
        min_sample_size=1,
        prob_rotate=1.0,
        prob_flip_leftright=1.0,
    )
    uninterrupted.patchpool.append(Image.new("RGB", (8, 8), (255, 0, 0)))
    images = [Image.new("RGB", (32, 64), (value, value, value)) for value in (32, 96, 160)]

    def run_epoch(transform):
        random.seed(123)
        transform.set_epoch(7)
        return [np.asarray(transform(image)).copy() for image in images]

    uninterrupted_outputs = run_epoch(uninterrupted)
    resumed_outputs = run_epoch(resumed)

    for uninterrupted_output, resumed_output in zip(
        uninterrupted_outputs,
        resumed_outputs,
    ):
        np.testing.assert_array_equal(uninterrupted_output, resumed_output)


def test_canonical_train_resize_is_owned_by_random_translation():
    direct = build_train_transforms(
        (384, 128),
        preprocess="resize",
        random_erasing=0,
        color_jitter=False,
        random_patch=False,
        color_augmentation=False,
    )
    padded = build_train_transforms(
        (384, 128),
        preprocess="resize_pad",
        random_erasing=0,
        color_jitter=False,
        random_patch=False,
        color_augmentation=False,
    )

    assert not any(isinstance(op, T.Resize) for op in direct.transforms)
    assert isinstance(direct.transforms[0], T.RandomHorizontalFlip)
    assert isinstance(direct.transforms[1], Random2DTranslation)
    assert isinstance(padded.transforms[0], ResizePad)
    assert isinstance(padded.transforms[1], T.RandomHorizontalFlip)
    assert isinstance(padded.transforms[2], Random2DTranslation)


def test_post_normalization_random_erasing_uses_zero_fill():
    transform = build_train_transforms(
        random_erasing=1.0,
        color_jitter=False,
        random_patch=False,
        color_augmentation=False,
    )

    erasing = transform.transforms[-1]
    assert isinstance(erasing, T.RandomErasing)
    assert erasing.value == 0.0


def test_cuda_train_loader_uses_seeded_nonpersistent_workers_and_generator(tmp_path):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        device="cuda",
        num_workers=2,
        seed=42,
    )
    split = SimpleNamespace(samples=_samples())
    dataset = SimpleNamespace(train=split, query=split, gallery=split)

    train_loader = trainer._build_train_loader(dataset)
    query_loader, gallery_loader = trainer._build_test_loaders(dataset)

    assert train_loader.sampler.seed == 42
    assert train_loader.worker_init_fn is _seed_data_worker
    assert train_loader.generator is trainer._train_generator
    for loader in (train_loader, query_loader, gallery_loader):
        assert loader.num_workers == 2
        assert loader.persistent_workers is False


def test_source_balance_train_loader_uses_source_sampler(tmp_path):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501,mot17_1501",
        data_dir=str(tmp_path),
        source_balance="market1501:2,2;mot17_1501:2,2",
        seed=42,
    )
    split = SimpleNamespace(samples=_source_samples())
    dataset = SimpleNamespace(train=split, query=split, gallery=split)

    train_loader = trainer._build_train_loader(dataset)

    assert isinstance(train_loader.sampler, SourceBalancedPKSampler)
    assert train_loader.batch_size == 8
    assert train_loader.sampler.seed == 42


def test_cpu_and_mps_honor_requested_nonpersistent_workers(tmp_path):
    split = SimpleNamespace(samples=_samples())
    dataset = SimpleNamespace(train=split, query=split, gallery=split)

    for device in ("cpu", "mps"):
        trainer = ReIDTrainer(
            model_name="csl_tinyvit_7m",
            dataset_name="market1501",
            data_dir=str(tmp_path),
            device=device,
            num_workers=4,
        )
        train_loader = trainer._build_train_loader(dataset)
        query_loader, gallery_loader = trainer._build_test_loaders(dataset)

        assert trainer.requested_num_workers == 4
        assert trainer.num_workers == 4
        for loader in (train_loader, query_loader, gallery_loader):
            assert loader.num_workers == 4
            assert loader.persistent_workers is False


def test_anatomical_target_generation_follows_zero_decay_schedule():
    trainer = object.__new__(ReIDTrainer)
    trainer.seed = 42
    trainer._train_generator = torch.Generator()
    trainer.anatomical_auxiliary = True
    trainer.anatomical_target_type = "learned_pose_concat_ema"
    trainer.anatomical_student_start_epoch = 0
    trainer.anatomical_student_ramp_end_epoch = 0
    trainer.anatomical_decay_start_epoch = 140
    trainer.anatomical_decay_end_epoch = 170
    trainer.anatomical_fine_start_epoch = 20
    trainer.anatomical_fine_ramp_end_epoch = 40
    enabled = []
    dataset = SimpleNamespace(
        set_anatomical_targets_enabled=enabled.append,
        set_epoch=lambda _epoch: None,
    )
    loader = SimpleNamespace(dataset=dataset, sampler=None)

    trainer._seed_training_epoch(169, loader)
    trainer._seed_training_epoch(170, loader)
    trainer._seed_training_epoch(200, loader)

    assert enabled == [True, False, False]


def test_anatomical_deployment_stays_active_after_auxiliary_decay():
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_deployment = True
    trainer.anatomical_target_type = "learned_pose_concat_ema"
    trainer.anatomical_student_start_epoch = 0
    trainer.anatomical_student_ramp_end_epoch = 0
    trainer.anatomical_decay_start_epoch = 140
    trainer.anatomical_decay_end_epoch = 170
    trainer.anatomical_fine_start_epoch = 20
    trainer.anatomical_fine_ramp_end_epoch = 40

    assert trainer._anatomical_training_active(200) is True


def test_resume_contract_rejects_worker_count_rng_change():
    saved = {"data": {"num_workers": 0}}
    requested = {"data": {"num_workers": 4}}

    assert contract_differences(saved, requested) == [
        "data.num_workers: saved=0, requested=4"
    ]


def test_clear_memory_uses_threshold_and_device_cache(monkeypatch, tmp_path):
    calls = {"gc": 0, "cuda": 0, "mps": 0}
    monkeypatch.setattr(
        trainer_module.gc,
        "collect",
        lambda: calls.__setitem__("gc", calls["gc"] + 1),
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: calls.__setitem__("cuda", calls["cuda"] + 1),
    )
    monkeypatch.setattr(
        torch.mps,
        "empty_cache",
        lambda: calls.__setitem__("mps", calls["mps"] + 1),
    )

    cpu_trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        device="cpu",
    )
    mps_trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        device="mps",
    )
    cuda_trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        device="cuda",
    )

    assert not cpu_trainer._clear_memory(force=True)
    assert calls == {"gc": 0, "cuda": 0, "mps": 0}

    monkeypatch.setattr(mps_trainer, "_memory_utilization", lambda: 0.50)
    assert not mps_trainer._clear_memory(threshold=0.90)
    assert calls == {"gc": 0, "cuda": 0, "mps": 0}

    monkeypatch.setattr(mps_trainer, "_memory_utilization", lambda: 0.95)
    assert mps_trainer._clear_memory(threshold=0.90)
    assert calls == {"gc": 1, "cuda": 0, "mps": 1}

    assert cuda_trainer._clear_memory(force=True)
    assert calls == {"gc": 2, "cuda": 1, "mps": 1}


def test_memory_utilization_uses_process_accelerator_memory(monkeypatch, tmp_path):
    cuda_trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        device="cuda",
    )
    mps_trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        device="mps",
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=1000),
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 750)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 2000)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 1000)

    assert cuda_trainer._memory_utilization() == 0.75
    assert mps_trainer._memory_utilization() == 0.50


def test_handle_oom_clears_gradients_and_memory(monkeypatch, tmp_path):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_7m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        device="mps",
    )
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    cleared = []
    monkeypatch.setattr(
        trainer,
        "_clear_memory",
        lambda **kwargs: cleared.append(kwargs) or True,
    )

    assert trainer._handle_oom(RuntimeError("MPS backend out of memory"), optimizer)
    assert parameter.grad is None
    assert cleared == [{"force": True}]
    assert not trainer._handle_oom(RuntimeError("unrelated failure"), optimizer)


def test_rng_state_roundtrip_restores_all_process_rngs():
    ReIDTrainer._seed_everything(91)
    state = ReIDTrainer._capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1))

    ReIDTrainer._seed_everything(7)
    ReIDTrainer._restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])
