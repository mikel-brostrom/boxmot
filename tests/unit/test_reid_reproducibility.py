import os
import random
from types import SimpleNamespace

import numpy as np
import torch

import boxmot.reid.training.trainer as trainer_module
from boxmot.reid.datasets.base import ReIDSample
from boxmot.reid.datasets.sampler import PKSampler, SourceBalancedPKSampler, parse_source_balance
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


def test_cpu_and_mps_force_zero_nonpersistent_workers(tmp_path):
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
        assert trainer.num_workers == 0
        for loader in (train_loader, query_loader, gallery_loader):
            assert loader.num_workers == 0
            assert loader.persistent_workers is False


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
