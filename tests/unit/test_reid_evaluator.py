from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot.engine.reid import comparison as comparison_module
from boxmot.engine.reid import evaluator as evaluator_module
from boxmot.engine.reid.base import BasePredictor, BaseValidator
from boxmot.reid.training.evaluator import compute_distance_matrix


def test_reid_evaluator_uses_base_validator_and_predictor_contracts():
    assert issubclass(evaluator_module.ReIDValidator, BaseValidator)
    assert issubclass(evaluator_module.ReIDEmbeddingPredictor, BasePredictor)


def _evidence_packet(global_feature, parts, visibility, rarity, roles, nullness):
    return np.asarray(
        [
            *global_feature,
            *np.asarray(parts, dtype=np.float32).reshape(-1),
            *visibility,
            *rarity,
            *np.asarray(roles, dtype=np.float32).reshape(-1),
            *nullness,
        ],
        dtype=np.float32,
    )


def test_evidence_sinkhorn_distance_reranks_only_topk_gallery():
    query = np.stack(
        [
            _evidence_packet(
                [1.0, 0.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [1.0, 1.0],
                [1.0, 1.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [0.0, 0.0],
            )
        ]
    )
    gallery = np.stack(
        [
            _evidence_packet(
                [1.0, 0.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [1.0, 1.0],
                [1.0, 1.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [0.0, 0.0],
            ),
            _evidence_packet(
                [0.0, 1.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [1.0, 1.0],
                [1.0, 1.0],
                [[1.0, 0.0], [0.0, 1.0]],
                [0.0, 0.0],
            ),
        ]
    )

    distmat = compute_distance_matrix(
        query,
        gallery,
        metric="evidence_sinkhorn",
        part_dim=2,
        part_count=2,
        role_count=2,
        beta=0.2,
        topk=1,
        sinkhorn_iters=5,
        sinkhorn_temperature=0.1,
    )

    assert distmat.shape == (1, 2)
    assert distmat[0, 0] < distmat[0, 1]
    assert distmat[0, 1] == pytest.approx(1.0)


def test_eval_reid_overrides_inference_feature_and_writes_mode_json(monkeypatch, tmp_path):
    weights = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": {},
            "model_name": "csl_tinyvit_23m",
            "num_classes": 10,
            "preprocess": "resize_pad",
            "inference_feature": "concat_bn",
            "feat_dim": 512,
        },
        weights,
    )
    (tmp_path / "hparams.json").write_text(
        '{"data": {"img_size": [384, 128], "preprocess": "resize"}, "evaluation": {"flip_tta": true}}'
    )
    output_dir = tmp_path / "evals"

    class _Split:
        samples = [object()]

    class _Dataset:
        query = _Split()
        gallery = _Split()
        num_train_pids = 10

    class _TorchDataset:
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return torch.zeros(3, 384, 128), 1, index

    class _Head:
        inference_feature = "concat_bn"

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = _Head()

        def load_state_dict(self, state_dict, strict=False):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    built = {}
    transforms = {}
    extracted = []

    def fake_build_model(name, weights_path, **kwargs):
        model = _Model()
        model.head.inference_feature = kwargs["inference_feature"]
        built.update(name=name, weights_path=weights_path, kwargs=kwargs, model=model)
        return model

    def fake_build_test_transforms(img_size, preprocess):
        transforms.update(img_size=img_size, preprocess=preprocess)
        return object()

    def fake_extract_features(model, dataloader, device, desc="Extracting", flip_tta=False, normalize=True):
        extracted.append((model.head.inference_feature, flip_tta, normalize))
        return (
            np.ones((1, 512), dtype=np.float32),
            np.asarray([1]),
            np.asarray([0 if len(extracted) == 1 else 1]),
        )

    monkeypatch.setattr(evaluator_module, "build_dataset", lambda dataset, data_dir: _Dataset())
    monkeypatch.setattr(evaluator_module, "ReIDImageDataset", _TorchDataset)
    monkeypatch.setattr(evaluator_module, "build_test_transforms", fake_build_test_transforms)
    monkeypatch.setattr(evaluator_module.ReIDModelRegistry, "build_model", fake_build_model)
    monkeypatch.setattr(evaluator_module, "extract_features", fake_extract_features)
    monkeypatch.setattr(
        evaluator_module,
        "_benchmark_inference_latency",
        lambda *args, **kwargs: {
            "latency_device": "cpu",
            "latency_warmup": 2,
            "latency_iters": 7,
            "latency_batch_size": 1,
            "latency_ms_per_batch": 1.25,
            "latency_ms_per_image": 1.25,
            "latency_includes_flip_tta": True,
        },
    )

    result = evaluator_module.main(
        SimpleNamespace(
            weights=str(weights),
            model=None,
            dataset="market1501",
            data_dir=str(tmp_path),
            preprocess=None,
            imgsz=None,
            inference_feature="raw_mean",
            flip_tta=None,
            device="cpu",
            batch_size=1,
            num_workers=0,
            latency_warmup=2,
            latency_iters=7,
            output=str(output_dir),
        )
    )

    assert built["kwargs"]["inference_feature"] == "raw_mean"
    assert built["model"].head.inference_feature == "raw_mean"
    assert transforms == {"img_size": (384, 128), "preprocess": "resize_pad"}
    assert extracted == [("raw_mean", True, True), ("raw_mean", True, True)]
    assert result["inference_feature"] == "raw_mean"
    assert result["feature_dim"] == 512
    assert result["latency_device"] == "cpu"
    assert result["latency_ms_per_image"] == 1.25
    assert (output_dir / "eval_csl_tinyvit_23m_market1501_raw_mean.json").exists()


def test_compare_reid_runs_cross_domain_pairs_and_writes_summary(monkeypatch, tmp_path):
    market_run = tmp_path / "market_run"
    duke_run = tmp_path / "duke_run"
    market_run.mkdir()
    duke_run.mkdir()
    market_weights = market_run / "best.pt"
    duke_weights = duke_run / "best.pt"
    torch.save({"state_dict": {}, "model_name": "fake_reid", "dataset": "market1501"}, market_weights)
    torch.save({"state_dict": {}, "model_name": "fake_reid", "dataset": "duke"}, duke_weights)

    market_data = tmp_path / "Market-1501-v15.09.15"
    duke_data = tmp_path / "DukeMTMC-reID"
    market_data.mkdir()
    duke_data.mkdir()
    calls = []

    def fake_eval_main(args):
        calls.append(args)
        return {
            "model": args.model or "fake_reid",
            "weights": args.weights,
            "dataset": args.dataset,
            "inference_feature": args.inference_feature,
            "mAP": 0.75 if args.dataset == "duke" else 0.65,
            "rank1": 0.85 if args.dataset == "duke" else 0.80,
            "rank5": 0.90,
            "rank10": 0.95,
            "latency_device": args.device,
            "latency_ms_per_image": 2.5 if args.dataset == "duke" else 3.5,
            "latency_ms_per_batch": 10.0,
            "latency_batch_size": 4,
        }

    monkeypatch.setattr(comparison_module.evaluator, "main", fake_eval_main)

    def fake_write_map_latency_plot(results, path):
        path.write_text("plot")
        return path

    monkeypatch.setattr(comparison_module, "_write_map_latency_plot", fake_write_map_latency_plot)
    output_dir = tmp_path / "comparison"

    result = comparison_module.main(
        SimpleNamespace(
            weights=(str(market_weights), str(duke_weights)),
            target=(f"market1501={market_data}", f"duke={duke_data}"),
            label=(),
            model=(),
            include_same_dataset=False,
            preprocess=None,
            imgsz=None,
            inference_feature="evidence_sinkhorn",
            flip_tta=None,
            device="cpu",
            batch_size=4,
            num_workers=0,
            latency_warmup=2,
            latency_iters=7,
            continue_on_error=False,
            output=str(output_dir),
        )
    )

    assert [(Path(call.weights).parent.name, call.dataset) for call in calls] == [
        ("market_run", "duke"),
        ("duke_run", "market1501"),
    ]
    assert all(call.inference_feature == "evidence_sinkhorn" for call in calls)
    assert result["summary"] == {
        "models": 2,
        "targets": 2,
        "rows": 4,
        "evaluated": 2,
        "skipped": 2,
        "failed": 0,
        "cross_domain_only": True,
        "map_latency_plot": str(output_dir / "map_vs_latency.png"),
    }
    assert [row["status"] for row in result["results"]].count("skipped_same_dataset") == 2
    assert [row["status"] for row in result["results"]].count("ok") == 2
    assert (output_dir / "map_vs_latency.png").exists()
    assert (output_dir / "cross_domain_results.json").exists()
    assert (output_dir / "cross_domain_results.md").exists()
