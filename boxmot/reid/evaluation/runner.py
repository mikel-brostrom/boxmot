"""Reusable workflow for standalone ReID model evaluation.

Loads a trained checkpoint and evaluates on query/gallery splits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.datasets import build_dataset
from boxmot.reid.datasets.torch_dataset import ReIDImageDataset
from boxmot.reid.datasets.transforms import build_test_transforms
from boxmot.reid.training.evaluator import (
    compute_distance_matrix,
    evaluate_ranking,
    extract_features,
    visibility_part_count,
)
from boxmot.utils import logger as LOGGER


@dataclass(frozen=True, slots=True)
class ReIDEvaluationConfig:
    """Resolved inputs for evaluating one ReID checkpoint."""

    weights: str | Path
    dataset: str
    data_dir: str | Path
    model: str | None = None
    preprocess: str | None = None
    imgsz: int | tuple[int, int] | list[int] | None = None
    inference_feature: str | None = None
    flip_tta: bool | None = None
    device: str = "cpu"
    batch_size: int = 64
    num_workers: int = 4
    latency_warmup: int = 5
    latency_iters: int = 30


@dataclass(frozen=True, slots=True)
class ReIDPredictionResult:
    """Feature extraction output for a query/gallery validation run."""

    q_feats: object
    q_pids: object
    q_camids: object
    g_feats: object
    g_pids: object
    g_camids: object
    latency: dict


def _load_hparams(weights_path: Path) -> dict:
    """Load hparams.json saved next to a checkpoint, if present."""
    hparams_path = weights_path.parent / "hparams.json"
    if not hparams_path.exists():
        return {}
    try:
        return json.loads(hparams_path.read_text())
    except json.JSONDecodeError:
        LOGGER.warning(f"Could not parse {hparams_path}; using checkpoint/default eval settings")
        return {}


def _hparams_value(hparams: dict, key: str, default=None):
    """Return hparams key from flat or nested layouts."""
    if key in hparams:
        return hparams[key]

    nested_paths = {
        "img_size": ("data", "img_size"),
        "preprocess": ("data", "preprocess"),
        "flip_tta": ("evaluation", "flip_tta"),
        "branch_metric_part_weight": ("model", "branch", "metric_part_weight"),
        "evidence_sinkhorn_iters": ("model", "evidence", "sinkhorn_iters"),
        "evidence_sinkhorn_temperature": ("model", "evidence", "sinkhorn_temperature"),
        "evidence_rerank_topk": ("model", "evidence", "rerank_topk"),
        "evidence_num_roles": ("model", "head", "evidence_num_roles"),
    }
    path = nested_paths.get(key)
    if not path:
        return default

    cur = hparams
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _synchronize_device(device: torch.device) -> None:
    """Synchronize accelerator work before wall-clock timing reads."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


@torch.no_grad()
def _benchmark_inference_latency(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    flip_tta: bool,
    warmup: int = 5,
    iterations: int = 30,
) -> dict:
    """Benchmark model-forward latency on one real evaluation batch."""
    warmup = max(0, int(warmup))
    iterations = int(iterations)
    if iterations <= 0:
        return {}

    try:
        imgs, *_ = next(iter(dataloader))
    except StopIteration:
        return {}

    imgs = imgs.to(device)
    batch_size = int(imgs.shape[0])
    if batch_size <= 0:
        return {}

    def forward_once() -> torch.Tensor:
        feats = model(imgs)
        if flip_tta:
            feats_flip = model(torch.flip(imgs, dims=[3]))
            feats = (feats + feats_flip) / 2.0
        return feats

    model.eval()
    for _ in range(warmup):
        _ = forward_once()
    _synchronize_device(device)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = forward_once()
    _synchronize_device(device)
    elapsed = time.perf_counter() - start

    batch_ms = elapsed * 1000.0 / iterations
    return {
        "latency_device": str(device),
        "latency_warmup": warmup,
        "latency_iters": iterations,
        "latency_batch_size": batch_size,
        "latency_ms_per_batch": round(float(batch_ms), 4),
        "latency_ms_per_image": round(float(batch_ms / batch_size), 4),
        "latency_includes_flip_tta": bool(flip_tta),
    }


class ReIDEmbeddingPredictor:
    """Predictor that extracts ReID embeddings from dataloaders."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        flip_tta: bool,
        normalize: bool,
    ) -> None:
        self.model = model
        self.device = device
        self.flip_tta = bool(flip_tta)
        self.normalize = bool(normalize)

    def inference(self, dataloader: torch.utils.data.DataLoader, *, desc: str = "Extracting"):
        """Extract embeddings for a dataloader."""
        return extract_features(
            self.model,
            dataloader,
            self.device,
            desc=desc,
            flip_tta=self.flip_tta,
            normalize=self.normalize,
        )

    def predict(self, dataloader: torch.utils.data.DataLoader, *, desc: str = "Extracting"):
        """Extract and postprocess embeddings for one dataloader."""

        return self.inference(dataloader, desc=desc)

    def benchmark_latency(
        self,
        dataloader: torch.utils.data.DataLoader,
        *,
        warmup: int,
        iterations: int,
    ) -> dict:
        """Benchmark model-forward latency on a representative dataloader batch."""
        return _benchmark_inference_latency(
            self.model,
            dataloader,
            self.device,
            flip_tta=self.flip_tta,
            warmup=warmup,
            iterations=iterations,
        )


class ReIDValidator:
    """Validator for a trained ReID checkpoint on a query/gallery dataset."""

    def __init__(self, config: ReIDEvaluationConfig) -> None:
        self.args = config

    def validate(self) -> dict[str, Any]:
        """Run setup, feature extraction, and metric evaluation."""

        self.setup()
        predictions = self.predict()
        results = self.evaluate(predictions)
        LOGGER.info(
            f"Results on {self.dataset_name}:  "
            f"mAP={results['mAP']:.2%}  "
            f"R1={results['rank1']:.2%}  "
            f"R5={results['rank5']:.2%}  "
            f"R10={results['rank10']:.2%}"
        )
        return results

    def setup(self) -> None:
        """Load checkpoint, model, dataset, and dataloaders."""
        self.weights_path = Path(self.args.weights)
        self.device = torch.device(getattr(self.args, "device", "cpu"))
        self.batch_size = getattr(self.args, "batch_size", 64)
        self.num_workers = getattr(self.args, "num_workers", 4)
        self.latency_warmup = getattr(self.args, "latency_warmup", 5)
        self.latency_iters = getattr(self.args, "latency_iters", 30)

        LOGGER.info(f"Loading checkpoint from {self.weights_path}")
        self.checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        self.hparams = _load_hparams(self.weights_path)

        self.model_name = getattr(self.args, "model", None) or self.checkpoint.get("model_name")
        if self.model_name is None:
            raise ValueError(
                "Cannot determine model architecture. Provide --model or use a checkpoint that stores 'model_name'."
            )

        self.num_classes = self.checkpoint.get("num_classes", -1)
        self.preprocess = (
            getattr(self.args, "preprocess", None)
            or self.checkpoint.get("preprocess")
            or _hparams_value(self.hparams, "preprocess", "resize")
        )
        flip_tta_arg = getattr(self.args, "flip_tta", None)
        self.flip_tta = (
            bool(_hparams_value(self.hparams, "flip_tta", False)) if flip_tta_arg is None else bool(flip_tta_arg)
        )
        checkpoint_model_kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(
            self.weights_path,
        )
        self.inference_feature_override = getattr(self.args, "inference_feature", None)
        if self.inference_feature_override:
            checkpoint_model_kwargs["inference_feature"] = self.inference_feature_override
        self.model_kwargs = ReIDModelRegistry.deployment_model_kwargs(
            self.model_name,
            checkpoint_model_kwargs,
        )

        self.dataset_name = self.args.dataset
        self.data_dir = self.args.data_dir
        LOGGER.info(f"Loading dataset '{self.dataset_name}' from {self.data_dir}")
        self.dataset = build_dataset(self.dataset_name, self.data_dir)
        if self.num_classes <= 0:
            self.num_classes = self.dataset.num_train_pids

        LOGGER.info(f"Building model '{self.model_name}' with {self.num_classes} classes")
        self.model = ReIDModelRegistry.build_model(
            self.model_name,
            self.weights_path,
            num_classes=self.num_classes,
            loss="softmax",
            pretrained=False,
            use_gpu=self.device.type != "cpu",
            **self.model_kwargs,
        )
        ReIDModelRegistry.load_deployment_weights(
            self.model,
            self.weights_path,
        )
        if (
            self.inference_feature_override
            and hasattr(self.model, "head")
            and hasattr(self.model.head, "inference_feature")
        ):
            self.model.head.inference_feature = self.inference_feature_override
        self.model = self.model.to(self.device)
        self.model.eval()

        self.img_size = getattr(self.args, "imgsz", None) or _hparams_value(self.hparams, "img_size") or (256, 128)
        if isinstance(self.img_size, int):
            self.img_size = (self.img_size, self.img_size // 2)
        elif isinstance(self.img_size, list):
            self.img_size = tuple(self.img_size)
        transform = build_test_transforms(self.img_size, preprocess=self.preprocess)

        self.query_ds = ReIDImageDataset(self.dataset.query.samples, transform=transform)
        self.gallery_ds = ReIDImageDataset(self.dataset.gallery.samples, transform=transform)
        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )
        self.query_loader = torch.utils.data.DataLoader(self.query_ds, **loader_kwargs)
        self.gallery_loader = torch.utils.data.DataLoader(self.gallery_ds, **loader_kwargs)

        self.inference_feature = self.model_kwargs.get("inference_feature")
        self.visibility_distance = self.inference_feature == "visibility_weighted_parts"
        self.evidence_distance = self.inference_feature == "evidence_sinkhorn"
        self.structured_distance = self.visibility_distance or self.evidence_distance
        self.predictor = ReIDEmbeddingPredictor(
            self.model,
            self.device,
            flip_tta=self.flip_tta,
            normalize=not self.structured_distance,
        )

    def predict(self) -> ReIDPredictionResult:
        """Extract query and gallery embeddings."""
        LOGGER.info(f"Extracting features ({len(self.query_ds)} query, {len(self.gallery_ds)} gallery)...")
        latency = self.predictor.benchmark_latency(
            self.query_loader,
            warmup=self.latency_warmup,
            iterations=self.latency_iters,
        )
        if latency:
            LOGGER.info(
                "Inference latency on "
                f"{latency['latency_device']}: "
                f"{latency['latency_ms_per_image']:.3f} ms/image "
                f"({latency['latency_ms_per_batch']:.3f} ms/batch, "
                f"batch={latency['latency_batch_size']})"
            )
        q_feats, q_pids, q_camids = self.predictor.predict(self.query_loader, desc="Query")
        g_feats, g_pids, g_camids = self.predictor.predict(self.gallery_loader, desc="Gallery")
        return ReIDPredictionResult(
            q_feats=q_feats,
            q_pids=q_pids,
            q_camids=q_camids,
            g_feats=g_feats,
            g_pids=g_pids,
            g_camids=g_camids,
            latency=latency,
        )

    def evaluate(self, predictions: ReIDPredictionResult) -> dict:
        """Compute ranking metrics from extracted embeddings."""
        LOGGER.info("Computing distance matrix and evaluating...")
        distmat = compute_distance_matrix(
            predictions.q_feats,
            predictions.g_feats,
            metric=(
                "evidence_sinkhorn"
                if self.evidence_distance
                else "visibility_weighted_parts"
                if self.visibility_distance
                else "cosine"
            ),
            part_dim=int(self.model_kwargs.get("feat_dim", 512)) if self.structured_distance else None,
            part_count=visibility_part_count(self.model_kwargs.get("head_parts", (1, 2)))
            if self.structured_distance
            else None,
            role_count=int(
                self.model_kwargs.get(
                    "evidence_num_roles",
                    _hparams_value(self.hparams, "evidence_num_roles", 8),
                )
            )
            if self.evidence_distance
            else None,
            beta=float(
                self.checkpoint.get(
                    "branch_metric_part_weight",
                    _hparams_value(self.hparams, "branch_metric_part_weight", 0.2),
                )
            ),
            topk=int(
                self.checkpoint.get("evidence_rerank_topk", _hparams_value(self.hparams, "evidence_rerank_topk", 100))
            )
            if self.evidence_distance
            else None,
            sinkhorn_iters=int(
                self.checkpoint.get(
                    "evidence_sinkhorn_iters",
                    _hparams_value(self.hparams, "evidence_sinkhorn_iters", 20),
                )
            ),
            sinkhorn_temperature=float(
                self.checkpoint.get(
                    "evidence_sinkhorn_temperature",
                    _hparams_value(self.hparams, "evidence_sinkhorn_temperature", 0.1),
                )
            ),
        )

        cmc, mAP = evaluate_ranking(
            distmat,
            predictions.q_pids,
            predictions.g_pids,
            predictions.q_camids,
            predictions.g_camids,
        )
        return {
            "model": self.model_name,
            "weights": str(self.weights_path),
            "dataset": self.dataset_name,
            "preprocess": self.preprocess,
            "img_size": list(self.img_size),
            "inference_feature": self.inference_feature,
            "feature_dim": int(predictions.q_feats.shape[1]),
            "flip_tta": self.flip_tta,
            "mAP": round(float(mAP), 4),
            "rank1": round(float(cmc[0]), 4) if len(cmc) > 0 else 0.0,
            "rank5": round(float(cmc[4]), 4) if len(cmc) > 4 else 0.0,
            "rank10": round(float(cmc[9]), 4) if len(cmc) > 9 else 0.0,
            **predictions.latency,
        }

def evaluate_reid(config: ReIDEvaluationConfig) -> dict[str, Any]:
    """Evaluate a trained ReID model on a dataset's query/gallery split."""

    return ReIDValidator(config).validate()


__all__ = (
    "ReIDEmbeddingPredictor",
    "ReIDEvaluationConfig",
    "ReIDPredictionResult",
    "ReIDValidator",
    "evaluate_reid",
)
