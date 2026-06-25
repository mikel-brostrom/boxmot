"""Engine entry point for standalone ReID model evaluation.

Invoked by the CLI ``eval-reid`` subcommand via ``main(args)``.
Loads a trained checkpoint and evaluates on query/gallery splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.datasets import build_dataset
from boxmot.reid.datasets.torch_dataset import ReIDImageDataset
from boxmot.reid.datasets.transforms import build_test_transforms
from boxmot.reid.training.evaluator import (
    compute_distance_matrix,
    evaluate_ranking,
    extract_features,
)
from boxmot.utils import logger as LOGGER


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


def _eval_json_name(model_name: str, dataset_name: str, inference_feature: str | None) -> str:
    """Return a stable eval result filename without overwriting feature-mode sweeps."""
    if inference_feature:
        return f"eval_{model_name}_{dataset_name}_{inference_feature}.json"
    return f"eval_{model_name}_{dataset_name}.json"


def main(args) -> dict:
    """Evaluate a trained ReID model on a dataset's query/gallery split.

    Args:
        args: Namespace with evaluation parameters.

    Returns:
        Dict with mAP, rank1, rank5, rank10.
    """
    weights_path = Path(args.weights)
    device = torch.device(getattr(args, "device", "cpu"))
    batch_size = getattr(args, "batch_size", 64)
    num_workers = getattr(args, "num_workers", 4)
    flip_tta_arg = getattr(args, "flip_tta", None)

    # 1. Load checkpoint
    LOGGER.info(f"Loading checkpoint from {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    hparams = _load_hparams(weights_path)

    # Resolve model name and num_classes from checkpoint or CLI
    model_name = getattr(args, "model", None) or checkpoint.get("model_name")
    if model_name is None:
        raise ValueError(
            "Cannot determine model architecture. Provide --model or use a checkpoint that stores 'model_name'."
        )

    num_classes = checkpoint.get("num_classes", -1)
    preprocess = (
        getattr(args, "preprocess", None)
        or checkpoint.get("preprocess")
        or _hparams_value(hparams, "preprocess", "resize")
    )
    flip_tta = bool(_hparams_value(hparams, "flip_tta", False)) if flip_tta_arg is None else bool(flip_tta_arg)
    model_kwargs = ReIDModelRegistry.get_checkpoint_model_kwargs(weights_path)
    inference_feature_override = getattr(args, "inference_feature", None)
    if inference_feature_override:
        model_kwargs["inference_feature"] = inference_feature_override

    # 2. Load dataset
    dataset_name = args.dataset
    data_dir = args.data_dir
    LOGGER.info(f"Loading dataset '{dataset_name}' from {data_dir}")
    dataset = build_dataset(dataset_name, data_dir)

    # If num_classes not in checkpoint, use dataset's
    if num_classes <= 0:
        num_classes = dataset.num_train_pids

    # 3. Build model and load weights
    LOGGER.info(f"Building model '{model_name}' with {num_classes} classes")
    model = ReIDModelRegistry.build_model(
        model_name,
        weights_path,
        num_classes=num_classes,
        loss="softmax",
        pretrained=False,
        use_gpu=device.type != "cpu",
        **model_kwargs,
    )
    model.load_state_dict(state_dict, strict=False)
    if inference_feature_override and hasattr(model, "head") and hasattr(model.head, "inference_feature"):
        model.head.inference_feature = inference_feature_override
    model = model.to(device)
    model.eval()

    # 4. Build test loaders
    img_size = getattr(args, "imgsz", None) or _hparams_value(hparams, "img_size") or (256, 128)
    if isinstance(img_size, int):
        img_size = (img_size, img_size // 2)
    elif isinstance(img_size, list):
        img_size = tuple(img_size)
    transform = build_test_transforms(img_size, preprocess=preprocess)

    query_ds = ReIDImageDataset(dataset.query.samples, transform=transform)
    gallery_ds = ReIDImageDataset(dataset.gallery.samples, transform=transform)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=num_workers > 0,
    )
    query_loader = torch.utils.data.DataLoader(query_ds, **loader_kwargs)
    gallery_loader = torch.utils.data.DataLoader(gallery_ds, **loader_kwargs)

    # 5. Extract features and evaluate
    LOGGER.info(f"Extracting features ({len(query_ds)} query, {len(gallery_ds)} gallery)...")
    q_feats, q_pids, q_camids = extract_features(model, query_loader, device, flip_tta=flip_tta)
    g_feats, g_pids, g_camids = extract_features(model, gallery_loader, device, flip_tta=flip_tta)

    LOGGER.info("Computing distance matrix and evaluating...")
    distmat = compute_distance_matrix(q_feats, g_feats)

    cmc, mAP = evaluate_ranking(distmat, q_pids, g_pids, q_camids, g_camids)

    results = {
        "model": model_name,
        "weights": str(weights_path),
        "dataset": dataset_name,
        "preprocess": preprocess,
        "img_size": list(img_size),
        "inference_feature": model_kwargs.get("inference_feature"),
        "feature_dim": int(q_feats.shape[1]),
        "flip_tta": flip_tta,
        "mAP": round(float(mAP), 4),
        "rank1": round(float(cmc[0]), 4) if len(cmc) > 0 else 0.0,
        "rank5": round(float(cmc[4]), 4) if len(cmc) > 4 else 0.0,
        "rank10": round(float(cmc[9]), 4) if len(cmc) > 9 else 0.0,
    }
    LOGGER.info(
        f"Results on {dataset_name}:  "
        f"mAP={results['mAP']:.2%}  "
        f"R1={results['rank1']:.2%}  "
        f"R5={results['rank5']:.2%}  "
        f"R10={results['rank10']:.2%}"
    )

    # 6. Optionally save results JSON next to weights
    output_dir = getattr(args, "output", None)
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / _eval_json_name(model_name, dataset_name, inference_feature_override)
    else:
        if inference_feature_override:
            json_path = weights_path.parent / f"eval_{dataset_name}_{inference_feature_override}.json"
        else:
            json_path = weights_path.parent / f"eval_{dataset_name}.json"
    json_path.write_text(json.dumps(results, indent=2))
    LOGGER.info(f"Saved evaluation results to {json_path}")

    return results
