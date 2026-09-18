"""Reusable model preparation for ReID export backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from boxmot.reid.backbones import get_backbone_spec
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.core.runtime import ReID
from boxmot.utils.devices import resolve_device


def default_export_img_size(weights: Path, model_name: str) -> tuple[int, int]:
    """Resolve a model's registered crop while preserving square vehicle crops."""
    weights_name = weights.name.lower()
    if "vehicleid" in weights_name or "veri" in weights_name:
        return (256, 256)
    checkpoint_size = ReIDModelRegistry.get_checkpoint_model_kwargs(weights).get("img_size")
    if checkpoint_size:
        return tuple(checkpoint_size)
    try:
        return get_backbone_spec(model_name).default_img_size
    except (KeyError, TypeError):
        return (256, 128)


def prepare_export_model(args: Any) -> tuple[torch.nn.Module, torch.Tensor]:
    """Load and warm a ReID model for an explicitly requested export."""
    args.device = resolve_device(args.device)
    include = tuple(str(fmt).lower() for fmt in (getattr(args, "include", ()) or ()))
    cpu_fp16_graph_export = bool(args.half and args.device.type == "cpu" and "engine" not in include)
    if args.half and args.device.type == "cpu" and not cpu_fp16_graph_export:
        raise AssertionError("--half TensorRT export requires GPU, use --device 0")

    backend_half = bool(args.half and not cpu_fp16_graph_export)
    reid = ReID(weights=args.weights, device=args.device, half=backend_half)
    model_name = ReIDModelRegistry.get_model_name(args.weights)
    model = reid.model.model.eval()

    if args.optimize and args.device.type != "cpu":
        raise AssertionError("--optimize not compatible with CUDA devices, use --device cpu")

    args.imgsz = default_export_img_size(args.weights, model_name)

    if backend_half:
        model = model.half()

    first_param = next(model.parameters(), None)
    if first_param is not None:
        model_dtype = first_param.dtype
    else:
        first_buffer = next(model.buffers(), None)
        model_dtype = first_buffer.dtype if first_buffer is not None else torch.float32
    dummy_input = torch.empty(
        args.batch_size,
        3,
        args.imgsz[0],
        args.imgsz[1],
        device=args.device,
        dtype=model_dtype,
    )
    for _ in range(2):
        _ = model(dummy_input)

    return model, dummy_input


__all__ = ("default_export_img_size", "prepare_export_model")
