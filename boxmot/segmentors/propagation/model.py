"""Construct the official EdgeTAM model without redundant pretrained weights."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from importlib import import_module
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from boxmot.segmentors.propagation.weights import resolve_edgetam_checkpoint
from boxmot.utils.devices import normalize_device, resolve_device

EDGETAM_SOURCE = "https://github.com/facebookresearch/EdgeTAM"
EDGETAM_REVISION = "7711e012a30a2402c4eaab637bdb00a521302c91"
_MODEL_CONFIG = "boxmot_official_edgetam"


def effective_precision(device: str | torch.device) -> str:
    """Use FP16 on MPS, BF16 on capable CUDA devices, and FP32 otherwise."""
    selected = torch.device(normalize_device(device))
    if selected.type == "mps":
        return "fp16"
    if selected.type == "cuda" and torch.cuda.get_device_capability(selected)[0] >= 8:
        return "bf16"
    return "fp32"


def _resolve_precision(selected: torch.device, precision: str | None) -> str:
    """Validate precision before allocating a model or entering autocast."""
    precision = effective_precision(selected) if precision is None else precision
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("EdgeTAM precision must be fp32, fp16, or bf16.")
    if precision != "fp32" and selected.type not in {"cuda", "mps"}:
        raise ValueError("EdgeTAM supports fp32 on CPU; reduced precision requires CUDA or MPS.")
    if precision == "bf16" and (selected.type != "cuda" or effective_precision(selected) != "bf16"):
        raise ValueError("EdgeTAM bf16 requires a CUDA device with bfloat16 support.")
    return precision


@contextmanager
def inference_context(device: str | torch.device, precision: str | None = None) -> Iterator[None]:
    """Run inference with a supported, local precision context and no global changes."""
    selected = torch.device(normalize_device(device))
    precision = _resolve_precision(selected, precision)
    autocast = (
        nullcontext()
        if precision == "fp32"
        else torch.autocast(selected.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
    )
    with torch.inference_mode(), autocast:
        yield


def postprocessing_metadata(device: str | torch.device) -> dict[str, Any]:
    """Describe upstream postprocessing, including optional CUDA hole filling."""
    selected = torch.device(normalize_device(device))
    # The extension is optional in the official distribution. CPU/MPS cannot
    # run its CUDA kernel; avoid repeated upstream warnings for those devices.
    extension = False
    if selected.type == "cuda" and find_spec("sam2") is not None and find_spec("sam2._C") is not None:
        # An optional compiled extension can exist but fail to load because its
        # PyTorch/CUDA ABI differs. Treat that specific optional capability as absent.
        try:
            extension = callable(getattr(import_module("sam2._C"), "get_connected_componnets", None))
        except (ImportError, OSError):
            extension = False
    return {
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
        "binarize_mask_from_pts_for_mem_enc": True,
        "requested_fill_hole_area": 8,
        "effective_fill_hole_area": 8 if extension else 0,
    }


def _register_model_config() -> str:
    """Register the official YAML because upstream wheels omit its config data."""
    config_store = import_module("hydra.core.config_store").ConfigStore.instance()
    omega_conf = import_module("omegaconf").OmegaConf
    resource = files("boxmot.segmentors.propagation").joinpath("edgetam.yaml")
    config_store.store(name=_MODEL_CONFIG, node=omega_conf.create(resource.read_text()), provider="boxmot")
    return _MODEL_CONFIG


def build_edgetam_predictor(
    checkpoint: str | Path, device: str | torch.device, *, precision: str | None = None
) -> Any:
    """Load one official predictor and strict full checkpoint, with no backbone download."""
    selected = resolve_device(device)
    precision = _resolve_precision(selected, precision)
    checkpoint_path = resolve_edgetam_checkpoint(checkpoint)
    if find_spec("sam2") is None:
        raise ModuleNotFoundError(
            "EdgeTAM requires the optional official EdgeTAM package; install BoxMOT's mask-guidance group. "
            "See docs/trackers/bytetrack.md."
        )
    builder = import_module("sam2.build_sam").build_sam2_video_predictor
    postprocessing = postprocessing_metadata(selected)
    predictor = builder(
        _register_model_config(),
        str(checkpoint_path),
        device=str(selected),
        hydra_overrides_extra=[
            "model.image_encoder.trunk._target_=boxmot.segmentors.propagation.backbone.InferenceTimmBackbone",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            f"++model.fill_hole_area={postprocessing['effective_fill_hole_area']}",
        ],
        # Supply all builder defaults explicitly: upstream appends fill_hole_area
        # after caller overrides, which would otherwise override our device gate.
        apply_postprocessing=False,
    )
    from boxmot.segmentors.propagation.perceiver import _forward_2d

    # Upstream's expanded latent view only works for one object. Bind the
    # batch-safe implementation without changing modules or checkpoint keys.
    perceiver = predictor.spatial_perceiver
    perceiver.forward_2d = MethodType(_forward_2d, perceiver)
    if precision == "fp16":
        predictor.half()
    if selected.type == "mps":
        from boxmot.segmentors.propagation.prompt import _embed_points

        # Apply only to this MPS model; do not patch the optional package
        # globally or replace its modules, parameters or checkpoint keys.
        encoder = predictor.sam_prompt_encoder
        encoder._embed_points = MethodType(_embed_points, encoder)
        if precision == "fp16":
            from boxmot.segmentors.propagation.inference import _run_memory_encoder, _run_single_frame_inference

            # Upstream stores memory as BF16, unnecessarily discarding FP16
            # mantissa bits. Preserve FP16 in both prompting and propagation.
            predictor._run_single_frame_inference = MethodType(_run_single_frame_inference, predictor)
            predictor._run_memory_encoder = MethodType(_run_memory_encoder, predictor)
    return predictor
