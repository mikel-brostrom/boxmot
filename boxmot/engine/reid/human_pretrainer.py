"""Human-privileged pretraining for the exact deployed TinyViT encoder.

The command consumes an offline JSON/JSONL manifest.  Each record points to an
RGB crop and a ``.pt`` target cache containing parser/pose part maps, a person
foreground mask, and frozen semantic-teacher features.  Privileged tensors are
used only to train ``patch_embed`` and ``layers``; the exported checkpoint has
no parser, pose, projection, or auxiliary-head dependency.

Example manifest line::

    {"image": "images/0001.jpg", "target": "targets/0001.pt"}

Each target cache must be a mapping with ``part_maps`` (P,H,W),
``foreground_mask`` (H,W), and ``teacher_features`` (C,Ht,Wt).
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from boxmot.reid.backbones.families.csl_tinyvit.model import CSLTinyViT
from boxmot.reid.backbones.families.csl_tinyvit.pretrained import (
    load_pretrained_tinyvit_checkpoint,
)
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.reid.training.human_pretraining import (
    export_tinyvit_backbone_checkpoint,
    foreground_aware_patch_target_weights,
    pose_parser_guided_whole_part_mask,
    semantic_teacher_feature_reconstruction_loss,
    two_view_masked_consistency_loss,
)
from boxmot.utils import logger as LOGGER

_PART_KEYS = ("part_maps", "anatomical_masks", "pose_part_maps")
_FOREGROUND_KEYS = ("foreground_mask", "person_mask")
_TEACHER_KEYS = ("teacher_features", "semantic_teacher_features", "features")

__all__ = (
    "HumanPretrainConfig",
    "HumanPretrainResult",
    "HumanPretrainingDataset",
    "forward_exact_tinyvit_encoder",
    "run_human_pretraining",
)


@dataclass(frozen=True)
class HumanPretrainConfig:
    """Resolved settings for offline human-centric encoder pretraining."""

    manifest: Path
    output: Path
    model_name: str = "csl_tinyvit_7m_v20"
    img_size: tuple[int, int] = (384, 128)
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    mask_ratio: float = 0.50
    consistency_weight: float = 0.50
    teacher_weight: float = 1.0
    foreground_weight: float = 1.0
    background_weight: float = 0.10
    workers: int = 4
    device: str = ""
    seed: int = 42
    pretrained: bool = True
    initial_weights: Path | None = None
    resume: Path | None = None
    amp: bool = True
    log_interval: int = 20

    def validate(self) -> None:
        """Reject configurations that would make the objective ambiguous."""
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Human-pretraining manifest does not exist: {self.manifest}")
        if not self.model_name.startswith("csl_tinyvit"):
            raise ValueError("Human pretraining requires a CSL-TinyViT model")
        if len(self.img_size) != 2 or min(self.img_size) <= 0:
            raise ValueError(f"img_size must be a positive H,W pair, got {self.img_size}")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 < self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in (0, 1]")
        if self.teacher_weight <= 0 or self.consistency_weight < 0:
            raise ValueError("teacher_weight must be positive and consistency_weight non-negative")
        if self.foreground_weight < 0 or self.background_weight < 0:
            raise ValueError("foreground/background weights must be non-negative")
        if self.workers < 0 or self.log_interval <= 0:
            raise ValueError("workers must be non-negative and log_interval positive")
        if self.initial_weights is not None and not self.initial_weights.is_file():
            raise FileNotFoundError(f"Initial TinyViT weights do not exist: {self.initial_weights}")
        if self.resume is not None and not self.resume.is_file():
            raise FileNotFoundError(f"Human-pretraining resume checkpoint does not exist: {self.resume}")
        if self.initial_weights is not None and self.resume is not None:
            raise ValueError("initial_weights and resume are mutually exclusive")


@dataclass(frozen=True)
class HumanPretrainResult:
    """Artifacts and final scalar produced by a completed pretraining run."""

    output_path: Path
    resume_path: Path
    epochs: int
    final_loss: float


def _first_tensor(mapping: Mapping[str, Any], keys: Sequence[str], *, source: Path) -> torch.Tensor:
    for key in keys:
        value = mapping.get(key)
        if torch.is_tensor(value):
            return value.detach().float()
    raise KeyError(f"Target cache {source} is missing a tensor named one of {tuple(keys)}")


def _read_target_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, Mapping):
        raise TypeError(f"Target cache must be a mapping, got {type(cache).__name__}: {path}")
    nested = cache.get("targets")
    if isinstance(nested, Mapping):
        cache = nested
    part_maps = _first_tensor(cache, _PART_KEYS, source=path)
    foreground = _first_tensor(cache, _FOREGROUND_KEYS, source=path)
    teacher = _first_tensor(cache, _TEACHER_KEYS, source=path)
    if part_maps.ndim == 4 and part_maps.shape[0] == 1:
        part_maps = part_maps.squeeze(0)
    if foreground.ndim == 3 and foreground.shape[0] == 1:
        foreground = foreground.squeeze(0)
    if teacher.ndim == 4 and teacher.shape[0] == 1:
        teacher = teacher.squeeze(0)
    if part_maps.ndim != 3:
        raise ValueError(f"part_maps must have shape (P,H,W), got {tuple(part_maps.shape)} in {path}")
    if foreground.ndim != 2:
        raise ValueError(f"foreground_mask must have shape (H,W), got {tuple(foreground.shape)} in {path}")
    if teacher.ndim != 3:
        raise ValueError(f"teacher_features must have shape (C,H,W), got {tuple(teacher.shape)} in {path}")
    if not torch.isfinite(teacher).all():
        raise ValueError(f"teacher_features contain non-finite values: {path}")
    return part_maps.clamp_min(0), foreground.clamp(0, 1), teacher


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    records: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(parsed, Mapping) and "samples" in parsed:
            records = parsed["samples"]
        elif isinstance(parsed, Mapping):
            records = [parsed]
        else:
            records = parsed
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest must contain a non-empty list of samples: {path}")
    if not all(isinstance(record, Mapping) for record in records):
        raise TypeError("Every human-pretraining manifest sample must be a mapping")
    return [dict(record) for record in records]


class HumanPretrainingDataset(Dataset):
    """RGB crops paired with offline pose/parser and semantic-teacher caches."""

    def __init__(self, manifest: str | Path, img_size: tuple[int, int]) -> None:
        self.manifest = Path(manifest)
        self.img_size = tuple(int(value) for value in img_size)
        records = _read_manifest(self.manifest)
        self.samples: list[tuple[Path, Path]] = []
        for index, record in enumerate(records):
            image_value = record.get("image")
            target_value = record.get("target", record.get("targets"))
            if not isinstance(image_value, str) or not isinstance(target_value, str):
                raise ValueError(f"Manifest sample {index} requires string 'image' and 'target' paths")
            image_path = Path(image_value)
            target_path = Path(target_value)
            if not image_path.is_absolute():
                image_path = self.manifest.parent / image_path
            if not target_path.is_absolute():
                target_path = self.manifest.parent / target_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
            if not target_path.is_file():
                raise FileNotFoundError(f"Manifest target cache does not exist: {target_path}")
            self.samples.append((image_path, target_path))

        first_parts, _, first_teacher = _read_target_cache(self.samples[0][1])
        self.num_parts = int(first_parts.shape[0])
        self.teacher_channels = int(first_teacher.shape[0])
        self.teacher_grid = tuple(int(value) for value in first_teacher.shape[-2:])
        self.resize = T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR)
        self.photometric = T.Compose(
            [
                T.ColorJitter(brightness=0.25, contrast=0.20, saturation=0.20, hue=0.05),
                T.RandomGrayscale(p=0.10),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, target_path = self.samples[index]
        with Image.open(image_path) as image_file:
            image = self.resize(image_file.convert("RGB"))
            view_a = self.photometric(image)
            view_b = self.photometric(image)
        part_maps, foreground, teacher = _read_target_cache(target_path)
        if part_maps.shape[0] != self.num_parts:
            raise ValueError(f"Inconsistent part count in {target_path}: {part_maps.shape[0]} != {self.num_parts}")
        if teacher.shape[0] != self.teacher_channels:
            raise ValueError(
                f"Inconsistent teacher width in {target_path}: {teacher.shape[0]} != {self.teacher_channels}"
            )
        part_maps = F.interpolate(
            part_maps.unsqueeze(0),
            size=self.img_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        foreground = (
            F.interpolate(
                foreground[None, None],
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        if teacher.shape[-2:] != self.teacher_grid:
            teacher = F.interpolate(
                teacher.unsqueeze(0),
                size=self.teacher_grid,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return {
            "view_a": view_a,
            "view_b": view_b,
            "part_maps": part_maps,
            "foreground_mask": foreground,
            "teacher_features": teacher,
        }


def forward_exact_tinyvit_encoder(model: CSLTinyViT, images: torch.Tensor) -> torch.Tensor:
    """Run only the native ``patch_embed``/``layers`` path and return B,C,H,W.

    This deliberately bypasses ReID necks, fusion paths, heads, and every
    privileged module.  Consequently every optimized model tensor is present
    in the backbone-only handoff checkpoint.
    """
    if bool(getattr(model, "identity_registers_enabled", False)):
        raise ValueError("Exact encoder pretraining does not include identity-register add-ons")
    if bool(getattr(model, "body_slots_enabled", False)):
        raise ValueError("Exact encoder pretraining does not include body-slot add-ons")
    tokens = model.patch_embed(images)
    output_size = (tokens.shape[2], tokens.shape[3])
    tokens, output_size = model.layers[0](tokens, output_size)
    width_merge = getattr(model, "stage1_width_merge", None)
    if width_merge is not None:
        tokens, output_size = width_merge(tokens, output_size)
    for layer in model.layers[1:]:
        tokens, output_size = layer(tokens, output_size)
    batch_size, token_count, channels = tokens.shape
    if token_count != output_size[0] * output_size[1]:
        raise RuntimeError(f"TinyViT token/grid mismatch: {token_count} != {output_size[0]}x{output_size[1]}")
    return tokens.view(batch_size, *output_size, channels).permute(0, 3, 1, 2).contiguous()


def _resolve_device(value: str) -> torch.device:
    if not value:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(dict(payload), temporary_path)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _resume_path_for(output: Path) -> Path:
    return output.with_name(f"{output.stem}.pretrain-last.pt")


def _serializable_config(config: HumanPretrainConfig) -> dict[str, Any]:
    values = asdict(config)
    for key in ("manifest", "output", "initial_weights", "resume"):
        value = values.get(key)
        values[key] = str(value) if value is not None else None
    values["img_size"] = list(config.img_size)
    return values


def run_human_pretraining(config: HumanPretrainConfig) -> HumanPretrainResult:
    """Train the exact TinyViT RGB encoder and export a strict local handoff."""
    config.validate()
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    dataset = HumanPretrainingDataset(config.manifest, config.img_size)
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=loader_generator,
    )

    use_model_zoo = config.pretrained and config.initial_weights is None and config.resume is None
    model = ReIDModelRegistry.build_model(
        name=config.model_name,
        weights=Path(f"{config.model_name}_human_pretrain.pt"),
        num_classes=1,
        loss="softmax",
        pretrained=use_model_zoo,
        use_gpu=device.type == "cuda",
        img_size=config.img_size,
    )
    if not isinstance(model, CSLTinyViT):
        raise TypeError(f"Expected CSLTinyViT, got {type(model).__name__}")
    if config.initial_weights is not None and config.resume is None:
        load_pretrained_tinyvit_checkpoint(model, config.initial_weights)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.patch_embed, model.layers):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    student_channels = int(getattr(model.layers[-1], "dim"))
    projector = nn.Conv2d(student_channels, dataset.teacher_channels, kernel_size=1, bias=False)
    nn.init.trunc_normal_(projector.weight, std=0.02)
    model = model.to(device)
    projector = projector.to(device)
    backbone_parameters = [
        parameter
        for module in (model.patch_embed, model.layers)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [*backbone_parameters, *projector.parameters()],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    mask_generator = torch.Generator().manual_seed(config.seed + 1)
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Human-pretraining resume checkpoint must be a mapping")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        projector.load_state_dict(checkpoint["projector_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if isinstance(scaler_state, Mapping):
            scaler.load_state_dict(scaler_state)
        if torch.is_tensor(checkpoint.get("torch_rng_state")):
            torch.set_rng_state(checkpoint["torch_rng_state"])
        cuda_rng_state = checkpoint.get("cuda_rng_state_all")
        if torch.cuda.is_available() and isinstance(cuda_rng_state, list):
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if torch.is_tensor(checkpoint.get("loader_generator_state")):
            loader_generator.set_state(checkpoint["loader_generator_state"])
        if torch.is_tensor(checkpoint.get("mask_generator_state")):
            mask_generator.set_state(checkpoint["mask_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if start_epoch >= config.epochs:
            raise ValueError(f"Resume epoch {start_epoch} is not below configured epochs={config.epochs}")

    resume_path = _resume_path_for(config.output)
    final_loss = float("nan")
    model.train()
    projector.train()
    for epoch in range(start_epoch, config.epochs):
        loss_sum = 0.0
        sample_count = 0
        for step, batch in enumerate(loader):
            view_a = batch["view_a"].to(device, non_blocking=True)
            view_b = batch["view_b"].to(device, non_blocking=True)
            part_maps = batch["part_maps"].to(device, non_blocking=True)
            foreground = batch["foreground_mask"].to(device, non_blocking=True)
            teacher_features = batch["teacher_features"].to(device, non_blocking=True)
            whole_mask = pose_parser_guided_whole_part_mask(
                part_maps,
                mask_ratio=config.mask_ratio,
                generator=mask_generator,
                foreground_mask=foreground,
                missing_target="error",
            )
            if whole_mask is None:
                raise RuntimeError("Validated part maps unexpectedly produced no whole-part mask")
            masked_view = whole_mask.apply(view_a, fill_value=0.0)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                masked_features = forward_exact_tinyvit_encoder(model, masked_view)
                clean_features = forward_exact_tinyvit_encoder(model, view_b)
                projected = projector(masked_features)
                teacher_target = F.interpolate(
                    teacher_features,
                    size=masked_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                target_weights = foreground_aware_patch_target_weights(
                    foreground,
                    masked_features.shape[-2:],
                    foreground_weight=config.foreground_weight,
                    background_weight=config.background_weight,
                )
                consistency_loss = two_view_masked_consistency_loss(
                    masked_features,
                    clean_features,
                    whole_mask.pixel_mask,
                    valid_samples=whole_mask.valid_samples,
                    channel_dim=1,
                    detach_view_b=True,
                )
                teacher_loss = semantic_teacher_feature_reconstruction_loss(
                    projected,
                    teacher_target,
                    whole_mask.pixel_mask,
                    target_weights=target_weights,
                    valid_samples=whole_mask.valid_samples,
                    channel_dim=1,
                    loss_type="cosine",
                    detach_teacher=True,
                    missing_target="error",
                )
                loss = config.consistency_weight * consistency_loss + config.teacher_weight * teacher_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([*backbone_parameters, *projector.parameters()], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = int(view_a.shape[0])
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
            if (step + 1) % config.log_interval == 0:
                LOGGER.info(
                    f"Human pretrain epoch {epoch + 1}/{config.epochs}, "
                    f"step {step + 1}/{len(loader)}: loss={loss_sum / sample_count:.5f}"
                )
        scheduler.step()
        final_loss = loss_sum / max(sample_count, 1)
        _atomic_torch_save(
            {
                "format": "boxmot-human-pretrain-resume-v1",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "projector_state_dict": projector.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
                "loader_generator_state": loader_generator.get_state(),
                "mask_generator_state": mask_generator.get_state(),
                "config": _serializable_config(config),
                "loss": final_loss,
            },
            resume_path,
        )
        LOGGER.info(f"Human pretrain epoch {epoch + 1}/{config.epochs} complete: loss={final_loss:.5f}")

    output_path = export_tinyvit_backbone_checkpoint(
        model,
        config.output,
        metadata={
            "objective": "pose-parser-guided-human-masked-pretraining",
            "model_name": config.model_name,
            "img_size": list(config.img_size),
            "epochs": config.epochs,
            "final_loss": final_loss,
            "manifest": str(config.manifest.resolve()),
            "privileged_inputs_exported": False,
        },
    )
    LOGGER.info(f"Exported exact TinyViT backbone handoff to {output_path}")
    return HumanPretrainResult(
        output_path=output_path,
        resume_path=resume_path,
        epochs=config.epochs,
        final_loss=final_loss,
    )


def _parse_img_size(value: str) -> tuple[int, int]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("image size must be H,W, for example 384,128")
    return parts


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone engine parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", dest="model_name", default="csl_tinyvit_7m_v20")
    parser.add_argument("--imgsz", type=_parse_img_size, default=(384, 128))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.50)
    parser.add_argument("--consistency-weight", type=float, default=0.50)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--foreground-weight", type=float, default=1.0)
    parser.add_argument("--background-weight", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-weights", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> HumanPretrainResult:
    """Run human pretraining from command-line arguments."""
    arguments = vars(build_parser().parse_args(argv))
    arguments["img_size"] = arguments.pop("imgsz")
    config = HumanPretrainConfig(**arguments)
    return run_human_pretraining(config)


if __name__ == "__main__":
    main()
