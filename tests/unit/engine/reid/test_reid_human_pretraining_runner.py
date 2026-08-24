"""Focused tests for the exact-encoder human-pretraining engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from boxmot.engine.reid import human_pretrainer
from boxmot.engine.reid.human_pretrainer import (
    HumanPretrainConfig,
    HumanPretrainingDataset,
    forward_exact_tinyvit_encoder,
    run_human_pretraining,
)
from boxmot.reid.backbones.families.csl_tinyvit import csl_tinyvit_7m_v20
from boxmot.reid.backbones.families.csl_tinyvit.pretrained import (
    load_pretrained_tinyvit_checkpoint,
)
from boxmot.reid.training.human_pretraining import export_tinyvit_backbone_checkpoint
from boxmot.reid.training.provenance import model_pretrained_provenance


def _write_sample(root: Path, name: str = "sample") -> tuple[Path, Path]:
    image_path = root / f"{name}.jpg"
    Image.new("RGB", (8, 16), color=(100, 120, 140)).save(image_path)
    target_path = root / f"{name}.pt"
    part_maps = torch.zeros(2, 16, 8)
    part_maps[0, :8] = 1
    part_maps[1, 8:] = 1
    torch.save(
        {
            "part_maps": part_maps,
            "foreground_mask": torch.ones(16, 8),
            "teacher_features": torch.randn(2, 4, 2),
        },
        target_path,
    )
    return image_path, target_path


def test_human_pretraining_dataset_resolves_relative_manifest_paths(tmp_path):
    image_path, target_path = _write_sample(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image": image_path.name, "target": target_path.name}) + "\n",
        encoding="utf-8",
    )

    dataset = HumanPretrainingDataset(manifest, (12, 6))
    sample = dataset[0]

    assert dataset.teacher_channels == 2
    assert sample["view_a"].shape == (3, 12, 6)
    assert sample["view_b"].shape == (3, 12, 6)
    assert sample["part_maps"].shape == (2, 12, 6)
    assert sample["foreground_mask"].shape == (12, 6)
    assert sample["teacher_features"].shape == (2, 4, 2)


def test_exact_encoder_forward_bypasses_reid_head_and_backpropagates():
    model = csl_tinyvit_7m_v20(
        num_classes=2,
        loss="softmax",
        pretrained=False,
        img_size=(64, 32),
    )
    output = forward_exact_tinyvit_encoder(model, torch.randn(1, 3, 64, 32))

    assert output.ndim == 4
    assert output.shape[1] == 320
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.patch_embed.parameters())
    assert all(parameter.grad is None for parameter in model.head.parameters())


def test_local_human_pretrained_checkpoint_loader_is_exact(tmp_path):
    source = csl_tinyvit_7m_v20(num_classes=2, loss="softmax", pretrained=False)
    with torch.no_grad():
        source.patch_embed.seq[0].c.weight.fill_(0.125)
    weights = export_tinyvit_backbone_checkpoint(source, tmp_path / "human.pt")
    target = csl_tinyvit_7m_v20(num_classes=3, loss="triplet", pretrained=False)

    load_pretrained_tinyvit_checkpoint(target, weights)

    required = {
        key
        for key in source.state_dict()
        if key.startswith(("patch_embed.", "layers."))
        and ".reid_adapters." not in key
        and not key.endswith((".attention_bias_h", ".attention_bias_w"))
    }
    assert target.pretrained_backbone_tensor_coverage == 1.0
    expected_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert target.pretrained_sha256 == expected_sha256
    assert model_pretrained_provenance(target)["sha256"] == expected_sha256
    assert required
    for key in required:
        torch.testing.assert_close(target.state_dict()[key], source.state_dict()[key])

    incomplete = tmp_path / "incomplete.pt"
    torch.save({"state_dict": {"patch_embed.invalid": torch.ones(1)}}, incomplete)
    with pytest.raises(RuntimeError, match="Incomplete local TinyViT"):
        load_pretrained_tinyvit_checkpoint(target, incomplete)


class _FakePatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)


class _FakeLayer(nn.Module):
    dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(4, 4, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        features = self.projection(features)
        return features.flatten(2).transpose(1, 2), output_size


class _FakeTinyViT(nn.Module):
    identity_registers_enabled = False
    body_slots_enabled = False
    stage1_width_merge = None

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _FakePatchEmbed()
        self.layers = nn.ModuleList([_FakeLayer()])
        self.head = nn.Linear(4, 1)


def test_human_pretraining_runner_exports_only_the_encoder(tmp_path, monkeypatch):
    image_path, target_path = _write_sample(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"image": image_path.name, "target": target_path.name}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(human_pretrainer, "CSLTinyViT", _FakeTinyViT)
    monkeypatch.setattr(
        human_pretrainer.ReIDModelRegistry,
        "build_model",
        lambda **kwargs: _FakeTinyViT(),
    )
    output = tmp_path / "human-backbone.pt"

    result = run_human_pretraining(
        HumanPretrainConfig(
            manifest=manifest,
            output=output,
            img_size=(16, 8),
            epochs=1,
            batch_size=1,
            workers=0,
            pretrained=False,
            amp=False,
            device="cpu",
            log_interval=1,
        )
    )

    checkpoint = torch.load(result.output_path, map_location="cpu", weights_only=True)
    assert result.output_path == output
    assert result.resume_path.is_file()
    assert torch.isfinite(torch.tensor(result.final_loss))
    assert checkpoint["metadata"]["privileged_inputs_exported"] is False
    assert checkpoint["state_dict"]
    assert all(key.startswith(("patch_embed.", "layers.")) for key in checkpoint["state_dict"])
    assert not any(key.startswith("head.") for key in checkpoint["state_dict"])

    resumed_output = tmp_path / "human-backbone-resumed.pt"
    resumed = run_human_pretraining(
        HumanPretrainConfig(
            manifest=manifest,
            output=resumed_output,
            img_size=(16, 8),
            epochs=2,
            batch_size=1,
            workers=0,
            pretrained=False,
            resume=result.resume_path,
            amp=False,
            device="cpu",
            log_interval=1,
        )
    )
    assert resumed.output_path == resumed_output
    assert torch.isfinite(torch.tensor(resumed.final_loss))
