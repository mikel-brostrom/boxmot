"""Focused tests for the offline HP-GRD teacher-signal extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from boxmot.engine.reid import teacher_extractor
from boxmot.engine.reid.privileged_cache import load_precomputed_teacher_tensors
from boxmot.engine.reid.teacher_extractor import (
    OfflineTeacherSignalDataset,
    PartMaskSignalStore,
    TeacherExtractionConfig,
    extract_teacher_signal_bundle,
    resolve_teacher_descriptor,
    run_teacher_extraction,
)
from boxmot.reid.datasets.anatomical import ANATOMICAL_PARTS
from boxmot.reid.training.trainer_components.privileged_graph import (
    DatasetSampleProvenance,
    PrivilegedGraphTeacherCache,
)


class _FiveDimensionalTeacher(nn.Module):
    """Small deterministic mapping-output teacher with an arbitrary width."""

    def forward(self, images: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        channel_mean = images.mean(dim=(2, 3))
        descriptor = torch.cat(
            (
                channel_mean,
                channel_mean[:, :1] - channel_mean[:, 1:2],
                channel_mean[:, 1:2] - channel_mean[:, 2:3],
            ),
            dim=1,
        )
        return {"teacher": {"descriptor": descriptor}}


def _write_images_and_index(root: Path) -> tuple[Path, tuple[DatasetSampleProvenance, ...]]:
    Image.new("RGB", (4, 8), color=(220, 40, 20)).save(root / "seven.png")
    Image.new("RGB", (4, 8), color=(20, 180, 80)).save(root / "forty-one.png")
    rows = [
        {"index": 7, "img_path": "seven.png", "pid": 1, "camid": 0},
        {"index": 41, "img_path": "forty-one.png", "pid": 2, "camid": 1},
    ]
    index = root / "samples.json"
    index.write_text(json.dumps(rows), encoding="utf-8")
    return index, tuple(DatasetSampleProvenance(**row) for row in rows)


def _mask_store() -> PartMaskSignalStore:
    # Store rows deliberately disagree with dataset row order. Stable indices,
    # not dataloader positions, must select the correct masks and confidence.
    masks = torch.zeros(2, 2, 8, 4)
    masks[0, 0, :, 2:] = 1  # stable index 41: right half
    masks[0, 1, :, :2] = 1
    masks[1, 0, :4] = 1  # stable index 7: top half
    # The second part is absent and must never generate a teacher intervention.
    visibility = torch.tensor([[0.8, 0.6], [0.9, 0.7]])
    confidence = torch.tensor([[0.7, 0.5], [0.8, 0.6]])
    return PartMaskSignalStore(
        sample_indices=torch.tensor([41, 7]),
        part_masks=masks,
        part_visibility=visibility,
        part_confidence=confidence,
        part_names=("upper", "lower"),
    )


def test_extractor_uses_stable_indices_and_like_for_like_mask_interventions(tmp_path) -> None:
    _, samples = _write_images_and_index(tmp_path)
    dataset = OfflineTeacherSignalDataset(
        samples,
        image_root=tmp_path,
        img_size=(8, 4),
        preprocess="resize",
        mask_store=_mask_store(),
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    teacher = _FiveDimensionalTeacher().train()

    tensors = extract_teacher_signal_bundle(
        teacher,
        dataloader,
        device="cpu",
        part_names=("upper", "lower"),
        descriptor_key="teacher.descriptor",
        include_leave_part_out=True,
        global_confidence_from_parts=True,
        normalize_descriptors=False,
        amp=False,
        max_intervention_batch=1,
    )

    assert teacher.training is True
    assert tensors["sample_indices"].tolist() == [7, 41]
    assert tensors["global_descriptors"].shape == (2, 5)
    assert tensors["part_descriptors"].shape == (2, 2, 5)
    assert tensors["leave_part_out_descriptors"].shape == (2, 2, 5)
    assert tensors["part_visibility"][0].tolist() == pytest.approx([0.9, 0.0])
    assert tensors["part_confidence"][0].tolist() == pytest.approx([0.8, 0.0])
    assert tensors["global_confidence"].tolist() == pytest.approx([0.85, 0.75])
    assert torch.count_nonzero(tensors["part_descriptors"][0, 1]) == 0
    assert torch.count_nonzero(tensors["leave_part_out_descriptors"][0, 1]) == 0

    first = dataset[0]
    image = first["images"].unsqueeze(0)
    mask = first["part_masks"][0][None, None]
    expected_part = teacher(image * mask)["teacher"]["descriptor"][0]
    expected_leave_out = teacher(image * (1 - mask))["teacher"]["descriptor"][0]
    torch.testing.assert_close(tensors["part_descriptors"][0, 0], expected_part)
    torch.testing.assert_close(tensors["leave_part_out_descriptors"][0, 0], expected_leave_out)

    # The exact downstream object accepts arbitrary teacher width because the
    # distillation objective transfers relation matrices, not coordinates.
    cache = PrivilegedGraphTeacherCache(part_names=("upper", "lower"), **tensors)
    assert cache.lookup([41, 7]).global_descriptors.shape == (2, 5)


def test_dataset_accepts_existing_anatomical_target_provider_and_resize_pad(tmp_path) -> None:
    _, samples = _write_images_and_index(tmp_path)

    def provider(index: int, size: tuple[int, int]) -> dict[str, torch.Tensor]:
        width, height = size
        masks = torch.zeros(2, height, width)
        masks[index, :, :] = 1
        return {
            "masks": masks,
            "visibility": torch.tensor([0.8, 0.9]),
            "reliability": torch.tensor([0.4, 0.7]),
        }

    dataset = OfflineTeacherSignalDataset(
        samples,
        image_root=tmp_path,
        img_size=(8, 8),
        preprocess="resize_pad",
        target_provider=provider,
    )
    first = dataset[0]
    second = dataset[1]

    assert first["images"].shape == (3, 8, 8)
    assert first["part_masks"].shape == (2, 8, 8)
    # 4x8 images are centered in the 8x8 canvas; padding carries no part mask.
    assert torch.count_nonzero(first["part_masks"][:, :, :2]) == 0
    assert torch.count_nonzero(first["part_masks"][:, :, 6:]) == 0
    assert first["part_visibility"].tolist() == pytest.approx([0.8, 0.0])
    assert second["part_confidence"].tolist() == pytest.approx([0.0, 0.7])


def test_descriptor_resolution_supports_nested_keys_and_spatial_teacher_features() -> None:
    spatial = torch.arange(2 * 7 * 3 * 2, dtype=torch.float32).reshape(2, 7, 3, 2)

    descriptor = resolve_teacher_descriptor({"packet": {"map": spatial}}, "packet.map")

    assert descriptor.shape == (2, 7)
    torch.testing.assert_close(descriptor, spatial.flatten(2).mean(dim=2))
    with pytest.raises(KeyError, match="descriptor path"):
        resolve_teacher_descriptor({"packet": {}}, "packet.missing")


def test_run_api_publishes_tensor_input_consumed_by_cache_builder(tmp_path) -> None:
    dataset_index, _ = _write_images_and_index(tmp_path)
    mask_path = tmp_path / "masks.pt"
    store = _mask_store()
    # Save in unsorted order to exercise stable-index lookup in the runner.
    torch.save(
        {
            "sample_indices": torch.tensor([41, 7]),
            "part_masks": store.part_masks.flip(0),
            "part_visibility": store.part_visibility.flip(0),
            "part_confidence": store.part_confidence.flip(0),
            "part_names": list(store.part_names),
        },
        mask_path,
    )
    output = tmp_path / "teacher-signals.pt"

    result = run_teacher_extraction(
        TeacherExtractionConfig(
            dataset_index=dataset_index,
            image_root=tmp_path,
            output=output,
            part_mask_input=mask_path,
            img_size=(8, 4),
            preprocess="resize",
            batch_size=2,
            workers=0,
            device="cpu",
            amp=False,
            descriptor_key="teacher.descriptor",
            include_leave_part_out=True,
        ),
        model=_FiveDimensionalTeacher(),
    )
    loaded = load_precomputed_teacher_tensors(output)

    assert result.output_path == output
    assert result.sample_count == 2
    assert result.part_count == 2
    assert result.global_dim == 5
    assert result.leave_part_out_dim == 5
    assert result.part_names == ("upper", "lower")
    assert loaded["sample_indices"].tolist() == [7, 41]
    assert set(loaded) == {
        "sample_indices",
        "global_descriptors",
        "part_descriptors",
        "part_visibility",
        "part_confidence",
        "leave_part_out_descriptors",
    }
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_teacher_extraction(
            TeacherExtractionConfig(
                dataset_index=dataset_index,
                image_root=tmp_path,
                output=output,
                part_mask_input=mask_path,
                img_size=(8, 4),
                preprocess="resize",
                workers=0,
                device="cpu",
            ),
            model=_FiveDimensionalTeacher(),
        )


def test_cli_loads_registered_teacher_without_downloads(tmp_path, monkeypatch, capsys) -> None:
    dataset_index, _ = _write_images_and_index(tmp_path)
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_checkpoint.write_bytes(b"local teacher provenance")
    masks = torch.ones(2, 1, 8, 4)
    mask_path = tmp_path / "masks.pt"
    torch.save({"sample_indices": torch.tensor([7, 41]), "part_masks": masks}, mask_path)
    output = tmp_path / "signals.pt"

    monkeypatch.setattr(
        teacher_extractor,
        "load_registered_teacher",
        lambda *args, **kwargs: (_FiveDimensionalTeacher(), "large_test_teacher", {"img_size": (8, 4)}),
    )
    monkeypatch.setattr(
        teacher_extractor.ReIDModelRegistry,
        "get_checkpoint_preprocess",
        lambda checkpoint: "resize",
    )

    assert (
        teacher_extractor.main(
            [
                "--teacher",
                str(teacher_checkpoint),
                "--dataset-index",
                str(dataset_index),
                "--image-root",
                str(tmp_path),
                "--part-mask-input",
                str(mask_path),
                "--output",
                str(output),
                "--workers",
                "0",
                "--device",
                "cpu",
                "--descriptor-key",
                "teacher.descriptor",
                "--part-names",
                "whole_person",
                "--no-amp",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["model_name"] == "large_test_teacher"
    assert summary["global_dim"] == 5
    assert summary["part_names"] == ["whole_person"]
    assert output.is_file()


def test_unnamed_six_part_mask_input_defaults_to_canonical_anatomical_order(tmp_path) -> None:
    dataset_index, _ = _write_images_and_index(tmp_path)
    mask_path = tmp_path / "six-parts.pt"
    masks = torch.ones(2, len(ANATOMICAL_PARTS), 8, 4)
    torch.save({"sample_indices": torch.tensor([7, 41]), "part_masks": masks}, mask_path)

    result = run_teacher_extraction(
        TeacherExtractionConfig(
            dataset_index=dataset_index,
            image_root=tmp_path,
            output=tmp_path / "canonical-signals.pt",
            part_mask_input=mask_path,
            img_size=(8, 4),
            preprocess="resize",
            batch_size=2,
            workers=0,
            device="cpu",
            amp=False,
            descriptor_key="teacher.descriptor",
        ),
        model=_FiveDimensionalTeacher(),
    )

    payload = torch.load(result.output_path, map_location="cpu", weights_only=True)
    assert result.part_names == ANATOMICAL_PARTS
    assert payload["part_names"] == list(ANATOMICAL_PARTS)
