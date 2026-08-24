"""Focused coverage for the V20 privileged pose/mask treatments."""

from __future__ import annotations

import torch
from PIL import Image

from boxmot.reid.datasets.base import ReIDSample
from boxmot.reid.datasets.torch_dataset import ReIDImageDataset
from boxmot.reid.training.trainer import ReIDTrainer


def test_query_relational_distillation_detaches_teacher_and_backpropagates():
    teacher = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.9, 0.0], [0.9, 0.1, 0.0]],
        ],
        requires_grad=True,
    )
    student = teacher.detach().clone()
    student[1, 0] = torch.tensor([0.0, 0.0, 1.0])
    student.requires_grad_()
    loss = ReIDTrainer._query_relational_distill_loss(
        student,
        teacher,
        torch.ones(4, 2),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
    )

    assert loss.item() > 0
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_query_relational_distillation_ignores_invisible_parts():
    teacher = torch.randn(4, 2, 8)
    student = teacher.clone()
    student[:, 1] = torch.randn_like(student[:, 1])
    reliability = torch.ones(4, 2)
    reliability[:, 1] = 0

    loss = ReIDTrainer._query_relational_distill_loss(
        student,
        teacher,
        reliability,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
    )

    assert loss.item() == 0.0


def test_clean_student_consistency_uses_clean_teacher_and_detaches_it():
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_accessory_query = False
    trainer.anatomical_local_scale_weight = 0.6
    trainer.anatomical_fine_scale_weight = 0.4
    clean_local = torch.randn(4, 2, 8, requires_grad=True)
    clean_fine = torch.randn(4, 2, 8, requires_grad=True)
    clean_descriptor = torch.randn(4, 12, requires_grad=True)
    student_local = clean_local.detach().clone()
    student_fine = clean_fine.detach().clone()
    student_descriptor = clean_descriptor.detach().clone()
    student_local[0, 0] = torch.randn(8)
    student_descriptor[0] = torch.randn(12)
    student_local.requires_grad_()
    student_fine.requires_grad_()
    student_descriptor.requires_grad_()
    student_features = {
        "_anatomical_query_student_tokens": student_local,
        "_anatomical_query_fine_student_tokens": student_fine,
        "norm_concat_bn": student_descriptor,
    }
    clean_features = {
        "_anatomical_query_teacher_tokens": clean_local,
        "_anatomical_query_teacher_valid": torch.ones(4, 2),
        "_anatomical_query_fine_teacher_tokens": clean_fine,
        "_anatomical_query_fine_teacher_valid": torch.ones(4, 2),
        "norm_concat_bn": clean_descriptor,
    }
    masks = torch.ones(4, 2, 6, 3)
    clean_targets = {
        "masks": masks,
        "foreground_mask": torch.ones(4, 1, 6, 3),
        "visibility": torch.ones(4, 2),
        "reliability": torch.ones(4, 2),
        "mask_valid": torch.ones(4, dtype=torch.bool),
        "valid": torch.ones(4, dtype=torch.bool),
    }

    loss = trainer._clean_teacher_student_consistency_loss(
        student_features,
        clean_features,
        clean_targets,
        torch.arange(4),
    )

    assert loss.item() > 0
    loss.backward()
    assert student_local.grad is not None
    assert student_descriptor.grad is not None
    assert clean_local.grad is None
    assert clean_fine.grad is None
    assert clean_descriptor.grad is None


def test_dataset_builds_clean_target_from_unmodified_source(tmp_path):
    image_path = tmp_path / "0001_c1s1_000001_00.jpg"
    Image.new("RGB", (4, 8), "white").save(image_path)

    class TargetTransform:
        def __init__(self, delta: float):
            self.delta = delta

        def apply_with_anatomical_target(self, image, target):
            transformed = dict(target)
            transformed["masks"] = target["masks"].clone() + self.delta
            return torch.zeros(3, 8, 4), transformed

    dataset = ReIDImageDataset(
        [ReIDSample(img_path=image_path, pid=1, camid=0)],
        transform=TargetTransform(1.0),
        return_clean_view=True,
        clean_transform=TargetTransform(2.0),
        return_clean_anatomical_target=True,
        anatomical_target_provider=lambda _index, _size: {
            "masks": torch.zeros(2, 8, 4)
        },
    )

    _, _, _, _, paired, target = dataset[0]

    assert paired is True
    assert torch.all(target["masks"] == 1)
    assert torch.all(target["_clean_view"]["masks"] == 2)
