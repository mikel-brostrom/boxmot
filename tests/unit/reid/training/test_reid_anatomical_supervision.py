"""Tests for training-only privileged anatomical supervision."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import nn

from boxmot.reid.backbones.families.csl_tinyvit.heads import MultiBranchHead
from boxmot.reid.backbones.families.csl_tinyvit.pooling import (
    AnatomicalAuxiliaryPool,
    DecoupledMaskedQueryTeacher,
    EMAAnatomicalAuxiliaryPool,
    PrivilegedMaskPoseAttentionAdapter,
)
from boxmot.reid.datasets.anatomical import (
    ANATOMICAL_CANONICAL_GRID_SIZE,
    ANATOMICAL_FLIP_PERMUTATION,
    ANATOMICAL_PARTS,
    COCO_KEYPOINT_FLIP_PERMUTATION,
    PoseAnatomicalTargetProvider,
)
from boxmot.reid.datasets.base import ReIDSample
from boxmot.reid.datasets.torch_dataset import ReIDImageDataset
from boxmot.reid.datasets.transforms import (
    EpochAwareCompose,
    Random2DTranslation,
)
from boxmot.reid.training.augmentations import (
    _build_anatomical_target_provider,
)
from boxmot.reid.training.config import (
    AugmentationConfig,
    ReIDTrainConfig,
    load_train_hparams,
)
from boxmot.reid.training.resume import contract_differences
from boxmot.reid.training.trainer import (
    ReIDTrainer,
    _cross_scale_role_relation_loss,
    _scale_aware_anatomical_targets,
)


def test_anatomical_forward_kwargs_are_empty_when_auxiliary_is_disabled():
    """An RGB control must not inherit pose requirements from its recipe."""
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = False
    trainer.anatomical_target_type = "learned_pose_concat_ema"

    assert trainer._anatomical_forward_kwargs(None, dtype=torch.float32) == {}


def test_cross_camera_contrastive_skips_candidate_empty_rows(monkeypatch):
    """Never differentiate logsumexp through an all-negative-infinity row."""
    tokens = torch.randn(6, 2, 16, requires_grad=True)
    pids = torch.tensor([0, 0, 1, 1, 2, 3])
    camera_ids = torch.tensor([0, 1, 0, 1, 0, 0])
    reliability = torch.zeros(6, 2)
    reliability[:4, 0] = 1.0
    reliability[4, 1] = 1.0
    original_logsumexp = torch.logsumexp

    def checked_logsumexp(values, *args, **kwargs):
        assert torch.isfinite(values).any(dim=1).all()
        return original_logsumexp(values, *args, **kwargs)

    monkeypatch.setattr(torch, "logsumexp", checked_logsumexp)
    loss = ReIDTrainer._cross_camera_part_contrastive_loss(
        tokens,
        pids,
        camera_ids,
        reliability,
        temperature=0.07,
    )
    loss.backward()

    assert loss.item() > 0
    assert torch.isfinite(tokens.grad).all()


def _coco_keypoints() -> list[list[float]]:
    return [
        [0.50, 0.08, 0.95],
        [0.47, 0.07, 0.95],
        [0.53, 0.07, 0.95],
        [0.43, 0.08, 0.95],
        [0.57, 0.08, 0.95],
        [0.38, 0.25, 0.95],
        [0.62, 0.25, 0.95],
        [0.30, 0.43, 0.95],
        [0.70, 0.43, 0.95],
        [0.25, 0.60, 0.95],
        [0.75, 0.60, 0.95],
        [0.43, 0.56, 0.95],
        [0.57, 0.56, 0.95],
        [0.41, 0.75, 0.95],
        [0.59, 0.75, 0.95],
        [0.39, 0.95, 0.95],
        [0.61, 0.95, 0.95],
    ]


def test_scale_aware_geometry_targets_are_normalized_and_finer_at_high_resolution():
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, 4),
        torch.tensor([-0.5, 0.5]),
        indexing="ij",
    )
    canonical_grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
        1,
        1,
        8,
        2,
    )
    masks = torch.ones(1, 1, 48, 16)
    valid = torch.ones(1, 1, 8, dtype=torch.bool)
    mask_valid = torch.ones(1, dtype=torch.bool)
    local_source = torch.randn(1, 2, 12, 4)
    fine_source = F.interpolate(
        local_source,
        size=(24, 8),
        mode="bilinear",
        align_corners=False,
    )

    local_routing, local_dense, local_valid, local_tokens = (
        _scale_aware_anatomical_targets(
            local_source,
            masks,
            canonical_grid,
            valid,
            mask_valid,
            fine_scale=False,
        )
    )
    fine_routing, fine_dense, fine_valid, fine_tokens = (
        _scale_aware_anatomical_targets(
            fine_source,
            masks,
            canonical_grid,
            valid,
            mask_valid,
            fine_scale=True,
        )
    )

    torch.testing.assert_close(
        local_routing.sum(dim=(-1, -2)),
        torch.ones(1, 1, 8),
    )
    torch.testing.assert_close(
        fine_routing.sum(dim=(-1, -2)),
        torch.ones(1, 1, 8),
    )
    torch.testing.assert_close(local_dense.sum(dim=(-1, -2)), torch.ones(1, 1))
    torch.testing.assert_close(fine_dense.sum(dim=(-1, -2)), torch.ones(1, 1))
    assert local_valid.all()
    assert fine_valid.all()
    assert local_tokens.shape == fine_tokens.shape == (1, 1, 8, 2)

    def normalized_vertical_variance(routing: torch.Tensor) -> torch.Tensor:
        height = routing.shape[-2]
        rows = (torch.arange(height) + 0.5) / height
        weights = routing[0, 0, 4].sum(dim=-1)
        mean = (weights * rows).sum()
        return (weights * (rows - mean).square()).sum()

    assert normalized_vertical_variance(fine_routing) < normalized_vertical_variance(local_routing)


def test_scale_aware_geometry_targets_keep_empty_fp16_cells_finite():
    source = torch.randn(2, 2, 8, 4, dtype=torch.float16)
    masks = torch.zeros(2, 6, 16, 8, dtype=torch.float16)
    canonical_grid = torch.zeros(2, 6, 8, 2, dtype=torch.float16)
    grid_valid = torch.zeros(2, 6, 8, dtype=torch.bool)
    mask_valid = torch.ones(2, dtype=torch.bool)

    routing, dense_target, routing_valid, teacher_tokens = (
        _scale_aware_anatomical_targets(
            source,
            masks,
            canonical_grid,
            grid_valid,
            mask_valid,
            fine_scale=False,
        )
    )

    assert routing.dtype == torch.float32
    assert dense_target.dtype == torch.float32
    assert teacher_tokens.dtype == torch.float32
    assert not routing_valid.any()
    assert torch.isfinite(routing).all()
    assert torch.isfinite(dense_target).all()
    assert torch.isfinite(teacher_tokens).all()
    assert torch.count_nonzero(routing) == 0
    assert torch.count_nonzero(dense_target) == 0
    assert torch.count_nonzero(teacher_tokens) == 0


def test_anatomical_token_width_must_divide_the_canonical_grid():
    with pytest.raises(
        ValueError,
        match="must be divisible by the eight canonical",
    ):
        ReIDTrainer(
            model_name="csl_tinyvit_11m",
            dataset_name="market1501",
            data_dir=".",
            anatomical_auxiliary=True,
            anatomical_metadata_dir="metadata",
            anatomical_token_dim=18,
        )


def test_cross_scale_alignment_preserves_scale_specific_feature_bases():
    torch.manual_seed(19)
    local_tokens = torch.randn(3, 6, 16)
    orthogonal, _ = torch.linalg.qr(torch.randn(16, 16))
    fine_tokens = local_tokens @ orthogonal
    reliability = torch.ones(3, 6)

    relational = _cross_scale_role_relation_loss(
        local_tokens,
        fine_tokens,
        reliability,
    )
    raw_alignment = (
        1.0
        - F.cosine_similarity(local_tokens, fine_tokens, dim=-1)
    ).mean()

    assert relational.item() < 1e-6
    assert raw_alignment.item() > 0.1
    fine_tokens[:, 0] = fine_tokens[:, 0] + 2.0
    assert (
        _cross_scale_role_relation_loss(
            local_tokens,
            fine_tokens,
            reliability,
        ).item()
        > 1e-4
    )


def test_pose_provider_rasterizes_six_anatomical_parts(tmp_path):
    image_root = tmp_path / "data"
    image_path = image_root / "bounding_box_train" / "0001_c1.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 128), "gray").save(image_path)

    metadata_root = tmp_path / "metadata"
    mask_path = metadata_root / "person" / "bounding_box_train" / "0001_c1.png"
    mask_path.parent.mkdir(parents=True)
    Image.new("L", (64, 128), 255).save(mask_path)
    payload = {
        "images": {
            "bounding_box_train/0001_c1.jpg": {
                "keypoints": _coco_keypoints(),
                "person_mask": "person/bounding_box_train/0001_c1.png",
            }
        }
    }
    metadata_root.mkdir(exist_ok=True)
    (metadata_root / "metadata.json").write_text(json.dumps(payload))

    samples = [ReIDSample(str(image_path), pid=1, camid=0)]
    provider = PoseAnatomicalTargetProvider(
        samples,
        image_root=image_root,
        metadata_root=metadata_root,
    )
    assert provider.matched_record_count == 1
    assert provider.pose_record_count == 1
    assert provider.qualified_pose_record_count == 1
    assert provider.person_mask_record_count == 1
    assert provider.nonempty_person_mask_record_count == 1
    assert provider.effective_supervision_record_count == 1
    assert provider.missing_person_mask_count == 0
    target = provider(0, (64, 128))

    assert tuple(target["masks"].shape) == (len(ANATOMICAL_PARTS), 128, 64)
    assert tuple(target["canonical_grid"].shape) == (
        len(ANATOMICAL_PARTS),
        *ANATOMICAL_CANONICAL_GRID_SIZE,
        2,
    )
    assert tuple(target["canonical_grid_valid"].shape) == (
        len(ANATOMICAL_PARTS),
        *ANATOMICAL_CANONICAL_GRID_SIZE,
    )
    assert tuple(target["canonical_grid_pose_valid"].shape) == (
        len(ANATOMICAL_PARTS),
        *ANATOMICAL_CANONICAL_GRID_SIZE,
    )
    assert target["pose_valid"].item() is True
    assert target["mask_valid"].item() is True
    assert target["valid"].item() is True
    assert torch.all(target["visibility"] > 0)
    assert torch.all(target["masks"].flatten(1).sum(dim=1) > 0)
    assert torch.all(target["canonical_grid_valid"])
    assert torch.all(target["canonical_grid_pose_valid"])
    assert torch.all(target["pose_reliability"] > 0)
    assert target["pose_mask_agreement"].item() == 1.0
    assert tuple(target["pose_keypoints"].shape) == (17, 3)
    assert target["pose_keypoints"][:, 2].count_nonzero().item() > 0
    torso_grid = target["canonical_grid"][1]
    assert torch.all(torso_grid[1:, :, 1] > torso_grid[:-1, :, 1])
    assert torch.all(torso_grid[:, 1, 0] > torso_grid[:, 0, 0])
    left_arm_grid = target["canonical_grid"][2]
    assert torch.all(left_arm_grid[1:, :, 1] > left_arm_grid[:-1, :, 1])
    assert not torch.equal(target["masks"][2], target["masks"][3])
    assert not torch.equal(target["masks"][4], target["masks"][5])
    cached = provider(0, (64, 128))
    assert torch.equal(cached["masks"], target["masks"])
    assert torch.equal(cached["canonical_grid"], target["canonical_grid"])
    assert torch.equal(
        cached["canonical_grid_valid"],
        target["canonical_grid_valid"],
    )
    assert torch.equal(
        cached["canonical_grid_pose_valid"],
        target["canonical_grid_pose_valid"],
    )
    assert torch.equal(
        cached["pose_keypoints"],
        target["pose_keypoints"],
    )
    assert len(provider._target_cache) == 1


def test_pose_provider_keeps_pose_teacher_targets_without_person_mask(
    tmp_path,
):
    image_root = tmp_path / "data"
    image_path = image_root / "bounding_box_train" / "0001_c1.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 128), "gray").save(image_path)
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    (metadata_root / "metadata.json").write_text(
        json.dumps(
            {
                "images": {
                    "bounding_box_train/0001_c1.jpg": {
                        "keypoints": _coco_keypoints(),
                    }
                }
            }
        )
    )
    provider = PoseAnatomicalTargetProvider(
        [ReIDSample(str(image_path), pid=1, camid=0)],
        image_root=image_root,
        metadata_root=metadata_root,
        pose_only_reliability=0.2,
    )
    assert provider.matched_record_count == 1
    assert provider.pose_record_count == 1
    assert provider.qualified_pose_record_count == 1
    assert provider.person_mask_record_count == 0
    assert provider.nonempty_person_mask_record_count == 0
    assert provider.effective_supervision_record_count == 1
    assert provider.missing_person_mask_count == 0

    target = provider(0, (64, 128))

    assert target["pose_valid"].item() is True
    assert target["mask_valid"].item() is False
    assert target["valid"].item() is False
    assert target["masks"].count_nonzero().item() > 0
    assert target["canonical_grid"].count_nonzero().item() > 0
    assert target["canonical_grid_valid"].count_nonzero().item() == 0
    assert target["canonical_grid_pose_valid"].all()
    assert target["pose_keypoints"][:, 2].count_nonzero().item() > 0
    assert target["visibility"].count_nonzero().item() > 0
    assert target["reliability"].count_nonzero().item() > 0
    assert torch.all(target["pose_reliability"] > 0)
    torch.testing.assert_close(
        target["reliability"],
        target["pose_reliability"] * 0.2,
    )
    assert target["pose_mask_agreement"].item() == 1.0


def test_anatomical_provider_rejects_insufficient_effective_coverage(
    tmp_path,
):
    image_root = tmp_path / "data"
    train_root = image_root / "bounding_box_train"
    train_root.mkdir(parents=True)
    image_paths = [train_root / f"000{i}_c1.jpg" for i in (1, 2)]
    for image_path in image_paths:
        Image.new("RGB", (64, 128), "gray").save(image_path)

    metadata_root = tmp_path / "metadata"
    mask_path = metadata_root / "person" / "bounding_box_train" / "0001_c1.png"
    mask_path.parent.mkdir(parents=True)
    Image.new("L", (64, 128), 255).save(mask_path)
    (metadata_root / "metadata.json").write_text(
        json.dumps(
            {
                "images": {
                    "bounding_box_train/0001_c1.jpg": {
                        "keypoints": _coco_keypoints(),
                        "person_mask": (
                            "person/bounding_box_train/0001_c1.png"
                        ),
                    }
                }
            }
        )
    )
    samples = [
        ReIDSample(str(path), pid=index, camid=0)
        for index, path in enumerate(image_paths)
    ]
    dataset = SimpleNamespace(train=SimpleNamespace(samples=samples))
    config = AugmentationConfig(
        anatomical_auxiliary=True,
        anatomical_metadata_dir=str(metadata_root),
        anatomical_pose_only_reliability=0.0,
        anatomical_min_effective_coverage=0.8,
    )

    with pytest.raises(ValueError, match="coverage is below"):
        _build_anatomical_target_provider(
            SimpleNamespace(
                anatomical_target_type="learned_pose_concat_ema",
                hpgrd_part_drop_weight=0.0,
            ),
            config,
            dataset,
            image_root=image_root,
        )


def test_pose_provider_loads_external_mask_without_pose_record(tmp_path):
    image_root = tmp_path / "data"
    image_path = image_root / "bounding_box_train" / "0001_c1.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 128), "gray").save(image_path)
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    (metadata_root / "metadata.json").write_text(
        json.dumps({"images": {}})
    )
    person_mask_dir = tmp_path / "highconf" / "bounding_box_train"
    person_mask_dir.mkdir(parents=True)
    Image.new("L", (64, 128), 255).save(
        person_mask_dir / "0001_c1.png"
    )

    provider = PoseAnatomicalTargetProvider(
        [ReIDSample(str(image_path), pid=1, camid=0)],
        image_root=image_root,
        metadata_root=metadata_root,
        person_mask_dir=person_mask_dir,
    )
    target = provider(0, (64, 128))

    assert provider.matched_record_count == 0
    assert provider.person_mask_record_count == 1
    assert provider.external_person_mask_record_count == 1
    assert target["mask_valid"].item() is True
    assert target["pose_valid"].item() is False
    assert target["foreground_mask"].shape == (1, 128, 64)
    assert target["foreground_mask"].all()
    assert target["masks"].count_nonzero().item() == 0


def test_pose_provider_discards_out_of_bounds_keypoint_confidence(tmp_path):
    image_root = tmp_path / "data"
    image_path = image_root / "bounding_box_train" / "0001_c1.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 128), "gray").save(image_path)
    metadata_root = tmp_path / "metadata"
    mask_path = (
        metadata_root
        / "person"
        / "bounding_box_train"
        / "0001_c1.png"
    )
    mask_path.parent.mkdir(parents=True)
    Image.new("L", (64, 128), 255).save(mask_path)
    keypoints = _coco_keypoints()
    for keypoint in keypoints:
        keypoint[0] = -0.1
    (metadata_root / "metadata.json").write_text(
        json.dumps(
            {
                "images": {
                    "bounding_box_train/0001_c1.jpg": {
                        "keypoints": keypoints,
                        "person_mask": (
                            "person/bounding_box_train/0001_c1.png"
                        ),
                    }
                }
            }
        )
    )
    provider = PoseAnatomicalTargetProvider(
        [ReIDSample(str(image_path), pid=1, camid=0)],
        image_root=image_root,
        metadata_root=metadata_root,
    )

    target = provider(0, (64, 128))

    assert target["pose_valid"].item() is False
    assert target["pose_reliability"].count_nonzero().item() == 0
    assert target["pose_keypoints"][:, 2].count_nonzero().item() == 0
    assert target["masks"].count_nonzero().item() == 0
    assert target["canonical_grid_valid"].count_nonzero().item() == 0
    assert target["canonical_grid_pose_valid"].count_nonzero().item() == 0
    assert target["pose_mask_agreement"].item() == 0.0


def test_pose_provider_loads_optional_accessory_mask(tmp_path):
    image_root = tmp_path / "data"
    image_path = image_root / "bounding_box_train" / "0001_c1.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 32), "gray").save(image_path)
    metadata_root = tmp_path / "metadata"
    bag_path = metadata_root / "bags" / "bounding_box_train" / "0001_c1.png"
    bag_path.parent.mkdir(parents=True)
    person_path = (
        metadata_root / "person" / "bounding_box_train" / "0001_c1.png"
    )
    person_path.parent.mkdir(parents=True)
    person = Image.new("L", (16, 32), 0)
    for y in range(32):
        for x in range(10):
            person.putpixel((x, y), 255)
    person.save(person_path)
    bag = Image.new("L", (16, 32), 0)
    for y in range(8, 24):
        for x in range(10, 16):
            bag.putpixel((x, y), 255)
    bag.save(bag_path)
    (metadata_root / "metadata.json").write_text(
        json.dumps(
            {
                "images": {
                    "bounding_box_train/0001_c1.jpg": {
                        "bag_mask": (
                            "bags/bounding_box_train/0001_c1.png"
                        ),
                        "person_mask": (
                            "person/bounding_box_train/0001_c1.png"
                        ),
                        "keypoints": _coco_keypoints(),
                    }
                }
            }
        )
    )
    provider = PoseAnatomicalTargetProvider(
        [ReIDSample(str(image_path), pid=1, camid=0)],
        image_root=image_root,
        metadata_root=metadata_root,
    )

    target = provider(0, (16, 32))
    cached = provider(0, (16, 32))

    assert provider.accessory_mask_record_count == 1
    assert target["accessory_valid"].item() is True
    assert target["accessory_visibility"].item() == 1.0
    assert target["accessory_reliability"].item() == 1.0
    assert target["accessory_mask"].shape == (1, 32, 16)
    assert target["accessory_mask"].count_nonzero().item() > 0
    assert torch.equal(
        cached["accessory_mask"],
        target["accessory_mask"],
    )
    assert len(provider._target_cache) == 1


def test_pose_provider_returns_rgb_only_target_without_pose_record(
    tmp_path,
):
    image_root = tmp_path / "data"
    image_path = image_root / "bounding_box_train" / "0001_c1.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 128), "gray").save(image_path)
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    (metadata_root / "metadata.json").write_text(json.dumps({"images": {}}))
    provider = PoseAnatomicalTargetProvider(
        [ReIDSample(str(image_path), pid=1, camid=0)],
        image_root=image_root,
        metadata_root=metadata_root,
    )
    assert provider.matched_record_count == 0
    assert provider.pose_record_count == 0

    target = provider(0, (64, 128))

    assert target["pose_valid"].item() is False
    assert target["mask_valid"].item() is False
    assert target["valid"].item() is False
    assert target["masks"].count_nonzero().item() == 0
    assert target["canonical_grid_pose_valid"].count_nonzero().item() == 0
    assert target["pose_keypoints"].count_nonzero().item() == 0
    assert target["pose_reliability"].count_nonzero().item() == 0


def test_horizontal_flip_swaps_anatomical_left_and_right():
    image = Image.new("RGB", (8, 8), "black")
    masks = torch.zeros(6, 8, 8)
    masks[0:2, 2:6, 2:6] = 1
    masks[2, :, :2] = 1
    masks[3, :, 6:] = 0.5
    masks[4, 4:, :2] = 1
    masks[5, 4:, 6:] = 0.5
    canonical_grid = torch.zeros(6, 4, 2, 2)
    canonical_grid[..., 0] = torch.tensor([-0.6, 0.4])
    canonical_grid[..., 1] = torch.linspace(
        -0.75,
        0.75,
        4,
    )[:, None]
    canonical_grid[3, ..., 0] += 0.1
    canonical_grid_valid = torch.ones(6, 4, 2, dtype=torch.bool)
    canonical_grid_valid[3, 0, 0] = False
    pose_keypoints = torch.tensor(_coco_keypoints())
    pose_keypoints[:, :2] = 2.0 * pose_keypoints[:, :2] - 1.0
    target = {
        "masks": masks,
        "foreground_mask": masks[:1].clone(),
        "accessory_mask": masks[2:3].clone(),
        "canonical_grid": canonical_grid,
        "canonical_grid_valid": canonical_grid_valid,
        "canonical_grid_pose_valid": canonical_grid_valid.clone(),
        "pose_keypoints": pose_keypoints,
        "visibility": torch.tensor([1.0, 1.0, 0.8, 0.4, 0.7, 0.3]),
        "reliability": torch.tensor([1.0, 1.0, 0.7, 0.3, 0.6, 0.2]),
        "valid": torch.tensor(True),
    }
    transform = EpochAwareCompose(
        [
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
        ]
    )

    _, flipped = transform.apply_with_anatomical_target(image, target)

    assert torch.all(flipped["masks"][2, :, :2] == 0.5)
    assert torch.all(flipped["masks"][3, :, 6:] == 1)
    assert torch.equal(
        flipped["foreground_mask"],
        torch.flip(target["foreground_mask"], dims=(-1,)),
    )
    assert torch.equal(
        flipped["accessory_mask"],
        torch.flip(target["accessory_mask"], dims=(-1,)),
    )
    assert torch.allclose(
        flipped["visibility"],
        torch.tensor([1.0, 1.0, 0.4, 0.8, 0.3, 0.7]),
    )
    assert torch.allclose(
        flipped["reliability"],
        torch.tensor([1.0, 1.0, 0.3, 0.7, 0.2, 0.6]),
    )
    expected_grid = canonical_grid[list(ANATOMICAL_FLIP_PERMUTATION)].clone()
    expected_grid[..., 0] *= -1
    expected_grid = torch.flip(expected_grid, dims=(-2,))
    assert torch.equal(flipped["canonical_grid"], expected_grid)
    expected_valid = torch.flip(
        canonical_grid_valid[list(ANATOMICAL_FLIP_PERMUTATION)],
        dims=(-1,),
    )
    assert torch.equal(
        flipped["canonical_grid_valid"],
        expected_valid,
    )
    assert torch.equal(
        flipped["canonical_grid_pose_valid"],
        expected_valid,
    )
    expected_pose = pose_keypoints[list(COCO_KEYPOINT_FLIP_PERMUTATION)].clone()
    expected_pose[:, 0] *= -1
    assert torch.equal(flipped["pose_keypoints"], expected_pose)
    assert torch.equal(
        flipped["mask_present"],
        flipped["masks"].flatten(1).amax(dim=1) > 1e-6,
    )

    provider = object.__new__(PoseAnatomicalTargetProvider)
    provider.compact_nonsemantic = True
    compact = provider.compact_target(flipped)

    assert torch.equal(compact["mask_present"], flipped["mask_present"])
    assert "masks" not in compact
    assert "foreground_mask" not in compact
    assert "accessory_mask" not in compact


def test_random_translation_updates_visibility_for_cropped_parts():
    image = Image.new("RGB", (8, 8), "black")
    masks = torch.zeros(6, 8, 8)
    masks[:, :, :4] = 1
    canonical_grid = torch.zeros(6, 4, 2, 2)
    canonical_grid[..., 0] = -0.5
    target = {
        "masks": masks,
        "canonical_grid": canonical_grid,
        "canonical_grid_valid": torch.ones(
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "canonical_grid_pose_valid": torch.ones(
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "pose_keypoints": torch.tensor(
            [[-0.5, 0.0, 1.0]] * 17,
        ),
        "visibility": torch.ones(6),
        "reliability": torch.ones(6),
        "valid": torch.tensor(True),
    }
    transform = Random2DTranslation(8, 8, p=1.0, scale=2.0)

    with (
        patch("random.random", return_value=0.0),
        patch("random.randint", side_effect=[8, 0]),
    ):
        _, translated = transform.apply_with_anatomical_target(
            image,
            target,
        )

    assert torch.all(translated["visibility"] < 0.05)
    assert not translated["canonical_grid_valid"].any()
    assert not translated["canonical_grid_pose_valid"].any()
    assert translated["pose_keypoints"][:, 2].count_nonzero().item() == 0


def test_dataset_compacts_nonsemantic_targets_after_alignment(tmp_path):
    image_path = tmp_path / "0001_c1.jpg"
    Image.new("RGB", (8, 8), "gray").save(image_path)

    class CompactProvider:
        compact_nonsemantic = True

        def __call__(self, _index, _size):
            return {
                "masks": torch.ones(6, 8, 8),
                "foreground_mask": torch.ones(1, 8, 8),
                "accessory_mask": torch.zeros(1, 8, 8),
                "canonical_grid": torch.zeros(6, 4, 2, 2),
                "canonical_grid_valid": torch.ones(
                    6,
                    4,
                    2,
                    dtype=torch.bool,
                ),
                "visibility": torch.ones(6),
                "reliability": torch.ones(6),
                "valid": torch.tensor(True),
            }

        def compact_target(self, target):
            return PoseAnatomicalTargetProvider.compact_target(
                self,
                target,
            )

    dataset = ReIDImageDataset(
        [ReIDSample(str(image_path), pid=1, camid=0)],
        transform=EpochAwareCompose([T.ToTensor()]),
        anatomical_target_provider=CompactProvider(),
    )

    _, _, _, target = dataset[0]

    assert target["mask_present"].all()
    assert "masks" not in target
    assert "foreground_mask" not in target
    assert "accessory_mask" not in target


def _model_style_initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


def test_anatomical_toggle_preserves_shared_head_initialization():
    kwargs = {
        "in_ch": (16, 16, 16),
        "feat_dim": 16,
        "num_classes": 4,
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "scale_balanced_branches": True,
        "hierarchical_scales": True,
        "inference_feature": "norm_concat_bn",
        "anatomical_token_dim": 16,
    }

    def build(
        enabled: bool,
        descriptor_distill: bool = False,
        multiscale: bool = False,
    ) -> MultiBranchHead:
        torch.manual_seed(1234)
        head = MultiBranchHead(
            **kwargs,
            anatomical_auxiliary=enabled,
            anatomical_descriptor_distill=descriptor_distill,
            anatomical_multiscale=multiscale,
        )
        head.apply(_model_style_initialize)
        head.reset_reid_initialization()
        return head

    control = build(False).state_dict()
    for anatomical_head in (
        build(True),
        build(True, True),
        build(True, True, True),
        build(True, False, True),
    ):
        anatomical = anatomical_head.state_dict()
        for key, value in control.items():
            assert torch.equal(value, anatomical[key]), key


def test_multiscale_anatomy_config_round_trips_to_trainer_kwargs():
    config = ReIDTrainConfig.from_flat_kwargs(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir=".",
        anatomical_multiscale=True,
        anatomical_local_scale_weight=0.60,
        anatomical_fine_scale_weight=0.40,
        anatomical_cross_scale_weight=0.05,
        anatomical_fine_start_epoch=40,
        anatomical_fine_ramp_end_epoch=80,
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_teacher_momentum=0.999,
        anatomical_branch_distill_weight=0.05,
        anatomical_branch_global_coefficient=0.20,
        anatomical_branch_coarse_coefficient=0.30,
        anatomical_branch_fine_coefficient=0.50,
    )

    trainer_kwargs = config.to_trainer_kwargs()

    assert config.model.anatomical_multiscale is True
    assert trainer_kwargs["anatomical_multiscale"] is True
    assert trainer_kwargs["anatomical_local_scale_weight"] == 0.60
    assert trainer_kwargs["anatomical_fine_scale_weight"] == 0.40
    assert trainer_kwargs["anatomical_cross_scale_weight"] == 0.05
    assert trainer_kwargs["anatomical_fine_start_epoch"] == 40
    assert trainer_kwargs["anatomical_fine_ramp_end_epoch"] == 80
    assert (
        trainer_kwargs["anatomical_target_type"]
        == "learned_pose_concat_ema"
    )
    assert trainer_kwargs["anatomical_teacher_momentum"] == 0.999
    assert trainer_kwargs["anatomical_branch_distill_weight"] == 0.05
    assert trainer_kwargs["anatomical_branch_global_coefficient"] == 0.20
    assert trainer_kwargs["anatomical_branch_coarse_coefficient"] == 0.30
    assert trainer_kwargs["anatomical_branch_fine_coefficient"] == 0.50


def test_anatomical_deployment_config_round_trips_to_trainer_kwargs():
    config = ReIDTrainConfig.from_flat_kwargs(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir=".",
        anatomical_deployment=True,
        anatomical_deployment_dim=64,
        anatomical_deployment_alpha=0.25,
        anatomical_deployment_id_weight=0.25,
        anatomical_deployment_metric_weight=0.10,
    )

    trainer_kwargs = config.to_trainer_kwargs()

    assert config.model.anatomical_deployment is True
    assert config.model.anatomical_deployment_dim == 64
    assert config.model.anatomical_deployment_alpha == 0.25
    assert trainer_kwargs["anatomical_deployment"] is True
    assert trainer_kwargs["anatomical_deployment_dim"] == 64
    assert trainer_kwargs["anatomical_deployment_alpha"] == 0.25
    assert trainer_kwargs["anatomical_deployment_id_weight"] == 0.25
    assert trainer_kwargs["anatomical_deployment_metric_weight"] == 0.10


def test_privileged_mask_pose_attention_config_round_trips():
    config = ReIDTrainConfig.from_flat_kwargs(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir=".",
        anatomical_auxiliary=True,
        anatomical_metadata_dir="pose",
        anatomical_person_mask_dir="masks",
        anatomical_target_type="privileged_mask_pose_attention",
        anatomical_multiscale=True,
        anatomical_foreground_weight=0.15,
        anatomical_attention_weight=0.10,
    )

    trainer_kwargs = config.to_trainer_kwargs()

    assert (
        config.model.anatomical_target_type
        == "privileged_mask_pose_attention"
    )
    assert config.augmentation.anatomical_person_mask_dir == "masks"
    assert trainer_kwargs["anatomical_foreground_weight"] == 0.15


def test_pose_semantic_teacher_config_round_trips():
    config = ReIDTrainConfig.from_flat_kwargs(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir=".",
        anatomical_auxiliary=True,
        anatomical_metadata_dir="pose",
        anatomical_person_mask_dir="masks",
        anatomical_target_type="learned_pose_semantic_fused_ema",
        anatomical_multiscale=True,
        anatomical_foreground_weight=0.03,
        anatomical_semantic_part_weight=0.05,
    )

    trainer_kwargs = config.to_trainer_kwargs()

    assert config.model.anatomical_target_type == (
        "learned_pose_semantic_fused_ema"
    )
    assert (
        config.augmentation.anatomical_semantic_part_weight
        == 0.05
    )
    assert trainer_kwargs["anatomical_foreground_weight"] == 0.03
    assert trainer_kwargs["anatomical_semantic_part_weight"] == 0.05


def test_decoupled_pose_parsing_teacher_config_round_trips():
    config = ReIDTrainConfig.from_flat_kwargs(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir=".",
        anatomical_target_type="decoupled_pose_parsing_teacher",
        anatomical_accessory_query=True,
        anatomical_query_distill_weight=0.05,
        anatomical_query_diversity_weight=0.01,
        anatomical_query_diversity_margin=0.10,
        anatomical_part_triplet_weight=0.03,
        anatomical_query_start_epoch=20,
        anatomical_query_ramp_end_epoch=50,
    )

    trainer_kwargs = config.to_trainer_kwargs()

    assert config.model.anatomical_accessory_query is True
    assert trainer_kwargs["anatomical_query_distill_weight"] == 0.05
    assert trainer_kwargs["anatomical_query_diversity_weight"] == 0.01
    assert trainer_kwargs["anatomical_query_diversity_margin"] == 0.10
    assert trainer_kwargs["anatomical_part_triplet_weight"] == 0.03
    assert trainer_kwargs["anatomical_query_start_epoch"] == 20
    assert trainer_kwargs["anatomical_query_ramp_end_epoch"] == 50


def test_default_fine_schedule_keeps_existing_multiscale_resume_compatible():
    saved = {
        "model": {
            "anatomical_multiscale": True,
            "anatomical_accessory_query": False,
            "anatomical_target_type": "deterministic_scale_aware_geometry",
        },
        "augmentation": {"anatomical_auxiliary": True},
    }
    requested = {
        "model": {
            "anatomical_multiscale": True,
            "anatomical_target_type": "deterministic_scale_aware_geometry",
        },
        "augmentation": {
            "anatomical_auxiliary": True,
            "anatomical_fine_start_epoch": 0,
            "anatomical_fine_ramp_end_epoch": 0,
            "anatomical_branch_distill_weight": 0.0,
            "anatomical_branch_global_coefficient": 0.20,
            "anatomical_branch_coarse_coefficient": 0.30,
            "anatomical_branch_fine_coefficient": 0.50,
            "anatomical_query_distill_weight": 0.0,
            "anatomical_query_diversity_weight": 0.0,
            "anatomical_query_diversity_margin": 0.10,
            "anatomical_part_triplet_weight": 0.0,
            "anatomical_query_start_epoch": 20,
            "anatomical_query_ramp_end_epoch": 50,
        },
    }

    assert contract_differences(saved, requested) == []
    requested["augmentation"]["anatomical_fine_start_epoch"] = 40
    requested["augmentation"]["anatomical_fine_ramp_end_epoch"] = 80
    differences = contract_differences(saved, requested)
    assert len(differences) == 2
    assert differences[0].startswith("augmentation.anatomical_fine_ramp_end_epoch:")
    assert differences[1].startswith("augmentation.anatomical_fine_start_epoch:")


def test_resume_contract_distinguishes_ema_and_geometry_teachers():
    saved = {
        "model": {
            "anatomical_multiscale": True,
            "anatomical_target_type": "learned_pose_concat_ema",
        },
        "augmentation": {"anatomical_auxiliary": True},
    }
    requested = {
        "model": {
            "anatomical_multiscale": True,
            "anatomical_target_type": "deterministic_scale_aware_geometry",
        },
        "augmentation": {"anatomical_auxiliary": True},
    }

    assert contract_differences(saved, requested) == [
        "model.anatomical_target_type: "
        "saved='learned_pose_concat_ema', "
        "requested='deterministic_scale_aware_geometry'"
    ]


def test_resume_contract_tracks_enabled_anatomical_branch_distillation():
    saved = {
        "model": {
            "anatomical_multiscale": True,
            "anatomical_target_type": "learned_pose_concat_ema",
        },
        "augmentation": {
            "anatomical_auxiliary": True,
            "anatomical_branch_distill_weight": 0.05,
            "anatomical_branch_global_coefficient": 0.20,
            "anatomical_branch_coarse_coefficient": 0.30,
            "anatomical_branch_fine_coefficient": 0.50,
        },
    }
    requested = {
        **saved,
        "augmentation": {
            **saved["augmentation"],
            "anatomical_branch_fine_coefficient": 0.40,
        },
    }

    assert contract_differences(saved, requested) == [
        "augmentation.anatomical_branch_fine_coefficient: "
        "saved=0.5, requested=0.4"
    ]


def test_resume_contract_tracks_enabled_anatomical_deployment():
    saved = {
        "model": {
            "anatomical_multiscale": True,
            "anatomical_target_type": "learned_pose_concat_ema",
            "anatomical_deployment": True,
            "anatomical_deployment_dim": 64,
            "anatomical_deployment_alpha": 0.25,
        },
        "augmentation": {
            "anatomical_auxiliary": True,
            "anatomical_deployment_id_weight": 0.25,
            "anatomical_deployment_metric_weight": 0.10,
        },
    }
    requested = {
        **saved,
        "model": {
            **saved["model"],
            "anatomical_deployment_alpha": 0.50,
        },
    }

    assert contract_differences(saved, requested) == [
        "model.anatomical_deployment_alpha: "
        "saved=0.25, requested=0.5"
    ]


def test_legacy_a11v8_hparams_infer_ema_teacher(tmp_path):
    (tmp_path / "hparams.json").write_text(
        json.dumps(
            {
                "augmentation": {
                    "anatomical_supervision": {
                        "enabled": True,
                        "teacher_momentum": 0.999,
                    }
                }
            }
        )
    )

    hparams = load_train_hparams(tmp_path)

    assert hparams["anatomical_target_type"] == "learned_pose_concat_ema"
    assert hparams["anatomical_teacher_momentum"] == 0.999


def test_ema_pose_heatmaps_preserve_joint_location_and_confidence():
    pose_keypoints = torch.zeros(1, 17, 3)
    row, column = 5, 3
    pose_keypoints[0, 0] = torch.tensor(
        (
            2.0 * (column + 0.5) / 6 - 1.0,
            2.0 * (row + 0.5) / 12 - 1.0,
            0.8,
        )
    )

    heatmaps = EMAAnatomicalAuxiliaryPool.pose_heatmaps(
        pose_keypoints,
        height=12,
        width=6,
    )

    assert tuple(heatmaps.shape) == (1, 29, 12, 6)
    assert heatmaps[0, 0].argmax().item() == row * 6 + column
    assert torch.isclose(
        heatmaps[0, 0, row, column],
        torch.tensor(0.8),
    )
    assert heatmaps[0, 1].count_nonzero().item() == 0
    assert heatmaps[0, 17:].count_nonzero().item() == 0


def test_branch_relational_distillation_is_cross_camera_and_reliability_gated():
    teacher = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        ),
        dim=1,
    )
    student = teacher.clone().requires_grad_(True)
    pids = torch.tensor([0, 0, 1, 1])
    camera_ids = torch.tensor([0, 1, 0, 1])
    reliability = torch.ones(4)

    matched = ReIDTrainer._cross_camera_relational_distill_loss(
        student,
        teacher,
        reliability,
        pids,
        camera_ids,
    )
    mismatched = ReIDTrainer._cross_camera_relational_distill_loss(
        student,
        teacher[[0, 2, 1, 3]],
        reliability,
        pids,
        camera_ids,
    )
    no_cross_camera = ReIDTrainer._cross_camera_relational_distill_loss(
        student,
        teacher,
        reliability,
        pids,
        torch.zeros(4, dtype=torch.long),
    )
    no_reliability = ReIDTrainer._cross_camera_relational_distill_loss(
        student,
        teacher,
        torch.zeros(4),
        pids,
        camera_ids,
    )

    assert matched.item() == pytest.approx(0.0, abs=1e-7)
    assert mismatched.item() > matched.item()
    assert no_cross_camera.item() == 0.0
    assert no_reliability.item() == 0.0


def test_branch_distillation_requires_the_learned_ema_teacher():
    with pytest.raises(
        ValueError,
        match="learned_pose_concat_ema",
    ):
        MultiBranchHead(
            (16, 16, 16),
            feat_dim=16,
            num_classes=4,
            head_parts=(1, 2, 4),
            part_pooling="stripes",
            scale_balanced_branches=True,
            hierarchical_scales=True,
            anatomical_auxiliary=True,
            anatomical_token_dim=16,
            anatomical_multiscale=True,
            anatomical_branch_distill=True,
            inference_feature="norm_concat_bn",
        )


def test_branch_distillation_adds_no_parameters_or_eval_path():
    kwargs = {
        "in_ch": (16, 16, 16),
        "feat_dim": 16,
        "num_classes": 4,
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "scale_balanced_branches": True,
        "hierarchical_scales": True,
        "anatomical_auxiliary": True,
        "anatomical_token_dim": 16,
        "anatomical_multiscale": True,
        "anatomical_target_type": "learned_pose_concat_ema",
        "inference_feature": "norm_concat_bn",
    }
    torch.manual_seed(91)
    control = MultiBranchHead(**kwargs)
    torch.manual_seed(91)
    branch_distilled = MultiBranchHead(
        **kwargs,
        anatomical_branch_distill=True,
    )

    control_state = control.state_dict()
    branch_state = branch_distilled.state_dict()
    assert control_state.keys() == branch_state.keys()
    for key in control_state:
        torch.testing.assert_close(
            control_state[key],
            branch_state[key],
            rtol=0,
            atol=0,
        )

    feature_maps = (
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 24, 8),
    )
    control.eval()
    branch_distilled.eval()
    with torch.no_grad():
        expected = control(feature_maps)
        actual = branch_distilled(feature_maps)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deployed_anatomical_tokens_are_rgb_only_at_inference():
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_deployment=True,
        anatomical_deployment_dim=8,
        anatomical_deployment_alpha=0.25,
        inference_feature="norm_concat_bn",
    )
    feature_maps = (
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 24, 8),
    )

    head.eval()
    with torch.no_grad():
        descriptor = head(feature_maps)

    assert descriptor.shape == (3, 48 + 6 * 8)
    torch.testing.assert_close(
        descriptor.norm(dim=1),
        torch.ones(3),
        rtol=1e-5,
        atol=1e-6,
    )
    expected_base_norm = 1.0 / (1.0 + 0.25**2) ** 0.5
    torch.testing.assert_close(
        descriptor[:, :48].norm(dim=1),
        torch.full((3,), expected_base_norm),
        rtol=1e-5,
        atol=1e-6,
    )


def test_privileged_attention_is_rgb_only_and_identity_initialized():
    kwargs = {
        "in_ch": (16, 16, 16),
        "feat_dim": 16,
        "num_classes": 4,
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "scale_balanced_branches": True,
        "hierarchical_scales": True,
        "inference_feature": "norm_concat_bn",
    }
    torch.manual_seed(151)
    control = MultiBranchHead(**kwargs)
    torch.manual_seed(151)
    privileged = MultiBranchHead(
        **kwargs,
        anatomical_auxiliary=True,
        anatomical_multiscale=True,
        anatomical_target_type="privileged_mask_pose_attention",
    )
    feature_maps = (
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 24, 8),
    )

    control.eval()
    privileged.eval()
    with torch.no_grad():
        expected = control(feature_maps)
        actual = privileged(feature_maps)

    assert actual.shape == expected.shape == (3, 48)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    privileged.train()
    _, features = privileged(feature_maps)
    assert features["_anatomical_attention"].shape == (3, 6, 12, 4)
    assert features["_anatomical_fine_attention"].shape == (
        3,
        6,
        24,
        8,
    )
    assert features["_anatomical_foreground_logits"].shape == (
        3,
        1,
        12,
        4,
    )
    assert features["_anatomical_gate_scale"].item() == 0.0


def test_privileged_attention_gate_ignores_unsupervised_logit_offsets():
    torch.manual_seed(153)
    adapter = PrivilegedMaskPoseAttentionAdapter(32).eval()
    adapter.foreground_gate_logit.data.fill_(1.0)
    adapter.part_gate_logit.data.fill_(1.0)
    feature_map = torch.randn(3, 32, 12, 4)

    with torch.no_grad():
        before = adapter(feature_map)
        offsets = torch.linspace(
            -8.0,
            8.0,
            adapter.num_parts,
        )[None, :, None, None]
        hook = adapter.part_predictor.register_forward_hook(
            lambda _module, _inputs, output: output + offsets
        )
        try:
            after = adapter(feature_map)
        finally:
            hook.remove()

    torch.testing.assert_close(after[0], before[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(after[1], before[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(after[2], before[2], rtol=1e-5, atol=1e-6)


def test_privileged_attention_gate_uses_predicted_part_visibility():
    torch.manual_seed(155)
    adapter = PrivilegedMaskPoseAttentionAdapter(32).eval()
    adapter.foreground_gate_logit.data.zero_()
    adapter.part_gate_logit.data.fill_(1.0)
    adapter.visibility_predictor.weight.data.zero_()
    adapter.foreground_predictor.weight.data.zero_()
    adapter.foreground_predictor.bias.data.zero_()
    feature_map = torch.randn(3, 32, 12, 4)

    with torch.no_grad():
        adapter.visibility_predictor.bias.fill_(-20.0)
        hidden = adapter(feature_map)[0]
        adapter.visibility_predictor.bias.fill_(20.0)
        visible = adapter(feature_map)[0]

    torch.testing.assert_close(hidden, feature_map, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(visible, feature_map, rtol=1e-5, atol=1e-6)


def test_privileged_attention_learns_mask_and_pose_gate_strengths_independently():
    torch.manual_seed(156)
    adapter = PrivilegedMaskPoseAttentionAdapter(32).eval()
    feature_map = torch.randn(3, 32, 12, 4)
    probe = torch.randn_like(feature_map)

    gated = adapter(feature_map)[0]
    (gated * probe).sum().backward()

    foreground_gradient = adapter.foreground_gate_logit.grad
    part_gradient = adapter.part_gate_logit.grad
    assert foreground_gradient is not None
    assert part_gradient is not None
    assert foreground_gradient.abs().item() > 0
    assert part_gradient.abs().item() > 0
    assert foreground_gradient.item() != part_gradient.item()


def test_privileged_attention_allows_one_cue_to_use_full_shared_budget():
    adapter = PrivilegedMaskPoseAttentionAdapter(32, max_scale=0.25).eval()
    adapter.foreground_predictor.weight.data.zero_()
    adapter.foreground_predictor.bias.data.fill_(20.0)
    adapter.foreground_gate_logit.data.fill_(20.0)
    adapter.part_gate_logit.data.zero_()
    feature_map = torch.ones(2, 32, 12, 4)

    with torch.no_grad():
        gated = adapter(feature_map)[0]

    torch.testing.assert_close(
        gated,
        torch.full_like(feature_map, 1.25),
        rtol=1e-5,
        atol=1e-6,
    )


def test_privileged_attention_combined_cues_stay_within_shared_budget():
    torch.manual_seed(158)
    adapter = PrivilegedMaskPoseAttentionAdapter(32, max_scale=0.25).eval()
    adapter.foreground_gate_logit.data.fill_(20.0)
    adapter.part_gate_logit.data.fill_(-20.0)
    feature_map = torch.ones(2, 32, 12, 4)

    with torch.no_grad():
        gated = adapter(feature_map)[0]

    assert (gated - feature_map).abs().max().item() <= 0.25 + 1e-6


def test_privileged_attention_loss_supervises_foreground_and_parts():
    torch.manual_seed(157)
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_multiscale=True,
        anatomical_target_type="privileged_mask_pose_attention",
        inference_feature="norm_concat_bn",
    )
    feature_maps = (
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 24, 8, requires_grad=True),
    )
    head.train()
    _, features = head(feature_maps)

    masks = torch.zeros(4, 6, 48, 16)
    for part_index in range(6):
        top = part_index * 7
        masks[:, part_index, top : top + 8] = 1
    targets = {
        "masks": masks,
        "foreground_mask": masks.amax(dim=1, keepdim=True),
        "visibility": torch.ones(4, 6),
        "reliability": torch.ones(4, 6),
        "pose_valid": torch.ones(4, dtype=torch.bool),
        "mask_valid": torch.ones(4, dtype=torch.bool),
        "valid": torch.ones(4, dtype=torch.bool),
    }
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_student_start_epoch = 0
    trainer.anatomical_student_ramp_end_epoch = 0
    trainer.anatomical_fine_start_epoch = 0
    trainer.anatomical_fine_ramp_end_epoch = 0
    trainer.anatomical_decay_start_epoch = 0
    trainer.anatomical_decay_end_epoch = 0
    trainer.anatomical_distill_weight = 0.10
    trainer.anatomical_foreground_weight = 0.15
    trainer.anatomical_attention_weight = 0.10
    trainer.anatomical_visibility_weight = 0.05
    trainer.anatomical_contrastive_weight = 0.10
    trainer.anatomical_local_scale_weight = 0.60
    trainer.anatomical_fine_scale_weight = 0.40
    trainer.anatomical_cross_scale_weight = 0.05
    trainer.anatomical_temperature = 0.07

    loss, components = trainer._privileged_mask_pose_attention_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        epoch=1,
        return_components=True,
    )
    loss.backward()

    assert loss.item() > 0
    assert components["distill"].item() > 0
    assert components["pose_teacher"].item() > 0
    assert components["attention"].item() > 0
    assert (
        head.anatomical_attention_adapter.foreground_predictor.weight.grad
        is not None
    )
    assert (
        head.anatomical_attention_adapter.part_predictor.weight.grad
        is not None
    )


def test_privileged_attention_excludes_fully_hidden_parts_from_spatial_losses():
    torch.manual_seed(159)
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_multiscale=True,
        anatomical_target_type="privileged_mask_pose_attention",
        inference_feature="norm_concat_bn",
    )
    head.train()
    _, features = head(
        (
            torch.randn(4, 16, 12, 4),
            torch.randn(4, 16, 12, 4),
            torch.randn(4, 16, 24, 8),
        )
    )
    masks = torch.ones(4, 6, 48, 16)
    targets = {
        "masks": masks,
        "foreground_mask": masks.amax(dim=1, keepdim=True),
        "visibility": torch.zeros(4, 6),
        "reliability": torch.ones(4, 6),
        "pose_valid": torch.ones(4, dtype=torch.bool),
        "mask_valid": torch.ones(4, dtype=torch.bool),
        "valid": torch.ones(4, dtype=torch.bool),
    }
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_student_start_epoch = 0
    trainer.anatomical_student_ramp_end_epoch = 0
    trainer.anatomical_fine_start_epoch = 0
    trainer.anatomical_fine_ramp_end_epoch = 0
    trainer.anatomical_decay_start_epoch = 0
    trainer.anatomical_decay_end_epoch = 0
    trainer.anatomical_distill_weight = 0.10
    trainer.anatomical_foreground_weight = 0.15
    trainer.anatomical_attention_weight = 0.10
    trainer.anatomical_visibility_weight = 0.05
    trainer.anatomical_contrastive_weight = 0.10
    trainer.anatomical_local_scale_weight = 0.60
    trainer.anatomical_fine_scale_weight = 0.40
    trainer.anatomical_cross_scale_weight = 0.05
    trainer.anatomical_temperature = 0.07

    _, components = trainer._privileged_mask_pose_attention_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        epoch=1,
        return_components=True,
    )

    assert components["visibility"].item() > 0
    assert components["attention"].item() == 0
    assert components["contrastive"].item() == 0
    assert components["cross_scale"].item() == 0
    assert components["valid_part_fraction"].item() == 0


def test_anatomical_deployment_preserves_all_shared_initialization():
    kwargs = {
        "in_ch": (16, 16, 16),
        "feat_dim": 16,
        "num_classes": 4,
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "scale_balanced_branches": True,
        "hierarchical_scales": True,
        "anatomical_auxiliary": True,
        "anatomical_token_dim": 16,
        "anatomical_multiscale": True,
        "anatomical_target_type": "learned_pose_concat_ema",
        "inference_feature": "norm_concat_bn",
    }
    torch.manual_seed(101)
    control = MultiBranchHead(**kwargs)
    torch.manual_seed(101)
    deployed = MultiBranchHead(
        **kwargs,
        anatomical_deployment=True,
        anatomical_deployment_dim=8,
    )

    deployed_state = deployed.state_dict()
    for key, value in control.state_dict().items():
        torch.testing.assert_close(
            deployed_state[key],
            value,
            rtol=0,
            atol=0,
        )


def test_deployed_anatomical_parts_receive_persistent_id_and_metric_gradients():
    torch.manual_seed(37)
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_deployment=True,
        anatomical_deployment_dim=8,
        anatomical_deployment_alpha=0.25,
        inference_feature="norm_concat_bn",
    )
    feature_maps = (
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 24, 8, requires_grad=True),
    )
    pose_keypoints = torch.tensor(_coco_keypoints())
    pose_keypoints[:, :2] = 2.0 * pose_keypoints[:, :2] - 1.0
    pose_keypoints = pose_keypoints[None].repeat(4, 1, 1)

    head.train()
    _, features = head(
        feature_maps,
        anatomical_pose=pose_keypoints,
    )
    assert features["norm_concat_bn"].shape == (4, 48 + 6 * 8)
    assert features["_anatomical_deployment_parts"].shape == (
        4,
        6,
        8,
    )
    assert features["_anatomical_deployment_visibility"].shape == (4, 6)

    trainer = object.__new__(ReIDTrainer)
    trainer.device = torch.device("cpu")
    trainer.anatomical_deployment = True
    trainer.anatomical_temperature = 0.07
    id_loss, metric_loss = trainer._anatomical_deployment_losses(
        nn.CrossEntropyLoss(),
        features,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
    )
    (0.25 * id_loss + 0.10 * metric_loss).backward()

    assert id_loss.item() > 0
    assert metric_loss.item() > 0
    assert (
        head.anatomical_deployment_projection.weight.grad is not None
    )
    assert (
        head.anatomical_deployment_projection.weight.grad.abs().sum().item()
        > 0
    )
    assert all(
        neck.classifier.weight.grad is not None
        and neck.classifier.weight.grad.abs().sum().item() > 0
        for neck in head.anatomical_deployment_necks
    )


def test_ema_pose_teacher_is_stop_gradient_and_updates_after_step():
    pool = EMAAnatomicalAuxiliaryPool(
        channels=16,
        token_dim=16,
        teacher_channels=24,
        multiscale=True,
    )
    student_source = torch.randn(2, 16, 8, 4, requires_grad=True)
    teacher_source = torch.randn(2, 24, 16, 8, requires_grad=True)
    pose_keypoints = torch.tensor(_coco_keypoints())
    pose_keypoints[:, :2] = 2.0 * pose_keypoints[:, :2] - 1.0

    outputs = pool(
        student_source,
        teacher_x=teacher_source,
        pose_keypoints=pose_keypoints[None].repeat(2, 1, 1),
    )
    online_teacher = outputs[2]
    ema_teacher = outputs[1]
    assert online_teacher.requires_grad
    assert not ema_teacher.requires_grad
    torch.testing.assert_close(online_teacher, ema_teacher)

    online_teacher.square().mean().backward()
    assert teacher_source.grad is None
    assert student_source.grad is None
    assert pool.online_pose_projection.weight.grad is not None
    assert pool.online_pose_encoder[0].weight.grad is not None
    assert all(
        parameter.grad is None
        for parameter in pool.ema_pose_encoder.parameters()
    )
    before = pool.ema_pose_projection.weight.detach().clone()
    with torch.no_grad():
        pool.online_pose_projection.weight.add_(1.0)
    pool.update_teacher(0.5)
    torch.testing.assert_close(
        pool.ema_pose_projection.weight,
        before + 0.5,
    )


def test_runtime_anatomy_gate_skips_pool_without_changing_retrieval_features():
    torch.manual_seed(43)
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
        inference_feature="norm_concat_bn",
    ).train()
    feature_maps = (
        torch.randn(4, 16, 12, 4),
        torch.randn(4, 16, 12, 4),
        torch.randn(4, 16, 24, 8),
    )
    pose_keypoints = torch.tensor(_coco_keypoints())
    pose_keypoints[:, :2] = 2.0 * pose_keypoints[:, :2] - 1.0
    pose_keypoints = pose_keypoints[None].repeat(4, 1, 1)
    calls = 0

    def count_pool_calls(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    hook = head.anatomical_auxiliary_pool.register_forward_hook(
        count_pool_calls
    )
    try:
        with torch.no_grad():
            head.set_anatomical_auxiliary_active(True)
            active_logits, active_features = head(
                feature_maps,
                anatomical_pose=pose_keypoints,
            )
            assert calls == 1
            head.set_anatomical_auxiliary_active(False)
            inactive_logits, inactive_features = head(
                feature_maps,
                anatomical_pose=pose_keypoints,
            )
    finally:
        hook.remove()

    assert calls == 1
    for active, inactive in zip(active_logits, inactive_logits, strict=True):
        torch.testing.assert_close(active, inactive, rtol=0, atol=0)
    torch.testing.assert_close(
        active_features["norm_concat_bn"],
        inactive_features["norm_concat_bn"],
        rtol=0,
        atol=0,
    )
    assert not any(
        key.startswith("_anatomical")
        for key in inactive_features
    )
    assert not any(
        key.endswith("anatomical_auxiliary_runtime_active")
        for key in head.state_dict()
    )


def test_pose_semantic_teacher_preserves_v8_inference_descriptor():
    kwargs = {
        "in_ch": (16, 16, 16),
        "feat_dim": 16,
        "num_classes": 4,
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "scale_balanced_branches": True,
        "hierarchical_scales": True,
        "anatomical_auxiliary": True,
        "anatomical_token_dim": 16,
        "anatomical_multiscale": True,
        "inference_feature": "norm_concat_bn",
    }
    torch.manual_seed(29)
    v8 = MultiBranchHead(
        **kwargs,
        anatomical_target_type="learned_pose_concat_ema",
    ).eval()
    torch.manual_seed(29)
    semantic = MultiBranchHead(
        **kwargs,
        anatomical_target_type="learned_pose_semantic_ema",
    ).eval()
    semantic_inference_calls = 0

    def count_semantic_inference_calls(
        _module,
        _inputs,
        _output,
    ) -> None:
        nonlocal semantic_inference_calls
        semantic_inference_calls += 1

    hooks = [
        prediction_head.register_forward_hook(
            count_semantic_inference_calls
        )
        for prediction_head in (
            semantic.anatomical_auxiliary_pool.semantic_prediction_heads
        )
    ]
    feature_maps = (
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 24, 8),
    )

    try:
        with torch.no_grad():
            expected = v8(feature_maps)
            actual = semantic(feature_maps)
    finally:
        for hook in hooks:
            hook.remove()

    assert actual.shape == expected.shape == (3, 48)
    assert semantic_inference_calls == 0
    assert semantic.anatomical_attention_adapter is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pose_teacher_supports_stage2_channel_representation_control(
    tmp_path,
):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        head_parts=(1, 2, 4),
        head_type="stage2_channel2",
        part_pooling="stripes",
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=True,
        anatomical_metadata_dir=str(tmp_path),
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_multiscale=True,
        anatomical_branch_distill_weight=0.0,
    )
    assert trainer.head_type == "stage2_channel2"

    head = MultiBranchHead(
        in_ch=(16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        specialist_mode="stage2_channel2",
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
        inference_feature="norm_concat_bn",
    )
    feature_maps = (
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 24, 8),
    )
    head.eval()
    with torch.no_grad():
        descriptor = head(feature_maps)

    assert descriptor.shape == (3, 48 + 2 * 128)
    torch.testing.assert_close(
        descriptor.norm(dim=1),
        torch.ones(3),
    )

    pose_keypoints = torch.tensor(_coco_keypoints())
    pose_keypoints[:, :2] = 2.0 * pose_keypoints[:, :2] - 1.0
    head.train()
    scores, features = head(
        feature_maps,
        anatomical_pose=pose_keypoints[None].repeat(3, 1, 1),
    )
    sum(score.square().mean() for score in scores).backward()

    assert len(scores) == 9
    assert "stage2_c1" in features
    assert "stage2_c2" in features
    assert "_anatomical_teacher_feature_map" in features
    channel_grad = head.stage2_channel_shared[0].weight.grad
    assert channel_grad is not None
    assert torch.isfinite(channel_grad).all()
    assert channel_grad.abs().sum().item() > 0


def test_pose_teacher_multiscale_channels_allocate_power_inside_every_scale(
    tmp_path,
):
    trainer = ReIDTrainer(
        model_name="csl_tinyvit_11m",
        dataset_name="market1501",
        data_dir=str(tmp_path),
        feature_fusion="global_final_parts_stage0_semantic_fine",
        attention_window_layout="rect",
        head_parts=(1, 2, 4),
        head_type="multiscale_channel2",
        multiscale_channel_alpha=0.5,
        part_pooling="stripes",
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        anatomical_auxiliary=True,
        anatomical_metadata_dir=str(tmp_path),
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_multiscale=True,
        anatomical_branch_distill_weight=0.0,
    )
    assert trainer.head_type == "multiscale_channel2"
    assert trainer.multiscale_channel_alpha == 0.5

    head = MultiBranchHead(
        in_ch=(16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        metric_feature="raw_concat",
        inference_feature="norm_concat_bn",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        specialist_mode="multiscale_channel2",
        multiscale_channel_alpha=0.5,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
    )
    feature_maps = (
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 24, 8),
    )
    head.eval()
    with torch.no_grad():
        descriptor = head(feature_maps)

    # v8 contributes 16 + 2*8 + 4*4 = 48 dimensions. Each source scale
    # additionally contributes two independently normalized 128-D summaries.
    assert descriptor.shape == (3, 48 + 6 * 128)
    torch.testing.assert_close(
        descriptor.norm(dim=1),
        torch.ones(3),
        atol=1e-5,
        rtol=1e-5,
    )
    spatial_slices = (
        descriptor[:, :16],
        descriptor[:, 16:32],
        descriptor[:, 32:48],
    )
    channel_slices = (
        descriptor[:, 48:304],
        descriptor[:, 304:560],
        descriptor[:, 560:816],
    )
    for spatial, channel in zip(
        spatial_slices,
        channel_slices,
        strict=True,
    ):
        spatial_power = spatial.square().sum(dim=1)
        channel_power = channel.square().sum(dim=1)
        torch.testing.assert_close(
            spatial_power + channel_power,
            torch.full((3,), 1.0 / 3.0),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            channel_power / (spatial_power + channel_power),
            torch.full((3,), 0.25),
            atol=1e-5,
            rtol=1e-5,
        )

    pose_keypoints = torch.tensor(_coco_keypoints())
    pose_keypoints[:, :2] = 2.0 * pose_keypoints[:, :2] - 1.0
    head.train()
    scores, features = head(
        feature_maps,
        anatomical_pose=pose_keypoints[None].repeat(3, 1, 1),
    )
    loss = sum(score.square().mean() for score in scores)
    loss = loss + features["raw_concat"].square().mean()
    loss.backward()

    assert len(scores) == 13
    assert features["raw_concat"].shape == (3, 816)
    assert {
        "global_c1",
        "global_c2",
        "coarse_c1",
        "coarse_c2",
        "fine_c1",
        "fine_c2",
        "_anatomical_teacher_feature_map",
    } <= features.keys()
    for projection in head.multiscale_channel_projections.values():
        channel_grad = projection[0].weight.grad
        assert channel_grad is not None
        assert torch.isfinite(channel_grad).all()
        assert channel_grad.abs().sum().item() > 0


def test_multiscale_channel_head_preserves_base_head_initialization():
    kwargs = {
        "in_ch": (16, 16, 16),
        "feat_dim": 16,
        "num_classes": 4,
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "metric_feature": "raw_concat",
        "inference_feature": "norm_concat_bn",
        "scale_balanced_branches": True,
        "hierarchical_scales": True,
    }
    torch.manual_seed(7)
    control = MultiBranchHead(**kwargs)
    control.reset_reid_initialization()
    torch.manual_seed(7)
    treatment = MultiBranchHead(
        **kwargs,
        specialist_mode="multiscale_channel2",
        multiscale_channel_alpha=0.5,
    )
    treatment.reset_reid_initialization()

    control_state = control.state_dict()
    treatment_state = treatment.state_dict()
    common_keys = set(control_state) & set(treatment_state)
    assert common_keys
    for key in common_keys:
        torch.testing.assert_close(
            treatment_state[key],
            control_state[key],
            rtol=0,
            atol=0,
        )


def test_decoupled_query_teacher_preserves_v8_inference_descriptor():
    kwargs = {
        "in_ch": (16, 16, 16),
        "feat_dim": 16,
        "num_classes": 4,
        "head_parts": (1, 2, 4),
        "part_pooling": "stripes",
        "scale_balanced_branches": True,
        "hierarchical_scales": True,
        "anatomical_auxiliary": True,
        "anatomical_token_dim": 16,
        "anatomical_multiscale": True,
        "inference_feature": "norm_concat_bn",
    }
    torch.manual_seed(41)
    v8 = MultiBranchHead(
        **kwargs,
        anatomical_target_type="learned_pose_concat_ema",
    ).eval()
    torch.manual_seed(41)
    decoupled = MultiBranchHead(
        **kwargs,
        anatomical_target_type="decoupled_pose_parsing_teacher",
    ).eval()
    query_calls = 0

    def count_query_calls(_module, _inputs, _output) -> None:
        nonlocal query_calls
        query_calls += 1

    hook = (
        decoupled.anatomical_auxiliary_pool.decoupled_query_teacher
        .register_forward_hook(count_query_calls)
    )
    feature_maps = (
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 24, 8),
    )
    try:
        with torch.no_grad():
            expected = v8(feature_maps)
            actual = decoupled(feature_maps)
    finally:
        hook.remove()

    assert actual.shape == expected.shape == (3, 48)
    assert query_calls == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_decoupled_query_losses_train_private_parsing_and_rgb_queries():
    torch.manual_seed(42)
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
        anatomical_target_type="decoupled_pose_parsing_teacher",
        inference_feature="norm_concat_bn",
    ).train()
    feature_maps = (
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 24, 8, requires_grad=True),
    )
    pose_keypoints = torch.tensor(_coco_keypoints())
    pose_keypoints[:, :2] = 2.0 * pose_keypoints[:, :2] - 1.0
    pose_keypoints = pose_keypoints[None].repeat(4, 1, 1)
    masks = torch.zeros(4, 6, 24, 8)
    for part_index in range(6):
        top = min(part_index * 4, 20)
        masks[:, part_index, top : top + 4] = 1
    _, features = head(
        feature_maps,
        anatomical_pose=pose_keypoints,
        anatomical_query_masks=masks,
    )

    assert features["_anatomical_query_student_tokens"].shape == (
        4,
        6,
        16,
    )
    assert features["_anatomical_query_teacher_tokens"].shape == (
        4,
        6,
        16,
    )
    assert features["_anatomical_query_part_logits"].shape == (
        4,
        6,
        12,
        4,
    )
    assert features["_anatomical_query_fine_part_logits"].shape == (
        4,
        6,
        24,
        8,
    )

    targets = {
        "masks": masks,
        "foreground_mask": masks.amax(dim=1, keepdim=True),
        "visibility": torch.ones(4, 6),
        "reliability": torch.ones(4, 6),
        "mask_valid": torch.ones(4, dtype=torch.bool),
        "valid": torch.ones(4, dtype=torch.bool),
    }
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_accessory_query = False
    trainer.anatomical_local_scale_weight = 0.60
    trainer.anatomical_fine_scale_weight = 0.40
    trainer.anatomical_query_diversity_margin = 0.10
    trainer.anatomical_query_start_epoch = 20
    trainer.anatomical_query_ramp_end_epoch = 50
    trainer.anatomical_student_start_epoch = 0
    trainer.anatomical_student_ramp_end_epoch = 50
    trainer.anatomical_decay_start_epoch = 120
    trainer.anatomical_decay_end_epoch = 170
    trainer.anatomical_foreground_weight = 0.03
    trainer.anatomical_visibility_weight = 0.05
    trainer.anatomical_query_distill_weight = 0.05
    trainer.anatomical_query_diversity_weight = 0.01
    trainer.anatomical_part_triplet_weight = 0.03
    trainer.margin = 0.3
    loss, components = trainer._decoupled_pose_parsing_query_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        epoch=50,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["query_distill"].item() > 0
    assert components["query_diversity"].item() >= 0
    assert components["part_triplet"].item() >= 0
    assert components["query_foreground"].item() > 0
    query_teacher = (
        head.anatomical_auxiliary_pool.decoupled_query_teacher
    )
    assert (
        query_teacher.parsing_adapters[0][0].weight.grad.abs().sum()
        > 0
    )
    assert query_teacher.part_predictors[0].weight.grad.abs().sum() > 0
    assert query_teacher.queries.grad.abs().sum() > 0
    assert feature_maps[1].grad.abs().sum() > 0
    assert feature_maps[2].grad.abs().sum() > 0


def test_decoupled_query_teacher_supports_optional_accessory():
    pool = DecoupledMaskedQueryTeacher(
        local_channels=16,
        fine_channels=16,
        token_dim=16,
        num_parts=7,
    )
    outputs = pool(
        torch.randn(2, 16, 12, 4),
        torch.randn(2, 16, 24, 8),
        torch.randn(2, 16, 12, 4),
        torch.randn(2, 16, 24, 8),
        torch.ones(2, 7, 24, 8),
    )

    assert outputs[0].shape == (2, 7, 16)
    assert outputs[6].shape == (2, 7, 16)
    assert outputs[5].shape == (2, 7, 12, 4)
    assert outputs[10].shape == (2, 1, 24, 8)
    assert outputs[11].shape == (2, 7, 24, 8)


def test_pose_semantic_teacher_losses_and_confidence_fusion_backpropagate():
    torch.manual_seed(30)
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_semantic_fused_ema",
        inference_feature="norm_concat_bn",
    ).train()
    feature_maps = (
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 24, 8, requires_grad=True),
    )
    pose_keypoints = torch.tensor(_coco_keypoints())
    pose_keypoints[:, :2] = 2.0 * pose_keypoints[:, :2] - 1.0
    pose_keypoints = pose_keypoints[None].repeat(4, 1, 1)
    _, features = head(
        feature_maps,
        anatomical_pose=pose_keypoints,
    )

    assert features["_anatomical_semantic_foreground_logits"].shape == (
        4,
        1,
        12,
        4,
    )
    assert features["_anatomical_semantic_part_logits"].shape == (
        4,
        6,
        12,
        4,
    )
    assert features[
        "_anatomical_semantic_fine_foreground_logits"
    ].shape == (4, 1, 24, 8)
    assert features["_anatomical_semantic_fine_part_logits"].shape == (
        4,
        6,
        24,
        8,
    )

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, 4),
        torch.tensor([-0.5, 0.5]),
        indexing="ij",
    )
    canonical_grid = torch.stack((grid_x, grid_y), dim=-1)
    masks = torch.zeros(4, 6, 24, 8)
    for part_index in range(6):
        top = part_index * 4
        masks[:, part_index, top : top + 5] = 1
    targets = {
        "masks": masks,
        "foreground_mask": masks.amax(dim=1, keepdim=True),
        "canonical_grid": canonical_grid[None, None].repeat(
            4,
            6,
            1,
            1,
            1,
        ),
        "canonical_grid_valid": torch.ones(
            4,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "canonical_grid_pose_valid": torch.ones(
            4,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "pose_keypoints": pose_keypoints,
        "visibility": torch.ones(4, 6),
        "reliability": torch.ones(4, 6),
        "pose_reliability": torch.full((4, 6), 0.75),
        "pose_mask_agreement": torch.full((4,), 0.8),
        "pose_valid": torch.ones(4, dtype=torch.bool),
        "mask_valid": torch.ones(4, dtype=torch.bool),
        "valid": torch.ones(4, dtype=torch.bool),
    }
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_target_type = "learned_pose_semantic_ema"
    trainer.anatomical_distill_weight = 0.10
    trainer.anatomical_attention_weight = 0.10
    trainer.anatomical_foreground_weight = 0.03
    trainer.anatomical_semantic_part_weight = 0.05
    trainer.anatomical_visibility_weight = 0.05
    trainer.anatomical_contrastive_weight = 0.10
    trainer.anatomical_descriptor_distill_weight = 0.0
    trainer.anatomical_branch_distill_weight = 0.0
    trainer.anatomical_pose_teacher_weight = 0.03
    trainer.anatomical_multiscale = True
    trainer.anatomical_local_scale_weight = 0.60
    trainer.anatomical_fine_scale_weight = 0.40
    trainer.anatomical_cross_scale_weight = 0.05
    trainer.anatomical_pose_only_reliability = 0.0
    trainer.anatomical_temperature = 0.07

    _, unfused_components = trainer._anatomical_auxiliary_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        epoch=50,
        return_components=True,
    )
    trainer.anatomical_target_type = (
        "learned_pose_semantic_fused_ema"
    )
    loss, fused_components = trainer._anatomical_auxiliary_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        epoch=50,
        return_components=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert fused_components["semantic_foreground"].item() > 0
    assert fused_components["semantic_part"].item() > 0
    assert not torch.allclose(
        fused_components["attention"],
        unfused_components["attention"],
    )
    semantic_heads = (
        head.anatomical_auxiliary_pool.semantic_prediction_heads
    )
    assert all(
        prediction_head.part_predictor.weight.grad is not None
        and prediction_head.part_predictor.weight.grad.abs().sum() > 0
        for prediction_head in semantic_heads
    )


def test_multiscale_ema_teacher_forward_and_loss_backpropagate():
    torch.manual_seed(31)
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
        anatomical_target_type="learned_pose_concat_ema",
        anatomical_branch_distill=True,
        inference_feature="norm_concat_bn",
    )
    feature_maps = (
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 24, 8, requires_grad=True),
    )
    pose_keypoints = torch.tensor(_coco_keypoints())
    pose_keypoints[:, :2] = 2.0 * pose_keypoints[:, :2] - 1.0
    pose_keypoints = pose_keypoints[None].repeat(4, 1, 1)

    head.train()
    _, features = head(
        feature_maps,
        anatomical_pose=pose_keypoints,
    )
    assert features["_anatomical_feature_map"].shape[-2:] == (12, 4)
    assert features["_anatomical_teacher_feature_map"].shape[-2:] == (
        24,
        8,
    )
    assert features["_anatomical_fine_feature_map"].shape[-2:] == (
        24,
        8,
    )
    branch_features = features["_anatomical_branch_features"]
    assert branch_features[0].shape == (4, 16)
    assert tuple(feature.shape for feature in branch_features[1]) == (
        (4, 8),
        (4, 8),
    )
    assert tuple(feature.shape for feature in branch_features[2]) == (
        (4, 4),
        (4, 4),
        (4, 4),
        (4, 4),
    )
    deployed_branches = (
        branch_features[0],
        *branch_features[1],
        *branch_features[2],
    )
    for branch_feature in deployed_branches:
        branch_feature.retain_grad()

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, 4),
        torch.tensor([-0.5, 0.5]),
        indexing="ij",
    )
    canonical_grid = torch.stack((grid_x, grid_y), dim=-1)
    targets = {
        "masks": torch.ones(4, 6, 24, 8),
        "canonical_grid": canonical_grid[None, None].repeat(
            4,
            6,
            1,
            1,
            1,
        ),
        "canonical_grid_valid": torch.ones(
            4,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "canonical_grid_pose_valid": torch.ones(
            4,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "pose_keypoints": pose_keypoints,
        "visibility": torch.ones(4, 6),
        "reliability": torch.ones(4, 6),
        "pose_reliability": torch.ones(4, 6),
        "pose_mask_agreement": torch.ones(4),
        "pose_valid": torch.ones(4, dtype=torch.bool),
        "mask_valid": torch.ones(4, dtype=torch.bool),
        "valid": torch.ones(4, dtype=torch.bool),
    }
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_target_type = "learned_pose_concat_ema"
    trainer.anatomical_distill_weight = 0.10
    trainer.anatomical_attention_weight = 0.10
    trainer.anatomical_visibility_weight = 0.05
    trainer.anatomical_contrastive_weight = 0.10
    trainer.anatomical_descriptor_distill_weight = 0.0
    trainer.anatomical_branch_distill_weight = 0.05
    trainer.anatomical_branch_global_coefficient = 0.20
    trainer.anatomical_branch_coarse_coefficient = 0.30
    trainer.anatomical_branch_fine_coefficient = 0.50
    trainer.anatomical_pose_teacher_weight = 0.03
    trainer.anatomical_multiscale = True
    trainer.anatomical_local_scale_weight = 0.60
    trainer.anatomical_fine_scale_weight = 0.40
    trainer.anatomical_cross_scale_weight = 0.05
    trainer.anatomical_pose_only_reliability = 0.0
    trainer.anatomical_temperature = 0.07

    dense_loss, dense_components = trainer._anatomical_auxiliary_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        epoch=1,
        return_components=True,
    )
    compact_targets = {
        key: value
        for key, value in targets.items()
        if key not in {"masks", "foreground_mask", "accessory_mask"}
    }
    compact_targets["mask_present"] = (
        targets["masks"].flatten(2).amax(dim=-1) > 1e-6
    )
    loss, components = trainer._anatomical_auxiliary_loss(
        features,
        compact_targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        epoch=1,
        return_components=True,
    )

    torch.testing.assert_close(loss, dense_loss)
    for key in components:
        torch.testing.assert_close(components[key], dense_components[key])
    loss.backward()

    assert torch.isfinite(loss)
    assert components["distill"].item() > 0
    assert components["attention"].item() > 0
    assert components["visibility"].item() > 0
    assert components["contrastive"].item() > 0
    assert components["branch_distill"].item() > 0
    assert components["branch_global"].item() > 0
    assert components["branch_coarse"].item() > 0
    assert components["branch_fine"].item() > 0
    assert components["pose_teacher"].item() > 0
    assert components["local_scale"].item() > 0
    assert components["fine_scale"].item() > 0
    assert all(
        branch_feature.grad is not None
        and branch_feature.grad.abs().sum().item() > 0
        for branch_feature in deployed_branches
    )
    assert head.anatomical_auxiliary_pool.online_pose_projection.weight.grad is not None
    assert all(
        parameter.grad is None
        for parameter in head.anatomical_auxiliary_pool.ema_pose_encoder.parameters()
    )


def test_anatomical_auxiliary_head_is_training_only():
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
    )
    features = (
        torch.randn(3, 16, 6, 2),
        torch.randn(3, 16, 12, 4),
        torch.randn(3, 16, 24, 8),
    )

    head.train()
    _, training_features = head(features)
    assert tuple(training_features["_anatomical_student_tokens"].shape) == (
        3,
        6,
        16,
    )
    assert tuple(training_features["_anatomical_attention"].shape) == (
        3,
        6,
        8,
        12,
        4,
    )
    assert torch.allclose(
        training_features["_anatomical_attention"].sum(dim=(-1, -2)),
        torch.ones(3, 6, 8),
        atol=1e-5,
    )
    assert training_features["_anatomical_feature_map"].shape == (3, 2, 12, 4)
    pool = head.anatomical_auxiliary_pool
    assert pool.role_queries.shape == (6, 2)
    assert pool.cell_embeddings.shape == (8, 2)
    assert torch.count_nonzero(pool.scale_query_offsets).item() == 0

    head.eval()
    with torch.no_grad():
        descriptor = head(features)
    assert tuple(descriptor.shape) == (3, 48)


def test_local_geometry_teacher_distills_into_deployed_descriptor():
    torch.manual_seed(29)
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_descriptor_distill=True,
        inference_feature="norm_concat_bn",
    )
    feature_maps = (
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 24, 8, requires_grad=True),
    )

    head.train()
    _, features = head(feature_maps)

    assert features["_anatomical_feature_map"].shape == (4, 2, 12, 4)
    assert tuple(features["norm_concat_bn"].shape) == (4, 48)
    assert tuple(features["_anatomical_final_student"].shape) == (4, 96)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, 4),
        torch.tensor([-0.5, 0.5]),
        indexing="ij",
    )
    canonical_grid = torch.stack((grid_x, grid_y), dim=-1)
    targets = {
        "masks": torch.ones(4, 6, 24, 8),
        "canonical_grid": canonical_grid[None, None].repeat(
            4,
            6,
            1,
            1,
            1,
        ),
        "canonical_grid_valid": torch.ones(
            4,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "visibility": torch.ones(4, 6),
        "reliability": torch.ones(4, 6),
        "valid": torch.ones(4, dtype=torch.bool),
    }
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_distill_weight = 0.0
    trainer.anatomical_attention_weight = 0.0
    trainer.anatomical_visibility_weight = 0.0
    trainer.anatomical_contrastive_weight = 0.0
    trainer.anatomical_descriptor_distill_weight = 1.0
    trainer.anatomical_pose_teacher_weight = 0.0
    trainer.anatomical_temperature = 0.07

    loss, components = trainer._anatomical_auxiliary_loss(
        features,
        targets,
        torch.arange(4),
        torch.arange(4),
        return_components=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["descriptor_distill"].item() > 0
    descriptor_projection = head.anatomical_auxiliary_pool.descriptor_projection
    assert descriptor_projection is not None
    assert descriptor_projection.weight.grad is not None
    assert descriptor_projection.weight.grad.abs().sum().item() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in head.bn_global.parameters()
    )

    head.eval()
    with torch.no_grad():
        descriptor = head(tuple(feature_map.detach() for feature_map in feature_maps))
    assert tuple(descriptor.shape) == (4, 48)


def test_anatomical_pool_uses_explicit_cells_and_scale_specific_offsets():
    torch.manual_seed(31)
    pool = AnatomicalAuxiliaryPool(
        channels=16,
        token_dim=16,
        fine_channels=24,
        multiscale=True,
    )
    student_source = torch.randn(2, 16, 8, 4, requires_grad=True)
    fine_source = torch.randn(2, 24, 16, 8, requires_grad=True)

    outputs = pool(
        student_source,
        fine_x=fine_source,
    )
    assert outputs[0].shape == (2, 2, 8, 4)
    assert outputs[1].shape == (2, 6, 16)
    assert outputs[2].shape == (2, 6, 8, 8, 4)
    assert outputs[4].shape == (2, 2, 16, 8)
    assert outputs[5].shape == (2, 6, 16)
    (outputs[1].square().mean() + outputs[5].square().mean()).backward()

    assert student_source.grad is not None
    assert student_source.grad.abs().sum().item() > 0
    assert fine_source.grad is not None
    assert fine_source.grad.abs().sum().item() > 0
    assert pool.role_queries.grad is not None
    assert pool.cell_embeddings.grad is not None
    assert pool.scale_query_offsets.grad is not None
    assert pool.scale_query_offsets.grad[0].abs().sum().item() > 0
    assert pool.scale_query_offsets.grad[1].abs().sum().item() > 0

    with torch.no_grad():
        baseline = pool(
            student_source.detach(),
            fine_x=fine_source.detach(),
        )
        pool.scale_query_offsets[1, :, 0].add_(0.5)
        changed = pool(
            student_source.detach(),
            fine_x=fine_source.detach(),
        )
    torch.testing.assert_close(baseline[2], changed[2])
    assert not torch.allclose(baseline[6], changed[6])


def test_geometry_teacher_is_causally_controlled_by_mask_and_pose():
    source = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.0, 1.0, 12),
            torch.linspace(-1.0, 1.0, 4),
            indexing="ij",
        ),
        dim=0,
    )[None].requires_grad_()
    masks = torch.zeros(1, 1, 24, 8)
    masks[..., :12, :] = 1
    shifted_masks = torch.zeros_like(masks)
    shifted_masks[..., 12:, :] = 1
    grid = torch.tensor(
        [[[[[-0.5, -0.75], [0.5, -0.75]], [[-0.5, -0.25], [0.5, -0.25]],
           [[-0.5, 0.25], [0.5, 0.25]], [[-0.5, 0.75], [0.5, 0.75]]]]]
    ).reshape(1, 1, 8, 2)
    valid = torch.ones(1, 1, 8, dtype=torch.bool)
    mask_valid = torch.ones(1, dtype=torch.bool)

    routing, _, _, tokens = _scale_aware_anatomical_targets(
        source,
        masks,
        grid,
        valid,
        mask_valid,
        fine_scale=False,
    )
    shifted_routing, _, _, shifted_tokens = (
        _scale_aware_anatomical_targets(
            source,
            shifted_masks,
            grid,
            valid,
            mask_valid,
            fine_scale=False,
        )
    )

    assert not torch.allclose(routing, shifted_routing)
    assert not torch.allclose(tokens, shifted_tokens)
    assert not tokens.requires_grad
    assert source.grad is None


def test_scale_aware_geometry_supervision_trains_rgb_student():
    torch.manual_seed(33)
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_descriptor_distill=True,
        inference_feature="norm_concat_bn",
    )
    feature_maps = (
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 24, 8, requires_grad=True),
    )
    head.train()
    _, features = head(feature_maps)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, 4),
        torch.tensor([-0.5, 0.5]),
        indexing="ij",
    )
    canonical_grid = torch.stack((grid_x, grid_y), dim=-1)
    targets = {
        "masks": torch.ones(4, 6, 24, 8),
        "canonical_grid": canonical_grid[None, None].repeat(
            4,
            6,
            1,
            1,
            1,
        ),
        "canonical_grid_valid": torch.ones(
            4,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "canonical_grid_pose_valid": torch.ones(
            4,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "visibility": torch.ones(4, 6),
        "reliability": torch.ones(4, 6),
        "pose_reliability": torch.ones(4, 6),
        "pose_mask_agreement": torch.ones(4),
        "pose_valid": torch.ones(4, dtype=torch.bool),
        "mask_valid": torch.ones(4, dtype=torch.bool),
        "valid": torch.ones(4, dtype=torch.bool),
    }
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_distill_weight = 0.20
    trainer.anatomical_attention_weight = 0.10
    trainer.anatomical_visibility_weight = 0.05
    trainer.anatomical_contrastive_weight = 0.10
    trainer.anatomical_descriptor_distill_weight = 0.10
    trainer.anatomical_pose_teacher_weight = 0.10
    trainer.anatomical_temperature = 0.07

    loss, components = trainer._anatomical_auxiliary_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        return_components=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["distill"].item() > 0
    assert components["attention"].item() > 0
    assert components["visibility"].item() > 0
    assert components["contrastive"].item() > 0
    assert components["descriptor_distill"].item() > 0
    assert components["pose_teacher"].item() > 0
    pool = head.anatomical_auxiliary_pool
    assert pool.feature_projection.weight.grad is not None
    assert pool.feature_projection.weight.grad.abs().sum().item() > 0
    assert pool.role_queries.grad is not None
    assert pool.role_queries.grad.abs().sum().item() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in head.bn_global.parameters()
    )

    head.eval()
    with torch.no_grad():
        descriptor = head(tuple(feature_map.detach() for feature_map in feature_maps))
    assert tuple(descriptor.shape) == (4, 48)


def test_multiscale_anatomy_supervises_local_and_fine_feature_maps():
    torch.manual_seed(37)
    head = MultiBranchHead(
        (16, 16, 16),
        feat_dim=16,
        num_classes=4,
        head_parts=(1, 2, 4),
        part_pooling="stripes",
        scale_balanced_branches=True,
        hierarchical_scales=True,
        anatomical_auxiliary=True,
        anatomical_token_dim=16,
        anatomical_multiscale=True,
        inference_feature="norm_concat_bn",
    )
    feature_maps = (
        torch.randn(4, 16, 6, 2, requires_grad=True),
        torch.randn(4, 16, 12, 4, requires_grad=True),
        torch.randn(4, 16, 24, 8, requires_grad=True),
    )
    head.train()
    _, features = head(feature_maps)

    assert tuple(features["_anatomical_student_tokens"].shape) == (4, 6, 16)
    assert tuple(features["_anatomical_fine_student_tokens"].shape) == (4, 6, 16)
    assert features["_anatomical_attention"].shape[-2:] == (12, 4)
    assert features["_anatomical_fine_attention"].shape[-2:] == (
        24,
        8,
    )

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, 4),
        torch.tensor([-0.5, 0.5]),
        indexing="ij",
    )
    canonical_grid = torch.stack((grid_x, grid_y), dim=-1)
    targets = {
        "masks": torch.ones(4, 6, 24, 8),
        "canonical_grid": canonical_grid[None, None].repeat(
            4,
            6,
            1,
            1,
            1,
        ),
        "canonical_grid_valid": torch.ones(
            4,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "visibility": torch.ones(4, 6),
        "reliability": torch.ones(4, 6),
        "valid": torch.ones(4, dtype=torch.bool),
    }
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_multiscale = True
    trainer.anatomical_local_scale_weight = 0.60
    trainer.anatomical_fine_scale_weight = 0.40
    trainer.anatomical_cross_scale_weight = 0.05
    trainer.anatomical_distill_weight = 0.10
    trainer.anatomical_attention_weight = 0.10
    trainer.anatomical_visibility_weight = 0.05
    trainer.anatomical_contrastive_weight = 0.10
    trainer.anatomical_descriptor_distill_weight = 0.0
    trainer.anatomical_pose_teacher_weight = 0.03
    trainer.anatomical_pose_only_reliability = 0.0
    trainer.anatomical_temperature = 0.07
    trainer.anatomical_student_start_epoch = 0
    trainer.anatomical_student_ramp_end_epoch = 50
    trainer.anatomical_fine_start_epoch = 40
    trainer.anatomical_fine_ramp_end_epoch = 80
    trainer.anatomical_decay_start_epoch = 120
    trainer.anatomical_decay_end_epoch = 170

    loss, components = trainer._anatomical_auxiliary_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        epoch=60,
        return_components=True,
    )
    expected_loss = (
        0.60 * components["local_scale"]
        + 0.50 * (0.40 * components["fine_scale"] + 0.05 * components["cross_scale"])
    )
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()

    assert torch.isfinite(loss)
    assert components["local_scale"].item() > 0
    assert components["fine_scale"].item() > 0
    assert components["cross_scale"].item() > 0
    assert feature_maps[1].grad is not None
    assert feature_maps[1].grad.abs().sum().item() > 0
    assert feature_maps[2].grad is not None
    assert feature_maps[2].grad.abs().sum().item() > 0


def test_multiscale_anatomical_loss_is_fp32_and_finite_under_cpu_autocast():
    dtype = torch.bfloat16
    feature_map = torch.randn(
        2,
        2,
        8,
        4,
        dtype=dtype,
        requires_grad=True,
    )
    fine_feature_map = torch.randn(
        2,
        2,
        16,
        8,
        dtype=dtype,
        requires_grad=True,
    )
    student_tokens = torch.randn(
        2,
        6,
        16,
        dtype=dtype,
        requires_grad=True,
    )
    fine_student_tokens = torch.randn(
        2,
        6,
        16,
        dtype=dtype,
        requires_grad=True,
    )
    attention = torch.full(
        (2, 6, 8, 8, 4),
        1.0 / 32,
        dtype=dtype,
        requires_grad=True,
    )
    fine_attention = torch.full(
        (2, 6, 8, 16, 8),
        1.0 / 128,
        dtype=dtype,
        requires_grad=True,
    )
    visibility_logits = torch.zeros(
        2,
        6,
        dtype=dtype,
        requires_grad=True,
    )
    fine_visibility_logits = torch.zeros(
        2,
        6,
        dtype=dtype,
        requires_grad=True,
    )
    features = {
        "_anatomical_feature_map": feature_map,
        "_anatomical_student_tokens": student_tokens,
        "_anatomical_attention": attention,
        "_anatomical_visibility_logits": visibility_logits,
        "_anatomical_fine_feature_map": fine_feature_map,
        "_anatomical_fine_student_tokens": fine_student_tokens,
        "_anatomical_fine_attention": fine_attention,
        "_anatomical_fine_visibility_logits": fine_visibility_logits,
    }
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, 4),
        torch.tensor([-0.5, 0.5]),
        indexing="ij",
    )
    canonical_grid = torch.stack((grid_x, grid_y), dim=-1)
    targets = {
        "masks": torch.ones(2, 6, 16, 8),
        "canonical_grid": canonical_grid[None, None].repeat(
            2,
            6,
            1,
            1,
            1,
        ),
        "canonical_grid_valid": torch.ones(
            2,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "visibility": torch.ones(2, 6),
        "reliability": torch.ones(2, 6),
        "valid": torch.ones(2, dtype=torch.bool),
    }
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_multiscale = True
    trainer.anatomical_local_scale_weight = 0.60
    trainer.anatomical_fine_scale_weight = 0.40
    trainer.anatomical_cross_scale_weight = 0.05
    trainer.anatomical_distill_weight = 0.10
    trainer.anatomical_attention_weight = 0.10
    trainer.anatomical_visibility_weight = 0.05
    trainer.anatomical_contrastive_weight = 0.10
    trainer.anatomical_descriptor_distill_weight = 0.0
    trainer.anatomical_pose_teacher_weight = 0.03
    trainer.anatomical_pose_only_reliability = 0.0
    trainer.anatomical_temperature = 0.07

    with torch.amp.autocast("cpu", dtype=dtype):
        loss, components = trainer._anatomical_auxiliary_loss(
            features,
            targets,
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            return_components=True,
        )
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(torch.isfinite(component) for component in components.values())
    assert feature_map.grad is None
    assert fine_feature_map.grad is None
    for tensor in (
        student_tokens,
        fine_student_tokens,
        attention,
        fine_attention,
        visibility_logits,
        fine_visibility_logits,
    ):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_v6_anatomical_schedule_warms_up_ramps_and_decays():
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_student_start_epoch = 20
    trainer.anatomical_student_ramp_end_epoch = 50
    trainer.anatomical_decay_start_epoch = 120
    trainer.anatomical_decay_end_epoch = 170

    assert trainer._anatomical_schedule_scales(1) == (0.0, 1.0)
    assert trainer._anatomical_schedule_scales(20) == (0.0, 1.0)
    assert trainer._anatomical_schedule_scales(35) == (0.5, 1.0)
    assert trainer._anatomical_schedule_scales(50) == (1.0, 1.0)
    assert trainer._anatomical_schedule_scales(120) == (1.0, 1.0)
    assert trainer._anatomical_schedule_scales(145) == (0.5, 0.5)
    assert trainer._anatomical_schedule_scales(170) == (0.0, 0.0)
    assert trainer._anatomical_schedule_scales(200) == (0.0, 0.0)


def test_anatomical_loss_skips_targets_before_student_schedule_starts():
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_student_start_epoch = 20
    trainer.anatomical_student_ramp_end_epoch = 50
    trainer.anatomical_fine_start_epoch = 0
    trainer.anatomical_fine_ramp_end_epoch = 0
    trainer.anatomical_decay_start_epoch = 120
    trainer.anatomical_decay_end_epoch = 170

    loss, components = trainer._anatomical_auxiliary_loss(
        {},
        None,
        torch.tensor([0]),
        torch.tensor([0]),
        epoch=10,
        return_components=True,
    )

    assert loss.item() == 0
    assert all(component.item() == 0 for component in components.values())


def test_fine_anatomical_schedule_can_ramp_after_local_student():
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_student_start_epoch = 0
    trainer.anatomical_student_ramp_end_epoch = 50
    trainer.anatomical_fine_start_epoch = 40
    trainer.anatomical_fine_ramp_end_epoch = 80
    trainer.anatomical_decay_start_epoch = 120
    trainer.anatomical_decay_end_epoch = 170

    for epoch, expected in ((20, 0.0), (40, 0.0), (60, 0.5), (80, 1.0), (145, 0.5), (170, 0.0)):
        student_scale, decay_scale = trainer._anatomical_schedule_scales(epoch)
        assert (
            trainer._anatomical_fine_schedule_scale(
                epoch,
                student_scale=student_scale,
                decay_scale=decay_scale,
            )
            == expected
        )

    trainer.anatomical_fine_start_epoch = 0
    trainer.anatomical_fine_ramp_end_epoch = 0
    student_scale, decay_scale = trainer._anatomical_schedule_scales(25)
    assert (
        trainer._anatomical_fine_schedule_scale(
            25,
            student_scale=student_scale,
            decay_scale=decay_scale,
        )
        == student_scale
    )


def test_geometry_teacher_ignores_mask_rejected_canonical_cells():
    batch_size, num_parts, channels = 2, 6, 2
    height, width = 8, 4
    feature_map = torch.randn(batch_size, channels, height, width)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, 4),
        torch.tensor([-0.5, 0.5]),
        indexing="ij",
    )
    canonical_grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
        1,
        1,
        8,
        2,
    ).repeat(
        batch_size, num_parts, 1, 1
    )
    changed_grid = canonical_grid.clone()
    changed_grid[..., 1:, :] = 10.0
    mask_grid_valid = torch.zeros(batch_size, num_parts, 8, dtype=torch.bool)
    mask_grid_valid[..., 0] = True
    masks = torch.ones(batch_size, num_parts, height, width)
    mask_valid = torch.ones(batch_size, dtype=torch.bool)

    expected = _scale_aware_anatomical_targets(
        feature_map,
        masks,
        canonical_grid,
        mask_grid_valid,
        mask_valid,
        fine_scale=False,
    )
    actual = _scale_aware_anatomical_targets(
        feature_map,
        masks,
        changed_grid,
        mask_grid_valid,
        mask_valid,
        fine_scale=False,
    )

    for expected_value, actual_value in zip(expected, actual, strict=True):
        torch.testing.assert_close(expected_value, actual_value)


def test_anatomical_loss_is_finite_and_backpropagates():
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_distill_weight = 0.20
    trainer.anatomical_attention_weight = 0.10
    trainer.anatomical_visibility_weight = 0.05
    trainer.anatomical_contrastive_weight = 0.10
    trainer.anatomical_descriptor_distill_weight = 0.0
    trainer.anatomical_pose_teacher_weight = 0.0
    trainer.anatomical_temperature = 0.07

    feature_map = torch.randn(4, 2, 6, 3)
    student_tokens = torch.randn(4, 6, 16, requires_grad=True)
    attention_logits = torch.randn(4, 6, 8, 6, 3, requires_grad=True)
    visibility_logits = torch.randn(4, 6, requires_grad=True)
    masks = torch.zeros(4, 6, 12, 6)
    for part in range(6):
        row = min(part * 2, 10)
        masks[:, part, row : row + 2] = 1
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-0.75, 0.75, 4),
        torch.tensor([-0.5, 0.5]),
        indexing="ij",
    )
    canonical_grid = torch.stack((grid_x, grid_y), dim=-1)
    canonical_grid = canonical_grid[None, None].repeat(4, 6, 1, 1, 1)
    targets = {
        "masks": masks,
        "canonical_grid": canonical_grid,
        "canonical_grid_valid": torch.ones(
            4,
            6,
            4,
            2,
            dtype=torch.bool,
        ),
        "visibility": torch.full((4, 6), 0.9),
        "reliability": torch.full((4, 6), 0.8),
        "valid": torch.ones(4, dtype=torch.bool),
    }
    features = {
        "_anatomical_feature_map": feature_map,
        "_anatomical_student_tokens": student_tokens,
        "_anatomical_attention": attention_logits.flatten(3).softmax(dim=-1).reshape_as(attention_logits),
        "_anatomical_visibility_logits": visibility_logits,
    }

    loss = trainer._anatomical_auxiliary_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert student_tokens.grad is not None
    assert attention_logits.grad is not None
    assert visibility_logits.grad is not None

    total, components = trainer._anatomical_auxiliary_loss(
        features,
        targets,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        return_components=True,
    )
    assert torch.isfinite(total)
    assert set(components) == {
        "distill",
        "attention",
        "visibility",
        "contrastive",
        "descriptor_distill",
        "pose_teacher",
        "local_scale",
        "fine_scale",
        "cross_scale",
        "valid_part_fraction",
        "cross_camera_anchor_fraction",
    }


def test_scale_aware_teacher_matches_exact_student_routing():
    trainer = object.__new__(ReIDTrainer)
    trainer.anatomical_auxiliary = True
    trainer.anatomical_distill_weight = 1.0
    trainer.anatomical_attention_weight = 1.0
    trainer.anatomical_visibility_weight = 0.0
    trainer.anatomical_contrastive_weight = 0.0
    trainer.anatomical_descriptor_distill_weight = 0.0
    trainer.anatomical_pose_teacher_weight = 0.0
    trainer.anatomical_temperature = 0.07

    batch_size, num_parts, channels = 2, 6, 2
    height, width = ANATOMICAL_CANONICAL_GRID_SIZE
    feature_map = torch.randn(batch_size, channels, height, width)
    grid_y, grid_x = torch.meshgrid(
        2.0 * (torch.arange(height) + 0.5) / height - 1.0,
        2.0 * (torch.arange(width) + 0.5) / width - 1.0,
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)
    canonical_grid = grid[None, None].repeat(
        batch_size,
        num_parts,
        1,
        1,
        1,
    )
    masks = torch.ones(
        batch_size,
        num_parts,
        height,
        width,
    )
    grid_valid = torch.ones(
        batch_size,
        num_parts,
        height * width,
        dtype=torch.bool,
    )
    student_attention, _, _, teacher_cells = (
        _scale_aware_anatomical_targets(
            feature_map,
            masks,
            canonical_grid.flatten(2, 3),
            grid_valid,
            torch.ones(batch_size, dtype=torch.bool),
            fine_scale=False,
        )
    )
    student_tokens = teacher_cells.flatten(2)
    targets = {
        "masks": masks,
        "canonical_grid": canonical_grid,
        "canonical_grid_valid": grid_valid.unflatten(
            -1,
            ANATOMICAL_CANONICAL_GRID_SIZE,
        ),
        "visibility": torch.ones(batch_size, num_parts),
        "reliability": torch.ones(batch_size, num_parts),
        "valid": torch.ones(batch_size, dtype=torch.bool),
    }
    features = {
        "_anatomical_feature_map": feature_map,
        "_anatomical_student_tokens": student_tokens,
        "_anatomical_attention": student_attention,
        "_anatomical_visibility_logits": torch.zeros(
            batch_size,
            num_parts,
        ),
    }

    total, components = trainer._anatomical_auxiliary_loss(
        features,
        targets,
        torch.arange(batch_size),
        torch.arange(batch_size),
        return_components=True,
    )

    assert total.abs().item() < 1e-5
    assert components["distill"].abs().item() < 1e-5
    assert components["attention"].abs().item() < 1e-5
