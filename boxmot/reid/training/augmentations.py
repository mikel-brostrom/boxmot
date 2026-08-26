"""Validation and construction of modular ReID training augmentations."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from boxmot.reid.datasets.anatomical import PoseAnatomicalTargetProvider
from boxmot.reid.datasets.pav_mosaic import (
    IndexedTransformCompose,
    PoseAlignedViewMosaic,
)
from boxmot.reid.datasets.transforms import (
    IdentityPreservingBackgroundMosaic,
    build_clean_train_transforms,
    build_train_transforms,
)
from boxmot.reid.training.config import AugmentationConfig
from boxmot.utils import logger as LOGGER


@dataclass(frozen=True)
class TrainingAugmentationBundle:
    """Runtime transforms and privileged targets consumed by the dataset."""

    image_transform: Any
    sample_transform: Any | None
    clean_transform: Any | None
    return_clean_view: bool
    return_clean_anatomical_target: bool
    anatomical_target_provider: PoseAnatomicalTargetProvider | None


def augmentation_config_from_options(options: object) -> AugmentationConfig:
    """Project a flat trainer/options object onto its typed augmentation config."""
    return AugmentationConfig(**{field.name: getattr(options, field.name) for field in fields(AugmentationConfig)})


def validate_augmentation_config(
    config: AugmentationConfig,
    *,
    epochs: int,
) -> None:
    """Validate every augmentation independently of trainer orchestration."""
    for name in ("random_grayscale", "random_erasing"):
        value = float(getattr(config, name))
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must satisfy 0 <= value <= 1")
    if config.random_crop_scale < 1.0:
        raise ValueError("random_crop_scale must be >= 1.0")

    if not 0 <= config.background_mosaic_probability <= 1:
        raise ValueError("background_mosaic_probability must satisfy 0 <= value <= 1")
    if (
        config.background_mosaic_start_epoch < 0
        or config.background_mosaic_ramp_end_epoch < config.background_mosaic_start_epoch
    ):
        raise ValueError("background mosaic epochs must satisfy 0 <= start_epoch <= ramp_end_epoch")
    if not (0 <= config.background_mosaic_min_foreground_ratio < config.background_mosaic_max_foreground_ratio <= 1):
        raise ValueError("background mosaic foreground ratios must satisfy 0 <= min < max <= 1")
    if config.background_mosaic_feather < 0 or config.background_mosaic_dilation < 0:
        raise ValueError("background mosaic feather and dilation must be non-negative")
    if not 0 <= config.background_mosaic_occluder_probability <= 1:
        raise ValueError("background_mosaic_occluder_probability must be in [0, 1]")
    if not (0 < config.background_mosaic_occluder_min_area <= config.background_mosaic_occluder_max_area <= 1):
        raise ValueError("background mosaic occluder area must satisfy 0 < min <= max <= 1")
    if config.background_mosaic and not config.background_mosaic_mask_dir:
        raise ValueError("background_mosaic_mask_dir is required when background mosaic is enabled")

    if not 0 <= config.same_id_part_mosaic_probability <= 1:
        raise ValueError("same_id_part_mosaic_probability must be in [0, 1]")
    if config.same_id_part_mosaic_max_regions not in {1, 2}:
        raise ValueError("same_id_part_mosaic_max_regions must be 1 or 2")
    if not (0 < config.same_id_part_mosaic_min_area <= config.same_id_part_mosaic_max_area <= 1):
        raise ValueError("same-ID part mosaic area must satisfy 0 < min_area <= max_area <= 1")
    if not 0 <= config.same_id_part_mosaic_boundary_jitter <= 0.1:
        raise ValueError("same_id_part_mosaic_boundary_jitter must be in [0, 0.1]")
    if not 0 <= config.same_id_part_mosaic_cross_camera_rate <= 1:
        raise ValueError("same_id_part_mosaic_cross_camera_rate must be in [0, 1]")
    if not 0 <= config.same_id_part_mosaic_min_unaltered <= 1:
        raise ValueError("same_id_part_mosaic_min_unaltered must be in [0, 1]")

    if not 0 <= config.pav_mosaic_probability <= 1:
        raise ValueError("pav_mosaic_probability must be in [0, 1]")
    if not 1 <= config.pav_mosaic_max_parts <= 7:
        raise ValueError("pav_mosaic_max_parts must be in [1, 7]")
    if not 0 < config.pav_mosaic_max_foreground_replacement <= 1:
        raise ValueError("pav_mosaic_max_foreground_replacement must be in (0, 1]")
    for name in (
        "pav_mosaic_cross_camera_rate",
        "pav_mosaic_different_pose_rate",
        "pav_mosaic_min_keypoint_confidence",
        "pav_mosaic_min_unaltered",
    ):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if config.pav_mosaic_warmup_epochs < 0:
        raise ValueError("pav_mosaic_warmup_epochs must be non-negative")
    if config.pav_mosaic and not 0 <= config.pav_mosaic_decay_start_epoch <= epochs:
        raise ValueError("pav_mosaic_decay_start_epoch must be within training")
    if not 0 <= config.pav_mosaic_final_probability_scale <= 1:
        raise ValueError("pav_mosaic_final_probability_scale must be in [0, 1]")
    if config.pav_consistency_weight < 0:
        raise ValueError("pav_consistency_weight must be non-negative")
    if config.clean_student_consistency_weight < 0:
        raise ValueError("clean_student_consistency_weight must be non-negative")
    if config.pav_mosaic and not config.pav_metadata_dir:
        raise ValueError("pav_metadata_dir is required when PAV mosaic is enabled")
    if config.pav_consistency_weight > 0 and not config.pav_mosaic:
        raise ValueError("PAV consistency requires PAV mosaic to be enabled")


def pav_requires_clean_view(
    config: AugmentationConfig,
    *,
    batch_size: int,
) -> bool:
    """Return whether PAV needs clean tensors for consistency or reversion."""
    if not config.pav_mosaic:
        return False
    if config.pav_consistency_weight > 0:
        return True
    max_augmented = int(math.floor(batch_size * (1.0 - config.pav_mosaic_min_unaltered)))
    attempt_probability = config.pav_mosaic_probability
    if config.background_mosaic:
        attempt_probability = 1.0 - ((1.0 - attempt_probability) * (1.0 - config.background_mosaic_probability))
    if max_augmented >= batch_size or attempt_probability <= 0:
        return False
    overflow_probability = sum(
        math.comb(batch_size, augmented_count)
        * attempt_probability**augmented_count
        * (1.0 - attempt_probability) ** (batch_size - augmented_count)
        for augmented_count in range(max_augmented + 1, batch_size + 1)
    )
    # Successful PAV applications are a subset of attempts. This binomial tail
    # is therefore a conservative bound for deciding whether to copy a batch.
    return overflow_probability >= 1e-6


def build_training_augmentation_bundle(
    options: object,
    dataset: object,
    *,
    batch_size: int,
) -> TrainingAugmentationBundle:
    """Build enabled augmentations in their canonical, identity-safe order."""
    config = augmentation_config_from_options(options)
    image_transform = build_train_transforms(
        getattr(options, "img_size"),
        preprocess=getattr(options, "preprocess"),
        color_jitter=config.color_jitter,
        gaussian_blur=config.gaussian_blur,
        random_grayscale=config.random_grayscale,
        random_erasing=(0.0 if config.same_id_part_mosaic else config.random_erasing),
        random_patch=config.random_patch,
        random_crop_scale=config.random_crop_scale,
        color_augmentation=config.color_augmentation,
    )
    image_root = getattr(dataset, "root", None)
    anatomical_target_provider = _build_anatomical_target_provider(
        options,
        config,
        dataset,
        image_root=image_root,
    )

    sample_transforms: list[Any] = []
    if config.background_mosaic:
        sample_transforms.append(
            _build_background_mosaic(
                config,
                dataset,
                image_root=image_root,
            )
        )
    if config.pav_mosaic:
        sample_transforms.append(
            _build_pav_mosaic(
                options,
                config,
                dataset,
                image_root=image_root,
            )
        )
    _log_same_id_part_mosaic(config)
    sample_transform = IndexedTransformCompose(sample_transforms) if sample_transforms else None

    clean_student_consistency = config.clean_student_consistency_weight > 0
    hpgrd_paired_view = float(getattr(options, "hpgrd_background_weight", 0.0)) > 0
    return_clean_view = (
        clean_student_consistency
        or hpgrd_paired_view
        or pav_requires_clean_view(
            config,
            batch_size=batch_size,
        )
    )
    if config.pav_mosaic:
        LOGGER.info(
            "PAV clean views "
            + (
                "enabled for consistency/reversion"
                if return_clean_view
                else ("disabled because consistency is off and batch overflow risk is below 1e-6")
            )
        )
    clean_transform = None
    if clean_student_consistency or hpgrd_paired_view:
        clean_transform = build_clean_train_transforms(
            getattr(options, "img_size"),
            preprocess=getattr(options, "preprocess"),
        )
        LOGGER.info("Deterministic clean teacher views enabled for augmented-student query/descriptor consistency")
    elif return_clean_view:
        clean_transform = build_train_transforms(
            getattr(options, "img_size"),
            preprocess=getattr(options, "preprocess"),
            color_jitter=config.color_jitter,
            gaussian_blur=config.gaussian_blur,
            random_grayscale=config.random_grayscale,
            random_erasing=config.random_erasing,
            random_patch=config.random_patch,
            random_crop_scale=config.random_crop_scale,
            color_augmentation=config.color_augmentation,
        )
    return TrainingAugmentationBundle(
        image_transform=image_transform,
        sample_transform=sample_transform,
        clean_transform=clean_transform,
        return_clean_view=return_clean_view,
        return_clean_anatomical_target=clean_student_consistency,
        anatomical_target_provider=anatomical_target_provider,
    )


def _build_anatomical_target_provider(
    options: object,
    config: AugmentationConfig,
    dataset: object,
    *,
    image_root: Any,
) -> PoseAnatomicalTargetProvider | None:
    hpgrd_needs_dense_parts = (
        float(getattr(options, "hpgrd_part_weight", 0.0)) > 0
        or float(getattr(options, "hpgrd_part_drop_weight", 0.0)) > 0
    )
    if not config.anatomical_auxiliary and not hpgrd_needs_dense_parts:
        return None
    if image_root is None:
        raise ValueError("Anatomical supervision currently requires one dataset with a concrete root")
    metadata_root = Path(str(config.anatomical_metadata_dir))
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"Anatomical metadata directory does not exist: {metadata_root}")
    provider = PoseAnatomicalTargetProvider(
        dataset.train.samples,
        image_root=image_root,
        metadata_root=metadata_root,
        person_mask_dir=config.anatomical_person_mask_dir,
        min_keypoint_confidence=config.anatomical_min_keypoint_confidence,
        pose_only_reliability=config.anatomical_pose_only_reliability,
        compact_nonsemantic=(
            str(getattr(options, "anatomical_target_type")).lower() == "learned_pose_concat_ema"
            and float(getattr(options, "hpgrd_part_weight", 0.0)) <= 0
            and float(getattr(options, "hpgrd_part_drop_weight", 0.0)) <= 0
        ),
    )
    if provider.matched_record_count == 0:
        raise ValueError(
            "Anatomical metadata does not match any training images. "
            f"dataset_root={image_root}, metadata={metadata_root}"
        )
    if provider.pose_record_count == 0:
        raise ValueError(
            f"Anatomical metadata has no valid COCO-17 pose records for the training images: {metadata_root}"
        )
    if provider.effective_supervision_record_count == 0:
        raise ValueError(
            "Anatomical metadata has no effectively supervised training "
            "records. Provide valid person masks or set a positive "
            "anatomical_pose_only_reliability."
        )
    sample_count = len(dataset.train.samples)
    effective_coverage = provider.effective_supervision_record_count / sample_count if sample_count else 0.0
    if effective_coverage < config.anatomical_min_effective_coverage:
        raise ValueError(
            "Anatomical supervision coverage is below the configured minimum: "
            f"effective={provider.effective_supervision_record_count}/"
            f"{sample_count} ({effective_coverage:.1%}), "
            f"required={config.anatomical_min_effective_coverage:.1%}. "
            "Check pose metadata, person masks, and dataset roots."
        )
    if provider.missing_person_mask_count:
        raise FileNotFoundError(
            "Anatomical metadata declares missing person masks: "
            f"count={provider.missing_person_mask_count}, "
            f"first={provider.first_missing_person_mask}"
        )
    if getattr(options, "anatomical_accessory_query") and provider.accessory_mask_record_count == 0:
        raise ValueError(f"Accessory query is enabled but no valid bag masks match the training set: {metadata_root}")
    fine_start = getattr(options, "anatomical_fine_start_epoch")
    fine_end = getattr(options, "anatomical_fine_ramp_end_epoch")
    fine_schedule = "shared" if fine_start == 0 and fine_end == 0 else f"{fine_start}->{fine_end}"
    LOGGER.info(
        "Privileged anatomical supervision enabled: "
        "parts=head,torso,left/right arms,left/right legs, "
        f"token_dim={getattr(options, 'anatomical_token_dim')}, "
        f"multiscale={getattr(options, 'anatomical_multiscale')}, "
        f"teacher={getattr(options, 'anatomical_target_type')}, "
        f"deployment={getattr(options, 'anatomical_deployment')}, "
        "branch_distill="
        f"{getattr(options, 'anatomical_branch_distill_weight'):g}, "
        f"fine_schedule={fine_schedule}, "
        f"coverage={provider.matched_record_count}/{len(dataset.train.samples)}, "
        f"effective={provider.effective_supervision_record_count}, "
        f"effective_coverage={effective_coverage:.1%}, "
        f"qualified_pose={provider.qualified_pose_record_count}, "
        f"person_masks={provider.person_mask_record_count}, "
        f"nonempty_person_masks={provider.nonempty_person_mask_record_count}, "
        f"external_person_masks={provider.external_person_mask_record_count}, "
        f"accessory_masks={provider.accessory_mask_record_count}, "
        f"metadata={metadata_root}"
    )
    return provider


def _build_pav_mosaic(
    options: object,
    config: AugmentationConfig,
    dataset: object,
    *,
    image_root: Any,
) -> PoseAlignedViewMosaic:
    if image_root is None:
        raise ValueError("PAV mosaic currently requires one dataset with a concrete root")
    metadata_root = Path(str(config.pav_metadata_dir))
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"PAV metadata directory does not exist: {metadata_root}")
    transform = PoseAlignedViewMosaic(
        dataset.train.samples,
        image_root=image_root,
        metadata_root=metadata_root,
        probability=config.pav_mosaic_probability,
        max_parts=config.pav_mosaic_max_parts,
        max_foreground_replacement=config.pav_mosaic_max_foreground_replacement,
        cross_camera_rate=config.pav_mosaic_cross_camera_rate,
        different_pose_rate=config.pav_mosaic_different_pose_rate,
        min_keypoint_confidence=config.pav_mosaic_min_keypoint_confidence,
        warmup_epochs=config.pav_mosaic_warmup_epochs,
        decay_start_epoch=config.pav_mosaic_decay_start_epoch,
        decay_end_epoch=getattr(options, "epochs"),
        final_probability_scale=config.pav_mosaic_final_probability_scale,
    )
    LOGGER.info(
        "Pose-aligned view mosaic enabled: "
        f"p={config.pav_mosaic_probability:.2f}, "
        f"parts=1-{config.pav_mosaic_max_parts}, "
        f"replacement<={config.pav_mosaic_max_foreground_replacement:.2f}, "
        f"cross_camera_rate={config.pav_mosaic_cross_camera_rate:.2f}, "
        f"metadata={metadata_root}"
    )
    return transform


def _build_background_mosaic(
    config: AugmentationConfig,
    dataset: object,
    *,
    image_root: Any,
) -> IdentityPreservingBackgroundMosaic:
    if image_root is None:
        raise ValueError("background mosaic currently requires one dataset with a concrete root")
    mask_root = Path(str(config.background_mosaic_mask_dir))
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Background mosaic mask directory does not exist: {mask_root}")
    primary_mask_root = mask_root / "primary"
    donor_mask_root = mask_root / "all_people"
    missing_mask_roots = [path for path in (primary_mask_root, donor_mask_root) if not path.is_dir()]
    if missing_mask_roots:
        raise FileNotFoundError(
            f"Background mosaic requires separate primary/ and all_people/ mask trees; missing: {missing_mask_roots}"
        )
    transform = IdentityPreservingBackgroundMosaic(
        dataset.train.samples,
        image_root=image_root,
        primary_mask_root=primary_mask_root,
        donor_mask_root=donor_mask_root,
        probability=config.background_mosaic_probability,
        start_epoch=config.background_mosaic_start_epoch,
        ramp_end_epoch=config.background_mosaic_ramp_end_epoch,
        min_foreground_ratio=config.background_mosaic_min_foreground_ratio,
        max_foreground_ratio=config.background_mosaic_max_foreground_ratio,
        feather=config.background_mosaic_feather,
        dilation=config.background_mosaic_dilation,
        occluder_probability=config.background_mosaic_occluder_probability,
        occluder_min_area=config.background_mosaic_occluder_min_area,
        occluder_max_area=config.background_mosaic_occluder_max_area,
    )
    LOGGER.info(
        "Identity-preserving background mosaic enabled: "
        f"p={config.background_mosaic_probability:.2f}, "
        f"ramp={config.background_mosaic_start_epoch}->"
        f"{config.background_mosaic_ramp_end_epoch}, "
        f"primary_masks={primary_mask_root}, donor_masks={donor_mask_root}"
    )
    return transform


def _log_same_id_part_mosaic(config: AugmentationConfig) -> None:
    if not config.same_id_part_mosaic:
        return
    LOGGER.info(
        "Cross-camera same-ID part mosaic enabled: "
        f"p={config.same_id_part_mosaic_probability:.2f}, "
        f"regions=1-{config.same_id_part_mosaic_max_regions}, "
        f"area={config.same_id_part_mosaic_min_area:.2f}-"
        f"{config.same_id_part_mosaic_max_area:.2f}, "
        f"cross_camera_rate={config.same_id_part_mosaic_cross_camera_rate:.2f}, "
        f"min_unaltered={config.same_id_part_mosaic_min_unaltered:.2f}"
    )


__all__ = [
    "TrainingAugmentationBundle",
    "augmentation_config_from_options",
    "build_training_augmentation_bundle",
    "pav_requires_clean_view",
    "validate_augmentation_config",
]
