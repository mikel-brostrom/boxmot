"""Tests for identity-preserving ReID background mosaic augmentation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from boxmot.reid.datasets.base import ReIDSample
from boxmot.reid.datasets.torch_dataset import ReIDImageDataset
from boxmot.reid.datasets.transforms import IdentityPreservingBackgroundMosaic


def _write_sample(
    image_root: Path,
    primary_mask_root: Path,
    donor_mask_root: Path,
    *,
    name: str,
    pid: int,
    background: tuple[int, int, int],
    secondary_people: bool = False,
) -> ReIDSample:
    relative_path = Path("bounding_box_train") / name
    image_path = image_root / relative_path
    primary_mask_path = (primary_mask_root / relative_path).with_suffix(".png")
    donor_mask_path = (donor_mask_root / relative_path).with_suffix(".png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    primary_mask_path.parent.mkdir(parents=True, exist_ok=True)
    donor_mask_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.full((16, 8, 3), background, dtype=np.uint8)
    image[2:14, 2:6] = (240, 0, 240)
    primary_mask = np.zeros((16, 8), dtype=np.uint8)
    primary_mask[2:14, 2:6] = 255
    donor_mask = primary_mask.copy()
    if secondary_people:
        for y1, y2 in ((0, 2), (14, 16)):
            image[y1:y2, 0:2] = (250, 250, 0)
            image[y1:y2, 6:8] = (250, 250, 0)
            donor_mask[y1:y2, 0:2] = 255
            donor_mask[y1:y2, 6:8] = 255
    Image.fromarray(image).save(image_path)
    Image.fromarray(primary_mask).save(primary_mask_path)
    Image.fromarray(donor_mask).save(donor_mask_path)
    return ReIDSample(str(image_path), pid=pid, camid=0)


def _fixture_samples(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[ReIDSample]]:
    image_root = tmp_path / "Market-1501"
    mask_root = tmp_path / "Market-1501-mosaic-masks"
    primary_mask_root = mask_root / "primary"
    donor_mask_root = mask_root / "all_people"
    colors = (
        (5, 10, 15),
        (20, 40, 60),
        (30, 60, 90),
        (40, 80, 120),
        (50, 100, 150),
    )
    samples = [
        _write_sample(
            image_root,
            primary_mask_root,
            donor_mask_root,
            name=f"{pid:04d}_c1s1_000001_00.png",
            pid=pid,
            background=color,
        )
        for pid, color in enumerate(colors)
    ]
    return image_root, primary_mask_root, donor_mask_root, samples


def test_background_mosaic_preserves_anchor_foreground_and_pid(tmp_path):
    image_root, primary_mask_root, donor_mask_root, samples = _fixture_samples(tmp_path)
    mosaic = IdentityPreservingBackgroundMosaic(
        samples,
        image_root=image_root,
        primary_mask_root=primary_mask_root,
        donor_mask_root=donor_mask_root,
        probability=1.0,
        start_epoch=0,
        ramp_end_epoch=0,
        min_foreground_ratio=0.1,
        max_foreground_ratio=0.9,
        feather=0,
        dilation=0,
        max_donor_tile_foreground=0.5,
    )
    mosaic.set_epoch(1)
    dataset = ReIDImageDataset(samples, sample_transform=mosaic)

    random.seed(7)
    augmented, pid, camid = dataset[0]
    output = np.asarray(augmented)
    anchor = np.asarray(Image.open(samples[0].img_path).convert("RGB"))
    foreground = np.zeros((16, 8), dtype=bool)
    foreground[2:14, 2:6] = True

    np.testing.assert_array_equal(output[foreground], anchor[foreground])
    assert np.any(output[~foreground] != anchor[~foreground])
    assert not np.any(np.all(output[~foreground] == (240, 0, 240), axis=1))
    assert (pid, camid) == (samples[0].pid, samples[0].camid)


def test_background_mosaic_probability_ramps_after_start_epoch(tmp_path):
    image_root, primary_mask_root, donor_mask_root, samples = _fixture_samples(tmp_path)
    mosaic = IdentityPreservingBackgroundMosaic(
        samples,
        image_root=image_root,
        primary_mask_root=primary_mask_root,
        donor_mask_root=donor_mask_root,
        probability=0.3,
        start_epoch=10,
        ramp_end_epoch=30,
    )

    mosaic.set_epoch(10)
    assert mosaic.effective_probability() == 0
    mosaic.set_epoch(20)
    assert mosaic.effective_probability() == 0.15
    mosaic.set_epoch(30)
    assert mosaic.effective_probability() == 0.3


def test_background_mosaic_rejects_tiny_anchor_mask(tmp_path):
    image_root, primary_mask_root, donor_mask_root, samples = _fixture_samples(tmp_path)
    anchor_mask = (
        primary_mask_root / "bounding_box_train" / Path(samples[0].img_path).name
    ).with_suffix(".png")
    tiny_mask = np.zeros((16, 8), dtype=np.uint8)
    tiny_mask[8, 4] = 255
    Image.fromarray(tiny_mask).save(anchor_mask)
    mosaic = IdentityPreservingBackgroundMosaic(
        samples,
        image_root=image_root,
        primary_mask_root=primary_mask_root,
        donor_mask_root=donor_mask_root,
        probability=1.0,
        start_epoch=0,
        ramp_end_epoch=0,
        feather=0,
        dilation=0,
    )
    mosaic.set_epoch(1)
    anchor = Image.open(samples[0].img_path).convert("RGB")

    random.seed(7)
    augmented = mosaic(anchor, 0)

    np.testing.assert_array_equal(np.asarray(augmented), np.asarray(anchor))


def test_background_mosaic_uses_all_people_mask_for_donor_cleanup(tmp_path):
    image_root, primary_mask_root, donor_mask_root, samples = _fixture_samples(tmp_path)
    samples[1] = _write_sample(
        image_root,
        primary_mask_root,
        donor_mask_root,
        name=Path(samples[1].img_path).name,
        pid=samples[1].pid,
        background=(20, 40, 60),
        secondary_people=True,
    )
    mosaic = IdentityPreservingBackgroundMosaic(
        samples,
        image_root=image_root,
        primary_mask_root=primary_mask_root,
        donor_mask_root=donor_mask_root,
        probability=1.0,
        start_epoch=0,
        ramp_end_epoch=0,
        max_donor_tile_foreground=0.5,
    )

    random.seed(3)
    tile = mosaic._background_tile(samples[1], target_size=(8, 16))

    assert tile is not None
    output = np.asarray(tile)
    assert not np.any(np.all(output == (250, 250, 0), axis=2))


def test_context_mosaic_can_add_real_person_boundary_occluder(tmp_path):
    image_root, primary_mask_root, donor_mask_root, samples = _fixture_samples(tmp_path)
    for sample in samples[1:]:
        path = Path(sample.img_path)
        donor = np.asarray(Image.open(path).convert("RGB")).copy()
        donor[2:14, 2:6] = (250, 25, 25)
        Image.fromarray(donor).save(path)
    mosaic = IdentityPreservingBackgroundMosaic(
        samples,
        image_root=image_root,
        primary_mask_root=primary_mask_root,
        donor_mask_root=donor_mask_root,
        probability=0.0,
        start_epoch=0,
        ramp_end_epoch=0,
        occluder_probability=1.0,
        occluder_min_area=0.2,
        occluder_max_area=0.2,
    )
    mosaic.set_epoch(1)
    anchor = Image.open(samples[0].img_path).convert("RGB")

    random.seed(4)
    output, applied = mosaic.apply_with_status(anchor, 0)
    pixels = np.asarray(output)

    assert applied is True
    assert np.any(np.all(pixels == (250, 25, 25), axis=2))
