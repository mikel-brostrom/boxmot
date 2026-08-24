import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from boxmot.reid.datasets.anatomical import PoseAnatomicalTargetProvider
from boxmot.reid.datasets.base import ReIDSample
from boxmot.reid.datasets.pav_mosaic import PoseAlignedViewMosaic
from boxmot.reid.datasets.torch_dataset import ReIDImageDataset


def _keypoints() -> list[list[float]]:
    coordinates = (
        (0.50, 0.12),
        (0.45, 0.11),
        (0.55, 0.11),
        (0.40, 0.13),
        (0.60, 0.13),
        (0.35, 0.30),
        (0.65, 0.30),
        (0.27, 0.45),
        (0.73, 0.45),
        (0.20, 0.60),
        (0.80, 0.60),
        (0.40, 0.55),
        (0.60, 0.55),
        (0.40, 0.75),
        (0.60, 0.75),
        (0.40, 0.95),
        (0.60, 0.95),
    )
    return [[x, y, 0.99] for x, y in coordinates]


def _make_fixture(tmp_path):
    image_root = tmp_path / "Market-1501"
    train_root = image_root / "bounding_box_train"
    train_root.mkdir(parents=True)
    colors = ((0, 0, 0), (220, 20, 20), (20, 220, 20))
    samples = []
    records = {}
    for index, color in enumerate(colors):
        path = train_root / f"{index:04d}_c{index + 1}s1_000001_00.jpg"
        Image.new("RGB", (40, 80), color).save(path)
        samples.append(
            ReIDSample(
                img_path=str(path),
                pid=0 if index < 2 else 1,
                camid=index,
            )
        )
        relative = path.relative_to(image_root)
        mask_path = (tmp_path / "metadata" / "person" / relative).with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((80, 40), 255, dtype=np.uint8)).save(mask_path)
        records[relative.as_posix()] = {
            "keypoints": _keypoints(),
            "person_mask": (Path("person") / relative).with_suffix(".png").as_posix(),
        }
    metadata_root = tmp_path / "metadata"
    (metadata_root / "metadata.json").write_text(
        json.dumps({"version": 1, "images": records}),
        encoding="utf-8",
    )
    return image_root, metadata_root, samples


def test_pav_mosaic_uses_only_same_identity_donors(tmp_path):
    image_root, metadata_root, samples = _make_fixture(tmp_path)
    transform = PoseAlignedViewMosaic(
        samples,
        image_root=image_root,
        metadata_root=metadata_root,
        probability=1.0,
        max_parts=1,
        max_foreground_replacement=1.0,
        cross_camera_rate=1.0,
        different_pose_rate=0.0,
        warmup_epochs=0,
        decay_start_epoch=200,
        decay_end_epoch=200,
        feather=0.0,
    )
    transform.set_epoch(1)
    random.seed(3)

    anchor = Image.open(samples[0].img_path).convert("RGB")
    output, applied = transform.apply_with_status(anchor, 0)
    pixels = np.asarray(output)

    assert applied is True
    assert np.any(pixels[..., 0] > 100)
    assert not np.any(pixels[..., 1] > 100)


def test_pav_probability_warms_up_and_decays(tmp_path):
    image_root, metadata_root, samples = _make_fixture(tmp_path)
    transform = PoseAlignedViewMosaic(
        samples,
        image_root=image_root,
        metadata_root=metadata_root,
        probability=0.25,
        warmup_epochs=40,
        decay_start_epoch=170,
        decay_end_epoch=200,
        final_probability_scale=0.5,
    )

    transform.set_epoch(20)
    assert transform.effective_probability() == 0.125
    transform.set_epoch(170)
    assert transform.effective_probability() == 0.25
    transform.set_epoch(200)
    assert transform.effective_probability() == 0.125


def test_reid_dataset_can_return_clean_and_augmented_views(tmp_path):
    image_root, metadata_root, samples = _make_fixture(tmp_path)
    transform = PoseAlignedViewMosaic(
        samples,
        image_root=image_root,
        metadata_root=metadata_root,
        probability=1.0,
        max_parts=1,
        max_foreground_replacement=1.0,
        warmup_epochs=0,
        decay_start_epoch=200,
        decay_end_epoch=200,
        feather=0.0,
    )
    transform.set_epoch(1)
    dataset = ReIDImageDataset(
        samples,
        transform=lambda image: torch.from_numpy(np.asarray(image).copy()),
        sample_transform=transform,
        return_clean_view=True,
        clean_transform=lambda image: torch.from_numpy(np.asarray(image).copy()),
    )
    random.seed(5)

    augmented, pid, camid, clean, applied = dataset[0]

    assert (pid, camid, applied) == (0, 0, True)
    assert not torch.equal(augmented, clean)


def test_pav_feathering_stays_inside_replacement_mask(tmp_path, monkeypatch):
    image_root, metadata_root, samples = _make_fixture(tmp_path)
    transform = PoseAlignedViewMosaic(
        samples,
        image_root=image_root,
        metadata_root=metadata_root,
        probability=1.0,
        max_parts=1,
        max_foreground_replacement=0.1,
        warmup_epochs=0,
        decay_start_epoch=200,
        decay_end_epoch=200,
        feather=0.8,
    )
    transform.set_epoch(1)
    part_mask = np.zeros((80, 40), dtype=bool)
    part_mask[30:40, 10:30] = True
    warped = np.full((80, 40, 3), (220, 20, 20), dtype=np.uint8)
    monkeypatch.setattr(
        transform,
        "_part_layer",
        lambda *args, **kwargs: (warped, part_mask.copy()),
    )
    random.seed(9)

    anchor = Image.open(samples[0].img_path).convert("RGB")
    output, applied = transform.apply_with_status(anchor, 0)
    changed = np.any(np.asarray(output) != np.asarray(anchor), axis=2)

    assert applied is True
    assert changed.any()
    assert not changed[~part_mask].any()
    assert changed.sum() / (80 * 40) <= 0.1


def test_pav_does_not_report_layer_clipped_to_empty_as_applied(
    tmp_path,
    monkeypatch,
):
    image_root, metadata_root, samples = _make_fixture(tmp_path)
    transform = PoseAlignedViewMosaic(
        samples,
        image_root=image_root,
        metadata_root=metadata_root,
        probability=1.0,
        max_parts=1,
        max_foreground_replacement=1.0,
        warmup_epochs=0,
        decay_start_epoch=200,
        decay_end_epoch=200,
        feather=0.8,
    )
    transform.set_epoch(1)
    anchor_record = transform._record(0)
    anchor_mask_path = metadata_root / anchor_record["person_mask"]
    anchor_mask = np.zeros((80, 40), dtype=np.uint8)
    anchor_mask[25:65, 15:25] = 255
    Image.fromarray(anchor_mask).save(anchor_mask_path)
    part_mask = np.zeros((80, 40), dtype=bool)
    part_mask[:5, :5] = True
    warped = np.full((80, 40, 3), 220, dtype=np.uint8)
    monkeypatch.setattr(
        transform,
        "_part_layer",
        lambda *args, **kwargs: (warped, part_mask.copy()),
    )
    random.seed(11)

    anchor = Image.open(samples[0].img_path).convert("RGB")
    output, applied = transform.apply_with_status(anchor, 0)

    assert applied is False
    np.testing.assert_array_equal(np.asarray(output), np.asarray(anchor))


def test_pav_cross_camera_rate_controls_candidate_pool(tmp_path):
    image_root, metadata_root, samples = _make_fixture(tmp_path)
    samples[1].camid = samples[0].camid
    samples[2].pid = samples[0].pid
    transform = PoseAlignedViewMosaic(
        samples,
        image_root=image_root,
        metadata_root=metadata_root,
        probability=1.0,
        cross_camera_rate=0.0,
        different_pose_rate=0.0,
        warmup_epochs=0,
        decay_start_epoch=200,
        decay_end_epoch=200,
    )

    assert transform._select_donor(0, "torso", set()) == 1
    transform.cross_camera_rate = 1.0
    assert transform._select_donor(0, "torso", set()) == 2


def test_pav_and_anatomical_provider_share_manifest_records(tmp_path):
    image_root, metadata_root, samples = _make_fixture(tmp_path)
    transform = PoseAlignedViewMosaic(
        samples,
        image_root=image_root,
        metadata_root=metadata_root,
    )
    provider = PoseAnatomicalTargetProvider(
        samples,
        image_root=image_root,
        metadata_root=metadata_root,
    )

    assert transform.records is provider.records
