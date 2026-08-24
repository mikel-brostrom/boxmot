"""Tests for the Market-1501 gray-background conversion helpers."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tools.create_market1501_person_masks import (
    clone_structure_and_non_images,
    discover_images,
    gray_background,
    main,
    resolve_market_root,
    resolve_output_root,
    select_mosaic_person_masks,
    validate_paths,
    verify_clone_layout,
)


def test_select_mosaic_person_masks_separates_primary_from_all_people():
    masks = torch.zeros(3, 10, 10)
    masks[0, 3:7, 3:7] = 1
    masks[1, :5, :5] = 1
    masks[2, 2:8, 2:8] = 1
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            cls=torch.tensor([0, 0, 2]),
            conf=torch.tensor([0.9, 0.9, 1.0]),
            xyxy=torch.tensor(
                [
                    [3.0, 3.0, 7.0, 7.0],
                    [0.0, 0.0, 5.0, 5.0],
                    [2.0, 2.0, 8.0, 8.0],
                ]
            ),
        ),
        masks=SimpleNamespace(data=masks),
    )

    selected = select_mosaic_person_masks(
        result,
        (10, 10),
        person_class=0,
        mask_threshold=0.5,
    )

    assert selected is not None
    primary, all_people = selected
    expected_primary = np.zeros((10, 10), dtype=bool)
    expected_primary[3:7, 3:7] = True
    expected_all_people = expected_primary.copy()
    expected_all_people[:5, :5] = True
    np.testing.assert_array_equal(primary, expected_primary)
    np.testing.assert_array_equal(all_people, expected_all_people)


def test_select_mosaic_person_masks_includes_only_nearby_bags():
    masks = torch.zeros(3, 12, 12)
    masks[0, 3:9, 3:8] = 1
    masks[1, 5:9, 8:10] = 1
    masks[2, :2, :2] = 1
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            cls=torch.tensor([0, 24, 26]),
            conf=torch.tensor([0.9, 0.8, 0.8]),
            xyxy=torch.tensor(
                [
                    [3.0, 3.0, 8.0, 9.0],
                    [8.0, 5.0, 10.0, 9.0],
                    [0.0, 0.0, 2.0, 2.0],
                ]
            ),
        ),
        masks=SimpleNamespace(data=masks),
    )

    selected = select_mosaic_person_masks(
        result,
        (12, 12),
        person_class=0,
        mask_threshold=0.5,
        bag_classes=(24, 26, 28),
        bag_proximity=0.1,
    )

    assert selected is not None
    primary, all_people = selected
    expected = np.zeros((12, 12), dtype=bool)
    expected[3:9, 3:8] = True
    expected[5:9, 8:10] = True
    np.testing.assert_array_equal(primary, expected)
    np.testing.assert_array_equal(all_people, expected)


def test_gray_background_preserves_foreground_and_replaces_background():
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    output, effective_mask = gray_background(image, mask, gray=127)

    np.testing.assert_array_equal(output[1, 1], image[1, 1])
    np.testing.assert_array_equal(output[~mask], np.full((8, 3), 127, dtype=np.uint8))
    np.testing.assert_array_equal(effective_mask, mask)


def test_market_root_resolution_discovery_and_output_safety(tmp_path):
    market = tmp_path / "datasets" / "Market-1501-v15.09.15"
    train = market / "bounding_box_train"
    gallery = market / "bounding_box_test"
    query = market / "query"
    gt_bbox = market / "gt_bbox"
    gt_query = market / "gt_query"
    train.mkdir(parents=True)
    gallery.mkdir()
    query.mkdir()
    gt_bbox.mkdir()
    gt_query.mkdir()
    (train / "0001_c1.jpg").touch()
    (gallery / "-1_c1.jpg").touch()
    (query / "0001_c2.png").touch()
    (gt_bbox / "0002_c1.jpg").touch()
    (market / "README.txt").touch()

    assert resolve_market_root(tmp_path / "datasets") == market
    assert resolve_output_root(tmp_path / "Market-1501-gray") == tmp_path / "Market-1501-gray"
    assert [path.relative_to(market).as_posix() for path in discover_images(market)] == [
        "bounding_box_test/-1_c1.jpg",
        "bounding_box_train/0001_c1.jpg",
        "gt_bbox/0002_c1.jpg",
        "query/0001_c2.png",
    ]
    validate_paths(market, tmp_path / "Market-1501-gray")
    with pytest.raises(ValueError, match="outside the source"):
        validate_paths(market, market / "gray")

    clone = tmp_path / "Market-1501-gray"
    clone.mkdir(parents=True)
    clone_structure_and_non_images(market, clone)
    assert (clone / "README.txt").is_file()
    assert (clone / "gt_query").is_dir()
    for image in discover_images(market):
        destination = clone / image.relative_to(market)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()
    verify_clone_layout(market, clone)


def test_masks_only_writes_only_accepted_train_masks(tmp_path, monkeypatch):
    source = tmp_path / "Market-1501-v15.09.15"
    train = source / "bounding_box_train"
    train.mkdir(parents=True)
    (source / "query").mkdir()
    accepted = train / "0001_c1s1_000001_00.jpg"
    rejected = train / "0002_c1s1_000001_00.jpg"
    Image.fromarray(np.full((16, 8, 3), 64, dtype=np.uint8)).save(accepted)
    Image.fromarray(np.full((16, 8, 3), 96, dtype=np.uint8)).save(rejected)

    person_mask = torch.zeros(2, 16, 8)
    person_mask[0, 2:14, 2:6] = 1
    person_mask[1, 0:4, 0:2] = 1
    accepted_result = SimpleNamespace(
        boxes=SimpleNamespace(
            cls=torch.tensor([0, 0]),
            conf=torch.tensor([0.9, 0.8]),
            xyxy=torch.tensor(
                [
                    [2.0, 2.0, 6.0, 14.0],
                    [0.0, 0.0, 2.0, 4.0],
                ]
            ),
        ),
        masks=SimpleNamespace(data=person_mask),
    )
    rejected_result = SimpleNamespace(boxes=None, masks=None)

    class FakeYOLO:
        def __init__(self, model):
            assert model == "fake-seg.pt"

        def predict(self, **kwargs):
            assert kwargs["conf"] == 0.5
            return [
                accepted_result if Path(path).name == accepted.name else rejected_result
                for path in kwargs["source"]
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    output = tmp_path / "Market-1501-mosaic-highconf"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_market1501_person_masks",
            "--source",
            str(source),
            "--output",
            str(output),
            "--model",
            "fake-seg.pt",
            "--batch-size",
            "2",
            "--conf",
            "0.50",
            "--masks-only",
        ],
    )

    assert main() == 0
    mask_root = tmp_path / "Market-1501-mosaic-highconf-masks"
    assert not output.exists()
    primary_path = (
        mask_root / "primary" / "bounding_box_train" / accepted.with_suffix(".png").name
    )
    all_people_path = (
        mask_root / "all_people" / "bounding_box_train" / accepted.with_suffix(".png").name
    )
    assert primary_path.is_file()
    assert all_people_path.is_file()
    assert not (
        mask_root / "primary" / "bounding_box_train" / rejected.with_suffix(".png").name
    ).exists()
    primary = np.asarray(Image.open(primary_path)) >= 128
    all_people = np.asarray(Image.open(all_people_path)) >= 128
    assert np.any(all_people & ~primary)
    assert np.all(~primary | all_people)

    report = json.loads(
        (tmp_path / "Market-1501-mosaic-highconf-masking-report.json").read_text()
    )
    assert report["masks_only"] is True
    assert report["scope"] == ["bounding_box_train"]
    assert report["mask_layout"]["primary"] == "primary"
    assert report["mask_layout"]["all_people"] == "all_people"
    assert report["settings"]["conf"] == 0.5
    assert report["images_selected"] == 2
    assert report["images_written"] == 1
    assert report["missing_masks"] == [
        "bounding_box_train/0002_c1s1_000001_00.jpg"
    ]
