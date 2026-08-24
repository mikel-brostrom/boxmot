"""Tests for Market-1501 dataset-root resolution."""

from pathlib import Path

from boxmot.reid.datasets.market1501 import Market1501


def _write_market_splits(root: Path) -> None:
    for split in ("bounding_box_train", "bounding_box_test", "query"):
        directory = root / split
        directory.mkdir(parents=True)
        (directory / "0001_c1s1_000001_00.jpg").touch()


def test_market1501_accepts_explicit_root_with_custom_name(tmp_path):
    original = tmp_path / "Market-1501-v15.09.15"
    segmented = tmp_path / "Market-1501-gray"
    _write_market_splits(original)
    _write_market_splits(segmented)

    dataset = Market1501(str(segmented))

    assert dataset.root == segmented
    assert all(Path(sample.img_path).is_relative_to(segmented) for sample in dataset.train.samples)


def test_market1501_accepts_parent_of_canonical_root(tmp_path):
    canonical = tmp_path / "datasets" / "Market-1501-v15.09.15"
    _write_market_splits(canonical)

    dataset = Market1501(str(tmp_path / "datasets"))

    assert dataset.root == canonical
