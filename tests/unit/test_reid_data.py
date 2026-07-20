import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

from boxmot.engine.reid import data as reid_data


def _write_market_fixture(root: Path) -> None:
    for dirname in ("bounding_box_train", "bounding_box_test", "query"):
        split_dir = root / dirname
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "0001_c1s1_000001_00.jpg").write_bytes(b"jpg")
    (root / "readme.txt").write_text("market fixture\n", encoding="utf-8")


def test_ensure_builtin_reid_dataset_downloads_market1501_archive(monkeypatch, tmp_path):
    source_root = tmp_path / "source" / "Market-1501-v15.09.15"
    _write_market_fixture(source_root)
    (source_root / "bounding_box_train" / "._0001_c1s1_000001_00.jpg").write_bytes(b"resource fork")

    archive_path = tmp_path / "Market-1501-v15.09.15.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_root, arcname="Market-1501-v15.09.15")

    def fake_download_hf_file(repo_id, repo_type, filename, local_dir, description=None):
        assert repo_id == "Lekim89/market1501"
        assert repo_type == "dataset"
        assert filename == "Market-1501-v15.09.15.tar.gz"
        assert description == "Downloading market1501 dataset"
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        dst = local_dir / filename
        shutil.copy2(archive_path, dst)
        return dst

    monkeypatch.setattr(reid_data, "download_hf_file", fake_download_hf_file)

    dataset_root = reid_data.ensure_builtin_reid_dataset("market1501", tmp_path / "cache")

    assert dataset_root == (tmp_path / "cache" / "Market-1501-v15.09.15").resolve()
    assert (dataset_root / "bounding_box_train" / "0001_c1s1_000001_00.jpg").is_file()
    assert not (dataset_root / "bounding_box_train" / "._0001_c1s1_000001_00.jpg").exists()
    assert (dataset_root / "bounding_box_test" / "0001_c1s1_000001_00.jpg").is_file()
    assert (dataset_root / "query" / "0001_c1s1_000001_00.jpg").is_file()


def test_resolve_market1501_train_data_defaults_to_reid_cache(monkeypatch, tmp_path):
    calls = []

    def fake_default_reid_data_root():
        return tmp_path / "boxmot" / "datasets" / "reid"

    def fake_ensure(name, cache_root=None):
        calls.append((name, Path(cache_root)))
        return Path(cache_root) / "Market-1501-v15.09.15"

    monkeypatch.setattr(reid_data, "default_reid_data_root", fake_default_reid_data_root)
    monkeypatch.setattr(reid_data, "ensure_builtin_reid_dataset", fake_ensure)

    args = SimpleNamespace(
        data=(),
        dataset="market1501",
        data_dir=None,
        data_specs=(),
        resume=None,
        train_explicit_keys=(),
    )

    resolved = reid_data.resolve_reid_train_data(args)

    expected_root = (tmp_path / "boxmot" / "datasets" / "reid").resolve()
    assert resolved.data_dir == str(expected_root)
    assert calls == [("market1501", expected_root)]
