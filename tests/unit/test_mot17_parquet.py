from __future__ import annotations

import boxmot.data.mot17_parquet as mot17_parquet


class _SeqInfo:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def iterrows(self):
        yield from enumerate(self._rows)


def test_mot17_split_marker_requires_readable_images(tmp_path):
    split_dir = tmp_path / "ablation"
    seq_dir = split_dir / "MOT17-02-FRCNN"
    img_dir = seq_dir / "img1"
    target_dir = tmp_path / "images" / "train" / "MOT17-02" / "img1"
    img_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nseqLength=2\n")
    (target_dir / "000001.jpg").write_bytes(b"image")
    (img_dir / "000001.jpg").symlink_to((target_dir / "000001.jpg").resolve())
    (img_dir / "000002.jpg").symlink_to((target_dir / "000002.jpg").resolve())

    assert mot17_parquet._split_layout_complete(split_dir) is False

    (target_dir / "000002.jpg").write_bytes(b"image")

    assert mot17_parquet._split_layout_complete(split_dir) is True


def test_mot17_image_download_refreshes_stale_marker(monkeypatch, tmp_path):
    seq_info = _SeqInfo([{"sequence": "MOT17-02", "seq_length": 2}])
    img_dir = tmp_path / "images" / "train" / "MOT17-02" / "img1"
    img_dir.mkdir(parents=True)
    marker = tmp_path / "images" / "train" / ".hf_download_complete"
    marker.touch()
    calls = []

    def fake_snapshot_download(repo_id, subfolder, dest_root, *, status_fn=None, description=None):
        calls.append((repo_id, subfolder, dest_root, status_fn, description))
        (img_dir / "000001.jpg").write_bytes(b"image")
        (img_dir / "000002.jpg").write_bytes(b"image")

    monkeypatch.setattr(mot17_parquet, "snapshot_download_hf_subfolder", fake_snapshot_download)

    mot17_parquet._download_images(tmp_path, "train", seq_info, "train")

    assert calls == [
        (mot17_parquet.PARQUET_REPO, "images/train", tmp_path, None, "Downloading MOT17 images (train)...")
    ]
    assert marker.exists()


def test_mot17_image_download_trusts_marker_when_required_images_exist(monkeypatch, tmp_path):
    seq_info = _SeqInfo([{"sequence": "MOT17-02", "seq_length": 2}])
    img_dir = tmp_path / "images" / "train" / "MOT17-02" / "img1"
    img_dir.mkdir(parents=True)
    (img_dir / "000001.jpg").write_bytes(b"image")
    (img_dir / "000002.jpg").write_bytes(b"image")
    (tmp_path / "images" / "train" / ".hf_download_complete").touch()

    def fail_snapshot_download(*_args, **_kwargs):
        raise AssertionError("complete image cache should not be downloaded")

    monkeypatch.setattr(mot17_parquet, "snapshot_download_hf_subfolder", fail_snapshot_download)

    mot17_parquet._download_images(tmp_path, "train", seq_info, "train")
