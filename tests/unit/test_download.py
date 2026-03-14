from pathlib import Path

from boxmot.utils.download import _sync_trackeval_dataset_overlays


def test_sync_trackeval_dataset_overlays_copies_all_mmot_files(tmp_path):
    dest = tmp_path / "trackeval_root"
    _sync_trackeval_dataset_overlays(dest)

    utils_dir = Path("boxmot/utils")
    expected = {
        dest / "trackeval" / "datasets" / "mmot_rgb.py": utils_dir / "custom_mot_challenge_obb.py",
        dest / "trackeval" / "datasets" / "mmot_8ch.py": utils_dir / "custom_mot_challenge_obb.py",
        dest / "trackeval" / "datasets" / "__init__.py": utils_dir / "trackeval_datasets_init.py",
    }

    for target, source in expected.items():
        assert target.exists()
        assert target.read_text() == source.read_text()
