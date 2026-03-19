from pathlib import Path

from boxmot.utils.download import _sync_trackeval_dataset_overlays


def test_sync_trackeval_dataset_overlays_copies_all_mmot_files(tmp_path):
    dest = tmp_path / "trackeval_root"
    _sync_trackeval_dataset_overlays(dest)

    utils_dir = Path("boxmot/utils/evaluation")
    expected = {
        dest / "trackeval" / "datasets" / "mmot_rgb.py": utils_dir / "custom_mot_challenge_obb.py",
        dest / "trackeval" / "datasets" / "__init__.py": utils_dir / "trackeval_datasets_init.py",
    }

    for target, source in expected.items():
        assert target.exists()
        assert target.read_text() == source.read_text()

    assert not (dest / "trackeval" / "datasets" / "mmot_8ch.py").exists()


def test_sync_trackeval_dataset_overlays_patches_numpy_builtin_aliases(tmp_path):
    dest = tmp_path / "trackeval_root"
    dataset_file = dest / "trackeval" / "datasets" / "mot_challenge_2d_box.py"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text(
        "import numpy as np\n"
        "def load(read_data):\n"
        "    return np.asarray(read_data, dtype=np.float), np.asarray(read_data, dtype=np.int)\n"
    )

    _sync_trackeval_dataset_overlays(dest)

    patched = dataset_file.read_text()
    assert "np.float" not in patched
    assert "np.int" not in patched
    assert "dtype=float" in patched
    assert "dtype=int" in patched
