from __future__ import annotations

import numpy as np

from boxmot.data.cache import (
    REID_CROP_SCHEMA_VERSION,
    find_existing_reid_cache_file,
    reid_cache_dir_candidates,
    reid_cache_key,
    reid_preprocess_cache_key,
)


def test_reid_cache_key_is_readable_and_separates_python_from_cpp(tmp_path, monkeypatch):
    monkeypatch.delenv("BOXMOT_REID_BACKEND", raising=False)
    model = tmp_path / "lmbn_n_duke.pt"

    python_key = reid_cache_key(model, tracker_backend="python")
    cpp_key = reid_cache_key(model, tracker_backend="cpp")

    assert python_key == "python/lmbn_n_duke-pt-pytorch"
    assert cpp_key == "cpp/lmbn_n_duke-pt-ort"


def test_reid_cache_key_separates_same_named_models_by_content(tmp_path):
    first = tmp_path / "first" / "shared.pt"
    second = tmp_path / "second" / "shared.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first model weights")
    second.write_bytes(b"second model weights")

    first_key = reid_cache_key(first, tracker_backend="python")
    second_key = reid_cache_key(second, tracker_backend="python")

    assert first_key.startswith("python/shared-pt-pytorch-w")
    assert second_key.startswith("python/shared-pt-pytorch-w")
    assert first_key != second_key


def test_reid_preprocess_cache_key_versions_the_crop_contract():
    assert reid_preprocess_cache_key("resize") == f"resize-cropv{REID_CROP_SCHEMA_VERSION}"


def test_reid_cache_candidates_put_canonical_first_and_gate_legacy(tmp_path):
    embeddings_root = tmp_path / "embs"
    model = tmp_path / "lmbn_n_duke.pt"
    canonical = embeddings_root / reid_cache_key(model, tracker_backend="python") / reid_preprocess_cache_key("resize")
    legacy = embeddings_root / "lmbn_n_duke" / "resize"

    canonical_only = reid_cache_dir_candidates(
        embeddings_root,
        model,
        reid_preprocess="resize",
        tracker_backend="python",
    )
    with_legacy = reid_cache_dir_candidates(
        embeddings_root,
        model,
        reid_preprocess="resize",
        tracker_backend="python",
        allow_legacy=True,
    )

    assert canonical_only[0] == canonical
    assert with_legacy[0] == canonical
    assert legacy not in canonical_only
    assert with_legacy[-1] == legacy


def test_find_existing_reid_cache_skips_partial_canonical_for_row_aligned_legacy(tmp_path):
    embeddings_root = tmp_path / "embs"
    model = tmp_path / "lmbn_n_duke.pt"
    sequence_name = "MOT17-02-FRCNN"
    candidates = reid_cache_dir_candidates(
        embeddings_root,
        model,
        reid_preprocess="resize",
        tracker_backend="python",
        allow_legacy=True,
    )
    canonical_file = candidates[0] / f"{sequence_name}.npy"
    legacy_file = candidates[-1] / f"{sequence_name}.npy"
    canonical_file.parent.mkdir(parents=True)
    legacy_file.parent.mkdir(parents=True)
    np.save(canonical_file, np.ones((1, 4), dtype=np.float32))
    np.save(legacy_file, np.ones((2, 4), dtype=np.float32))

    found = find_existing_reid_cache_file(
        embeddings_root,
        model,
        sequence_name,
        reid_preprocess="resize",
        tracker_backend="python",
        expected_rows=2,
        allow_legacy=True,
    )

    assert found == legacy_file
    assert (
        find_existing_reid_cache_file(
            embeddings_root,
            model,
            sequence_name,
            reid_preprocess="resize",
            tracker_backend="python",
            expected_rows=2,
            allow_legacy=False,
        )
        is None
    )


def test_unhashed_flattened_crop_cache_requires_legacy_trust(tmp_path):
    embeddings_root = tmp_path / "embs"
    model = tmp_path / "lmbn_n_duke.pt"
    model.write_bytes(b"current model weights")
    sequence_name = "MOT17-02-FRCNN"
    unhashed_file = embeddings_root / "lmbn_n_duke_pt_pytorch_py_cropv2" / "resize" / f"{sequence_name}.npy"
    unhashed_file.parent.mkdir(parents=True)
    np.save(unhashed_file, np.ones((2, 4), dtype=np.float32))

    untrusted = find_existing_reid_cache_file(
        embeddings_root,
        model,
        sequence_name,
        reid_preprocess="resize",
        tracker_backend="python",
        expected_rows=2,
        allow_legacy=False,
    )
    trusted = find_existing_reid_cache_file(
        embeddings_root,
        model,
        sequence_name,
        reid_preprocess="resize",
        tracker_backend="python",
        expected_rows=2,
        allow_legacy=True,
    )

    assert untrusted is None
    assert trusted == unhashed_file


def test_find_existing_reid_cache_rejects_nonempty_zero_width_embeddings(tmp_path):
    embeddings_root = tmp_path / "embs"
    model = tmp_path / "lmbn_n_duke.pt"
    model.write_bytes(b"model weights")
    sequence_name = "MOT17-02-FRCNN"
    canonical_file = (
        reid_cache_dir_candidates(
            embeddings_root,
            model,
            reid_preprocess="resize",
            tracker_backend="python",
        )[0]
        / f"{sequence_name}.npy"
    )
    canonical_file.parent.mkdir(parents=True)
    np.save(canonical_file, np.empty((2, 0), dtype=np.float32))

    assert (
        find_existing_reid_cache_file(
            embeddings_root,
            model,
            sequence_name,
            reid_preprocess="resize",
            tracker_backend="python",
            expected_rows=2,
        )
        is None
    )
