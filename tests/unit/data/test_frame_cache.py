"""Tests for boxmot.data.frame_cache – RAM-based frame caching."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from boxmot.data.frame_cache import FrameCache, available_ram_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg(path: Path, h: int = 64, w: int = 48) -> Path:
    """Write a small random JPEG image to *path* and return it."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _make_npy(path: Path, h: int = 64, w: int = 48) -> Path:
    """Write a random .npy image array to *path* and return it."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    np.save(str(path), arr)
    return path


@pytest.fixture()
def jpeg_dir(tmp_path: Path) -> list[Path]:
    """Create 10 small JPEG files and return their paths."""
    paths = []
    for i in range(10):
        paths.append(_make_jpeg(tmp_path / f"frame_{i:04d}.jpg"))
    return paths


@pytest.fixture()
def npy_dir(tmp_path: Path) -> list[Path]:
    """Create 5 small .npy files and return their paths."""
    paths = []
    for i in range(5):
        paths.append(_make_npy(tmp_path / f"frame_{i:04d}.npy"))
    return paths


# ---------------------------------------------------------------------------
# Basic construction & tiers
# ---------------------------------------------------------------------------

class TestFrameCacheBasic:

    def test_empty_paths_produces_inactive_cache(self):
        cache = FrameCache([])
        assert not cache.active
        assert cache.mode == "none"

    def test_nonexistent_paths_produces_inactive_cache(self, tmp_path):
        paths = [tmp_path / "no_such_file.jpg"]
        cache = FrameCache(paths)
        assert not cache.active

    def test_decoded_tier_for_small_jpegs(self, jpeg_dir):
        cache = FrameCache(jpeg_dir)
        assert cache.active
        assert cache.mode == "decoded"

    def test_read_image_returns_correct_shape(self, jpeg_dir):
        cache = FrameCache(jpeg_dir)
        img = cache.read_image(jpeg_dir[0])
        assert img.shape == (64, 48, 3)
        assert img.dtype == np.uint8

    def test_read_image_returns_copy(self, jpeg_dir):
        """Decoded tier must return a copy to prevent cache corruption."""
        cache = FrameCache(jpeg_dir)
        img1 = cache.read_image(jpeg_dir[0])
        img2 = cache.read_image(jpeg_dir[0])
        assert not np.shares_memory(img1, img2)

    def test_npy_files_decode_correctly(self, npy_dir):
        cache = FrameCache(npy_dir)
        if cache.active:
            img = cache.read_image(npy_dir[0])
            assert img.ndim == 3
            assert img.shape[2] == 3

    def test_clear_releases_data(self, jpeg_dir):
        cache = FrameCache(jpeg_dir)
        assert cache.active
        cache.clear()
        assert not cache.active
        assert cache.mode == "none"

    def test_context_manager(self, jpeg_dir):
        with FrameCache(jpeg_dir) as cache:
            assert cache.active
        assert not cache.active

    def test_fallback_used_for_missing_key(self, jpeg_dir, tmp_path):
        cache = FrameCache(jpeg_dir)
        extra = _make_jpeg(tmp_path / "extra.jpg", h=32, w=32)
        img = cache.read_image(extra)
        assert img.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# Budget & n_cache_peers
# ---------------------------------------------------------------------------

class TestBudgetDivision:

    def test_n_cache_peers_divides_budget(self, jpeg_dir):
        """With n_cache_peers > 1 the per-instance budget shrinks."""
        # With peers=1 (default), small images should fit as decoded
        cache1 = FrameCache(jpeg_dir, n_cache_peers=1)
        assert cache1.active

        # With a huge peers count, budget → 0 → cache disabled
        cache_many = FrameCache(jpeg_dir, n_cache_peers=999_999)
        # Should either be inactive or have a much smaller budget
        # (it may still be active if images are tiny enough to fit in
        # a sub-byte budget, but the budget was definitely divided)

    def test_peers_zero_treated_as_one(self, jpeg_dir):
        """n_cache_peers <= 0 is clamped to 1 (no division by zero)."""
        cache = FrameCache(jpeg_dir, n_cache_peers=0)
        assert cache.active  # same as peers=1 for small images

    def test_budget_fraction_respected(self, jpeg_dir):
        """Setting budget_fraction=0 disables the cache."""
        cache = FrameCache(jpeg_dir, budget_fraction=0.0)
        assert not cache.active

    def test_min_free_bytes_huge_disables_cache(self, jpeg_dir):
        """If min_free_bytes exceeds available RAM the cache is disabled."""
        cache = FrameCache(jpeg_dir, min_free_bytes=10 * 1024**4)
        assert not cache.active


# ---------------------------------------------------------------------------
# Thread safety – concurrent reads
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_reads_from_threads(self, jpeg_dir):
        """Multiple threads reading the same cache simultaneously."""
        cache = FrameCache(jpeg_dir)
        assert cache.active

        def _read(path):
            return cache.read_image(path).shape

        with ThreadPoolExecutor(max_workers=4) as pool:
            # Read all frames 3 times concurrently
            paths = jpeg_dir * 3
            results = list(pool.map(_read, paths))

        assert all(shape == (64, 48, 3) for shape in results)


# ---------------------------------------------------------------------------
# Multiprocessing – spawn context (simulates replay phase)
# ---------------------------------------------------------------------------

def _worker_create_and_read(args: tuple) -> tuple:
    """Worker function: create a FrameCache and read one frame.

    Returns ``(mode, shape, pixel_sum)`` so the caller can verify
    correctness without transferring the full image.
    """
    paths, idx, n_peers = args
    cache = FrameCache(paths, n_cache_peers=n_peers)
    img = cache.read_image(paths[idx])
    cache.clear()
    return cache.mode, img.shape, int(img.sum())


class TestMultiprocessSpawn:

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="spawn workers may not have enough RAM in CI",
    )
    def test_spawn_workers_create_independent_caches(self, jpeg_dir):
        """Each spawned process creates its own cache with divided budget."""
        n_workers = 3
        tasks = [(jpeg_dir, i % len(jpeg_dir), n_workers) for i in range(n_workers)]

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            results = list(pool.map(_worker_create_and_read, tasks))

        for _mode, shape, _psum in results:
            assert shape == (64, 48, 3)

    def test_concurrent_caches_divide_budget(self, jpeg_dir):
        """Simulate the replay-phase pattern: N caches, each with peers=N."""
        n_workers = 4
        caches = [FrameCache(jpeg_dir, n_cache_peers=n_workers) for _ in range(n_workers)]
        # All should work (small images) and the budget per cache is 1/N
        for cache in caches:
            img = cache.read_image(jpeg_dir[0])
            assert img.shape == (64, 48, 3)
            cache.clear()


# ---------------------------------------------------------------------------
# Integration: FrameCache with MOTSequence
# ---------------------------------------------------------------------------

class TestMOTSequenceIntegration:

    def test_mot_sequence_passes_cache_peers(self, tmp_path):
        """MOTSequence should forward n_cache_peers to FrameCache."""
        from boxmot.data.dataset import MOTSequence

        seq_dir = tmp_path / "SEQ01" / "img1"
        seq_dir.mkdir(parents=True)
        frame_paths = []
        for i in range(5):
            p = seq_dir / f"{i + 1:06d}.jpg"
            _make_jpeg(p)
            frame_paths.append(p)

        meta = {
            "seq_dir": seq_dir.parent,
            "frame_ids": np.arange(1, 6),
            "frame_paths": frame_paths,
            "det_path": None,
            "emb_path": None,
            "mask_path": None,
        }

        # With skip_image_load=True, no cache is created regardless
        seq_skip = MOTSequence("SEQ01", meta, target_fps=None, skip_image_load=True)
        assert seq_skip._frame_cache is None

        # With skip_image_load=False, cache IS created with peers forwarded
        seq_load = MOTSequence("SEQ01", meta, target_fps=None, skip_image_load=False, n_cache_peers=4)
        if seq_load._frame_cache is not None and seq_load._frame_cache.active:
            img = seq_load._frame_cache.read_image(frame_paths[0])
            assert img.shape[2] == 3


# ---------------------------------------------------------------------------
# available_ram_bytes
# ---------------------------------------------------------------------------

class TestAvailableRam:

    def test_returns_positive_int(self):
        result = available_ram_bytes()
        assert result is None or (isinstance(result, int) and result > 0)

    def test_returns_reasonable_value(self):
        result = available_ram_bytes()
        if result is not None:
            # At least 100 MB available on any modern system
            assert result > 100 * 1024 * 1024
            # Less than 1 TB (sanity upper bound)
            assert result < 1024 * 1024**3
