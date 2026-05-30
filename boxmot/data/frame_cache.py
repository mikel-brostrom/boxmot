"""RAM-based frame cache for faster image I/O during evaluation.

The cache operates in two tiers, chosen automatically based on available RAM:

1. **Pre-decoded** (best): images are read *and* decoded in parallel during
   preload.  :meth:`~FrameCache.read_image` returns a ``numpy`` copy from a
   dict lookup — near-zero latency.
2. **Raw bytes** (good): only the raw file bytes are held in RAM.
   ``read_image()`` still runs ``cv2.imdecode`` on each call but avoids disk
   I/O entirely.
3. **No cache** (baseline): direct disk reads via the fallback function.
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from boxmot.utils import logger as LOGGER


def available_ram_bytes() -> int | None:
    """Best-effort estimate of available system RAM in bytes.

    Tries ``psutil`` first (most accurate and cross-platform).  Falls back to
    ``/proc/meminfo`` on Linux, ``vm_stat`` on macOS, and finally to half the
    total physical RAM via :func:`os.sysconf`.
    """
    # psutil (optional dependency – most accurate)
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass

    # Linux: /proc/meminfo
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass

    # macOS: vm_stat
    try:
        import subprocess

        output = subprocess.check_output(  # noqa: S603,S607
            ["vm_stat"], text=True, timeout=5,
        )
        lines = output.strip().split("\n")
        page_size = int(lines[0].split("page size of ")[1].split(" ")[0])
        stats: dict[str, int] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            try:
                stats[key.strip()] = int(val.strip().rstrip("."))
            except ValueError:
                pass
        free = stats.get("Pages free", 0)
        inactive = stats.get("Pages inactive", 0)
        return (free + inactive) * page_size
    except Exception:  # noqa: BLE001
        pass

    # Fallback: half of total physical RAM
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // 2
    except (ValueError, OSError, AttributeError):
        pass

    return None


_COMPRESSED_SUFFIXES = frozenset((".jpg", ".jpeg", ".png", ".webp"))

#: Default headroom reserved for the OS, Python working set, GPU driver,
#: model weights, and batch tensors.  2 GiB is enough for a typical
#: evaluation pipeline (YOLO + ReID + tracker on CPU/GPU).
_DEFAULT_MIN_FREE_BYTES = 2 * 1024**3  # 2 GiB


class FrameCache:
    """Two-tier RAM cache: pre-decoded arrays or raw file bytes.

    The cache automatically picks the most aggressive tier that fits in
    the RAM budget.  The budget is the *lesser* of:

    * ``available × budget_fraction``
    * ``available − min_free_bytes``

    This ensures maximum utilisation while always leaving a safe headroom
    for the OS, Python working set, GPU driver, and model weights.

    Parameters
    ----------
    frame_paths:
        Paths to the image files to cache.
    budget_fraction:
        Upper bound on the fraction of *currently available* RAM the
        cache may occupy.
    min_free_bytes:
        Absolute minimum RAM (bytes) that must remain free after the
        cache is populated.  Defaults to 2 GiB.
    n_cache_peers:
        Number of ``FrameCache`` instances expected to coexist at the
        same time (e.g. concurrent worker processes).  The per-instance
        budget is divided by this value so the total stays within the
        system budget.  Defaults to 1 (no sharing).
    n_threads:
        Threads for parallel I/O and decoding during preload.
        Defaults to ``min(os.cpu_count(), 12)``.
    fallback:
        ``(Path) -> ndarray`` called when a path is not cached or
        decoding fails.  Defaults to an internal disk reader.
    """

    def __init__(
        self,
        frame_paths: list[Path],
        *,
        budget_fraction: float = 0.8,
        min_free_bytes: int = _DEFAULT_MIN_FREE_BYTES,
        n_cache_peers: int = 1,
        n_threads: int | None = None,
        fallback: Optional[Callable[[Path], np.ndarray]] = None,
    ):
        self._decoded: dict[str, np.ndarray] = {}
        self._raw: dict[str, bytes] = {}
        self._mode: str = "none"  # "none" | "decoded" | "raw"
        self._fallback = fallback or self._read_from_disk
        self._total_bytes = 0

        if not frame_paths:
            return

        if n_threads is None:
            n_threads = min(os.cpu_count() or 4, 12)

        # --- compute total raw size on disk ---
        total_raw = 0
        valid: list[Path] = []
        for p in frame_paths:
            try:
                total_raw += p.stat().st_size
                valid.append(p)
            except OSError:
                pass

        if not valid:
            return

        available = available_ram_bytes()
        if available is None:
            return

        # Budget = min(fraction-based cap, headroom-based cap), ≥ 0.
        budget = max(0, min(available * budget_fraction,
                            available - min_free_bytes))
        # Divide among concurrent peers so total stays within bounds.
        budget = budget // max(1, n_cache_peers)

        if budget <= 0:
            LOGGER.info(
                f"Frame cache disabled: {available / 1e9:.1f} GB available, "
                f"need \u2265{min_free_bytes / 1e9:.1f} GB headroom"
            )
            return

        # --- Tier 1: pre-decode (parallel read + decode → dict lookup) ---
        # Estimate decoded size: JPEG ~15:1 compression, .npy ≈1:1.
        has_compressed = any(p.suffix.lower() in _COMPRESSED_SUFFIXES for p in valid)
        est_decoded = total_raw * (15 if has_compressed else 2)

        if est_decoded < budget:
            self._preload_decoded(valid, n_threads)
            if self._mode == "decoded":
                return

        # --- Tier 2: raw bytes (parallel read → cv2.imdecode on access) ---
        if total_raw < budget:
            self._preload_raw(valid, n_threads)
            return

        LOGGER.info(
            f"Frame cache disabled: ~{est_decoded / 1e9:.1f} GB decoded / "
            f"{total_raw / 1e9:.1f} GB raw, budget {budget / 1e9:.1f} GB"
        )

    # ------------------------------------------------------------------
    # preload
    # ------------------------------------------------------------------

    def _preload_decoded(self, paths: list[Path], n_threads: int) -> None:
        """Read and JPEG-decode all frames in parallel threads.

        ``cv2.imread`` / ``cv2.imdecode`` release the GIL, so multiple
        threads achieve true parallelism for the CPU-bound decode work.
        """

        def _work(p: Path) -> tuple[str, np.ndarray | None]:
            try:
                return str(p), self._decode_from_file(p)
            except Exception:  # noqa: BLE001
                return str(p), None

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for key, img in pool.map(_work, paths):
                if img is not None:
                    self._decoded[key] = img

        if not self._decoded:
            return

        self._mode = "decoded"
        self._total_bytes = sum(a.nbytes for a in self._decoded.values())
        LOGGER.info(
            f"Frame cache [decoded]: {len(self._decoded)} frames "
            f"({self._total_bytes / 1e6:.0f} MB) in RAM"
        )

    def _preload_raw(self, paths: list[Path], n_threads: int) -> None:
        """Read raw file bytes in parallel threads."""

        def _work(p: Path) -> tuple[str, bytes]:
            return str(p), p.read_bytes()

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for key, data in pool.map(_work, paths):
                self._raw[key] = data

        self._mode = "raw"
        self._total_bytes = sum(len(v) for v in self._raw.values())
        LOGGER.info(
            f"Frame cache [raw bytes]: {len(self._raw)} frames "
            f"({self._total_bytes / 1e6:.0f} MB) in RAM"
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether the cache is populated and serving from RAM."""
        return self._mode != "none"

    @property
    def mode(self) -> str:
        """Current tier: ``"decoded"``, ``"raw"``, or ``"none"``."""
        return self._mode

    def read_image(self, path: Path) -> np.ndarray:
        """Return a decoded image array via the fastest available source.

        * **decoded** tier — dict lookup + ``ndarray.copy()`` (memcpy).
        * **raw** tier — ``cv2.imdecode`` from cached bytes (no disk I/O).
        * **fallback** — original disk reader.
        """
        key = str(path)

        if self._mode == "decoded":
            img = self._decoded.get(key)
            if img is not None:
                return img.copy()

        if self._mode == "raw":
            raw = self._raw.get(key)
            if raw:
                try:
                    return self._decode_bytes(path, raw)
                except Exception:  # noqa: BLE001
                    pass

        return self._fallback(path)

    # ------------------------------------------------------------------
    # decode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_bytes(path: Path, raw: bytes) -> np.ndarray:
        """Decode from in-memory bytes (raw-bytes tier)."""
        suffix = Path(path).suffix if isinstance(path, str) else path.suffix
        if suffix == ".npy":
            arr = np.load(io.BytesIO(raw))
            if arr.ndim == 3 and arr.shape[2] == 8:
                arr = arr[:, :, [1, 2, 4]]
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            elif arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return arr
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode image: {path}")
        return img

    @staticmethod
    def _decode_from_file(path: Path) -> np.ndarray:
        """Read + decode from disk (used during parallel preload)."""
        if path.suffix == ".npy":
            arr = np.load(str(path))
            if arr.ndim == 3 and arr.shape[2] == 8:
                arr = arr[:, :, [1, 2, 4]]
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            elif arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return arr
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    @staticmethod
    def _read_from_disk(path: Path) -> np.ndarray:
        """Fallback disk reader."""
        return FrameCache._decode_from_file(path)

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Release all cached data."""
        self._decoded.clear()
        self._raw.clear()
        self._mode = "none"
        self._total_bytes = 0

    def __enter__(self) -> "FrameCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.clear()
