#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Utility script to download and extract BoxMOT releases and MOT evaluation tools.
"""

import concurrent.futures
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional
from zipfile import BadZipFile, ZipFile

import gdown
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from boxmot.utils import logger as LOGGER
from boxmot.utils.rich.core.ui import print_text
from boxmot.utils.rich.workflow.progress import RichTqdm as tqdm

_download_status_state = threading.local()


def set_download_status_fn(status_fn: Any) -> None:
    """Register a callback that download helpers should use for progress reporting.

    The callback is typically a :class:`WorkflowDetailCallback` whose ``.bar()``
    method routes progress updates into an active Rich workflow panel.
    Pass ``None`` to clear the registration.
    """
    _download_status_state.status_fn = status_fn


def get_download_status_fn() -> Any:
    """Return the currently registered download status callback, if any."""
    return getattr(_download_status_state, "status_fn", None)


def get_http_session(retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    """Create HTTP session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        backoff_factor=backoff_factor
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _has_workflow_bar(status_fn: Any) -> bool:
    """Return True if ``status_fn`` exposes a ``.bar()`` context manager."""
    return status_fn is not None and callable(getattr(status_fn, "bar", None))


def download_file(
    url: str,
    dest: Path,
    chunk_size: int = 8192,
    overwrite: bool = False,
    timeout: int = 10,
    *,
    status_fn: Any = None,
) -> Path:
    """
    Download a file from a URL to a destination path, with progress and logging.
    Returns the path to the downloaded file.

    When ``status_fn`` exposes ``.bar()`` (i.e. a ``WorkflowDetailCallback``),
    the download progress is rendered inside the active workflow panel
    instead of via tqdm — this prevents tqdm's carriage-return updates from
    racing with Rich's repaints.
    """
    if dest.exists() and not overwrite:
        LOGGER.debug(f"Cached: {dest.name}")
        return dest

    if status_fn is None:
        status_fn = get_download_status_fn()

    # Ensure parent dir
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Downloading {dest.name}...")

    if "drive.google.com" in url or "drive.usercontent.google.com" in url:
        gdown.download(
            url=url,
            output=str(dest),
            quiet=False,
            fuzzy=True,
        )
    else:
        session = get_http_session()
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total = int(response.headers.get("Content-Length", 0)) or None

        if _has_workflow_bar(status_fn):
            with open(dest, "wb") as f, status_fn.bar(
                f"Downloading {dest.name}", total, unit="B"
            ) as advance:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        advance(len(chunk))
        else:
            with open(dest, "wb") as f, tqdm(
                total=total or 0,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {dest.name}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        if total is not None:
            written = dest.stat().st_size
            if written < total:
                try:
                    dest.unlink()
                except OSError:
                    pass
                raise IOError(
                    f"Truncated download for {dest.name}: "
                    f"got {written} bytes, expected {total}. The partial file "
                    f"has been removed; re-run the command to retry."
                )

    return dest


def download_files_parallel(
    items: "list[tuple[str, Path]]",
    *,
    overwrite: bool = False,
    max_workers: int | None = None,
) -> "list[Path]":
    """Download multiple files concurrently.

    When a workflow status callback exposing ``parallel_bars`` is registered
    via :func:`set_download_status_fn`, all pending downloads share a single
    Rich panel with one progress bar per file. Otherwise (or when only one
    file actually needs downloading) this falls back to sequential
    :func:`download_file` calls so cached files cost nothing.
    """
    pending: list[tuple[str, Path]] = []
    for url, dest in items:
        if overwrite or not dest.exists():
            pending.append((url, dest))

    if not pending:
        return [dest for _, dest in items]

    status_fn = get_download_status_fn()
    can_parallel = (
        len(pending) > 1
        and status_fn is not None
        and callable(getattr(status_fn, "parallel_bars", None))
    )

    if not can_parallel:
        for url, dest in pending:
            download_file(url, dest, overwrite=overwrite)
        return [dest for _, dest in items]

    descriptions = [f"Downloading {dest.name}" for _, dest in pending]
    workers = max_workers or len(pending)

    with status_fn.parallel_bars(descriptions, unit="B") as task_callbacks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    download_file,
                    url,
                    dest,
                    overwrite=overwrite,
                    status_fn=task_cb,
                )
                for (url, dest), task_cb in zip(pending, task_callbacks)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    return [dest for _, dest in items]


def extract_zip(
    zip_path: Path,
    extract_to: Path,
    overwrite: bool = False,
    *,
    status_fn: Any = None,
) -> None:
    """
    Extract a ZIP archive to a target directory.

    When ``status_fn`` exposes ``.bar()``, the extraction progress is
    rendered inside the active workflow panel instead of via tqdm.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    if status_fn is None:
        status_fn = get_download_status_fn()

    try:
        with ZipFile(zip_path, 'r') as zf:
            members = zf.infolist()
            total_files = len(members)

            if not overwrite:
                already = [member.filename for member in members if (extract_to / member.filename).exists()]
                if len(already) == total_files:
                    LOGGER.debug(f"Cached: {zip_path.name} (extracted)")
                    return

            LOGGER.info(f"Extracting {zip_path.name}...")
            extract_to.mkdir(parents=True, exist_ok=True)

            if _has_workflow_bar(status_fn):
                with status_fn.bar(f"Extracting {zip_path.name}", total_files) as advance:
                    for member in members:
                        target = extract_to / member.filename
                        if not (target.exists() and not overwrite):
                            zf.extract(member, extract_to)
                        advance(1)
            else:
                for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
                    target = extract_to / member.filename
                    if target.exists() and not overwrite:
                        continue
                    zf.extract(member, extract_to)

    except BadZipFile:
        LOGGER.error(f"Corrupt ZIP: {zip_path.name}")
        try:
            zip_path.unlink()
        except FileNotFoundError:
            pass
        raise


def extract_tar(
    tar_path: Path,
    extract_to: Path,
    overwrite: bool = False,
    *,
    status_fn: Any = None,
) -> None:
    """
    Extract a TAR archive to a target directory.

    When ``status_fn`` exposes ``.bar()``, the extraction progress is
    rendered inside the active workflow panel instead of via tqdm.
    """
    import tarfile

    if not tar_path.is_file():
        raise FileNotFoundError(f"TAR file not found: {tar_path}")

    if status_fn is None:
        status_fn = get_download_status_fn()

    try:
        with tarfile.open(tar_path) as tf:
            members = tf.getmembers()

            extract_to.mkdir(parents=True, exist_ok=True)

            # Filter out already-extracted members when not overwriting
            if overwrite:
                to_extract = members
            else:
                to_extract = [m for m in members if not (extract_to / m.name).exists()]

            if not to_extract:
                LOGGER.debug(f"Cached: {tar_path.name} (all extracted)")
                return

            LOGGER.info(f"Extracting {tar_path.name}...")

            # Use extractall for sequential I/O (much faster than per-file extract)
            num_to_extract = len(to_extract)
            if _has_workflow_bar(status_fn):
                _count = [0]
                with status_fn.bar(f"Extracting {tar_path.name}", num_to_extract) as advance:
                    def _filter_with_progress(member, dest_path):
                        result = tarfile.data_filter(member, dest_path)
                        _count[0] += 1
                        advance(1)
                        return result
                    tf.extractall(path=extract_to, members=to_extract, filter=_filter_with_progress)
            else:
                _pbar = tqdm(total=num_to_extract, desc=f"Extracting {tar_path.name}")
                def _filter_with_tqdm(member, dest_path):
                    result = tarfile.data_filter(member, dest_path)
                    _pbar.update(1)
                    return result
                try:
                    tf.extractall(path=extract_to, members=to_extract, filter=_filter_with_tqdm)
                finally:
                    _pbar.close()

    except tarfile.TarError:
        LOGGER.error(f"Corrupt TAR: {tar_path.name}")
        try:
            tar_path.unlink()
        except FileNotFoundError:
            pass
        raise


def download_hf_dataset(repo_id: str, dest: Path, overwrite: bool = False, status_fn: Any = None) -> None:
    """
    Download a dataset from HuggingFace Hub to the given destination.

    Requires ``huggingface_hub`` to be installed (``pip install huggingface_hub``).

    Args:
        repo_id: HuggingFace dataset repo ID (e.g. "user/dataset").
        dest: Local directory to save the dataset into.
        overwrite: If True, re-download even if *dest* already exists.
        status_fn: Optional workflow status callback with ``.bar()`` support.
    """

    dest_exists = dest.exists() or dest.with_suffix('').exists()

    # Check for incomplete downloads in the HF cache — if any .incomplete files
    # remain, the previous download was interrupted and should be resumed.
    hf_cache_dir = dest / ".cache" / "huggingface" / "download"
    has_incomplete = hf_cache_dir.is_dir() and any(hf_cache_dir.rglob("*.incomplete"))

    if dest_exists and not overwrite and not has_incomplete:
        LOGGER.debug(f"HF dataset already present at {dest}")
        return

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        LOGGER.info("Installing huggingface_hub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import HfApi, snapshot_download

    from huggingface_hub.hf_api import RepoFile

    # Get file list with real sizes upfront
    api = HfApi()
    files = [
        f for f in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True)
        if isinstance(f, RepoFile)
    ]
    num_files = len(files)
    total_size = sum(f.size or (f.lfs.size if f.lfs else 0) for f in files)

    LOGGER.info(f"Downloading HuggingFace dataset {repo_id} "
                f"({num_files} files, {total_size / 1e9:.1f} GB) ...")

    if _has_workflow_bar(status_fn):
        # Use the Rich workflow bar so progress is visible inside the eval panel.
        _ctx = status_fn.bar(
            f"Downloading {repo_id} ({num_files} files)",
            total_size,
            unit="B",
        )
        _advance = _ctx.__enter__()

        import time as _time

        from tqdm.auto import tqdm as base_tqdm

        class _WorkflowTqdm(base_tqdm):
            """tqdm shim that forwards byte progress to the workflow bar."""
            _FLUSH_BYTES = 1 << 20   # flush to Rich every 1 MiB
            _FLUSH_SECS = 0.08       # … or every 80 ms (keeps spinner ~12 fps)

            def __init__(self, *args, **kwargs):
                kwargs.pop("name", None)
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)
                self._pending = 0
                self._last_flush = _time.monotonic()

            def update(self, n=1):
                self._pending += n
                now = _time.monotonic()
                if self._pending >= self._FLUSH_BYTES or (now - self._last_flush) >= self._FLUSH_SECS:
                    _advance(self._pending)
                    self._pending = 0
                    self._last_flush = now
                return super().update(n)

            def close(self):
                if self._pending > 0:
                    _advance(self._pending)
                    self._pending = 0
                super().close()

        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(dest),
                tqdm_class=_WorkflowTqdm,
            )
        finally:
            _ctx.__exit__(None, None, None)
    else:
        from tqdm.auto import tqdm as base_tqdm

        class _TqdmKnownTotal(base_tqdm):
            """tqdm wrapper that injects pre-computed totals for HF progress bars."""
            _lock_total = False

            def __init__(self, *args, **kwargs):
                kwargs.pop("name", None)
                desc = kwargs.get("desc", "")
                if desc.startswith("Downloading"):
                    kwargs["total"] = total_size
                    kwargs["desc"] = "Downloading"
                elif desc.startswith("Fetching"):
                    kwargs["total"] = num_files
                    kwargs["desc"] = f"Fetching {num_files} files"
                super().__init__(*args, **kwargs)
                if desc.startswith("Downloading"):
                    self._lock_total = True

            def __setattr__(self, name, value):
                if name == "total" and self._lock_total:
                    return
                super().__setattr__(name, value)

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            tqdm_class=_TqdmKnownTotal,
        )

    LOGGER.debug(f"HF dataset ready at {dest}")


def _hf_subfolder_file_count(repo_id: str, subfolder: str) -> int:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.hf_api import RepoFile

        api = HfApi()
        files = [
            f
            for f in api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=subfolder,
                recursive=True,
            )
            if isinstance(f, RepoFile)
        ]
        return len(files)
    except Exception:
        return 0


def snapshot_download_hf_subfolder(
    repo_id: str,
    subfolder: str,
    dest_root: Path,
    *,
    status_fn: Any = None,
    description: str | None = None,
) -> None:
    """Download a Hugging Face dataset subfolder with one aggregated progress task."""
    subfolder = str(subfolder).strip("/")
    if not subfolder:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        LOGGER.info("Installing huggingface_hub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    snapshot_kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": str(dest_root),
        "allow_patterns": [f"{subfolder}/**"],
    }
    num_files = _hf_subfolder_file_count(repo_id, subfolder)
    progress_description = description or f"Downloading {repo_id}/{subfolder}"

    def _make_tqdm_aggregator(rich_tqdm: type) -> type:
        shared_fetch_task = [None]

        class _TqdmAggregated(rich_tqdm):
            """Collapse Hugging Face byte/fetch tqdm instances into one file-count task."""

            _lock = None

            @classmethod
            def get_lock(cls):
                if cls._lock is None:
                    from threading import RLock

                    cls._lock = RLock()
                return cls._lock

            @classmethod
            def set_lock(cls, lock):
                cls._lock = lock

            def __init__(self, iterable=None, *args: Any, **kwargs: Any) -> None:
                kwargs.pop("name", None)
                kwargs.pop("disable", None)
                kwargs.pop("unit_scale", None)
                desc = str(kwargs.get("desc", ""))
                self._iterable = iterable
                self._total = int(kwargs.get("total") or 0)
                self.n = 0
                self._task_id = None

                if desc.startswith("Downloading"):
                    return

                if desc.startswith("Fetching"):
                    if num_files > 0:
                        kwargs["total"] = num_files
                        kwargs["desc"] = f"Fetching {num_files} files"
                    elif "desc" not in kwargs:
                        kwargs["desc"] = progress_description

                    if shared_fetch_task[0] is None:
                        super().__init__(iterable, *args, **kwargs)
                        shared_fetch_task[0] = self._task_id
                    else:
                        self._task_id = shared_fetch_task[0]
                    return

                super().__init__(iterable, *args, **kwargs)

            @property
            def total(self):
                return self._total

            @total.setter
            def total(self, value):
                self._total = int(value) if value else 0

            def update(self, n=1) -> None:
                if n is None:
                    return
                n = int(n)
                if n == 0:
                    return
                if self._task_id is not None:
                    super().update(n)
                else:
                    self.n += n

            def refresh(self) -> None:
                pass

            def set_description(self, desc: str, refresh: bool = True) -> None:
                if self._task_id is not None:
                    super().set_description(desc, refresh=refresh)

            def close(self) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc: Any):
                pass

            def __iter__(self):
                if self._iterable is None:
                    return self
                for item in self._iterable:
                    yield item
                    self.update(1)

            def __len__(self):
                if hasattr(self._iterable, "__len__"):
                    return len(self._iterable)
                raise TypeError

        return _TqdmAggregated

    if status_fn is not None and callable(getattr(status_fn, "tqdm_proxy", None)):
        with status_fn.tqdm_proxy(progress_description, unit="files") as rich_tqdm:
            snapshot_download(tqdm_class=_make_tqdm_aggregator(rich_tqdm), **snapshot_kwargs)
        return

    from boxmot.utils.rich.workflow.progress import RichTqdm

    snapshot_download(tqdm_class=_make_tqdm_aggregator(RichTqdm), **snapshot_kwargs)


def download_hf_dataset_subfolder(
    repo_id: str,
    subfolder: str,
    dest_root: Path,
    overwrite: bool = False,
    status_fn: Any = None,
) -> None:
    """Download a specific subfolder from a Hugging Face dataset repo."""
    subfolder = str(subfolder).strip("/")
    if not subfolder:
        return

    target = dest_root / subfolder
    marker = target / ".hf_download_complete"
    if not overwrite and marker.exists():
        LOGGER.debug(f"HF dataset subfolder already present at {target}")
        return
    if not overwrite and target.is_dir() and any(path.name != ".hf_download_complete" for path in target.iterdir()):
        marker.touch()
        LOGGER.debug(f"HF dataset subfolder already populated at {target}")
        return

    message = f"Downloading {repo_id}/{subfolder} ..."
    if status_fn is not None:
        status_fn(message)
    else:
        print_text(message, stderr=True)

    snapshot_download_hf_subfolder(repo_id, subfolder, dest_root, status_fn=status_fn)

    # Mark download as complete so subsequent runs skip it
    target.mkdir(parents=True, exist_ok=True)
    marker.touch()
    LOGGER.debug(f"HF dataset subfolder ready at {target}")


def download_eval_data(
    *,
    runs_url: Optional[str] = None,
    dataset_url: str,
    dataset_dest: Path,
    overwrite: bool = False,
    runs_check_path: Optional[Path] = None,
    status_fn: Callable[[str], None] | None = None,
) -> None:
    """
    Download and extract benchmark evaluation data.
    If `runs_url` is truthy, downloads+unzips runs.zip; otherwise skips it.
    If `runs_check_path` exists, skips the runs download entirely.
    Always downloads+unzips the benchmark data.
    """
    message = "Setting up evaluation data..."
    if status_fn is not None:
        status_fn(message)
    else:
        print_text(message, stderr=True)

    # Optional runs data — skip if user already has their own dets/embs
    if runs_url:
        if runs_url.startswith("hf://"):
            parts = runs_url[len("hf://"):].split("/")
            repo_id = "/".join(parts[:2])
            subfolder = "/".join(parts[2:])
            if subfolder:
                download_hf_dataset_subfolder(
                    repo_id,
                    subfolder,
                    Path("."),
                    overwrite=overwrite,
                    status_fn=status_fn,
                )
            else:
                download_hf_dataset(repo_id, Path("."), overwrite=overwrite, status_fn=status_fn)
        elif runs_check_path is not None and Path(runs_check_path).exists():
            # Legacy ZIP workflow: skip if the expected directory already exists
            LOGGER.debug(f"Skipping runs.zip download: {runs_check_path} already exists.")
        else:
            runs_zip = download_file(
                runs_url, Path("runs.zip"), overwrite=overwrite, status_fn=status_fn
            )
            extract_zip(runs_zip, Path("."), overwrite=overwrite, status_fn=status_fn)

    if not dataset_url:
        return

    # HuggingFace dataset (hf://owner/repo[/subfolder])
    if dataset_url.startswith("hf://"):
        parts = dataset_url[len("hf://"):].split("/")
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        if subfolder:
            download_hf_dataset_subfolder(
                repo_id,
                subfolder,
                dataset_dest,
                overwrite=overwrite,
                status_fn=status_fn,
            )
        else:
            download_hf_dataset(repo_id, dataset_dest, overwrite=overwrite, status_fn=status_fn)

        # Extract any tar archives found after HF download (e.g. SportsMOT)
        tar_search_root = dataset_dest / subfolder if subfolder else dataset_dest
        tar_files = list(tar_search_root.rglob("*.tar")) + list(tar_search_root.rglob("*.tar.gz"))
        for tar_file in sorted(tar_files):
            # Quick check: skip tar if its stem directory already exists and is non-empty
            stem = tar_file.stem
            if stem.endswith(".tar"):
                stem = stem[:-4]  # handle .tar.gz
            extracted_dir = tar_file.parent / stem
            if not overwrite and extracted_dir.is_dir() and any(extracted_dir.iterdir()):
                LOGGER.debug(f"Cached: {tar_file.name} (directory already exists)")
                continue
            extract_tar(tar_file, tar_file.parent, overwrite=overwrite, status_fn=status_fn)
        return

    # benchmark ZIP
    benchmark_zip = download_file(
        dataset_url, dataset_dest, overwrite=overwrite, status_fn=status_fn
    )
    extract_zip(benchmark_zip, dataset_dest.parent, overwrite=overwrite, status_fn=status_fn)

    LOGGER.debug(f"Benchmark data ready at: {dataset_dest.parent}")
