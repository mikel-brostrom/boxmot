#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Utility script to download and extract BoxMOT releases and MOT evaluation tools.
"""

import concurrent.futures
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional
from zipfile import BadZipFile, ZipFile

import gdown
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from boxmot.utils import logger as LOGGER
from boxmot.utils.rich.ui import print_text


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


def _patch_trackeval_numpy_aliases(dest: Path) -> None:
    """Patch deprecated NumPy builtin aliases in a downloaded TrackEval tree."""
    package_root = dest / "trackeval"
    if not package_root.exists():
        return

    replacements = (
        (r"\bnp\.float\b", "float"),
        (r"\bnp\.int\b", "int"),
        (r"\bnp\.bool\b", "bool"),
        (r"\bnp\.object\b", "object"),
        (r"\bnp\.str\b", "str"),
    )

    for py_file in package_root.rglob("*.py"):
        content = py_file.read_text()
        patched = content
        for pattern, repl in replacements:
            patched = re.sub(pattern, repl, patched)
        if patched != content:
            py_file.write_text(patched)


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
        # Google Drive: use gdown (handles confirm tokens automatically).
        # gdown's tqdm bar goes straight to stderr and would corrupt an
        # active Rich Live region. When a workflow callback exposes a
        # ``tqdm_proxy`` helper, monkey-patch the gdown module's tqdm with
        # a Rich-backed shim so the progress is rendered inside the panel.
        if status_fn is not None and callable(getattr(status_fn, "tqdm_proxy", None)):
            import importlib

            # ``gdown.download`` is the public function exposed in
            # ``gdown/__init__.py``, so ``import gdown.download`` returns the
            # function (not the submodule). Use importlib to reach the
            # underlying ``gdown.download`` module that owns ``import tqdm``.
            gdown_download_mod = importlib.import_module("gdown.download")

            original_tqdm_module = gdown_download_mod.tqdm
            # gdown also writes "Downloading...", "From:", "To:" etc with
            # ``print(..., file=sys.stderr)``. Replace ``print`` in gdown's
            # module namespace with a no-op for the duration of the call so
            # those messages don't leak to the terminal and corrupt the
            # active Rich Live region. ``contextlib.redirect_stderr`` is too
            # aggressive — it would also intercept the writes Rich is doing
            # for its own progress bar refreshes through other paths.
            original_print = gdown_download_mod.__dict__.get("print", print)

            def _silent_print(*args: Any, **kwargs: Any) -> None:
                return None

            with status_fn.tqdm_proxy(f"Downloading {dest.name}", unit="B") as rich_tqdm:

                class _TqdmShim:
                    tqdm = rich_tqdm

                    def __getattr__(self, name: str) -> Any:
                        return getattr(original_tqdm_module, name)

                gdown_download_mod.tqdm = _TqdmShim()
                gdown_download_mod.__dict__["print"] = _silent_print
                try:
                    gdown.download(
                        url=url,
                        output=str(dest),
                        quiet=False,
                        fuzzy=True,
                    )
                finally:
                    gdown_download_mod.tqdm = original_tqdm_module
                    if original_print is print:
                        gdown_download_mod.__dict__.pop("print", None)
                    else:
                        gdown_download_mod.__dict__["print"] = original_print
        else:
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
def _sync_trackeval_dataset_overlays(dest: Path) -> None:
    """Overlay the vendored TrackEval OBB dataset adapters with the tracked copies."""
    _patch_trackeval_numpy_aliases(dest)

    source_dir = Path(__file__).resolve().parent / "evaluation"
    overlays = [
        (source_dir / "custom_mot_challenge_obb.py", dest / "trackeval" / "datasets" / "mmot_rgb.py"),
        (source_dir / "trackeval_datasets_init.py", dest / "trackeval" / "datasets" / "__init__.py"),
    ]
    for src, dst in overlays:
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and src.resolve() == dst.resolve():
            continue
        shutil.copyfile(src, dst)


def download_trackeval(dest: Path, branch: str = "main", overwrite: bool = False) -> None:
    """
    Download and set up the TrackEval repository into the given destination folder.

    Args:
        dest (Path): target directory for TrackEval (e.g. boxmot/engine/trackeval)
        branch (str): Git branch to download (default "master")
        overwrite (bool): if True, force re-download even if dest already exists
    """
    # If already exists and we're not overwriting, skip
    if dest.exists() and not overwrite:
        _sync_trackeval_dataset_overlays(dest)
        LOGGER.debug("TrackEval already present")
        return

    LOGGER.info("Downloading TrackEval...")
    repo_url = "https://github.com/Annzstbl/MMOT"
    zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
    zip_file = dest.parent / f"MMOT-{branch}.zip"

    # Download the archive
    zip_path = download_file(zip_url, zip_file, overwrite=overwrite)

    # Extract into the parent folder
    extract_zip(zip_path, dest.parent, overwrite=overwrite)

    # GitHub unpacks to "MMOT-<branch>"; TrackEval lives inside that repo.
    extracted = None
    for d in dest.parent.iterdir():
        if d.is_dir() and d.name.lower().startswith("mmot") and d.name.lower().endswith(f"-{branch}"):
            extracted = d
            break

    if extracted is None:
        LOGGER.warning("Couldn't locate extracted MMOT folder")
    else:
        trackeval_src = extracted / "TrackEval"
        if not trackeval_src.exists():
            LOGGER.warning("Couldn't locate TrackEval inside extracted MMOT archive")
        else:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(trackeval_src), str(dest))
        if extracted.exists():
            shutil.rmtree(extracted)

    # Clean up the downloaded zip
    try:
        zip_file.unlink()
    except FileNotFoundError:
        pass

    _sync_trackeval_dataset_overlays(dest)

    LOGGER.debug("TrackEval setup complete")

    
def download_hf_dataset(repo_id: str, dest: Path, overwrite: bool = False) -> None:
    """
    Download a dataset from HuggingFace Hub to the given destination.

    Requires ``huggingface_hub`` to be installed (``pip install huggingface_hub``).

    Args:
        repo_id: HuggingFace dataset repo ID (e.g. "user/dataset").
        dest: Local directory to save the dataset into.
        overwrite: If True, re-download even if *dest* already exists.
    """

    dest_exists = dest.exists() or dest.with_suffix('').exists()

    if dest_exists and not overwrite:
        LOGGER.debug(f"HF dataset already present at {dest}")
        return

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        LOGGER.info("Installing huggingface_hub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import HfApi, snapshot_download

    from tqdm.auto import tqdm as base_tqdm
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
        local_dir=str(dest.parent),
        tqdm_class=_TqdmKnownTotal,
    )
    LOGGER.debug(f"HF dataset ready at {dest}")


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
    Download & extract TrackEval evaluation data.
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
        if runs_check_path is not None and Path(runs_check_path).exists():
            LOGGER.debug(f"Skipping runs.zip download: {runs_check_path} already exists.")
        else:
            runs_zip = download_file(
                runs_url, Path("runs.zip"), overwrite=overwrite, status_fn=status_fn
            )
            extract_zip(runs_zip, Path("."), overwrite=overwrite, status_fn=status_fn)

    if not dataset_url:
        return

    # HuggingFace dataset (hf://owner/repo/subfolder)
    if dataset_url.startswith("hf://"):
        parts = dataset_url[len("hf://"):].split("/")
        repo_id = "/".join(parts[:2])        # e.g. "Fleyderer/FastTracker-Benchmark-MOT"
        download_hf_dataset(repo_id, dataset_dest, overwrite=overwrite)
        return

    # benchmark ZIP
    benchmark_zip = download_file(
        dataset_url, dataset_dest, overwrite=overwrite, status_fn=status_fn
    )
    extract_zip(benchmark_zip, dataset_dest.parent, overwrite=overwrite, status_fn=status_fn)

    LOGGER.debug(f"Benchmark data ready at: {dataset_dest.parent}")
